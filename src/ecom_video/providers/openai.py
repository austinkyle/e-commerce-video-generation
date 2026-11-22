from __future__ import annotations

import base64
import json
import mimetypes
import re
from pathlib import Path

import httpx

from ecom_video.config import require_key
from ecom_video.creative_playbook import CONTINUITY_RULES, STARTING_STATE_RULES, format_lesson_reminders
from ecom_video.models import AnalysisResult, hook_direction

MODEL = "gpt-5.4-mini-2026-03-17"
URL = "https://api.openai.com/v1/responses"
SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "summary": {"type": "string"},
        "questions": {"type": "array", "items": {"type": "string"}},
        "shots": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "properties": {
                "index": {"type": "integer", "minimum": 1},
                "start_seconds": {"type": ["number", "null"], "minimum": 0},
                "end_seconds": {"type": ["number", "null"], "exclusiveMinimum": 0},
                "creative_role": {"type": "string"},
                "visual_description": {"type": "string"},
                "starting_state": {"type": "string"}, "action": {"type": "string"},
                "ending_state": {"type": "string"}, "camera": {"type": "string"},
                "target_duration_seconds": {"type": ["integer", "null"], "minimum": 1, "maximum": 15},
                "edit_duration_seconds": {"type": ["number", "null"], "minimum": 0.5, "maximum": 15},
                "product_interaction": {"type": "string"}, "hook_or_payoff": {"type": "string"},
                "audio_requested": {"type": "boolean"},
                "audio_direction": {"type": "string"},
                "continuity_group": {"type": "string"},
                "character_groups": {"type": "array", "items": {"type": "string"}},
                "continuity_notes": {"type": "array", "items": {"type": "string"}},
                "uncertainties": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["index", "start_seconds", "end_seconds", "creative_role", "visual_description",
                         "starting_state", "action", "ending_state", "target_duration_seconds", "edit_duration_seconds", "camera", "product_interaction",
                         "hook_or_payoff", "audio_requested", "audio_direction", "continuity_group", "character_groups", "continuity_notes", "uncertainties"],
        }},
    }, "required": ["summary", "questions", "shots"],
}

REVISION_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "summary": {"type": "string"},
        "changed_shot_indices": {"type": "array", "items": {"type": "integer", "minimum": 1}},
        "shots": SCHEMA["properties"]["shots"],
    },
    "required": ["summary", "changed_shot_indices", "shots"],
}


def _image_data(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def analyze_reference(
    frames: list[Path], instruction: str, product_facts: str = "",
    reference_duration_seconds: float | None = None,
    hook: str = "recommend",
    clip_count: int | None = None,
    seconds_per_clip: int | None = None,
    duration_selection: str = "fixed",
    reference_has_audio: bool | None = None,
) -> AnalysisResult:
    if not frames:
        raise ValueError("At least one representative frame is required")
    if len(frames) > 24:
        raise ValueError("At most 24 representative frames may be analyzed")
    duration_note = (
        f"Source-video duration: {reference_duration_seconds:.2f} seconds. "
        if reference_duration_seconds is not None else ""
    )
    if clip_count is not None and seconds_per_clip is not None:
        if duration_selection == "director":
            target_note = (
                f"Requested target output: up to {clip_count} clips. The creator selected Director decides for duration; choose each target_duration_seconds (1-15s) and edit_duration_seconds individually based on action completion. Do not apply a fixed duration to every shot. "
            )
        elif hook in {"in_store_reaction", "wet_works"}:
            target_note = (
                f"Requested target clip count {clip_count} is a maximum, not a required count; the interface duration {seconds_per_clip}s is a fallback only. "
                "Choose one generated clip per meaningful cut/action and set each target_duration_seconds (1-15s) and edit_duration_seconds "
                "(estimated trimmed use in the final edit) separately. Never equate source timestamps, generated duration, and final edit duration. "
            )
        else:
            target_note = (
                f"Requested target output: up to {clip_count} generated clips, with {seconds_per_clip} seconds as the fallback per target clip. "
                "These target durations are separate from absolute source-video timestamps and intended final edit durations. "
            )
    else:
        target_note = ""
    reference_audio_note = (
        "Reference audio stream metadata: present. The supplied analysis contains video frames rather than the audio waveform; use visible physical actions and creator instructions to plan plausible synchronized sounds, and do not claim to identify the source track or dialogue. "
        if reference_has_audio else
        "Reference audio stream metadata: absent or unavailable. Select only natural sounds supported by visible physical actions and creator instructions. "
    )
    in_store_analysis_guidance = ""
    if hook in {"in_store_reaction", "wet_works"}:
        in_store_analysis_guidance = (
            "For In-Store Reaction, keep the immediate reaction, motivated gesture, and quick product-revealing reframe "
            "together in one target clip when the reference and available duration support it. Derive pacing from the reference; "
            "do not split reaction gestures without a genuinely distinct scene. Allocate later clips to product-specific actions "
            "that advance the story."
        )
    content: list[dict] = [{"type": "input_text", "text": (
        "Analyze this complete short-form product video as reference for an original execution. "
        "Identify a small set of distinct meaningful shots with approximate absolute source-video timestamps, creative role, "
        "starting state, concrete physical action, ending state or payoff, camera framing/movement, product interaction, "
        "and a per-shot native-audio decision. Set audio_requested true only when synchronized natural sound materially supports an audible action, reaction, mechanism, or ambience; set it false for static shots, no useful sound, or explicit silence. In audio_direction name only plausible sounds and their physical sync points. Never invent dialogue or music unless the creator requests it, and never claim sound is present until the provider output is verified. "
        "continuity group/anchors, and uncertainties. Do not invent product capabilities. Return targeted questions only "
        "when missing facts materially affect physical plausibility or fidelity. Preserve structure, not exact footage. "
        "Treat the complete creator instructions as authoritative. Keep the summary concise (about 700 characters) "
        "and use it for shared requirements such as product identity, character appearance, environment, lighting, "
        "continuity, realism, and prohibitions. Every timestamp is measured from the start of the source video, "
        "never the duration of a generated target clip. Match intervals to timestamped frame evidence. Do not reuse "
        "one interval for distinct scenes. Only give a source interval when a shot corresponds to visible reference footage. "
        "For a shot created primarily from written instructions with no exact matching segment, set both timestamps to null. "
        "Keep all intervals within the source-video duration. Adapt each shot's visual description, action, and product interaction "
        "to the requested execution; use source footage only for allowed framing and rhythm. Do not carry source-only "
        "subjects or actions into the target shot when the creator instructions replace them. Keep each shot field "
        "concise and preserve its scene-specific requirements in the relevant fields. Give every shot a purposeful role "
        "in a progression such as hook, demonstration, or reaction when the source and brief support it. Avoid repetitive "
        "product shots. Specify visible states before and after each action, and give reactions a believable visual payoff. "
        "Assign continuity_group to the recurring environment only; use a different group for intentional location changes. "
        "Assign stable character_groups to each recurring target person or animal across every scene, including location, "
        "camera angle, pose, and wardrobe changes. Reuse the same character group when the creator requests the same "
        "person in a new outfit or environment. Base these groups on the requested target cast, not people who appear only "
        "in the source video; creator-requested character replacements take precedence over source identities. Describe visual "
        "anchors in continuity_notes, and the image workflow will pass the earliest established starting image as the identity reference. "
        "Keep movement physically plausible and preserve visible product details.\n"
        f"{reference_audio_note}Creator instructions: {instruction}\nProduct facts (may be incomplete): {product_facts}\n"
        f"Selected hook direction: {hook_direction(hook)}\n"
        "Make each shot's creative_role distinct and explain its unique contribution to one coherent advertisement. "
        "Do not repeat the same product pose. For supplied imagery, treat visible product details as authoritative.\n"
        f"{duration_note}{target_note}Representative frames span the full source video ({len(frames)} total). "
        "Cover meaningful scenes from the opening through the actual final scene and use the ending for a real payoff when present; "
        f"do not stop analysis early or omit ending footage to match target clip count. {in_store_analysis_guidance} "
        "For each scene describe the exact camera movement, physical interaction, mechanism and causal result. "
        "Preserve reference visual rhythm and mechanics while describing original target footage. "
        f"{STARTING_STATE_RULES} {CONTINUITY_RULES} {format_lesson_reminders(hook)} "
        "Keep starting_state limited to what is visibly true at frame zero; put subsequent movement only in action."
    )}]
    for frame in frames:
        match = re.search(r"-(\d+(?:\.\d+)?)s\.", frame.name)
        note = f"Frame timestamp in source video: {float(match.group(1)):.3f} seconds. " if match else ""
        content.append({"type": "input_text", "text": note + "Reference video frame follows."})
        content.append({"type": "input_image", "image_url": _image_data(frame), "detail": "high"})
    response = httpx.post(
        URL,
        headers={"Authorization": f"Bearer {require_key('OPENAI_API_KEY')}"},
        json={"model": MODEL, "input": [{"role": "user", "content": content}],
              "text": {"format": {"type": "json_schema", "name": "reference_analysis", "strict": True, "schema": SCHEMA}}},
        timeout=180,
    )
    response.raise_for_status()
    payload = response.json()
    analyze_reference.last_usage = payload.get("usage", {})
    text = next((item.get("text") for output in payload.get("output", [])
                 for item in output.get("content", []) if item.get("type") == "output_text"), None)
    if not text:
        raise RuntimeError("Analysis returned no structured result")
    return AnalysisResult.model_validate_json(text)


analyze_reference.last_usage = {}


def direct_original(instructions: str, answers: list[str], hook: str = "recommend") -> AnalysisResult:
    """Produce either a small set of targeted questions or a validated original shot plan."""
    in_store_director_guidance = ""
    if hook in {"in_store_reaction", "wet_works"}:
        in_store_director_guidance = (
            "Keep the estimated final edit length distinct from generated duration. "
            "Do not split reaction gestures into separate shots unless a genuinely distinct scene intervenes. "
            "Do not spend the first second establishing the store or hold an empty reaction. "
            "Follow with product-specific scenes that advance the story; unboxing and setup are optional. "
            "For demonstrations identify the actual mechanism, trigger, force or movement, and resulting state. "
            "Assign recurring target cast with character_groups independently of setting continuity. Later scenes may use different people only when their character_groups say so. "
            "Do not carry the hook shopper into later clips unless the scene fields say they recur. "
            "Use a generic or fictional store unless actual distribution is established."
        )
    response = httpx.post(
        URL,
        headers={"Authorization": f"Bearer {require_key('OPENAI_API_KEY')}"},
        json={
            "model": MODEL,
            "input": [{"role": "user", "content": [{"type": "input_text", "text": (
                "Act as a practical ecommerce commercial director. Interview only for missing details that affect product identity, physical plausibility, safety, or the visual payoff. Never repeat a fact or question present in the brief or previous answers. Treat 'you decide' and 'surprise me' as permission to choose and state a reasonable assumption. Packaging questions are only relevant when packaging appears. Ask at most five concise questions in one turn. If critical information is still missing, return questions and an empty shots array. Otherwise choose a purposeful shot count from distinct actions/cuts, typically three to six without padding to a fixed count. For every shot specify visible starting state, ordered physical action, ending payoff, camera movement, environment, product interaction, continuity anchors, audio_requested, audio_direction, target_duration_seconds, and edit_duration_seconds. Choose per-shot durations when the creator requests Director decides; otherwise use null when the creator has not specified a duration. Keep beginning and ending states physically possible. Never invent product functionality. Plan only useful, natural, synchronized sound; spoken dialogue requires an explicit request, and music is not generated by this workflow. Output remains silent for shots where no native sound is selected. Source times must be null. "
                f"{STARTING_STATE_RULES} {CONTINUITY_RULES} {format_lesson_reminders(hook)} Keep starting_state limited to frame zero and put later movements only in action.\n"
                f"{in_store_director_guidance}\n"
                f"Hook: {hook_direction(hook)}\nInitial brief: {instructions}\nPrevious answers: {json.dumps(answers)}"
            )}]}],
            "text": {"format": {"type": "json_schema", "name": "original_director_turn", "strict": True, "schema": SCHEMA}},
        }, timeout=180,
    )
    response.raise_for_status()
    payload = response.json()
    direct_original.last_usage = payload.get("usage", {})
    text = next((item.get("text") for output in payload.get("output", []) for item in output.get("content", []) if item.get("type") == "output_text"), None)
    if not text:
        raise RuntimeError("Creative Director returned no structured result")
    return AnalysisResult.model_validate_json(text)


direct_original.last_usage = {}


def revise_creative_plan(instruction: str, plan: dict, confirmed_context: dict) -> dict:
    """Revise only the requested shots while returning a complete structured plan."""
    response = httpx.post(
        URL,
        headers={"Authorization": f"Bearer {require_key('OPENAI_API_KEY')}"},
        json={"model": MODEL,
              "input": [{"role": "user", "content": [{"type": "input_text", "text": (
                  "Revise this existing ecommerce video plan. Return every shot in the same order and with the same indices. "
                  "Change only the shots required by the user's request; preserve all other shots exactly. Preserve the original concept, confirmed product specifications, hook unless explicitly changed, prior interview answers, continuity facts, target_duration_seconds, edit_duration_seconds, and audio_requested unless the requested shot change affects its audible actions. Do not restart the interview or invent product features. Each shot needs its creative role, visible scene, starting state, concrete physical action, ending payoff, camera, product interaction, audio_requested, audio_direction, continuity group and notes, and source timestamps when already present. Plan only useful natural sounds and synchronize them with physical actions; do not invent dialogue or music. "
                  "List only intentionally changed shot indices in changed_shot_indices. Keep summary coherent.\n"
                  f"Confirmed context: {json.dumps(confirmed_context, ensure_ascii=False)}\n"
                  f"Current plan: {json.dumps(plan, ensure_ascii=False)}\n"
                  f"Requested refinement: {instruction}"
              )}]}],
              "text": {"format": {"type": "json_schema", "name": "creative_plan_revision", "strict": True, "schema": REVISION_SCHEMA}}},
        timeout=180,
    )
    response.raise_for_status()
    payload = response.json()
    revise_creative_plan.last_usage = payload.get("usage", {})
    text = next((item.get("text") for output in payload.get("output", []) for item in output.get("content", []) if item.get("type") == "output_text"), None)
    if not text:
        raise RuntimeError("Creative plan revision returned no structured result")
    return json.loads(text)


revise_creative_plan.last_usage = {}
