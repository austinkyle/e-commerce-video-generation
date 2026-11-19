from __future__ import annotations

import base64
import json
import mimetypes
import re
from pathlib import Path

import httpx

from ecom_video.config import require_key
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
                "product_interaction": {"type": "string"}, "hook_or_payoff": {"type": "string"},
                "continuity_group": {"type": "string"},
                "continuity_notes": {"type": "array", "items": {"type": "string"}},
                "uncertainties": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["index", "start_seconds", "end_seconds", "creative_role", "visual_description",
                         "starting_state", "action", "ending_state", "camera", "product_interaction",
                         "hook_or_payoff", "continuity_group", "continuity_notes", "uncertainties"],
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
) -> AnalysisResult:
    if not frames:
        raise ValueError("At least one representative frame is required")
    if len(frames) > 24:
        raise ValueError("At most 24 representative frames may be analyzed")
    duration_note = (
        f"Source-video duration: {reference_duration_seconds:.2f} seconds. "
        if reference_duration_seconds is not None else ""
    )
    content: list[dict] = [{"type": "input_text", "text": (
        "Analyze this complete short-form product video as reference for an original execution. "
        "Identify a small set of distinct meaningful shots with approximate absolute source-video timestamps, creative role, "
        "starting state, concrete physical action, ending state or payoff, camera framing/movement, product interaction, "
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
        "Assign the same stable continuity_group to shots that share a setting and recurring person, animal, or product; "
        "describe those anchors in continuity_notes. Use a different group for an intentional location change. Do not force "
        "identical settings across distinct locations. Keep movement physically plausible and preserve visible product details.\n"
        f"Creator instructions: {instruction}\nProduct facts (may be incomplete): {product_facts}\n"
        f"Selected hook direction: {hook_direction(hook)}\n"
        "Make each shot's creative_role distinct and explain its unique contribution to one coherent advertisement. "
        "Do not repeat the same product pose. For supplied imagery, treat visible product details as authoritative.\n"
        f"{duration_note}Representative frames span the full source video ({len(frames)} total)."
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
    response = httpx.post(
        URL,
        headers={"Authorization": f"Bearer {require_key('OPENAI_API_KEY')}"},
        json={
            "model": MODEL,
            "input": [{"role": "user", "content": [{"type": "input_text", "text": (
                "Act as a practical ecommerce commercial director. Interview only for missing details that affect product identity, physical plausibility, safety, or the visual payoff. Never repeat a fact or question present in the brief or previous answers. Treat 'you decide' and 'surprise me' as permission to choose and state a reasonable assumption. Packaging questions are only relevant when packaging appears. Ask at most five concise questions in one turn. If critical information is still missing, return questions and an empty shots array. Otherwise return a coherent sequence of three to five distinct purposeful shots with clear hook, reveal/demo and payoff roles as appropriate. For every shot specify visible starting state, ordered physical action, ending payoff, camera, environment implied by the brief, product interaction, and continuity anchors. For physical demonstrations, state which working face or product edge meets the surface, the grip and support, the visible contact point, and the single primary movement where practical. Keep beginning and ending states physically possible. If a difficult transformation is unreliable in one continuous take, recommend a simpler action or separate before-and-after shots while preserving the story. Choose a camera angle that makes the product mechanism and contact visible. Never invent product functionality. Keep silent audio as the default. Source times must be null.\n"
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
                  "Change only the shots required by the user's request; preserve all other shots exactly. Preserve the original concept, confirmed product specifications, hook unless explicitly changed, prior interview answers, and continuity facts. Do not restart the interview or invent product features. Each shot needs its creative role, visible scene, starting state, concrete physical action, ending payoff, camera, product interaction, continuity group and notes, and source timestamps when already present. "
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
