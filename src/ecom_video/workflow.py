from __future__ import annotations

import json
import hashlib
import re
from decimal import Decimal
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Protocol

from pydantic import BaseModel, Field, model_validator

from ecom_video.costs import IMAGE_1K_USD, KIE_USD_PER_CREDIT, GenerationEstimate, estimate_generation, estimate_within_approved_amount
from ecom_video.creative_playbook import (
    IN_STORE_REACTION_IMAGE_RULES,
    IN_STORE_REACTION_VIDEO_RULES,
)
from ecom_video.media import make_mp4, make_silent_mp4, probe_video
from ecom_video.models import AnalysisResult, AudioMode, ShotPlan
from ecom_video.prompt_limits import KIE_VIDEO_PROMPT_SAFE_CHARACTERS, kie_video_prompt_length

AUTO_GENERATION_APPROVAL_LIMIT_USD = Decimal("0.50")
MAX_CREATIVE_INSTRUCTIONS = 20_000
IN_STORE_OPENING_IMAGE_RULES = (
    "Opening reaction image: show understated, spontaneous anticipation with relaxed facial features. "
    "Use the matching-scene reference to preserve identity, clothing, and setting only; do not copy its expression or pose."
)


def creative_instruction_length(instructions: str) -> int:
    """Match the browser textarea's UTF-16 character-counting behavior."""
    return len(instructions.encode("utf-16-le")) // 2


class PlannedClip(BaseModel):
    index: int = Field(ge=1)
    title: str
    source_shot_index: int = Field(ge=1)
    source_start_seconds: float | None = Field(default=None, ge=0)
    source_end_seconds: float | None = Field(default=None, gt=0)
    creative_role: str = ""
    visual_description: str
    starting_state: str = ""
    action: str
    product_interaction: str = ""
    ending_state: str = ""
    scene_payoff: str = ""
    camera: str
    continuity_group: str = ""
    character_groups: list[str] = Field(default_factory=list)
    continuity_notes: list[str] = Field(default_factory=list)
    duration_seconds: int = Field(ge=1, le=15)
    edit_duration_seconds: float | None = Field(default=None, ge=0.5, le=15)
    audio_mode: AudioMode = "auto"
    audio_requested: bool | None = None
    preserve_audio: bool = False
    audio_direction: str = "Silent output; no audio track requested."
    starting_image_prompt: str
    video_prompt: str
    environment_references: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_source_interval(self) -> "PlannedClip":
        if (self.source_start_seconds is None) != (self.source_end_seconds is None):
            raise ValueError("source start and end times must both be provided or both be null")
        if self.source_start_seconds is not None and self.source_end_seconds <= self.source_start_seconds:
            raise ValueError("source end time must be after source start time")
        return self

    @property
    def audio_enabled(self) -> bool:
        if self.audio_mode == "enabled":
            return True
        if self.audio_mode == "silent":
            return False
        # audio_requested is the Creative Director's Auto decision. None keeps
        # saved pre-audio plans compatible with their legacy preserve_audio flag.
        return self.audio_requested if self.audio_requested is not None else self.preserve_audio


class MultiClipPlan(BaseModel):
    summary: str
    environment: str
    instructions: str
    product_facts: str
    product_references: list[str] = Field(default_factory=list)
    continuity_references: list[str] = Field(default_factory=list)
    hook: str = "custom"
    clips: list[PlannedClip] = Field(min_length=1)


def _product_hidden_at_start(text: str) -> bool:
    nouns = r"(?:product|item|display|package|packaging|bowl|device|toy)"
    hidden_claims = (
        rf"\bno\s+{nouns}\b.{{0,32}}\bvisible\b",
        rf"\b{nouns}\b.{{0,48}}\b(?:not visible|(?:mostly )?outside (?:the )?(?:opening )?frame|off[- ]screen|concealed|hidden)\b",
        rf"\b(?:not visible|(?:mostly )?outside (?:the )?(?:opening )?frame|off[- ]screen|concealed|hidden)\b.{{0,48}}\b{nouns}\b",
    )
    for clause in re.split(r"[.!?;\n]", text):
        # Imperatives are prompt constraints, not affirmative statements of the
        # opening composition. Do not let them suppress useful identity anchors.
        if re.search(r"\b(?:do not|don't|never|avoid|exclude)\b", clause, re.I):
            continue
        if any(re.search(pattern, clause, re.I) for pattern in hidden_claims):
            return True
    return False


def _product_is_off_frame_at_start(clip: "PlannedClip") -> bool:
    return _product_hidden_at_start(clip.starting_state or clip.starting_image_prompt)


def starting_image_revision_prompt(plan: "MultiClipPlan", clip: "PlannedClip", instructions: str = "") -> str:
    """Refresh saved In-Store image guidance without changing the saved creative plan."""
    prompt = clip.starting_image_prompt
    in_store = plan.hook in {"in_store_reaction", "wet_works"}
    if in_store:
        if clip.index == 1:
            prompt = re.sub(
                r"\bboth look ahead with surprised expressions\b",
                "both look ahead with understated, natural anticipation",
                prompt,
                flags=re.IGNORECASE,
            )
            prompt = re.sub(
                r"Reference roles:.*?(?=\sShow only (?:this )?(?:initial state|details visible at the start of the video)\.|$)",
                "Reference roles: use the approved same-scene image for character identity, clothing, and warehouse environment only; do not copy its expression or pose. The coffee's upright held state follows the saved frame-zero direction. ",
                prompt,
                flags=re.IGNORECASE | re.DOTALL,
            )
            prompt += f" {IN_STORE_OPENING_IMAGE_RULES}"
        prompt += f" {IN_STORE_REACTION_IMAGE_RULES}"
    if instructions.strip():
        prompt += f"\nRequested revision: {instructions.strip()}"
    return prompt


def product_identity_clip_index(plan: MultiClipPlan) -> int:
    """Pick a stable product-visible frame to anchor image continuity."""
    ranked: list[tuple[int, int]] = []
    for clip in plan.clips:
        role = _provider_text(clip.creative_role).lower()
        score = 0
        if any(word in role for word in ("reveal", "demo", "demonstration", "product", "unbox", "close-up", "detail")):
            score += 4
        if any(word in role for word in ("hook", "discovery")):
            score -= 2
        # Only the affirmative frame-zero state controls visibility. Searching all
        # prompt text for negation misclassifies instructions such as "do not hide".
        if _product_hidden_at_start(clip.starting_state):
            score -= 8
        ranked.append((score, clip.index))
    return max(ranked, key=lambda item: (item[0], -item[1]))[1]


def _image_reference_prompt(
    prompt: str,
    has_product_reference: bool,
    has_continuity_reference: bool = False,
    reference_roles: list[str] | None = None,
) -> str:
    if not has_product_reference and not has_continuity_reference:
        return prompt
    directions = list(reference_roles or [])
    if not reference_roles:
        if has_product_reference:
            directions.append("Product references define product appearance only; ignore their backgrounds")
        if has_continuity_reference:
            directions.append("continuity references define faces, wardrobe, location, and pose")
    return f"{prompt}\nReferences: {'; '.join(directions)}. Follow the stated frame-zero scene."


def _provider_text(value: str) -> str:
    return " ".join(value.split())


def compile_video_prompt(plan: MultiClipPlan, clip: PlannedClip, product_reference_count: int = 0) -> str:
    """Compact an oversized saved prompt from its structured shot direction.

    The approved plan remains unchanged. Required facts are preserved whole; if
    they cannot fit, fail before any generation request instead of truncating.
    """
    notes = list(dict.fromkeys(_provider_text(note) for note in clip.continuity_notes if note.strip()))
    in_store = plan.hook in {"in_store_reaction", "wet_works"}
    if not 0 <= product_reference_count <= 6:
        raise ValueError("Grok video prompts support at most six additional product references")
    is_opening = in_store and clip.index == 1 and any(
        word in clip.creative_role.lower() for word in ("reaction", "hook", "reveal", "discovery")
    )
    ad_concept = plan.summary
    if is_opening:
        ad_concept = (
            "In-Store Reaction hook: preserve this scene's approved shopper, product, and retail setting; "
            "build one continuous reaction-to-reveal shot."
        )
    elif in_store:
        ad_concept = (
            "In-Store Reaction ad: the opening reaction/reveal is complete; this clip advances the product-specific story. "
            "Follow this clip's scene and continuity notes for its cast and setting; do not carry the hook shopper into this clip unless explicitly named."
        )
    duration_line = f"Generated clip length: {clip.duration_seconds}s." if in_store else ""
    if in_store and clip.edit_duration_seconds is not None:
        duration_line += f" Intended final edit use: {clip.edit_duration_seconds:g}s; the final edit may trim this generated clip."
    physical_instruction = (
        "Physical reliability: preserve the starting state and exact product mechanism; show the real contact, support, trigger, or force and its causal result. Keep movement physically plausible."
        if in_store else
        "Physical reliability: preserve the exact starting orientation; place the working surface toward the fabric with a visible contact point; show a natural grip and support. Use one primary physical movement where practical."
    )
    audio_enabled = clip.audio_enabled
    video_type = "smartphone video" if audio_enabled else "silent smartphone video"
    product_facts = _provider_text(plan.product_facts)
    if in_store:
        # The video request receives image URLs, not source-video frames. Keep
        # provider-facing facts honest about the references actually attached.
        product_facts = re.sub(
            r"\s*When no product photos are supplied, use the reference video frames\.?",
            "", product_facts, flags=re.I,
        )
    parts = [
        f"Create one continuous {clip.duration_seconds}-second vertical 9:16, 720p {video_type} from the supplied starting image.",
        *([duration_line] if duration_line else []),
        f"Ad concept: {_provider_text(ad_concept)}",
        f"Product identity and fixed specifications: {product_facts}",
        f"Creative purpose: {_provider_text(clip.creative_role)}",
        f"Scene: {_provider_text(clip.visual_description)}",
        f"Starting physical state: {_provider_text(clip.starting_state)}",
        f"Ordered physical action: {_provider_text(clip.action)}",
        f"Ending state and payoff: {_provider_text(clip.ending_state)}",
    ]
    if in_store and product_reference_count:
        product_labels = "@image2" if product_reference_count == 1 else f"@image2-@image{product_reference_count + 1}"
        parts.append(
            f"Refs: @image1=start/scene; {product_labels}=product only; ignore ref bg."
        )
    if not in_store:
        parts.append(physical_instruction)
    product_interaction = _provider_text(clip.product_interaction if in_store and clip.product_interaction else clip.action)
    if product_interaction and (not in_store or product_interaction != _provider_text(clip.action)):
        parts.append(f"Product interaction and mechanism: {product_interaction}")
    parts.append(f"Camera framing and movement: {_provider_text(clip.camera)}")
    if is_opening:
        parts.append(
            f"Hook direction: immediate, emotionally legible shopper reaction; one motivated gesture and quick product reveal; "
            f"natural handheld smartphone movement, all within this continuous {clip.duration_seconds}-second clip."
        )
    elif in_store:
        parts.append("Cast continuity: preserve the person already visible in the supplied starting image; do not replace them with the opening shopper unless this clip explicitly calls for that same person.")
    if in_store:
        parts.append(IN_STORE_REACTION_VIDEO_RULES)
    if in_store:
        if clip.scene_payoff and _provider_text(clip.scene_payoff) != _provider_text(clip.ending_state):
            parts.append(f"Product-specific payoff: {_provider_text(clip.scene_payoff)}")
    if audio_enabled:
        parts.append(
            "Native audio direction: " + _provider_text(clip.audio_direction)
            + " Synchronize each selected native sound to the visible event; no music, score, voiceover, or added dialogue."
        )
    else:
        parts.append("Audio is disabled for this clip; produce no audio stream, dialogue, music, or sound effects.")
    if notes:
        parts.append("Visual continuity: " + "; ".join(notes))
    if clip.source_start_seconds is not None and clip.source_end_seconds is not None:
        parts.append(
            f"Reference segment {clip.source_start_seconds:.1f}-{clip.source_end_seconds:.1f}s informs framing rhythm only; do not copy its subject, product, or setting."
        )
    revision = re.search(r"(?:^|\n)Clip-specific revision:\s*(.+)", clip.video_prompt, flags=re.DOTALL)
    if revision:
        parts.append(f"Clip-specific revision: {_provider_text(revision.group(1))}")
    if in_store:
        parts.append("Keep identities and product geometry stable. No cuts, morphing, or impossible motion.")
    else:
        parts.append("Keep identity, packaging, proportions, and physical contact consistent. Make the mechanism visible. Avoid complex transformations; if the action requires a difficult before-to-after change, depict the simple observable action and end state without inventing intermediate mechanics. No cuts, morphing, or impossible motion.")
    compacted = "\n".join(parts)
    length = kie_video_prompt_length(compacted)
    if length > KIE_VIDEO_PROMPT_SAFE_CHARACTERS:
        raise ValueError(
            f"Clip {clip.index} cannot be safely compacted: required details need {length} prompt characters; "
            f"the safe Kie budget is {KIE_VIDEO_PROMPT_SAFE_CHARACTERS} (provider maximum "
            "4096). Shorten the product specifications, shot action, or continuity notes in the "
            "Creative Plan before generating."
        )
    return compacted


def _has_positive_state_claim(text: str, pattern: str) -> bool:
    """Match an affirmative claim, ignoring negations in the same short clause."""
    for match in re.finditer(pattern, text, flags=re.I):
        clause_start = max(
            text.rfind(mark, 0, match.start()) for mark in (".", "!", "?", ";", ",")
        ) + 1
        clause_end = min(
            (position for mark in (".", "!", "?", ";", ",")
             if (position := text.find(mark, match.end())) != -1),
            default=len(text),
        )
        clause = text[clause_start:clause_end]
        if re.search(r"\b(?:no|not|never|without|avoid|exclude|don't|do not|doesn't|does not)\b", clause, flags=re.I):
            continue
        if (re.search(r"\b(?:later|afterward|afterwards|subsequently|during video|video action|will)\b", clause, flags=re.I)
                and not re.search(r"\b(?:frame zero|initial state|starting state)\b", clause, flags=re.I)):
            continue
        return True
    return False


def _frame_zero_prompt_claims(prompt: str) -> str:
    """Read only explicit frame-zero clauses; ignore later-action and prohibition text."""
    markers = re.finditer(r"\b(?:frame[- ]zero|at frame zero|initial physical state)\b", prompt, flags=re.I)
    clauses = []
    for marker in markers:
        tail = prompt[marker.start():]
        clause = re.split(r"[.!?\n]", tail, maxsplit=1)[0]
        clauses.append(clause)
    return " ".join(clauses)


def validate_starting_image_state(plan: MultiClipPlan) -> None:
    """Reject conflicting positive claims within frame zero, not later video actions."""
    for clip in plan.clips:
        # The structured starting_state is the canonical frame-zero description.
        # Add only prompt clauses explicitly labeled as frame zero. Never inspect
        # clip.action, video_prompt, later-action instructions, or generic negatives.
        initial = _provider_text(
            f"{clip.starting_state} {_frame_zero_prompt_claims(clip.starting_image_prompt)}"
        ).lower()
        conflicts = (
            (
                r"\b(?:cup|coffee)\b.{0,45}\b(?:held|holding|upright|intact|securely)\b",
                r"\b(?:cup|coffee|drink)\b.{0,35}\b(?:falling|fallen|dropped|spilling|spilled|splash(?:ing|ed)?)\b|\b(?:falling|fallen|dropped|spilling|spilled|splash(?:ing|ed)?)\b.{0,30}\b(?:cup|coffee|drink)\b",
                "an intact held drink conflicts with a falling or spilled drink",
            ),
            (
                r"\bfloor\b.{0,25}\bdry\b",
                r"\b(?:floor\b.{0,25}\bwet\b|spill(?:ing|ed)?|splash(?:ing|ed)?)\b",
                "a dry floor conflicts with a spill or splash",
            ),
            (
                r"\b(?:product|device|village|lights?)\b.{0,30}\b(?:unlit|off|dark)\b|\b(?:unlit|switched off)\b.{0,25}\b(?:product|device|village|lights?)\b",
                r"\b(?:product|device|village|lights?)\b.{0,30}\b(?:illuminated|lit|glowing|on)\b|\b(?:illuminated|glowing)\b.{0,25}\b(?:product|device|village|lights?)\b",
                "an unlit product conflicts with an illuminated product",
            ),
            (
                r"\b(?:box|package|packaging)\b.{0,30}\b(?:closed|sealed|unopened)\b|\b(?:closed|sealed|unopened)\b.{0,25}\b(?:box|package|packaging)\b",
                r"\b(?:box|package|packaging)\b.{0,30}\b(?:open|opened|unboxed)\b|\b(?:open|opened|unboxed)\b.{0,25}\b(?:box|package|packaging)\b",
                "a closed package conflicts with an open package",
            ),
            (
                r"\b(?:warehouse|retail aisle|retail shelving)\b",
                r"\b(?:white studio|studio backdrop|seamless backdrop|empty white room)\b",
                "a retail setting conflicts with a studio-white background",
            ),
        )
        for initial_pattern, opposite_pattern, message in conflicts:
            if (_has_positive_state_claim(initial, initial_pattern)
                    and _has_positive_state_claim(initial, opposite_pattern)):
                raise ValueError(f"Clip {clip.index} starting-image contradiction: {message}.")
        hidden = _product_hidden_at_start(initial)
        visible = _has_positive_state_claim(
            initial, r"\b(?:product (?:is )?(?:visible|in frame|revealed)|visible product)\b"
        )
        if hidden and visible:
            raise ValueError(
                f"Clip {clip.index} starting-image contradiction: a hidden product conflicts with a visible product."
            )


class KieWorkflowClient(Protocol):
    def upload(self, path: Path) -> str: ...
    def create_image(self, prompt: str, input_urls: list[str], aspect_ratio: str = "9:16") -> str: ...
    def create_masked_image_edit(self, prompt: str, input_urls: list[str], mask_url: str, aspect_ratio: str = "9:16") -> str: ...
    def create_video(self, prompt: str, image_url: str, duration_seconds: int, **kwargs) -> str: ...
    def wait_for_task(self, task_id: str, on_state: Callable[[str], None] | None = None) -> dict: ...
    def download_result(self, task_data: dict, destination: Path) -> Path: ...


def _validate_reference_paths(paths: list[Path]) -> list[str]:
    resolved: list[str] = []
    for path in paths:
        path = path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        resolved.append(str(path))
    return resolved


def _make_prompts(
    shot: ShotPlan,
    index: int,
    shared_requirements: str,
    product_facts: str,
    environment: str,
    seconds_per_clip: int,
    continuity_requirements: str,
    hook: str = "custom",
) -> tuple[str, str]:
    creative_role = shot.creative_role.strip() or "Advance the product story with a distinct scene."
    starting_state = shot.starting_state.strip() or shot.visual_description
    ending_state = shot.ending_state.strip() or shot.hook_or_payoff
    scene_environment = environment.strip().rstrip(".")
    visual_description = shot.visual_description.strip().rstrip(".")
    image_continuity_anchors = "; ".join(dict.fromkeys(note.strip() for note in shot.continuity_notes if note.strip()))
    is_in_store = hook in {"in_store_reaction", "wet_works"}
    target_duration = shot.target_duration_seconds or seconds_per_clip
    edit_duration = shot.edit_duration_seconds
    in_store_opening = is_in_store and index == 1 and any(
        marker in shot.creative_role.lower() for marker in ("reaction", "hook", "reveal", "discovery")
    )
    scoped_requirements = shared_requirements
    if is_in_store and not in_store_opening:
        scoped_requirements = (
            "Preserve product appearance and overall handheld style. This scene's starting image, scene description, and continuity notes define its people and location. "
            "Do not reuse the opening shopper in this scene unless explicitly described."
        )
    scene_direction = (
        f"Creative role: {creative_role}\n"
        f"Target scene: {visual_description}\n"
        f"Starting state: {starting_state}\n"
        f"Concrete action: {shot.action}\n"
        f"Ending state or payoff: {ending_state}\n"
        f"Product interaction: {shot.product_interaction}\n"
        f"Scene payoff: {shot.hook_or_payoff}\n" +
        (
            f"Native audio plan: {shot.audio_direction}. Keep every sound synchronized with the visible action; no music or added dialogue."
            if shot.audio_requested else
            f"Audio plan: silent. No native sounds were selected. Direction notes: {shot.audio_direction}"
        )
    )
    if in_store_opening:
        scene_direction += (
            f"\nOpening-hook unit: complete the shopper's immediate reaction, one motivated gesture, and a fast, "
            f"clearly identifiable product reveal inside this single {target_duration}-second target clip. "
            "Do not allocate a second clip to another reaction gesture. Source timestamps describe the reference segment, "
            "not seconds within this generated target clip."
        )
    has_source_segment = shot.start_seconds is not None and shot.end_seconds is not None
    source_segment = f"reference segment {shot.start_seconds:.1f}-{shot.end_seconds:.1f}s" if has_source_segment else None
    video_camera_direction = (
        f"The {source_segment} suggests this camera framing: {shot.camera}."
        if source_segment else f"Use this requested camera framing: {shot.camera}; no exact reference segment applies."
    )
    image_prompt = (
        "Photorealistic vertical 9:16 starting frame. "
        f"Scene: {scene_environment}. Shot: {visual_description}. "
        f"People and continuity: {image_continuity_anchors or creative_role}. "
        f"Product: {product_facts}. Frame zero: {starting_state}. "
        f"Framing: {shot.camera}. Show only details visible at the start of the video."
        + (f" {IN_STORE_REACTION_IMAGE_RULES}" if is_in_store else "")
    )
    duration_direction = (
        f"Target generated clip duration: {target_duration} seconds. Intended final edit use: {edit_duration if edit_duration is not None else target_duration} seconds. "
        "Final edit duration may be shorter than the generated clip. Source-video timestamps refer only to the original reference.\n"
        if is_in_store else f"Target clip duration: {target_duration} seconds.\n"
    )
    in_store_video_guidance = (
        "Show a clear progression from the defined starting state through the concrete physical action to the defined "
        "ending state/payoff during this clip. Avoid a mostly static product pose. Use framing and camera movement "
        "that make the real mechanism, contact/trigger, and cause-and-effect easy to see. Preserve the identity and appearance "
        "of people visible in the supplied starting image and do not substitute the hook shopper unless explicitly specified. "
        if is_in_store else
        "Show a clear progression from the defined starting state through the concrete physical action to the defined "
        "ending state/payoff during this clip. Avoid a mostly static product pose. Use framing and camera movement "
        "that make the real mechanism, contact/trigger, and cause-and-effect easy to see. "
    )
    video_prompt = (
        f"Create one continuous {target_duration}-second vertical 9:16, 720p, silent smartphone-style "
        f"video from the provided starting image. Keep the intended environment as {scene_environment}. The specific scene is: {visual_description}. "
        "Continue the depicted action naturally from the starting image. Preserve this clip's approved environment and lighting.\n"
        f"{duration_direction}"
        f"Shared creator requirements distilled from the complete brief: {scoped_requirements}\n"
        f"Scene-specific target direction: {scene_direction}\n"
        f"Continuity requirements: {continuity_requirements}\n"
        "Use the analyzed source segment for framing rhythm only; do not copy its subject, product, "
        "action, or location. Follow the target creative direction and references. "
        f"This is target shot {index}; follow the Shot {index} direction above. {video_camera_direction} "
        f"{in_store_video_guidance}"
        "Prefer one primary physical movement per shot. If a difficult transformation cannot be shown believably in one clip, "
        "simplify the observable action or recommend separate before-and-after shots without losing the intended story. Keep physical contact and movement believable. Keep the product "
        "accurate and unchanged. Keep the product design accurate and consistent with its reference images. "
        "Do not invent unseen product structure or capabilities. No cuts, morphing, or duplicated limbs. "
        "Honor explicit instructions about visible text, packaging, and branding. Audio is disabled and this output is silent; do not claim or attempt to add sound."
    )
    return image_prompt, video_prompt


def normalize_reference_segments(analysis: AnalysisResult, reference_duration_seconds: float) -> AnalysisResult:
    """Keep only valid, distinct source intervals; never treat target duration as source time."""
    normalized = analysis.model_copy(deep=True)
    seen: set[tuple[float, float]] = set()
    for shot in normalized.shots:
        start, end = shot.start_seconds, shot.end_seconds
        if start is None or end is None:
            shot.start_seconds = shot.end_seconds = None
            continue
        interval = (start, end)
        if end <= start or end > reference_duration_seconds or interval in seen:
            shot.start_seconds = shot.end_seconds = None
            continue
        seen.add(interval)
    return normalized


def fit_reference_analysis_to_clip_count(
    analysis: AnalysisResult,
    clip_count: int,
    *,
    hook: str = "recommend",
    seconds_per_clip: int = 5,
) -> AnalysisResult:
    """Group adjacent source beats into target clips without dropping the reference ending."""
    if clip_count < 1 or seconds_per_clip < 1:
        raise ValueError("clip_count and seconds_per_clip must be positive")
    fitted = analysis.model_copy(deep=True)
    shots = sorted(fitted.shots, key=lambda shot: (shot.start_seconds is None, shot.start_seconds or 0, shot.index))

    def category(shot: ShotPlan) -> str:
        role = shot.creative_role.lower()
        if any(word in role for word in ("rising", "load", "demonstrat", "use", "toss")):
            return "demonstration"
        if any(word in role for word in ("hook", "reaction", "reveal", "discovery")):
            return "hook"
        if any(word in role for word in ("install", "setup", "mount", "transition")):
            return "setup"
        if any(word in role for word in ("payoff", "cta", "final", "result", "ending")):
            return "payoff"
        return "other"

    def merge(first: ShotPlan, second: ShotPlan, *, opening_hook: bool = False) -> ShotPlan:
        merged = first.model_copy(deep=True)
        merged.end_seconds = second.end_seconds
        if opening_hook:
            merged.creative_role = "In-Store Reaction and Product Reveal"
            merged.visual_description = (
                f"One continuous in-store reaction hook: {first.visual_description} "
                f"The camera quickly reframes to identify the product: {second.visual_description}"
            )
            merged.action = (
                "Open with the shopper already reacting to the display. Immediately complete the motivated gesture: "
                f"{second.action[:1].lower() + second.action[1:].rstrip('. ')}. The handheld camera follows that gesture "
                "in a quick reframe and lands on a clearly identifiable view of the product before this target clip ends."
            )
            merged.starting_state = "The shopper is already stopped beside the display and visibly reacting; frame her and the product's location so the camera can reframe immediately."
            merged.ending_state = "The quick handheld reframe lands on a clearly identifiable product display, resolving what caused the reaction."
            merged.camera = "One vertical handheld smartphone shot: close-to-medium reaction framing, then a quick natural pan/reframe to the identifiable product display."
            merged.product_interaction = second.product_interaction
            merged.hook_or_payoff = "Immediate reaction resolves into a fast, visually identifiable product reveal within this opening clip."
        else:
            merged.creative_role = f"{first.creative_role.strip()} and {second.creative_role.strip()}".strip()
            merged.visual_description = f"{first.visual_description} Then {second.visual_description[:1].lower() + second.visual_description[1:]}"
            merged.action = f"{first.action.rstrip('. ')}; then {second.action[:1].lower() + second.action[1:].rstrip('. ')}."
            merged.ending_state = second.ending_state or second.hook_or_payoff
            merged.product_interaction = f"{first.product_interaction} Then {second.product_interaction}"
            merged.hook_or_payoff = second.hook_or_payoff
            merged.camera = second.camera if second.camera else first.camera
        merged.start_seconds = first.start_seconds
        planned_lengths = [value for value in (first.target_duration_seconds, second.target_duration_seconds) if value is not None]
        edit_lengths = [value for value in (first.edit_duration_seconds, second.edit_duration_seconds) if value is not None]
        if planned_lengths:
            combined_target = sum(planned_lengths)
            merged.target_duration_seconds = min(5, combined_target) if opening_hook else min(15, combined_target)
        if edit_lengths:
            edit_cap = merged.target_duration_seconds or seconds_per_clip
            merged.edit_duration_seconds = min(sum(edit_lengths), edit_cap)
        merged.continuity_group = first.continuity_group if first.continuity_group == second.continuity_group else ""
        combined_notes = list(dict.fromkeys([*first.continuity_notes, *second.continuity_notes]))
        merged.continuity_notes = combined_notes[:2] if opening_hook else combined_notes
        merged.uncertainties = list(dict.fromkeys([*first.uncertainties, *second.uncertainties]))
        merged.audio_direction = "; ".join(dict.fromkeys((first.audio_direction, second.audio_direction)))
        merged.audio_requested = first.audio_requested or second.audio_requested
        return merged

    is_in_store = hook in {"in_store_reaction", "wet_works"}
    if is_in_store and len(shots) > 1 and category(shots[0]) == "hook":
        # Keep the opening reaction and the reveal beat as one target clip.
        while len(shots) > 1 and category(shots[1]) == "hook":
            first, second = shots[0], shots[1]
            if first.continuity_group and second.continuity_group and first.continuity_group != second.continuity_group:
                break
            shots[:2] = [merge(first, second, opening_hook=True)]

    while len(shots) > clip_count:
        candidates: list[tuple[int, int, int]] = []
        for index, (first, second) in enumerate(zip(shots, shots[1:])):
            same_group = bool(first.continuity_group and first.continuity_group == second.continuity_group)
            same_category = category(first) == category(second)
            score = (4 if same_group else 0) + (3 if same_category else 0)
            if category(second) == "payoff" and index + 2 < len(shots):
                score -= 2
            if first.start_seconds is not None and second.end_seconds is not None:
                span = second.end_seconds - first.start_seconds
                score -= max(0, int(span // max(seconds_per_clip * 2, 1)))
            candidates.append((score, index, int(same_group and same_category)))
        _, merge_index, opening = max(candidates, key=lambda item: (item[0], item[2], -item[1]))
        first, second = shots[merge_index:merge_index + 2]
        shots[merge_index:merge_index + 2] = [merge(first, second, opening_hook=bool(is_in_store and merge_index == 0 and category(first) == category(second) == "hook"))]

    for index, shot in enumerate(shots, start=1):
        shot.index = index
        if category(shot) == "setup" and "transition" in shot.creative_role.lower():
            action = shot.action.lower()
            if any(word in action for word in ("install", "mount", "hang", "steadies", "steady")):
                shot.creative_role = "Product Setup and Installation"
    fitted.shots = shots
    return fitted


def build_multiclip_plan(
    analysis: AnalysisResult,
    instructions: str,
    product_facts: str,
    environment: str,
    clip_count: int,
    seconds_per_clip: int,
    product_references: list[Path] | None = None,
    continuity_references: list[Path] | None = None,
    hook: str = "custom",
) -> MultiClipPlan:
    """Build distinct, editable target clips from analyzed reference shots."""
    if clip_count < 1:
        raise ValueError("A clip plan requires at least one clip")
    if clip_count > len(analysis.shots):
        raise ValueError("clip_count cannot exceed the number of distinct analyzed shots")
    if len({shot.index for shot in analysis.shots[:clip_count]}) != clip_count:
        raise ValueError("clip_count requires distinct analyzed shots with unique indices")
    if seconds_per_clip < 1 or seconds_per_clip > 15:
        raise ValueError("seconds_per_clip must be between 1 and 15")
    if not instructions.strip():
        raise ValueError("creative instructions are required")
    if creative_instruction_length(instructions) > MAX_CREATIVE_INSTRUCTIONS:
        raise ValueError("creative instructions must be 20,000 characters or fewer")
    if not product_facts.strip():
        raise ValueError("product facts are required")
    if not environment.strip():
        raise ValueError("intended video environment is required")

    product_paths = _validate_reference_paths(product_references or [])
    continuity_paths = _validate_reference_paths(continuity_references or [])

    def continuity_for(shot: ShotPlan) -> str:
        group = shot.continuity_group.strip()
        if not group:
            return "No other shot is linked to this continuity group; follow this shot's own setting and references."
        related = [candidate for candidate in analysis.shots if candidate.index != shot.index and candidate.continuity_group.strip() == group]
        anchors = list(dict.fromkeys(note for candidate in [shot, *related] for note in candidate.continuity_notes if note.strip()))
        related_scenes = [f"Related shot {candidate.index}: {candidate.visual_description}" for candidate in related]
        details = anchors + related_scenes
        suffix = " Shared anchors: " + " ".join(details) if details else " Preserve the same setting and recurring entities across linked shots."
        return f"Environment continuity group '{group}'. Keep this setting and its visible environment consistent across this group.{suffix}"

    clips: list[PlannedClip] = []
    for index, shot in enumerate(analysis.shots[:clip_count], start=1):
        image_prompt, video_prompt = _make_prompts(
            shot, index, analysis.summary, product_facts, environment, seconds_per_clip, continuity_for(shot), hook,
        )
        source_title = (
            f"from reference segment {shot.start_seconds:.1f}-{shot.end_seconds:.1f}s"
            if shot.start_seconds is not None and shot.end_seconds is not None
            else f"· {shot.creative_role.strip() or 'creative direction'}"
        )
        clips.append(PlannedClip(
            index=index,
            title=f"Shot {index} {source_title}",
            source_shot_index=shot.index,
            source_start_seconds=shot.start_seconds,
            source_end_seconds=shot.end_seconds,
            creative_role=shot.creative_role or "Advance the product story with a distinct scene.",
            visual_description=shot.visual_description,
            starting_state=shot.starting_state or shot.visual_description,
            action=shot.action,
            product_interaction=shot.product_interaction,
            ending_state=shot.ending_state or shot.hook_or_payoff,
            scene_payoff=shot.hook_or_payoff,
            camera=shot.camera,
            continuity_group=shot.continuity_group,
            character_groups=shot.character_groups,
            continuity_notes=shot.continuity_notes,
            duration_seconds=shot.target_duration_seconds or seconds_per_clip,
            edit_duration_seconds=shot.edit_duration_seconds,
            audio_mode="auto",
            audio_requested=shot.audio_requested,
            preserve_audio=shot.audio_requested,
            audio_direction=shot.audio_direction,
            starting_image_prompt=image_prompt,
            video_prompt=video_prompt,
        ))
    return MultiClipPlan(
        summary=analysis.summary,
        environment=environment,
        instructions=instructions,
        product_facts=product_facts,
        product_references=product_paths,
        continuity_references=continuity_paths,
        hook=hook,
        clips=clips,
    )


def estimate_multiclip_generation(
    plan: MultiClipPlan,
    analysis_usd: Decimal = Decimal("0"),
) -> GenerationEstimate:
    return estimate_generation(
        seconds=sum(clip.duration_seconds for clip in plan.clips),
        images=len(plan.clips),
        analysis_usd=analysis_usd,
    )


def generation_approval_required(
    estimate: GenerationEstimate,
    *,
    pricing_verified: bool = True,
    estimate_reliable: bool = True,
) -> bool:
    return (
        estimate.total_usd > AUTO_GENERATION_APPROVAL_LIMIT_USD
        or not pricing_verified
        or not estimate_reliable
    )


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug[:48].rstrip("-") or "shot"


def _task_credits(task_data: dict) -> Decimal | None:
    value = task_data.get("creditsConsumed", task_data.get("consumeCredits", task_data.get("consumedCredits")))
    return Decimal(str(value)) if value is not None else None


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _record_task_failure(record: dict, error: Exception) -> None:
    task_data = getattr(error, "task_data", None)
    if isinstance(task_data, dict):
        credits = _task_credits(task_data)
        record.update(
            credits_consumed=str(credits) if credits is not None else None,
            status="failed",
            completed_at=_timestamp(),
            provider_state=task_data.get("state", "fail"),
            provider_failure={
                "code": task_data.get("failCode"),
                "message": task_data.get("failMsg"),
            },
        )
    else:
        record["last_checked_at"] = _timestamp()


def _write_usage_manifest(output_dir: Path, tasks: list[dict], expected_task_count: int) -> None:
    credits = [Decimal(task["credits_consumed"]) for task in tasks if task.get("credits_consumed") is not None]
    logical_tasks = {(task.get("clip_index"), task.get("kind")) for task in tasks}
    completed_tasks = {(task.get("clip_index"), task.get("kind")) for task in tasks
                       if task.get("status", "success") == "success"}
    completed_count = len(completed_tasks)
    complete = len(logical_tasks) == expected_task_count and len(credits) == len(tasks)
    total_credits = sum(credits, Decimal("0")) if complete else None
    manifest = {
        "provider": "Kie.ai",
        "pricing_basis": "provider-reported task credits; USD equivalent at $0.005 per credit",
        "expected_generation_tasks": expected_task_count,
        "completed_generation_tasks": completed_count,
        "tasks": tasks,
        "total_credits_consumed": str(total_credits) if total_credits is not None else None,
        "total_usd_equivalent": str(total_credits * KIE_USD_PER_CREDIT) if total_credits is not None else None,
        "all_task_charges_reported": complete,
    }
    (output_dir / "generation-usage.json").write_text(json.dumps(manifest, indent=2) + "\n")


def find_saved_mask_for_scene(outputs_dir: Path, source_image: Path) -> Path | None:
    """Find a prior edit mask only when its recorded first/source image matches exactly."""
    source_image = source_image.expanduser().resolve()
    if not source_image.is_file():
        return None
    source_hash = hashlib.sha256(source_image.read_bytes()).hexdigest()
    for manifest_path in sorted(outputs_dir.expanduser().resolve().glob("**/generation-usage.json")):
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        for task in manifest.get("tasks", []):
            request = task.get("request", {})
            references = request.get("references", [])
            mask_info = request.get("mask") or {}
            if not references or references[0].get("sha256") != source_hash or not mask_info.get("filename"):
                continue
            mask_path = (manifest_path.parent / Path(mask_info["filename"]).name).resolve()
            if not mask_path.is_file():
                continue
            expected_mask_hash = mask_info.get("sha256")
            if expected_mask_hash and hashlib.sha256(mask_path.read_bytes()).hexdigest() != expected_mask_hash:
                continue
            return mask_path
    return None


def _masked_edit_prompt(clip: PlannedClip, scene_prompt: str) -> str:
    return (
        "Edit the first input image using its attached alpha mask. The first input is the selected scene base and environment only. "
        "Transparent mask pixels are editable; preserve the environment and every pixel outside that region. "
        "Use the following scene-specific starting-frame direction for the edited region, including its intended action, pose, "
        "expression, wardrobe, and lighting: "
        f"{clip.visual_description} Frame zero: {clip.starting_state} {scene_prompt} "
        "Every character reference is for identity only: preserve recognizable face, skin tone, hair, age, and body proportions; "
        "ignore that reference's pose, expression, wardrobe, and environment. "
        "Every product reference is for product appearance only, including construction; ignore its person, pose, lighting, and background. "
        "Match the edited subjects to the selected scene's camera perspective, practical lighting, and natural smartphone appearance."
    )


def run_masked_image_edit(
    plan: MultiClipPlan,
    clip: PlannedClip,
    scene_base: Path,
    mask_image: Path,
    product_references: list[Path],
    character_references: list[Path],
    output_dir: Path,
    client: KieWorkflowClient,
    scene_prompt: str | None = None,
    resume: bool = False,
) -> Path:
    """Run one opt-in Sunburst 2.6 masked edit with scene, character, and product roles separated."""
    scene_base = scene_base.expanduser().resolve()
    mask_image = mask_image.expanduser().resolve()
    character_paths = _validate_reference_paths(character_references)
    product_paths = _validate_reference_paths(product_references)
    if not scene_base.is_file() or not mask_image.is_file():
        raise FileNotFoundError(scene_base if not scene_base.is_file() else mask_image)
    if str(scene_base) in character_paths or str(scene_base) in product_paths:
        raise ValueError("The selected scene base cannot also serve as a character or product reference.")
    input_paths = [scene_base, *(Path(path) for path in character_paths), *(Path(path) for path in product_paths)]
    if not 1 <= len(input_paths) <= 16:
        raise ValueError("Masked image editing requires 1 to 16 distinct reference images.")

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"clip-{clip.index:02d}-{_slug(clip.action)}-starting.png"
    manifest_path = output_dir / "generation-usage.json"
    task_record = None
    if manifest_path.is_file():
        try:
            saved = json.loads(manifest_path.read_text())
            task_record = next((item for item in saved.get("tasks", []) if item.get("kind") == "starting_image"), None)
        except (OSError, ValueError, json.JSONDecodeError):
            saved = {}
        if task_record:
            if task_record.get("status") == "success" and output_path.is_file():
                return output_path
            if task_record.get("status") != "submitted" or not task_record.get("task_id"):
                raise RuntimeError("This masked edit has no safely resumable provider task; it was not resubmitted.")

    if task_record is None:
        roles = ["scene environment only"] + ["character identity only"] * len(character_paths) + ["product identity only"] * len(product_paths)
        input_urls = [client.upload(path) for path in input_paths]
        mask_url = client.upload(mask_image)
        prompt = _masked_edit_prompt(clip, scene_prompt or clip.starting_image_prompt)
        request = {
            "model": "gpt-image-2-6/sunburst-image-to-image",
            "prompt": prompt,
            "references": [
                {"filename": path.name, "role": role, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                for path, role in zip(input_paths, roles)
            ],
            "compiled_input_url_order": [path.name for path in input_paths],
            "mask": {
                "filename": mask_image.name,
                "role": "editable pixels have transparent alpha; opaque alpha protects the selected scene",
                "sha256": hashlib.sha256(mask_image.read_bytes()).hexdigest(),
            },
            "parameters": {"aspect_ratio": "9:16", "resolution": "1K", "quality": "medium", "output_format": "png", "background": "opaque"},
        }
        task_record = {"clip_index": clip.index, "kind": "starting_image", "task_id": None,
                       "credits_consumed": None, "status": "submitting", "submitted_at": _timestamp(), "request": request}
        manifest = {"provider": "Kie.ai", "model": request["model"],
                    "pricing_basis": "provider-reported task credits; USD equivalent at $0.005 per credit",
                    "expected_generation_tasks": 1, "completed_generation_tasks": 0,
                    "all_task_charges_reported": False, "tasks": [task_record]}
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        try:
            task_id = client.create_masked_image_edit(prompt, input_urls, mask_url, aspect_ratio="9:16")
        except Exception as exc:
            task_record["status"] = "submission_unknown" if getattr(exc, "outcome_uncertain", False) else "failed"
            task_record["error"] = str(exc)
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
            raise
        task_record.update(task_id=task_id, status="submitted")
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    else:
        manifest = saved

    try:
        task_data = client.wait_for_task(task_record["task_id"])
    except Exception as exc:
        _record_task_failure(task_record, exc)
        manifest["tasks"] = [task_record]
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        raise
    client.download_result(task_data, output_path)
    credits = _task_credits(task_data)
    task_record.update(
        credits_consumed=str(credits) if credits is not None else None,
        usd_equivalent=str(credits * KIE_USD_PER_CREDIT) if credits is not None else None,
        status="success", completed_at=_timestamp(), provider_state=task_data.get("state", "success"),
    )
    manifest.update(
        completed_generation_tasks=1,
        all_task_charges_reported=credits is not None,
        total_credits_consumed=str(credits) if credits is not None else None,
        total_usd_equivalent=str(credits * KIE_USD_PER_CREDIT) if credits is not None else None,
        result={"filename": output_path.name, "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest()},
        tasks=[task_record],
    )
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return output_path


def run_multiclip_generation(
    plan: MultiClipPlan,
    product_references: list[Path],
    continuity_references: list[Path],
    output_dir: Path,
    client: KieWorkflowClient,
    approved_max_usd: Decimal | None = None,
    pricing_verified: bool = True,
    estimate_reliable: bool = True,
    analysis_usd: Decimal = Decimal("0"),
    on_progress: Callable[[dict], None] | None = None,
    resume: bool = False,
    images_only: bool = False,
    starting_images: dict[int, Path] | None = None,
    continuity_reference_roles: list[str] | None = None,
    character_references: list[Path] | None = None,
) -> list[Path]:
    """Generate one starting image and one audio-configurable clip per plan shot.

    Jobs at or below $0.50 proceed without an extra confirmation when configured
    provider rates and the estimate are reliable. Larger or uncertain jobs need
    an explicit maximum approved amount that covers the full estimate.
    """
    estimate = estimate_multiclip_generation(plan, analysis_usd=analysis_usd)
    output_dir = output_dir.expanduser().resolve()
    operation_cost = estimate.total_usd
    if starting_images and not images_only:
        # Reviewed starting images and the reference analysis are already paid
        # for. This operation is only the set of planned video requests.
        operation_cost = sum(
            (Decimal(str(clip.duration_seconds)) * Decimal("0.0225") for clip in plan.clips),
            Decimal("0"),
        )
    elif resume and images_only:
        missing_images = sum(
            1 for clip in plan.clips
            if not (starting_images and clip.index in starting_images)
            and not any(output_dir.glob(f"clip-{clip.index:02d}-*-starting.png"))
        )
        operation_cost = IMAGE_1K_USD * missing_images
    approval_needed = (operation_cost > AUTO_GENERATION_APPROVAL_LIMIT_USD
                       or not pricing_verified or not estimate_reliable)
    if approval_needed and approved_max_usd is None:
        reasons = []
        if operation_cost > AUTO_GENERATION_APPROVAL_LIMIT_USD:
            reasons.append(f"estimate ${operation_cost:.4f} exceeds ${AUTO_GENERATION_APPROVAL_LIMIT_USD:.2f}")
        if not pricing_verified:
            reasons.append("provider pricing is unverified")
        if not estimate_reliable:
            reasons.append("the estimate is unreliable")
        raise PermissionError(f"Explicit approval is required because {' and '.join(reasons)}")
    if approved_max_usd is not None and not estimate_within_approved_amount(operation_cost, approved_max_usd):
        raise PermissionError(
            f"Estimated cost ${operation_cost:.4f} exceeds approved maximum ${approved_max_usd:.4f}"
        )

    reaction_format = plan.hook in {"in_store_reaction", "wet_works"}
    product_paths = [Path(path).expanduser().resolve() for path in (product_references or plan.product_references)]
    if reaction_format and images_only and all(_product_is_off_frame_at_start(clip) for clip in plan.clips):
        product_paths = []
    # Validate every provider-facing prompt and frame-zero state before provider work.
    validate_starting_image_state(plan)
    # Image-only work has no video request; an oversized video prompt must not
    # block starting-image creation or regeneration.
    video_prompts = {} if images_only else {
        clip.index: compile_video_prompt(
            plan, clip, product_reference_count=min(6, len(product_paths)) if reaction_format else 0
        ) for clip in plan.clips
    }
    continuity_paths = [Path(path).expanduser().resolve() for path in (continuity_references or plan.continuity_references)]
    character_paths = [Path(path).expanduser().resolve() for path in (character_references or [])]
    environment_paths_by_clip = {
        clip.index: [Path(path).expanduser().resolve() for path in clip.environment_references]
        for clip in plan.clips
    }
    if output_dir.exists() and any(output_dir.iterdir()) and not resume:
        raise FileExistsError(f"Output directory is not empty; refusing to overwrite: {output_dir}")
    all_environment_paths = list(dict.fromkeys(
        path for paths in environment_paths_by_clip.values() for path in paths
    ))
    for path in product_paths + continuity_paths + character_paths + all_environment_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    reference_urls = [client.upload(path) for path in product_paths + continuity_paths + character_paths]
    product_reference_urls = reference_urls[:len(product_paths)]
    continuity_reference_urls = reference_urls[len(product_paths):len(product_paths) + len(continuity_paths)]
    character_reference_urls = reference_urls[len(product_paths) + len(continuity_paths):]
    reference_filenames = {
        url: path.name for path, url in zip(product_paths + continuity_paths + character_paths, reference_urls)
    }
    request_roles_by_url = {
        **{url: "product identity only" for url in product_reference_urls},
        **{
            url: role for url, role in zip(
                continuity_reference_urls,
                continuity_reference_roles or ["project continuity input"] * len(continuity_paths),
            )
        },
        **{url: "character identity only" for url in character_reference_urls},
    }
    prompt_roles_by_url = {
        **{url: "product details only; ignore background" for url in product_reference_urls},
        **{
            url: role for url, role in zip(
                continuity_reference_urls,
                continuity_reference_roles or ["continuity references define faces, wardrobe, location, and pose"] * len(continuity_paths),
            )
        },
        **{url: "character identity only; preserve this character's face, skin tone, hair, age, and body; allow the current scene's wardrobe and setting" for url in character_reference_urls},
    }
    environment_urls_by_path: dict[Path, str] = {}
    for path in all_environment_paths:
        url = client.upload(path)
        environment_urls_by_path[path] = url
        reference_filenames[url] = path.name
        request_roles_by_url[url] = "scene environment only"
        prompt_roles_by_url[url] = (
            "scene environment only; preserve the intended location, background, and practical lighting; "
            "ignore people, faces, pose, wardrobe, and product"
        )
    environment_urls_by_clip = {
        index: [environment_urls_by_path[path] for path in paths]
        for index, paths in environment_paths_by_clip.items()
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    silent_outputs: list[Path] = []
    prepared_images: dict[int, Path] = {}
    manifest_path = output_dir / "generation-usage.json"
    persisted_expected_task_count: int | None = None
    if resume and manifest_path.is_file():
        try:
            prior_manifest = json.loads(manifest_path.read_text())
            usage_tasks = prior_manifest.get("tasks", [])
            persisted_expected_task_count = prior_manifest.get("expected_generation_tasks")
        except (OSError, ValueError, json.JSONDecodeError):
            usage_tasks = []
    else:
        usage_tasks = []
    expected_task_count = len(plan.clips) * 2
    if starting_images and not persisted_expected_task_count:
        # A clip-only revision that reuses an approved image spends on video only.
        expected_task_count = len(plan.clips) + sum(
            clip.index not in starting_images for clip in plan.clips
        )
    if persisted_expected_task_count:
        # Preserve the original image-plus-video task total across reviewed-image resumes.
        expected_task_count = int(persisted_expected_task_count)
    continuity_group_images: dict[str, str] = {}
    character_group_images: dict[str, str] = {}
    identity_clip_index = product_identity_clip_index(plan)
    product_identity_url: str | None = None

    def notify(stage: str, clip: PlannedClip, provider_state: str | None = None) -> None:
        if on_progress:
            on_progress({
                "stage": stage,
                "clip_index": clip.index,
                "clip_count": len(plan.clips),
                "provider_state": provider_state,
            })

    # In reaction formats, build the hook and same-scene continuity images before
    # product references. Shot-level visibility below omits irrelevant product
    # photos from off-frame hooks; other formats retain product-first ordering.
    if reaction_format:
        ordered_clips = sorted(plan.clips, key=lambda clip: (clip.index != 1, clip.index))
    elif any(clip.character_groups for clip in plan.clips):
        # Character references are established in timeline order before later scenes consume them.
        ordered_clips = sorted(plan.clips, key=lambda clip: clip.index)
    else:
        ordered_clips = sorted(plan.clips, key=lambda clip: (clip.index != identity_clip_index, clip.index))
    for clip in ordered_clips:
        stem = f"clip-{clip.index:02d}-{_slug(clip.action)}"
        silent_output = output_dir / f"{stem}.mp4"
        supplied_image = Path(starting_images[clip.index]).expanduser().resolve() if starting_images and clip.index in starting_images else None
        if supplied_image is not None and not supplied_image.is_file():
            raise FileNotFoundError(supplied_image)
        if resume and silent_output.is_file() and not images_only:
            saved_info = probe_video(silent_output)
            if saved_info["has_audio"] != clip.audio_enabled:
                expected = "native audio" if clip.audio_enabled else "no audio"
                raise RuntimeError(
                    f"Saved clip {clip.index} does not match its audio setting; expected {expected}. "
                    "The saved file was not accepted as complete and no automatic generation retry was started."
                )
            silent_outputs.append(silent_output)
            local_start = supplied_image or next(iter(output_dir.glob(f"clip-{clip.index:02d}-*-starting.png")), None)
            if local_start:
                local_url = client.upload(local_start)
                reference_filenames[local_url] = local_start.name
                if clip.continuity_group.strip():
                    continuity_group_images[clip.continuity_group.strip()] = local_url
                for character_group in clip.character_groups:
                    if character_group.strip():
                        character_group_images.setdefault(character_group.strip(), local_url)
                if clip.index == identity_clip_index:
                    product_identity_url = local_url
            continue
        notify("preparing_images", clip)
        continuity_roles = continuity_reference_roles or ["project continuity input"] * len(continuity_paths)
        clip_environment_urls = environment_urls_by_clip.get(clip.index, [])
        scene_reference_urls = list(dict.fromkeys(continuity_reference_urls + clip_environment_urls))
        clip_product_urls = list(product_reference_urls)
        group = clip.continuity_group.strip()
        if group and group in continuity_group_images and len(scene_reference_urls) + len(clip_product_urls) < 16:
            group_url = continuity_group_images[group]
            if group_url not in scene_reference_urls:
                scene_reference_urls.append(group_url)
                request_roles_by_url[group_url] = "environment continuity only"
                prompt_roles_by_url[group_url] = "environment continuity only; preserve setting and background, not cast identity"
        scene_character_urls = list(character_reference_urls)
        for character_group in clip.character_groups:
            character_url = character_group_images.get(character_group.strip())
            if (character_url and character_url not in scene_character_urls
                    and len(scene_reference_urls) + len(scene_character_urls) + len(clip_product_urls) < 16):
                scene_character_urls.append(character_url)
                request_roles_by_url[character_url] = "character identity only"
                prompt_roles_by_url[character_url] = (
                    "character identity only; preserve this character's face, skin tone, hair, age, and body; "
                    "allow the current scene's wardrobe and setting"
                )
        # A single image URL cannot act as two incompatible provider references.
        # Keep it in the character role when a cast anchor is also the scene anchor.
        scene_reference_urls = [url for url in scene_reference_urls if url not in scene_character_urls]
        product_hidden = reaction_format and _product_is_off_frame_at_start(clip)
        if product_hidden:
            clip_product_urls = []
        identity_excluded = _product_hidden_at_start(clip.starting_state)
        if (clip.index != identity_clip_index and not identity_excluded and product_identity_url
                and product_identity_url not in scene_character_urls
                and len(scene_reference_urls) + len(clip_product_urls) < 16):
            if product_identity_url not in clip_product_urls:
                clip_product_urls.append(product_identity_url)
                request_roles_by_url[product_identity_url] = "product identity only"
                prompt_roles_by_url[product_identity_url] = "product identity only; ignore background"
        image_references = (
            scene_reference_urls + scene_character_urls + clip_product_urls if reaction_format else
            clip_product_urls + scene_reference_urls + scene_character_urls
        )
        explicit_roles = (
            [f"{reference_filenames.get(url, Path(url.rsplit('/', 1)[-1].split('?', 1)[0]).name)} = {prompt_roles_by_url.get(url, 'project reference')}"
             for url in image_references]
            if continuity_reference_roles is not None or clip_environment_urls or scene_character_urls else []
        )
        image_prompt = _image_reference_prompt(
            clip.starting_image_prompt,
            has_product_reference=bool(clip_product_urls),
            has_continuity_reference=bool(scene_reference_urls or scene_character_urls),
            reference_roles=explicit_roles,
        )
        request_info = {
            "prompt": image_prompt,
            "references": [
                {
                    "filename": reference_filenames.get(url, Path(url.rsplit("/", 1)[-1].split("?", 1)[0]).name),
                    "role": request_roles_by_url.get(url, "project reference"),
                }
                for url in image_references
            ],
            "revision_note": next((line.split(":", 1)[1].strip() for line in reversed(image_prompt.splitlines()) if line.lower().startswith(("requested revision:", "clip-specific revision:"))), None),
        }
        starting_image = supplied_image or (next(iter(output_dir.glob(f"clip-{clip.index:02d}-*-starting.png")), None) if resume else None)
        if starting_image is None:
            image_record = next((task for task in usage_tasks if task.get("clip_index") == clip.index and task.get("kind") == "starting_image"), None)
            if image_record and resume and images_only and image_record.get("status") in {"failed", "fail"}:
                # A terminal task cannot recover by polling it again. Keep its receipt
                # for accounting, and create one fresh attempt only after explicit resume.
                image_record = None
            if image_record:
                if not image_record.get("task_id"):
                    raise RuntimeError("The starting-image submission has no saved provider task ID. Reconcile its status before retrying to avoid duplicate charges.")
                image_task = image_record["task_id"]
            else:
                try:
                    image_task = (
                        client.create_image(image_prompt, image_references, aspect_ratio="9:16")
                        if image_references else client.create_text_image(image_prompt, aspect_ratio="9:16")
                    )
                except Exception as exc:
                    if getattr(exc, "outcome_uncertain", False):
                        usage_tasks.append({"clip_index": clip.index, "kind": "starting_image", "task_id": None, "credits_consumed": None, "status": "submission_unknown", "submitted_at": _timestamp(), "request": request_info})
                        _write_usage_manifest(output_dir, usage_tasks, expected_task_count)
                    raise
                image_record = {
                    "clip_index": clip.index, "kind": "starting_image", "task_id": image_task,
                    "credits_consumed": None, "status": "submitted", "submitted_at": _timestamp(),
                    "request": request_info,
                }
                usage_tasks.append(image_record)
                _write_usage_manifest(output_dir, usage_tasks, expected_task_count)
            try:
                image_data = client.wait_for_task(image_task, on_state=lambda state: notify("preparing_images", clip, state))
            except Exception as exc:
                _record_task_failure(image_record, exc)
                _write_usage_manifest(output_dir, usage_tasks, expected_task_count)
                raise
            image_record.update(credits_consumed=str(_task_credits(image_data)) if _task_credits(image_data) is not None else None, status="success", completed_at=_timestamp())
            _write_usage_manifest(output_dir, usage_tasks, expected_task_count)
            starting_image = client.download_result(image_data, output_dir / f"{stem}-starting.png")

        prepared_images[clip.index] = starting_image
        starting_url = client.upload(starting_image)
        reference_filenames[starting_url] = starting_image.name
        if group:
            continuity_group_images[group] = starting_url
        for character_group in clip.character_groups:
            if character_group.strip():
                character_group_images.setdefault(character_group.strip(), starting_url)
        if clip.index == identity_clip_index:
            product_identity_url = starting_url
        if images_only:
            continue
        notify("generating_clips", clip)
        video_record = next((task for task in usage_tasks if task.get("clip_index") == clip.index and task.get("kind") == "video"), None) if resume else None
        if video_record:
            if not video_record.get("task_id"):
                raise RuntimeError("The video submission has no saved provider task ID. Reconcile its status before retrying to avoid duplicate charges.")
            video_task = video_record["task_id"]
        else:
            try:
                video_product_urls = product_reference_urls[:6] if reaction_format else []
                video_request = {
                    "prompt": video_prompts[clip.index],
                    "references": [
                        {"filename": starting_image.name, "role": "exact starting frame; sets frame-zero scene and cast"},
                        *[
                            {"filename": reference_filenames.get(url, Path(url.rsplit('/', 1)[-1].split('?', 1)[0]).name),
                             "role": "product identity only; ignore background"}
                            for url in video_product_urls
                        ],
                    ],
                    "duration_seconds": clip.duration_seconds,
                    "aspect_ratio": "9:16",
                    "resolution": "720p",
                    "audio_enabled": clip.audio_enabled,
                }
                video_task = client.create_video(
                    video_prompts[clip.index], starting_url, clip.duration_seconds,
                    aspect_ratio="9:16", resolution="720p",
                    enable_audio=None if clip.audio_enabled else False,
                    reference_image_urls=video_product_urls,
                )
            except Exception as exc:
                if getattr(exc, "outcome_uncertain", False):
                    usage_tasks.append({"clip_index": clip.index, "kind": "video", "task_id": None, "credits_consumed": None, "status": "submission_unknown", "submitted_at": _timestamp(), "request": video_request})
                    _write_usage_manifest(output_dir, usage_tasks, expected_task_count)
                raise
            video_record = {"clip_index": clip.index, "kind": "video", "task_id": video_task, "credits_consumed": None, "status": "submitted", "submitted_at": _timestamp(), "request": video_request}
            usage_tasks.append(video_record)
            _write_usage_manifest(output_dir, usage_tasks, expected_task_count)
        try:
            video_data = client.wait_for_task(video_task, on_state=lambda state: notify("generating_clips", clip, state))
        except Exception as exc:
            _record_task_failure(video_record, exc)
            _write_usage_manifest(output_dir, usage_tasks, expected_task_count)
            raise
        video_credits = _task_credits(video_data)
        video_record.update(credits_consumed=str(video_credits) if video_credits is not None else None, status="success", completed_at=_timestamp())
        _write_usage_manifest(output_dir, usage_tasks, expected_task_count)
        provider_path = output_dir / f"{stem}-provider.mp4"
        provider_video = provider_path if resume and provider_path.is_file() else client.download_result(video_data, provider_path)
        notify("processing_outputs", clip)
        if clip.audio_enabled:
            make_mp4(provider_video, silent_output, preserve_audio=True)
        else:
            make_silent_mp4(provider_video, silent_output)
        silent_outputs.append(silent_output)
    if images_only:
        return [prepared_images[clip.index] for clip in plan.clips]
    return [output_dir / f"clip-{clip.index:02d}-{_slug(clip.action)}.mp4" for clip in plan.clips]
