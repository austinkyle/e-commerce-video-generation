from __future__ import annotations

import json
import re
from decimal import Decimal
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Protocol

from pydantic import BaseModel, Field, model_validator

from ecom_video.costs import KIE_USD_PER_CREDIT, GenerationEstimate, estimate_generation
from ecom_video.media import make_silent_mp4
from ecom_video.models import AnalysisResult, ShotPlan
from ecom_video.prompt_limits import KIE_VIDEO_PROMPT_SAFE_CHARACTERS, kie_video_prompt_length

AUTO_GENERATION_APPROVAL_LIMIT_USD = Decimal("1.00")
MAX_CREATIVE_INSTRUCTIONS = 20_000


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
    ending_state: str = ""
    camera: str
    continuity_group: str = ""
    continuity_notes: list[str] = Field(default_factory=list)
    duration_seconds: int = Field(ge=1, le=15)
    starting_image_prompt: str
    video_prompt: str

    @model_validator(mode="after")
    def validate_source_interval(self) -> "PlannedClip":
        if (self.source_start_seconds is None) != (self.source_end_seconds is None):
            raise ValueError("source start and end times must both be provided or both be null")
        if self.source_start_seconds is not None and self.source_end_seconds <= self.source_start_seconds:
            raise ValueError("source end time must be after source start time")
        return self


class MultiClipPlan(BaseModel):
    summary: str
    environment: str
    instructions: str
    product_facts: str
    product_references: list[str] = Field(default_factory=list)
    continuity_references: list[str] = Field(default_factory=list)
    hook: str = "custom"
    clips: list[PlannedClip] = Field(min_length=1)


def product_identity_clip_index(plan: MultiClipPlan) -> int:
    """Pick a stable product-visible frame to anchor image continuity."""
    exclusions = re.compile(r"\b(no|without|before|not|doesn't|does not)\b.{0,48}\b(product|bowl|item|package|packaging)\b", re.I)
    ranked: list[tuple[int, int]] = []
    for clip in plan.clips:
        role = _provider_text(clip.creative_role).lower()
        details = " ".join((clip.visual_description, clip.starting_state, clip.action, clip.starting_image_prompt))
        score = 0
        if any(word in role for word in ("reveal", "demo", "demonstration", "product", "unbox", "close-up", "detail")):
            score += 4
        if any(word in role for word in ("hook", "discovery")):
            score -= 2
        if exclusions.search(details):
            score -= 8
        ranked.append((score, clip.index))
    return max(ranked, key=lambda item: (item[0], -item[1]))[1]


def _image_reference_prompt(prompt: str, has_identity_reference: bool) -> str:
    if not has_identity_reference:
        return prompt
    return (
        f"{prompt}\n\nReference-image direction: use the supplied product reference to preserve only the product's "
        "identity, color, shape, materials, and branding. Follow this shot's requested pose, physical state, "
        "camera angle, and location; do not copy the reference image's framing or background."
    )


def _provider_text(value: str) -> str:
    return " ".join(value.split())


def compile_video_prompt(plan: MultiClipPlan, clip: PlannedClip) -> str:
    """Compact an oversized saved prompt from its structured shot direction.

    The approved plan remains unchanged. Required facts are preserved whole; if
    they cannot fit, fail before any generation request instead of truncating.
    """
    notes = list(dict.fromkeys(_provider_text(note) for note in clip.continuity_notes if note.strip()))
    parts = [
        f"Create one continuous {clip.duration_seconds}-second vertical 9:16, 720p silent smartphone video from the supplied starting image.",
        f"Ad concept: {_provider_text(plan.summary)}",
        f"Product identity and fixed specifications: {_provider_text(plan.product_facts)}",
        f"Creative purpose: {_provider_text(clip.creative_role)}",
        f"Scene: {_provider_text(clip.visual_description)}",
        f"Starting physical state: {_provider_text(clip.starting_state)}",
        f"Ordered physical action: {_provider_text(clip.action)}",
        f"Ending state and payoff: {_provider_text(clip.ending_state)}",
        "Physical reliability: preserve the exact starting orientation; place the working surface toward the fabric with a visible contact point; show a natural grip and support. Use one primary physical movement where practical.",
        f"Product interaction: {_provider_text(clip.action)}",
        f"Camera framing and movement: {_provider_text(clip.camera)}",
    ]
    if notes:
        parts.append("Visual continuity: " + "; ".join(notes))
    if clip.source_start_seconds is not None and clip.source_end_seconds is not None:
        parts.append(
            f"Reference segment {clip.source_start_seconds:.1f}-{clip.source_end_seconds:.1f}s informs framing rhythm only; do not copy its subject, product, or setting."
        )
    revision = re.search(r"(?:^|\n)Clip-specific revision:\s*(.+)", clip.video_prompt, flags=re.DOTALL)
    if revision:
        parts.append(f"Clip-specific revision: {_provider_text(revision.group(1))}")
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


class KieWorkflowClient(Protocol):
    def upload(self, path: Path) -> str: ...
    def create_image(self, prompt: str, input_urls: list[str], aspect_ratio: str = "9:16") -> str: ...
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
) -> tuple[str, str]:
    creative_role = shot.creative_role.strip() or "Advance the product story with a distinct scene."
    starting_state = shot.starting_state.strip() or shot.visual_description
    ending_state = shot.ending_state.strip() or shot.hook_or_payoff
    scene_direction = (
        f"Creative role: {creative_role}\n"
        f"Target scene: {shot.visual_description}\n"
        f"Starting state: {starting_state}\n"
        f"Concrete action: {shot.action}\n"
        f"Ending state or payoff: {ending_state}\n"
        f"Product interaction: {shot.product_interaction}\n"
        f"Scene payoff: {shot.hook_or_payoff}"
    )
    has_source_segment = shot.start_seconds is not None and shot.end_seconds is not None
    source_segment = f"reference segment {shot.start_seconds:.1f}-{shot.end_seconds:.1f}s" if has_source_segment else None
    camera_direction = (
        f"Camera framing inspired by {source_segment}: {shot.camera}."
        if source_segment else f"Use this requested camera framing: {shot.camera}; no exact reference segment applies."
    )
    video_camera_direction = (
        f"The {source_segment} suggests this camera framing: {shot.camera}."
        if source_segment else f"Use this requested camera framing: {shot.camera}; no exact reference segment applies."
    )
    image_prompt = (
        "Create one photorealistic vertical 9:16 starting frame for a separately editable short clip. "
        f"Set the scene in {environment}. The specific scene is: {shot.visual_description}. "
        "The room, background, and lighting in this starting image must match the intended video; do not use a "
        "plain white or studio-white background. Keep the visible product and character appearance "
        "consistent with the supplied product and continuity references.\n"
        f"Shared creator requirements distilled from the complete brief: {shared_requirements}\n"
        f"Scene-specific target direction: {scene_direction}\n"
        f"Continuity requirements: {continuity_requirements}\n"
        f"Product details to preserve exactly: {product_facts}\n"
        "Use the analyzed source segment for framing rhythm only, never its subject, product, or setting. "
        f"This is target shot {index}; follow the Shot {index} direction above. {camera_direction} "
        "Show the defined starting state and product interaction in the starting pose. Keep the product design accurate "
        "and consistent with its reference images. "
        "When a tool touches a surface, orient its actual working face toward the surface and show the true contact edge; "
        "use a natural, supported hand grip and an initial pose that can physically continue into the action. "
        "Use the continuity notes for recurring entities and preserve intentional scene changes. Do not invent product construction "
        "or capabilities."
    )
    video_prompt = (
        f"Create one continuous {seconds_per_clip}-second vertical 9:16, 720p, silent smartphone-style "
        f"video from the provided starting image. Keep the intended environment as {environment}. The specific scene is: {shot.visual_description}. "
        "Continue the depicted action naturally from the starting image. Preserve this clip's approved environment and lighting.\n"
        f"Target clip duration: {seconds_per_clip} seconds. Source-video timestamps refer only to the original reference.\n"
        f"Shared creator requirements distilled from the complete brief: {shared_requirements}\n"
        f"Scene-specific target direction: {scene_direction}\n"
        f"Continuity requirements: {continuity_requirements}\n"
        f"Product details to preserve exactly: {product_facts}\n"
        "Use the analyzed source segment for framing rhythm only; do not copy its subject, product, "
        "action, or location. Follow the target creative direction and references. "
        f"This is target shot {index}; follow the Shot {index} direction above. {video_camera_direction} "
        "Show a clear progression from the defined starting state through the concrete physical action to the defined "
        "ending state/payoff during this clip. Avoid a mostly static product pose. Use framing and camera movement "
        "that make the mechanism and surface contact point easy to see. Preserve starting tool orientation and realistic grip. "
        "Prefer one primary physical movement per shot. If a difficult transformation cannot be shown believably in one clip, "
        "simplify the observable action or recommend separate before-and-after shots without losing the intended story. Keep physical contact and movement believable. Keep the product "
        "accurate and unchanged. Keep the product design accurate and consistent with its reference images. "
        "Do not invent unseen product structure or capabilities. No cuts, morphing, or duplicated limbs. "
        "Honor explicit instructions about visible text, packaging, and branding."
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


def build_multiclip_plan(
    analysis: AnalysisResult,
    instructions: str,
    product_facts: str,
    environment: str,
    clip_count: int,
    seconds_per_clip: int,
    product_references: list[Path] | None = None,
    continuity_references: list[Path] | None = None,
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
        return f"Continuity group '{group}'. Keep people, animals, product appearance, and environment consistent across this group.{suffix}"

    clips: list[PlannedClip] = []
    for index, shot in enumerate(analysis.shots[:clip_count], start=1):
        image_prompt, video_prompt = _make_prompts(
            shot, index, analysis.summary, product_facts, environment, seconds_per_clip, continuity_for(shot),
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
            ending_state=shot.ending_state or shot.hook_or_payoff,
            camera=shot.camera,
            continuity_group=shot.continuity_group,
            continuity_notes=shot.continuity_notes,
            duration_seconds=seconds_per_clip,
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
        )
    else:
        record["last_checked_at"] = _timestamp()


def _write_usage_manifest(output_dir: Path, tasks: list[dict], expected_task_count: int) -> None:
    credits = [Decimal(task["credits_consumed"]) for task in tasks if task.get("credits_consumed") is not None]
    completed_count = sum(task.get("status", "success") == "success" for task in tasks)
    complete = len(tasks) == expected_task_count and len(credits) == expected_task_count
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
) -> list[Path]:
    """Generate one starting image and one silent, editable clip per plan shot.

    Jobs at or below $1.00 proceed without an extra confirmation when configured
    provider rates and the estimate are reliable. Larger or uncertain jobs need
    an explicit maximum approved amount that covers the full estimate.
    """
    estimate = estimate_multiclip_generation(plan, analysis_usd=analysis_usd)
    if generation_approval_required(
        estimate,
        pricing_verified=pricing_verified,
        estimate_reliable=estimate_reliable,
    ) and approved_max_usd is None:
        reasons = []
        if estimate.total_usd > AUTO_GENERATION_APPROVAL_LIMIT_USD:
            reasons.append(f"estimate ${estimate.total_usd:.4f} exceeds ${AUTO_GENERATION_APPROVAL_LIMIT_USD:.2f}")
        if not pricing_verified:
            reasons.append("provider pricing is unverified")
        if not estimate_reliable:
            reasons.append("the estimate is unreliable")
        raise PermissionError(f"Explicit approval is required because {' and '.join(reasons)}")
    if approved_max_usd is not None and approved_max_usd < estimate.total_usd:
        raise PermissionError(
            f"Estimated cost ${estimate.total_usd:.4f} exceeds approved maximum ${approved_max_usd:.4f}"
        )

    # Validate every provider-facing prompt before image generation or uploads.
    video_prompts = {clip.index: compile_video_prompt(plan, clip) for clip in plan.clips}

    product_paths = [Path(path).expanduser().resolve() for path in (product_references or plan.product_references)]
    continuity_paths = [Path(path).expanduser().resolve() for path in (continuity_references or plan.continuity_references)]
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not resume:
        raise FileExistsError(f"Output directory is not empty; refusing to overwrite: {output_dir}")
    for path in product_paths + continuity_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    reference_urls = [client.upload(path) for path in product_paths + continuity_paths]
    output_dir.mkdir(parents=True, exist_ok=True)
    silent_outputs: list[Path] = []
    prepared_images: dict[int, Path] = {}
    manifest_path = output_dir / "generation-usage.json"
    if resume and manifest_path.is_file():
        try:
            usage_tasks = json.loads(manifest_path.read_text()).get("tasks", [])
        except (OSError, ValueError, json.JSONDecodeError):
            usage_tasks = []
    else:
        usage_tasks = []
    expected_task_count = len(plan.clips) * 2
    continuity_group_images: dict[str, str] = {}
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

    # Establish one product-visible image before other shots so even shots in
    # different locations/groups inherit the same product identity.
    ordered_clips = sorted(plan.clips, key=lambda clip: (clip.index != identity_clip_index, clip.index))
    for clip in ordered_clips:
        stem = f"clip-{clip.index:02d}-{_slug(clip.action)}"
        silent_output = output_dir / f"{stem}.mp4"
        supplied_image = Path(starting_images[clip.index]).expanduser().resolve() if starting_images and clip.index in starting_images else None
        if supplied_image is not None and not supplied_image.is_file():
            raise FileNotFoundError(supplied_image)
        if resume and silent_output.is_file() and not images_only:
            silent_outputs.append(silent_output)
            local_start = supplied_image or next(iter(output_dir.glob(f"clip-{clip.index:02d}-*-starting.png")), None)
            if local_start:
                local_url = client.upload(local_start)
                if clip.continuity_group.strip():
                    continuity_group_images[clip.continuity_group.strip()] = local_url
                if clip.index == identity_clip_index:
                    product_identity_url = local_url
            continue
        notify("preparing_images", clip)
        image_references = list(reference_urls)
        group = clip.continuity_group.strip()
        if group and group in continuity_group_images and len(image_references) < 16:
            image_references.append(continuity_group_images[group])
        identity_excluded = bool(re.search(
            r"\b(no|without|before|not|doesn't|does not)\b.{0,48}\b(product|bowl|item|package|packaging)\b",
            " ".join((clip.visual_description, clip.starting_state, clip.action, clip.starting_image_prompt)),
            flags=re.I,
        ))
        if clip.index != identity_clip_index and not identity_excluded and product_identity_url and len(image_references) < 16:
            if product_identity_url not in image_references:
                image_references.append(product_identity_url)
        image_prompt = _image_reference_prompt(clip.starting_image_prompt, bool(image_references))
        starting_image = supplied_image or (next(iter(output_dir.glob(f"clip-{clip.index:02d}-*-starting.png")), None) if resume else None)
        if starting_image is None:
            image_record = next((task for task in usage_tasks if task.get("clip_index") == clip.index and task.get("kind") == "starting_image"), None)
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
                        usage_tasks.append({"clip_index": clip.index, "kind": "starting_image", "task_id": None, "credits_consumed": None, "status": "submission_unknown", "submitted_at": _timestamp()})
                        _write_usage_manifest(output_dir, usage_tasks, expected_task_count)
                    raise
                image_record = {"clip_index": clip.index, "kind": "starting_image", "task_id": image_task, "credits_consumed": None, "status": "submitted", "submitted_at": _timestamp()}
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
        if group:
            continuity_group_images[group] = starting_url
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
                video_task = client.create_video(
                    video_prompts[clip.index], starting_url, clip.duration_seconds,
                    aspect_ratio="9:16", resolution="720p",
                )
            except Exception as exc:
                if getattr(exc, "outcome_uncertain", False):
                    usage_tasks.append({"clip_index": clip.index, "kind": "video", "task_id": None, "credits_consumed": None, "status": "submission_unknown", "submitted_at": _timestamp()})
                    _write_usage_manifest(output_dir, usage_tasks, expected_task_count)
                raise
            video_record = {"clip_index": clip.index, "kind": "video", "task_id": video_task, "credits_consumed": None, "status": "submitted", "submitted_at": _timestamp()}
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
        make_silent_mp4(provider_video, silent_output)
        silent_outputs.append(silent_output)
    if images_only:
        return [prepared_images[clip.index] for clip in plan.clips]
    return [output_dir / f"clip-{clip.index:02d}-{_slug(clip.action)}.mp4" for clip in plan.clips]
