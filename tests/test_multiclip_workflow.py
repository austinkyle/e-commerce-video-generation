from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from ecom_video.models import AnalysisResult
from ecom_video import cli
from ecom_video.workflow import (
    build_multiclip_plan,
    compile_video_prompt,
    estimate_multiclip_generation,
    run_multiclip_generation,
    normalize_reference_segments,
)
from ecom_video.costs import GenerationEstimate


def analysis_fixture() -> AnalysisResult:
    return AnalysisResult.model_validate({
        "summary": "A short product demonstration with a reveal.",
        "questions": [],
        "shots": [
            {
                "index": index,
                "start_seconds": (index - 1) * 2,
                "end_seconds": index * 2,
                "visual_description": description,
                "action": action,
                "camera": camera,
                "product_interaction": "The hands keep the product steady.",
                "hook_or_payoff": "A clear product reveal.",
                "uncertainties": [],
            }
            for index, description, action, camera in [
                (1, "Woman settles a kitten in the pet pouch", "Adjust the kitten in the pouch", "Handheld medium shot"),
                (2, "Close garment detail", "Adjust the opening", "Handheld close-up"),
                (3, "Centered reveal", "Let the kitten settle", "Handheld portrait"),
            ]
        ],
    })


def test_plan_uses_distinct_reference_shots_and_same_environment_for_image_and_video():
    plan = build_multiclip_plan(
        analysis_fixture(),
        instructions="Show a calm kitten already partly supported in the pouch.",
        product_facts="Red Santa hoodie with circular opening and belt graphic.",
        environment="naturally lit living room",
        clip_count=3,
        seconds_per_clip=5,
    )

    assert len(plan.clips) == 3
    assert len({clip.source_shot_index for clip in plan.clips}) == 3
    for clip in plan.clips:
        assert "naturally lit living room" in clip.starting_image_prompt
        assert "naturally lit living room" in clip.video_prompt
        assert "studio-white" in clip.starting_image_prompt
        assert "baby" not in clip.starting_image_prompt.lower()
        assert "baby" not in clip.video_prompt.lower()
        assert clip.duration_seconds == 5


def test_plan_rejects_more_clips_than_distinct_analyzed_shots():
    with pytest.raises(ValueError, match="distinct analyzed shots"):
        build_multiclip_plan(
            analysis_fixture(), "creative direction", "product details",
            "living room", clip_count=4, seconds_per_clip=5,
        )


def test_original_null_source_segments_get_descriptive_titles():
    analysis = analysis_fixture()
    for shot in analysis.shots:
        shot.start_seconds = shot.end_seconds = None
    plan = build_multiclip_plan(analysis, "original concept", "product", "specific scenes", 3, 5)
    assert all(clip.source_start_seconds is None and clip.source_end_seconds is None for clip in plan.clips)
    assert all("reference segment" not in clip.title for clip in plan.clips)


def test_source_video_intervals_remain_distinct_from_generated_clip_duration():
    plan = build_multiclip_plan(
        analysis_fixture(), "Creative direction", "Product details", "Living room", 3, 5,
    )

    assert [(clip.source_start_seconds, clip.source_end_seconds) for clip in plan.clips] == [
        (0.0, 2.0), (2.0, 4.0), (4.0, 6.0),
    ]
    assert [clip.duration_seconds for clip in plan.clips] == [5, 5, 5]
    assert "0.0-2.0s" in plan.clips[0].title
    assert "4.0-6.0s" in plan.clips[2].video_prompt


def test_creative_direction_only_shot_has_no_invented_source_timestamps():
    analysis = analysis_fixture()
    analysis.shots[1].start_seconds = None
    analysis.shots[1].end_seconds = None
    plan = build_multiclip_plan(
        analysis, "Add a new indoor reaction shot", "Product details", "Living room", 3, 5,
    )

    clip = plan.clips[1]
    assert clip.source_start_seconds is None
    assert clip.source_end_seconds is None
    assert "creative direction" in clip.title.lower()
    assert "no exact reference segment" in clip.video_prompt.lower()
    assert "target clip duration: 5 seconds" in clip.video_prompt.lower()


def test_repeated_or_out_of_bounds_source_ranges_are_unmapped_instead_of_reused():
    analysis = AnalysisResult.model_validate({
        "summary": "Three scenes.", "questions": [], "shots": [
            {**shot.model_dump(), "start_seconds": 0, "end_seconds": 5}
            for shot in analysis_fixture().shots
        ],
    })

    normalized = normalize_reference_segments(analysis, reference_duration_seconds=11.84)
    ranges = [(shot.start_seconds, shot.end_seconds) for shot in normalized.shots]
    assert ranges[0] == (0.0, 5.0)
    assert ranges[1:] == [(None, None), (None, None)]
    assert len({pair for pair in ranges if pair[0] is not None}) == 1


def test_multiclip_estimate_counts_one_image_and_video_per_clip():
    plan = build_multiclip_plan(
        analysis_fixture(), "creative direction", "product details",
        "living room", clip_count=3, seconds_per_clip=5,
    )

    estimate = estimate_multiclip_generation(plan)
    assert estimate.image_count == 3
    assert estimate.generated_seconds == 15
    assert estimate.total_usd == Decimal("0.4275")


def test_long_brief_is_retained_in_plan_but_generation_prompts_are_scene_specific():
    shared = "Shared requirements: preserve the same product design, woman, kitten, and living-room light."
    instructions = shared + "\n\n" + "DETAILED_CREATIVE_NOTE_DO_NOT_DROP. " * 500
    analysis = AnalysisResult.model_validate({
        "summary": shared,
        "questions": [],
        "shots": [{
            "index": index, "start_seconds": index - 1, "end_seconds": index,
            "visual_description": f"Target composition for shot {index}.",
            "action": f"Target-only action {index}: adjust the garment naturally.",
            "camera": f"Handheld framing {index}.",
            "product_interaction": "Maintain believable physical contact and fabric folds.",
            "hook_or_payoff": f"Reveal garment detail {index}.", "uncertainties": [],
        } for index in range(1, 4)],
    })

    plan = build_multiclip_plan(
        analysis, instructions, "Preserve the product shown in reference images.",
        "a naturally lit living room", 3, 5,
    )

    assert plan.instructions == instructions
    for index, clip in enumerate(plan.clips, start=1):
        assert shared in clip.starting_image_prompt and shared in clip.video_prompt
        assert f"Target-only action {index}" in clip.starting_image_prompt
        assert f"Target-only action {index}" in clip.video_prompt
        assert "DETAILED_CREATIVE_NOTE_DO_NOT_DROP" not in clip.starting_image_prompt
        assert "DETAILED_CREATIVE_NOTE_DO_NOT_DROP" not in clip.video_prompt
        assert len(clip.starting_image_prompt) < 4_000
        assert len(clip.video_prompt) < 4_000


def test_scene_prompts_encode_role_start_action_payoff_camera_and_product_constraints():
    data = analysis_fixture().model_dump()
    data["shots"][0].update({
        "creative_role": "Hook: discover the gecko toy on the store shelf.",
        "starting_state": "The woman notices the boxed gecko toy at eye level and reaches toward it.",
        "ending_state": "She holds the box toward camera with a surprised smile.",
        "continuity_group": "store-hook",
        "continuity_notes": ["Keep the same boxed toy and woman appearance during this shot."],
    })
    plan = build_multiclip_plan(
        AnalysisResult.model_validate(data), "Show a playful product story", "Preserve the toy design",
        "toy store", 1, 5,
    )
    clip = plan.clips[0]
    for prompt in (clip.starting_image_prompt, clip.video_prompt):
        assert "Creative role:" in prompt
        assert "Starting state:" in prompt
        assert "Concrete action:" in prompt
        assert "Ending state or payoff:" in prompt
        assert "Keep the product design accurate" in prompt
    assert "Hook: discover the gecko toy" in clip.video_prompt
    assert "She holds the box toward camera" in clip.video_prompt
    assert "Avoid a mostly static product pose" in clip.video_prompt


def test_successive_same_setting_clips_receive_shared_continuity_context_and_image(tmp_path):
    data = analysis_fixture().model_dump()
    data["shots"][0].update({"continuity_group": "store"})
    data["shots"][1].update({
        "continuity_group": "home-gecko-cat",
        "continuity_notes": ["Same living room, same gray tabby cat, same green-and-black gecko toy."],
    })
    data["shots"][2].update({
        "continuity_group": "home-gecko-cat",
        "continuity_notes": ["Continue with that same gray tabby cat, living room, and gecko toy."],
    })
    plan = build_multiclip_plan(
        AnalysisResult.model_validate(data), "Show the product story", "Keep the toy unchanged",
        "the scene described per shot", 3, 5,
    )
    assert "Same living room, same gray tabby cat" in plan.clips[1].video_prompt
    assert "same gray tabby cat" in plan.clips[2].video_prompt
    assert "home-gecko-cat" in plan.clips[2].video_prompt

    product = tmp_path / "product-reference.jpg"
    continuity = tmp_path / "approved-continuity.jpg"
    product.write_bytes(b"product")
    continuity.write_bytes(b"continuity")
    kie = FakeKie()
    def fake_silent(source: Path, destination: Path) -> Path:
        destination.write_bytes(source.read_bytes())
        return destination
    from unittest.mock import patch
    with patch("ecom_video.workflow.make_silent_mp4", fake_silent):
        run_multiclip_generation(plan, [product], [continuity], tmp_path / "outputs", kie)
    image_calls = [call for call in kie.calls if call[0] == "image"]
    assert len(image_calls) == 3
    assert len(image_calls[0][2]) == 2
    assert len(image_calls[1][2]) == 2
    assert any("clip-02-" in url for url in image_calls[2][2])


class FakeKie:
    def __init__(self):
        self.calls: list[tuple] = []

    def upload(self, path: Path) -> str:
        self.calls.append(("upload", path.name))
        return f"https://files.test/{path.name}"

    def create_image(self, prompt: str, input_urls: list[str], aspect_ratio: str = "9:16") -> str:
        self.calls.append(("image", prompt, input_urls, aspect_ratio))
        return f"image-task-{sum(call[0] == 'image' for call in self.calls)}"

    def create_text_image(self, prompt: str, aspect_ratio: str = "9:16") -> str:
        self.calls.append(("text_image", prompt, aspect_ratio))
        return f"image-task-{sum(call[0] in {'image', 'text_image'} for call in self.calls)}"

    def create_video(self, prompt: str, image_url: str, duration_seconds: int, **kwargs) -> str:
        self.calls.append(("video", prompt, image_url, duration_seconds, kwargs))
        return f"video-task-{sum(call[0] == 'video' for call in self.calls)}"

    def wait_for_task(self, task_id: str, on_state=None) -> dict:
        self.calls.append(("poll", task_id))
        if on_state:
            on_state("generating")
        credits = 6 if task_id.startswith("image") else 22.5
        return {
            "state": "success",
            "creditsConsumed": credits,
            "resultJson": json.dumps({"resultUrls": [f"https://files.test/{task_id}.mp4"]}),
        }

    def download_result(self, task_data: dict, destination: Path) -> Path:
        self.calls.append(("download", destination.name))
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"fixture")
        return destination


def test_generation_submits_each_clip_separately_and_reuses_references(tmp_path, monkeypatch):
    plan = build_multiclip_plan(
        analysis_fixture(), "creative direction", "product details",
        "living room", clip_count=3, seconds_per_clip=5,
    )
    product = tmp_path / "product.jpg"
    continuity = tmp_path / "approved-frame.jpg"
    product.write_bytes(b"product")
    continuity.write_bytes(b"continuity")
    kie = FakeKie()

    def fake_silent(source: Path, destination: Path) -> Path:
        destination.write_bytes(source.read_bytes())
        return destination

    monkeypatch.setattr("ecom_video.workflow.make_silent_mp4", fake_silent)
    outputs = run_multiclip_generation(
        plan,
        [product],
        [continuity],
        tmp_path / "clips",
        kie,
    )

    assert len(outputs) == 3
    assert all(path.is_file() for path in outputs)
    assert sum(call[0] == "image" for call in kie.calls) == 3
    assert sum(call[0] == "video" for call in kie.calls) == 3
    image_calls = [call for call in kie.calls if call[0] == "image"]
    assert all(len(call[2]) == 2 for call in image_calls)
    video_calls = [call for call in kie.calls if call[0] == "video"]
    assert [call[3] for call in video_calls] == [5, 5, 5]
    assert all(call[4]["resolution"] == "720p" for call in video_calls)
    assert all(call[4]["aspect_ratio"] == "9:16" for call in video_calls)
    usage = json.loads((tmp_path / "clips" / "generation-usage.json").read_text())
    assert usage["all_task_charges_reported"] is True
    assert usage["total_credits_consumed"] == "85.5"
    assert usage["total_usd_equivalent"] == "0.4275"


def test_image_review_phase_prepares_every_image_then_reuses_them_for_videos(tmp_path, monkeypatch):
    plan = build_multiclip_plan(analysis_fixture(), "direction", "product facts", "room", clip_count=3, seconds_per_clip=5)
    product = tmp_path / "product.jpg"; product.write_bytes(b"product")
    output_dir = tmp_path / "review"; kie = FakeKie()
    monkeypatch.setattr("ecom_video.workflow.make_silent_mp4", lambda src, dst: (dst.write_bytes(src.read_bytes()), dst)[1])

    starting_images = run_multiclip_generation(plan, [product], [], output_dir, kie, images_only=True)

    assert len(starting_images) == 3
    assert all(path.is_file() for path in starting_images)
    assert not [call for call in kie.calls if call[0] == "video"]
    saved_bytes = [path.read_bytes() for path in starting_images]
    outputs = run_multiclip_generation(
        plan, [product], [], output_dir, kie, resume=True,
        starting_images={clip.index: path for clip, path in zip(plan.clips, starting_images)},
    )

    assert len(outputs) == 3
    assert len([call for call in kie.calls if call[0] == "image"]) == 3
    assert len([call for call in kie.calls if call[0] == "video"]) == 3
    assert [path.read_bytes() for path in starting_images] == saved_bytes


def test_image_review_seeds_product_identity_across_different_scene_groups(tmp_path):
    plan = build_multiclip_plan(analysis_fixture(), "A dog bowl ad", "Forest-green collapsible bowl", "varied scenes", 3, 5)
    roles = ["Hook: dog waits before the bowl is revealed", "Product reveal", "Water-fill demonstration"]
    groups = ["trail-hook", "home-reveal", "forest-demo"]
    clips = []
    for clip, role, group in zip(plan.clips, roles, groups):
        clips.append(clip.model_copy(update={
            "creative_role": role,
            "continuity_group": group,
            "visual_description": "The bowl stays hidden until the reveal." if "Hook" in role else f"Show the forest-green bowl during {role.lower()}.",
            "starting_state": "No bowl visible yet." if "Hook" in role else "The same forest-green bowl is visible.",
            "starting_image_prompt": "A trail scene; no bowl visible yet." if "Hook" in role else f"Show the same forest-green bowl for the {role.lower()}.",
        }))
    plan = plan.model_copy(update={"clips": clips})
    kie = FakeKie()
    images = run_multiclip_generation(plan, [], [], tmp_path / "review", kie, images_only=True)

    assert len(images) == 3
    assert [path.name[:7] for path in images] == ["clip-01", "clip-02", "clip-03"]
    image_calls = [call for call in kie.calls if call[0] in {"image", "text_image"}]
    assert image_calls[0][0] == "text_image"  # the reveal image is prepared first as the anchor
    anchor_url = f"https://files.test/{images[1].name}"
    assert image_calls[1][0] == "text_image" and "reference-image direction" not in image_calls[1][1].lower()
    assert image_calls[2][0] == "image" and anchor_url in image_calls[2][2]
    assert "product reference" in image_calls[2][1].lower()


def test_resume_reuses_saved_identity_anchor_without_recreating_its_image(tmp_path):
    plan = build_multiclip_plan(analysis_fixture(), "A dog bowl ad", "Forest-green collapsible bowl", "varied scenes", 3, 5)
    clips = [clip.model_copy(update={"creative_role": role}) for clip, role in zip(plan.clips, ["Hook", "Product reveal", "Demonstration"])]
    plan = plan.model_copy(update={"clips": clips})
    output_dir = tmp_path / "resume"
    output_dir.mkdir()
    anchor = output_dir / "clip-02-anchor-starting.png"
    anchor.write_bytes(b"saved-anchor")
    kie = FakeKie()

    images = run_multiclip_generation(plan, [], [], output_dir, kie, images_only=True, resume=True)

    assert images[1] == anchor
    assert len([call for call in kie.calls if call[0] in {"image", "text_image"}]) == 2
    assert any(call[0] == "upload" and call[1] == anchor.name for call in kie.calls)


def test_compiled_prompt_prioritizes_contact_orientation_grip_and_simple_physical_action():
    plan = furglide_plan_with_long_provider_prompts()
    prompt = compile_video_prompt(plan, plan.clips[2])
    assert "working surface toward the fabric" in prompt
    assert "natural grip" in prompt
    assert "one primary physical movement" in prompt
    assert "visible contact point" in prompt


def test_failed_video_task_persists_provider_confirmed_credits_and_operation_time(tmp_path):
    plan = build_multiclip_plan(analysis_fixture(), "direction", "product facts", "room", clip_count=1, seconds_per_clip=5)
    reference = tmp_path / "product.jpg"
    reference.write_bytes(b"product")

    class FailedChargedVideo(FakeKie):
        def wait_for_task(self, task_id: str, on_state=None) -> dict:
            if task_id.startswith("video"):
                error = RuntimeError("Kie task failed: model rejected output")
                error.task_data = {"state": "fail", "creditsConsumed": 9, "failCode": "model_error"}
                raise error
            return super().wait_for_task(task_id, on_state)

    output_dir = tmp_path / "charged-failure"
    with pytest.raises(RuntimeError, match="Kie task failed"):
        run_multiclip_generation(plan, [reference], [], output_dir, FailedChargedVideo())

    manifest = json.loads((output_dir / "generation-usage.json").read_text())
    failed = next(task for task in manifest["tasks"] if task["kind"] == "video")
    assert failed["status"] == "failed"
    assert failed["credits_consumed"] == "9"
    assert failed["completed_at"]
    assert Decimal(manifest["total_credits_consumed"]) == Decimal("15")
    assert manifest["all_task_charges_reported"] is True


def test_uncertain_task_submission_is_recorded_and_not_automatically_resubmitted(tmp_path):
    plan = build_multiclip_plan(analysis_fixture(), "direction", "product facts", "room", clip_count=1, seconds_per_clip=5)
    reference = tmp_path / "product.jpg"
    reference.write_bytes(b"product")

    class UncertainVideoSubmission(FakeKie):
        def __init__(self):
            super().__init__()
            self.submissions = 0

        def create_video(self, prompt: str, image_url: str, duration_seconds: int, **kwargs) -> str:
            self.submissions += 1
            error = RuntimeError("provider response was lost")
            error.outcome_uncertain = True
            raise error

    provider = UncertainVideoSubmission()
    output_dir = tmp_path / "uncertain-submission"
    with pytest.raises(RuntimeError, match="provider response was lost"):
        run_multiclip_generation(plan, [reference], [], output_dir, provider)
    with pytest.raises(RuntimeError, match="Reconcile its status before retrying"):
        run_multiclip_generation(plan, [reference], [], output_dir, provider, resume=True)

    manifest = json.loads((output_dir / "generation-usage.json").read_text())
    uncertain = next(task for task in manifest["tasks"] if task["kind"] == "video")
    assert uncertain["status"] == "submission_unknown" and uncertain["task_id"] is None
    assert provider.submissions == 1


def test_resume_reuses_completed_clip_and_image_and_reconciles_submitted_image_task(tmp_path, monkeypatch):
    plan = build_multiclip_plan(
        analysis_fixture(), "creative direction", "product details",
        "living room", clip_count=3, seconds_per_clip=5,
    )
    output_dir = tmp_path / "clips"
    output_dir.mkdir()
    for filename, data in [
        ("clip-01-adjust-the-kitten-in-the-pouch-starting.png", b"image-1"),
        ("clip-01-adjust-the-kitten-in-the-pouch.mp4", b"video-1"),
        ("clip-02-adjust-the-opening-starting.png", b"image-2"),
    ]:
        (output_dir / filename).write_bytes(data)
    (output_dir / "generation-usage.json").write_text(json.dumps({
        "expected_generation_tasks": 6,
        "completed_generation_tasks": 3,
        "tasks": [
            {"clip_index": 1, "kind": "starting_image", "task_id": "image-done-1", "credits_consumed": "6", "status": "success"},
            {"clip_index": 1, "kind": "video", "task_id": "video-done-1", "credits_consumed": "22.5", "status": "success"},
            {"clip_index": 2, "kind": "starting_image", "task_id": "image-done-2", "credits_consumed": "6", "status": "success"},
            {"clip_index": 3, "kind": "starting_image", "task_id": "image-active-3", "credits_consumed": None, "status": "submitted"},
        ],
    }))
    kie = FakeKie()
    monkeypatch.setattr("ecom_video.workflow.make_silent_mp4", lambda src, dst: (dst.write_bytes(src.read_bytes()), dst)[1])

    outputs = run_multiclip_generation(plan, [], [], output_dir, kie, resume=True)

    assert len(outputs) == 3
    assert outputs[0].read_bytes() == b"video-1"
    assert not [call for call in kie.calls if call[0] == "image"]
    assert [call[1] for call in kie.calls if call[0] == "poll"] == [
        "video-task-1", "image-active-3", "video-task-2",
    ]
    assert sum(call[0] == "video" for call in kie.calls) == 2
    assert (output_dir / "clip-02-adjust-the-opening-starting.png").read_bytes() == b"image-2"
    manifest = json.loads((output_dir / "generation-usage.json").read_text())
    image3 = next(task for task in manifest["tasks"] if task["task_id"] == "image-active-3")
    assert image3["status"] == "success"
    assert manifest["completed_generation_tasks"] == 6
    assert manifest["total_credits_consumed"] == "85.5"


def furglide_plan_with_long_provider_prompts():
    source = analysis_fixture()
    fourth = source.shots[2].model_copy(update={"index": 4})
    shots = [
        source.shots[0].model_copy(update={
            "index": 1, "start_seconds": None, "end_seconds": None,
            "creative_role": "Wet Works discovery hook",
            "visual_description": "A shopper discovers FurGlide on a shelf in a fictional pet store.",
            "starting_state": "The FurGlide box sits on a generic store shelf.",
            "action": "The shopper notices the box, pauses, then lifts it for a closer look.",
            "ending_state": "The shopper holds the box just off the shelf.",
            "camera": "Vertical handheld phone, quick push-in, eye level.",
            "continuity_group": "store_discovery",
            "continuity_notes": ["Fictional pet store; dark-teal cream FurGlide box with paw mark."],
            "product_interaction": "Lift and inspect the box.",
            "hook_or_payoff": "Curious in-store discovery.",
        }),
        source.shots[1].model_copy(update={
            "index": 2, "start_seconds": None, "end_seconds": None,
            "creative_role": "Product reveal",
            "visual_description": "At home beside the couch, the shopper opens FurGlide packaging.",
            "starting_state": "Closed box rests beside the same fur-covered couch.",
            "action": "Open the top flaps, lift out the teal tool, and turn it once to show its ridges.",
            "ending_state": "Hold the tool above the open box with the couch behind it.",
            "camera": "Vertical handheld close-up with a small tilt as the box opens.",
            "continuity_group": "home_couch",
            "continuity_notes": ["Same shopper, home, daylight, couch, product and package in home scenes."],
            "product_interaction": "Open the box and lift out the tool.",
            "hook_or_payoff": "Reveal the reusable ridged cleaning surface.",
        }),
        source.shots[2].model_copy(update={
            "index": 3, "start_seconds": None, "end_seconds": None,
            "creative_role": "Demonstration",
            "visual_description": "In the same home, FurGlide is used on loose pet hair on the couch.",
            "starting_state": "The cushion is covered with loose hair; the tool is poised above it.",
            "action": "Press down and make short overlapping strokes; fur gathers along the silicone edge.",
            "ending_state": "Small fur clumps sit beside a visibly cleaner path.",
            "camera": "Vertical handheld medium close-up angled slightly down.",
            "continuity_group": "home_couch",
            "continuity_notes": ["Same couch, shopper, daylight and unchanged FurGlide design."],
            "product_interaction": "Apply firm strokes across upholstery.",
            "hook_or_payoff": "Show a plausible partial clean.",
        }),
        fourth.model_copy(update={
            "index": 4, "start_seconds": None, "end_seconds": None,
            "creative_role": "Payoff",
            "visual_description": "Close view of the same couch after the cleaning strokes.",
            "starting_state": "A cleaner fabric path and gathered fur clump remain beside the tool.",
            "action": "Lift the tool, remove the fur clump, and hold on the cleaned upholstery.",
            "ending_state": "Clean fabric fills frame; the removed fur is off to one side.",
            "camera": "Vertical handheld close-up with a slight pull-back.",
            "continuity_group": "home_couch",
            "continuity_notes": ["Same couch, shopper, daylight and unchanged FurGlide design."],
            "product_interaction": "Lift off the collected fur and reveal the textile.",
            "hook_or_payoff": "Satisfying realistic clean-fabric payoff.",
        }),
    ]
    analysis = AnalysisResult.model_validate({
        "summary": "FurGlide ad: in-store discovery, home unboxing, couch demonstration and realistic payoff.",
        "questions": [], "shots": [shot.model_dump() for shot in shots],
    })
    plan = build_multiclip_plan(
        analysis, "Keep the footage authentic and silent.",
        "FurGlide is a 4.5 x 3 x 1.5 inch reusable pet-hair remover with matte teal housing and dark-gray washable silicone ridges. Short firm strokes gather loose fur. No motor or adhesive. Conceptual dark-teal, cream and white box with FurGlide name and paw mark.",
        "Fictional pet store for the hook; the same home living room and couch for the unboxing, demonstration and payoff.",
        4, 5,
    )
    for clip in plan.clips:
        clip.video_prompt += "\n" + ("LOW-PRIORITY REDUNDANT CONTEXT. " * 160)
    return plan


def test_resume_compacts_long_creative_plan_video_prompts_for_clips_two_to_four(tmp_path, monkeypatch):
    from ecom_video.workflow import _slug

    plan = furglide_plan_with_long_provider_prompts()
    output_dir = tmp_path / "furglide"
    output_dir.mkdir()
    image1 = output_dir / f"clip-01-{_slug(plan.clips[0].action)}-starting.png"
    video1 = output_dir / f"clip-01-{_slug(plan.clips[0].action)}.mp4"
    image2 = output_dir / f"clip-02-{_slug(plan.clips[1].action)}-starting.png"
    image1.write_bytes(b"saved image 1")
    video1.write_bytes(b"saved video 1")
    image2.write_bytes(b"saved image 2")
    (output_dir / "generation-usage.json").write_text(json.dumps({"tasks": [
        {"clip_index": 1, "kind": "starting_image", "task_id": "image-done-1", "credits_consumed": "6", "status": "success"},
        {"clip_index": 1, "kind": "video", "task_id": "video-done-1", "credits_consumed": "22.5", "status": "success"},
        {"clip_index": 2, "kind": "starting_image", "task_id": "image-done-2", "credits_consumed": "6", "status": "success"},
    ]}))
    original_prompts = [clip.video_prompt for clip in plan.clips]
    kie = FakeKie()
    monkeypatch.setattr("ecom_video.workflow.make_silent_mp4", lambda src, dst: (dst.write_bytes(src.read_bytes()), dst)[1])

    outputs = run_multiclip_generation(plan, [], [], output_dir, kie, resume=True)

    video_requests = [call[1] for call in kie.calls if call[0] == "video"]
    assert len(outputs) == 4
    assert len(video_requests) == 3
    assert all(len(prompt) <= 3800 for prompt in video_requests)
    for clip in plan.clips[1:]:
        provider_prompt = compile_video_prompt(plan, clip)
        for required in (plan.product_facts, clip.starting_state, clip.action, clip.ending_state, clip.camera, *clip.continuity_notes):
            assert required in provider_prompt
    assert "LOW-PRIORITY REDUNDANT CONTEXT" in plan.clips[1].video_prompt
    assert [clip.video_prompt for clip in plan.clips] == original_prompts
    assert image1.read_bytes() == b"saved image 1"
    assert video1.read_bytes() == b"saved video 1"
    assert image2.read_bytes() == b"saved image 2"
    assert len([call for call in kie.calls if call[0] == "image"]) == 2


def test_compiler_fails_actionably_when_essential_details_cannot_fit_provider_budget():
    plan = furglide_plan_with_long_provider_prompts()
    plan.product_facts = "Unabridged confirmed product specification. " * 120

    with pytest.raises(ValueError, match="cannot be safely compacted.*Shorten the product specifications"):
        compile_video_prompt(plan, plan.clips[1])


def test_uncompactable_plan_is_rejected_before_any_provider_operation(tmp_path):
    plan = furglide_plan_with_long_provider_prompts()
    plan.product_facts = "Unabridged confirmed product specification. " * 120
    kie = FakeKie()

    with pytest.raises(ValueError, match="cannot be safely compacted"):
        run_multiclip_generation(plan, [], [], tmp_path / "clips", kie)

    assert kie.calls == []
    assert not (tmp_path / "clips").exists()


def test_compiler_preserves_selective_regeneration_revision():
    plan = furglide_plan_with_long_provider_prompts()
    revised = plan.clips[1].model_copy(update={
        "video_prompt": plan.clips[1].video_prompt + "\nClip-specific revision: Keep the branded box visible throughout the unboxing."
    })

    provider_prompt = compile_video_prompt(plan, revised)

    assert "Clip-specific revision: Keep the branded box visible throughout the unboxing." in provider_prompt
    assert len(provider_prompt) <= 3800


def test_runner_compacts_selective_regeneration_revision_before_video_submission(tmp_path, monkeypatch):
    plan = furglide_plan_with_long_provider_prompts()
    clip = plan.clips[1].model_copy(update={
        "video_prompt": plan.clips[1].video_prompt + "\nClip-specific revision: Keep the branded box visible throughout the unboxing."
    })
    single_plan = plan.model_copy(update={"clips": [clip]})
    kie = FakeKie()
    product = tmp_path / "product.jpg"
    product.write_bytes(b"product")
    monkeypatch.setattr("ecom_video.workflow.make_silent_mp4", lambda src, dst: (dst.write_bytes(src.read_bytes()), dst)[1])

    run_multiclip_generation(single_plan, [product], [], tmp_path / "regeneration", kie)

    submitted = next(call[1] for call in kie.calls if call[0] == "video")
    assert "Clip-specific revision: Keep the branded box visible throughout the unboxing." in submitted
    assert len(submitted) <= 3800


def test_generation_under_one_dollar_runs_without_an_additional_approval(tmp_path, monkeypatch):
    plan = build_multiclip_plan(
        analysis_fixture(), "creative direction", "product details",
        "living room", clip_count=3, seconds_per_clip=5,
    )
    product = tmp_path / "product.jpg"
    product.write_bytes(b"product")
    kie = FakeKie()

    monkeypatch.setattr("ecom_video.workflow.make_silent_mp4", lambda src, dst: (dst.write_bytes(src.read_bytes()), dst)[1])
    outputs = run_multiclip_generation(plan, [product], [], tmp_path / "clips", kie)

    assert len(outputs) == 3
    assert sum(call[0] == "video" for call in kie.calls) == 3


def test_one_dollar_threshold_requires_approval_only_above_limit():
    from ecom_video.workflow import generation_approval_required
    def estimate(total):
        return GenerationEstimate(1, 1, 1, Decimal(total), Decimal("0"), Decimal("0"))
    assert generation_approval_required(estimate("1.00")) is False
    assert generation_approval_required(estimate("1.0001")) is True
    assert generation_approval_required(estimate("0.10"), pricing_verified=False) is True
    assert generation_approval_required(estimate("0.10"), estimate_reliable=False) is True


def test_four_nine_second_clips_are_under_one_dollar(tmp_path, monkeypatch):
    plan = build_multiclip_plan(analysis_fixture(), "creative direction", "product details", "living room", 3, 9)
    product = tmp_path / "product.jpg"; product.write_bytes(b"product")
    kie = FakeKie()
    monkeypatch.setattr("ecom_video.workflow.make_silent_mp4", lambda src, dst: (dst.write_bytes(src.read_bytes()), dst)[1])
    assert len(run_multiclip_generation(plan, [product], [], tmp_path / "clips", kie)) == 3


def test_generation_over_fifty_cents_requires_approval_before_provider_calls(tmp_path):
    plan = build_multiclip_plan(
        analysis_fixture(), "creative direction", "product details",
        "living room", clip_count=3, seconds_per_clip=15,
    )
    product = tmp_path / "product.jpg"
    product.write_bytes(b"product")
    kie = FakeKie()

    with pytest.raises(PermissionError, match="approval"):
        run_multiclip_generation(plan, [product], [], tmp_path / "clips", kie)

    assert kie.calls == []


def test_generation_rejects_approval_below_estimate_without_provider_calls(tmp_path):
    plan = build_multiclip_plan(
        analysis_fixture(), "creative direction", "product details",
        "living room", clip_count=3, seconds_per_clip=5,
    )
    product = tmp_path / "product.jpg"
    product.write_bytes(b"product")
    kie = FakeKie()

    with pytest.raises(PermissionError, match="exceeds approved maximum"):
        run_multiclip_generation(
            plan, [product], [], tmp_path / "clips", kie,
            approved_max_usd=Decimal("0.42"),
        )

    assert kie.calls == []


def test_unverified_pricing_requires_approval_even_for_low_cost_job(tmp_path):
    plan = build_multiclip_plan(
        analysis_fixture(), "creative direction", "product details",
        "living room", clip_count=3, seconds_per_clip=5,
    )
    product = tmp_path / "product.jpg"
    product.write_bytes(b"product")
    kie = FakeKie()

    with pytest.raises(PermissionError, match="approval"):
        run_multiclip_generation(
            plan, [product], [], tmp_path / "clips", kie,
            pricing_verified=False,
        )

    assert kie.calls == []


def test_unreliable_estimate_requires_approval_even_for_low_cost_job(tmp_path):
    plan = build_multiclip_plan(
        analysis_fixture(), "creative direction", "product details",
        "living room", clip_count=3, seconds_per_clip=5,
    )
    product = tmp_path / "product.jpg"
    product.write_bytes(b"product")
    kie = FakeKie()

    with pytest.raises(PermissionError, match="estimate is unreliable"):
        run_multiclip_generation(
            plan, [product], [], tmp_path / "clips", kie,
            estimate_reliable=False,
        )

    assert kie.calls == []


def test_cli_starts_verified_job_below_threshold_without_approval_flag(tmp_path, monkeypatch):
    plan = build_multiclip_plan(
        analysis_fixture(), "creative direction", "product details",
        "living room", clip_count=3, seconds_per_clip=5,
    )
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(plan.model_dump_json())
    args = SimpleNamespace(
        plan=str(plan_path), output_dir=str(tmp_path / "clips"),
        approve_max_usd=None, require_approval=False, dry_run=False,
    )
    calls = {}
    fake_client = SimpleNamespace(close=lambda: None)
    monkeypatch.setattr("ecom_video.providers.kie.KieClient", lambda: fake_client)

    def fake_run(*args, **kwargs):
        calls["kwargs"] = kwargs
        return [tmp_path / "clips" / "clip-01.mp4"]

    monkeypatch.setattr(cli, "run_multiclip_generation", fake_run)
    cli._generate_multiclip(args)

    assert calls["kwargs"]["approved_max_usd"] is None
    assert calls["kwargs"]["pricing_verified"] is True
    assert calls["kwargs"]["estimate_reliable"] is True


def test_cli_dry_run_never_initializes_kie_or_calls_generation(tmp_path, monkeypatch):
    plan = build_multiclip_plan(
        analysis_fixture(), "creative direction", "product details",
        "living room", clip_count=3, seconds_per_clip=5,
    )
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(plan.model_dump_json())
    args = SimpleNamespace(
        plan=str(plan_path), output_dir=str(tmp_path / "clips"),
        approve_max_usd=None, require_approval=False, dry_run=True,
    )
    monkeypatch.setattr(
        "ecom_video.providers.kie.KieClient",
        lambda: pytest.fail("Kie must not be initialized for a dry run"),
    )
    monkeypatch.setattr(
        cli, "run_multiclip_generation",
        lambda *a, **kw: pytest.fail("generation runner must not be called for a dry run"),
    )

    cli._generate_multiclip(args)


@pytest.mark.parametrize("require_approval", [False, True])
def test_cli_holds_over_threshold_or_uncertain_job_without_approval(tmp_path, monkeypatch, require_approval):
    plan = build_multiclip_plan(
        analysis_fixture(), "creative direction", "product details",
        "living room", clip_count=3, seconds_per_clip=5 if require_approval else 15,
    )
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(plan.model_dump_json())
    args = SimpleNamespace(
        plan=str(plan_path), output_dir=str(tmp_path / "clips"),
        approve_max_usd=None, require_approval=require_approval, dry_run=False,
    )
    monkeypatch.setattr(
        "ecom_video.providers.kie.KieClient",
        lambda: pytest.fail("Kie must not be initialized while approval is required"),
    )
    monkeypatch.setattr(
        cli, "run_multiclip_generation",
        lambda *a, **kw: pytest.fail("generation runner must not be called before approval"),
    )

    cli._generate_multiclip(args)
