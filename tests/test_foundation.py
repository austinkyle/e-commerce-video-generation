from __future__ import annotations

import json
import subprocess
from decimal import Decimal
from pathlib import Path

import httpx
import pytest

from ecom_video.costs import estimate_generation
from ecom_video.media import make_mp4, make_silent_mp4, probe_video, sample_frames
from ecom_video.models import AnalysisResult
from ecom_video.providers import kie, openai


class FakeKieHTTP:
    def __init__(self, *args, **kwargs):
        self.calls = []

    def post(self, path, json):
        self.calls.append(("post", path, json))
        return httpx.Response(200, json={"code": 200, "data": {"taskId": "task-123"}},
                              request=httpx.Request("POST", f"https://api.kie.ai{path}"))

    def get(self, path, params):
        self.calls.append(("get", path, params))
        return httpx.Response(200, json={"code": 200, "data": {"state": "success", "creditsConsumed": 4.5}},
                              request=httpx.Request("GET", f"https://api.kie.ai{path}"))

    def close(self):
        pass


def test_openai_reference_analysis_uses_structured_schema_and_sampled_images(tmp_path, monkeypatch):
    frame = tmp_path / "frame-01-4.400s.jpg"
    frame.write_bytes(b"fake-image")
    result_json = {"summary": "A hand demonstrates the bottle.", "questions": [], "shots": [{
        "index": 1, "start_seconds": 4.4, "end_seconds": 6.8, "visual_description": "Close-up",
        "creative_role": "Product demonstration", "starting_state": "Bottle is centered in hand",
        "action": "Turn bottle", "ending_state": "Label faces camera", "camera": "Handheld close-up",
        "product_interaction": "Held in one hand", "hook_or_payoff": "Label reveal",
        "audio_requested": True, "audio_direction": "Natural cap click synchronized with the turn.",
        "continuity_group": "bottle-demo", "character_groups": ["model"], "continuity_notes": ["Keep the bottle and hand consistent."],
        "uncertainties": [],
    }]}

    class Response:
        def raise_for_status(self): pass
        def json(self): return {"output": [{"content": [{"type": "output_text", "text": json.dumps(result_json)}] }]}

    seen = {}
    def fake_post(url, **kwargs):
        seen.update(url=url, **kwargs)
        return Response()

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(openai.httpx, "post", fake_post)
    result = openai.analyze_reference(
        [frame], "Show a different model", "Bottle is 20 cm tall", reference_duration_seconds=11.84,
        hook="in_store_reaction", clip_count=4, seconds_per_clip=5, reference_has_audio=True,
    )
    assert isinstance(result, AnalysisResult)
    assert result.shots[0].end_seconds == 6.8
    assert seen["url"] == openai.URL
    assert seen["headers"]["Authorization"] == "Bearer test-key"
    request = seen["json"]
    assert request["model"] == openai.MODEL
    assert request["text"]["format"]["type"] == "json_schema"
    shot_schema = request["text"]["format"]["schema"]["properties"]["shots"]["items"]
    assert shot_schema["properties"]["start_seconds"]["type"] == ["number", "null"]
    assert {"creative_role", "starting_state", "action", "ending_state", "camera", "audio_requested", "audio_direction", "continuity_group", "character_groups", "continuity_notes"} <= set(shot_schema["required"])
    assert {"source_observation", "applied_user_changes"} <= set(shot_schema["required"])
    assert result.shots[0].audio_requested is True
    assert result.shots[0].audio_direction.startswith("Natural cap click")
    assert result.shots[0].character_groups == ["model"]
    image_part = next(part for part in request["input"][0]["content"] if part["type"] == "input_image")
    assert image_part["detail"] == "high"
    assert result.shots[0].start_seconds == 4.4
    analysis_prompt = request["input"][0]["content"][0]["text"]
    assert "11.84 seconds" in analysis_prompt
    assert "source-video timestamps" in analysis_prompt
    assert "concrete physical action" in analysis_prompt
    assert "Assign stable character_groups" in analysis_prompt
    assert "creator-requested character replacements take precedence" in analysis_prompt
    assert "put a concise faithful account of the unmodified source scene in source_observation" in analysis_prompt.lower()
    assert "list only explicit creator changes affecting that scene in applied_user_changes" in analysis_prompt.lower()
    assert "never omit or merge observed source scenes" in analysis_prompt.lower()
    assert "actual final scene" in analysis_prompt
    assert "Requested clip count 4 is a preference and cannot remove or merge a source scene" in analysis_prompt
    assert "5s interface duration only as a fallback" in analysis_prompt
    assert "Reference audio stream metadata: present" in analysis_prompt
    assert "rather than the audio waveform" in analysis_prompt
    assert "final edit duration" in analysis_prompt
    assert "Never equate absolute source timestamps with target or final edit duration" in analysis_prompt
    assert any("4.400 seconds" in part.get("text", "") for part in request["input"][0]["content"])


def test_reveal_reference_analysis_identifies_mechanism_and_keeps_source_scenes(tmp_path, monkeypatch):
    frame = tmp_path / "frame-01-3.200s.jpg"
    frame.write_bytes(b"fake-image")

    class Response:
        def raise_for_status(self): pass
        def json(self): return {"output": [{"content": [{"type": "output_text", "text": json.dumps({"summary": "Covered device revealed and activated.", "questions": [], "shots": []})}]}]}

    seen = {}
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(openai.httpx, "post", lambda url, **kwargs: (seen.update(kwargs) or Response()))
    openai.analyze_reference(
        [frame], "Replace the product only.", "Use supplied product images.", reference_duration_seconds=17.7,
        hook="countdown_reveal", clip_count=2, seconds_per_clip=5,
    )
    prompt = seen["json"]["input"][0]["content"][0]["text"]

    assert "3-2-1 Reveal is a curiosity-and-payoff format" in prompt
    assert "never infer a literal 3-2-1 countdown" in prompt.lower()
    assert "whether the reveal is physical or by edit" in prompt
    assert "preserve every distinct source scene and order" in prompt.lower()
    assert "only explicit creator substitutions" in prompt


def test_openai_reference_analysis_sends_forty_thousand_character_brief_without_truncation(tmp_path, monkeypatch):
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"fake-image")
    output = {"summary": "Shared continuity requirements.", "questions": [], "shots": [{
        "index": 1, "start_seconds": 0, "end_seconds": 3, "visual_description": "Target scene",
        "creative_role": "Hook", "starting_state": "Product is on shelf", "action": "Pick product up",
        "ending_state": "Product is held at chest height", "camera": "Handheld", "product_interaction": "Keep product stable",
        "hook_or_payoff": "Product reveal", "continuity_group": "store", "continuity_notes": [], "uncertainties": [],
    }]}
    class Response:
        def raise_for_status(self): pass
        def json(self): return {"output": [{"content": [{"type": "output_text", "text": json.dumps(output)}]}]}
    seen = {}
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(openai.httpx, "post", lambda url, **kwargs: (seen.update(kwargs) or Response()))
    brief_prefix = "Shared garment and character rules.\n"
    phrase = "Distinct request requirement. "
    brief = brief_prefix + phrase * ((40_000 - len(brief_prefix)) // len(phrase))
    brief += "x" * (40_000 - len(brief))

    openai.analyze_reference([frame], brief, "Product remains unchanged")

    analysis_text = seen["json"]["input"][0]["content"][0]["text"]
    assert len(brief) == 40_000
    assert brief in analysis_text
    assert "complete creator instructions as authoritative" in analysis_text
    assert "shared requirements" in analysis_text


def test_in_store_reaction_director_prompt_defines_hook_variation_and_story_sequence(monkeypatch):
    output = {"summary": "Store discovery followed by a product demonstration.", "questions": [], "shots": []}

    class Response:
        def raise_for_status(self): pass
        def json(self): return {"output": [{"content": [{"type": "output_text", "text": json.dumps(output)}]}]}

    seen = {}
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(openai.httpx, "post", lambda url, **kwargs: (seen.update(kwargs) or Response()))

    result = openai.direct_original("A useful product for pet owners", [], "in_store_reaction")

    assert result.summary.startswith("Store discovery")
    prompt = seen["json"]["input"][0]["content"][0]["text"]
    for requirement in (
        "first generated hook clip", "first second", "one continuous hook shot", "natural handheld smartphone movement",
        "gasp or hand over mouth", "pointing", "brief freeze", "drink drop is optional",
        "distinctive product mechanism", "never impose arbitrary fixed scene lengths", "output remains silent",
        "derive pacing and shot lengths from its rhythm",
        "unboxing and setup are optional", "use", "advance the story",
        "later scenes may use different people", "character_groups",
        "actual distribution is established",
    ):
        assert requirement.lower() in prompt.lower()


def test_non_in_store_director_does_not_receive_in_store_only_guidance(monkeypatch):
    output = {"summary": "A useful home product.", "questions": [], "shots": []}

    class Response:
        def raise_for_status(self): pass
        def json(self): return {"output": [{"content": [{"type": "output_text", "text": json.dumps(output)}]}]}

    seen = {}
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(openai.httpx, "post", lambda url, **kwargs: (seen.update(kwargs) or Response()))

    openai.direct_original("A product demo in a home", [], "recommend")

    prompt = seen["json"]["input"][0]["content"][0]["text"]
    assert "drink drop is optional" not in prompt.lower()
    assert "do not spend the first second establishing the store" not in prompt.lower()
    assert "actual distribution is established" not in prompt.lower()


def test_non_in_store_reference_analysis_does_not_receive_in_store_only_guidance(tmp_path, monkeypatch):
    output = {"summary": "A useful home product.", "questions": [], "shots": []}

    class Response:
        def raise_for_status(self): pass
        def json(self): return {"output": [{"content": [{"type": "output_text", "text": json.dumps(output)}]}]}

    frame = tmp_path / "frame-0.0s.jpg"
    frame.write_bytes(b"fake frame")
    seen = {}
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(openai.httpx, "post", lambda url, **kwargs: (seen.update(kwargs) or Response()))

    openai.analyze_reference([frame], "A home product demo", hook="recommend")

    prompt = seen["json"]["input"][0]["content"][0]["text"]
    assert "For In-Store Reaction, treat the immediate reaction" not in prompt
    assert "drink drop is optional" not in prompt.lower()

    openai.analyze_reference([frame], "An In-Store Reaction product discovery", hook="in_store_reaction")
    in_store_prompt = seen["json"]["input"][0]["content"][0]["text"]
    assert "analyze every distinct source scene independently" in in_store_prompt.lower()
    assert "ordered within-shot actions" in in_store_prompt.lower()


def test_legacy_wet_works_hook_keeps_in_store_reaction_direction():
    from ecom_video.models import hook_direction

    direction = hook_direction("wet_works")
    assert "In-Store Reaction" in direction
    assert "immediate" in direction.lower()
    assert "drink" in direction.lower()


def test_new_hook_identifier_is_accepted_by_original_job_and_legacy_identifier_is_preserved(tmp_path):
    from ecom_video.web_app import LocalJobService

    service = LocalJobService(tmp_path / "app-data", analyzer=lambda *args, **kwargs: None,
                              client_factory=lambda: object(), director=lambda *args: None)
    service._spawn = lambda target, *args: None

    new_job = service.create_original_job("A retro telephone", "in_store_reaction", 1, 5)
    legacy_job = service.create_original_job("A retro telephone", "wet_works", 1, 5)

    assert new_job["hook"] == "in_store_reaction"
    assert legacy_job["hook"] == "wet_works"
    assert service.get_job(legacy_job["id"])["hook"] == "wet_works"


def test_analysis_rejects_more_than_24_frames(tmp_path):
    paths = [tmp_path / f"{i}.jpg" for i in range(25)]
    with pytest.raises(ValueError, match="24"):
        openai.analyze_reference(paths, "instructions")


def test_analysis_schema_allows_no_exact_reference_segment():
    result = AnalysisResult.model_validate({"summary": "An original scene.", "questions": [], "shots": [{
        "index": 1, "start_seconds": None, "end_seconds": None, "visual_description": "A written-direction scene",
        "action": "Show the requested action", "camera": "Portrait framing",
        "product_interaction": "Hold the product naturally", "hook_or_payoff": "Reveal the product",
        "uncertainties": [],
    }]})
    assert result.shots[0].start_seconds is None
    assert result.shots[0].end_seconds is None


def test_kie_image_task_uses_documented_model_and_payload(monkeypatch):
    fake = FakeKieHTTP()
    monkeypatch.setenv("KIE_API_KEY", "test-key")
    monkeypatch.setattr(kie.httpx, "Client", lambda **kwargs: fake)
    client = kie.KieClient()
    task_id = client.create_image("Keep product shape", ["https://files.test/product.jpg"])
    method, endpoint, body = fake.calls[0]
    assert task_id == "task-123"
    assert endpoint == "/api/v1/jobs/createTask"
    assert body == {"model": "gpt-image-2-5-sunburst-image-to-image", "input": {
        "prompt": "Keep product shape", "input_urls": ["https://files.test/product.jpg"],
        "aspect_ratio": "9:16", "resolution": "1K", "background": "opaque"}}


def test_kie_masked_image_edit_uses_26_endpoint_and_first_reference_mask(monkeypatch):
    fake = FakeKieHTTP()
    monkeypatch.setenv("KIE_API_KEY", "test-key")
    monkeypatch.setattr(kie.httpx, "Client", lambda **kwargs: fake)
    client = kie.KieClient()
    task_id = client.create_masked_image_edit(
        "Change only facial expressions", ["https://files.test/v8.png"], "https://files.test/face-mask.png"
    )
    _, endpoint, body = fake.calls[0]
    assert task_id == "task-123"
    assert endpoint == "/api/v1/jobs/createTask"
    assert body == {"model": "gpt-image-2-6/sunburst-image-to-image", "input": {
        "prompt": "Change only facial expressions", "input_urls": ["https://files.test/v8.png"],
        "mask_url": "https://files.test/face-mask.png", "aspect_ratio": "9:16",
        "resolution": "1K", "quality": "medium", "output_format": "png", "background": "opaque"}}


def test_kie_masked_image_edit_rejects_missing_source_or_non_https_mask(monkeypatch):
    fake = FakeKieHTTP()
    monkeypatch.setenv("KIE_API_KEY", "test-key")
    monkeypatch.setattr(kie.httpx, "Client", lambda **kwargs: fake)
    client = kie.KieClient()
    with pytest.raises(ValueError, match="1 to 16"):
        client.create_masked_image_edit("Edit", [], "https://files.test/mask.png")
    with pytest.raises(ValueError, match="HTTPS"):
        client.create_masked_image_edit("Edit", ["https://files.test/v8.png"], "file:///mask.png")
    assert fake.calls == []


def test_kie_video_submission_is_silent_and_validated(monkeypatch):
    fake = FakeKieHTTP()
    monkeypatch.setenv("KIE_API_KEY", "test-key")
    monkeypatch.setattr(kie.httpx, "Client", lambda **kwargs: fake)
    client = kie.KieClient()
    assert client.create_video("Gentle hand movement", "https://files.test/start.jpg", 5) == "task-123"
    body = fake.calls[0][2]
    assert body["model"] == "grok-imagine-video-1-5-preview"
    assert body["input"]["duration"] == 5
    assert body["input"]["image_urls"] == ["https://files.test/start.jpg"]
    assert body["input"]["enable_audio"] is False


def test_kie_video_references_keep_starting_frame_first_and_limit_total_images(monkeypatch):
    fake = FakeKieHTTP()
    monkeypatch.setenv("KIE_API_KEY", "test-key")
    monkeypatch.setattr(kie.httpx, "Client", lambda **kwargs: fake)
    client = kie.KieClient()

    client.create_video(
        "Use @image2 for the product only.", "https://files.test/start.png", 5,
        resolution="720p", reference_image_urls=["https://files.test/product.png"],
    )
    body = fake.calls[0][2]
    assert body["input"]["image_urls"] == [
        "https://files.test/start.png", "https://files.test/product.png",
    ]

    with pytest.raises(ValueError, match="at most 7"):
        client.create_video(
            "Too many references", "https://files.test/start.png", 5,
            resolution="720p",
            reference_image_urls=[f"https://files.test/product-{i}.png" for i in range(7)],
        )
    assert len(fake.calls) == 1
    assert body["input"]["resolution"] == "720p"
    with pytest.raises(ValueError, match="1 and 15"):
        client.create_video("too long", "https://files.test/start.jpg", 16)


def test_kie_native_audio_uses_provider_default_without_disabling_flag(monkeypatch):
    fake = FakeKieHTTP()
    monkeypatch.setenv("KIE_API_KEY", "test-key")
    monkeypatch.setattr(kie.httpx, "Client", lambda **kwargs: fake)
    client = kie.KieClient()

    assert client.create_video("A cup falls and splashes", "https://files.test/start.jpg", 5, enable_audio=None) == "task-123"
    body = fake.calls[0][2]
    assert "enable_audio" not in body["input"]


def test_kie_video_prompt_rejects_over_limit_before_http_submission(monkeypatch):
    fake = FakeKieHTTP()
    monkeypatch.setenv("KIE_API_KEY", "test-key")
    monkeypatch.setattr(kie.httpx, "Client", lambda **kwargs: fake)
    client = kie.KieClient()

    with pytest.raises(ValueError, match="4096 characters"):
        client.create_video("x" * 4097, "https://files.test/start.jpg", 5)

    assert fake.calls == []


def test_kie_video_prompt_accepts_exact_provider_limit(monkeypatch):
    fake = FakeKieHTTP()
    monkeypatch.setenv("KIE_API_KEY", "test-key")
    monkeypatch.setattr(kie.httpx, "Client", lambda **kwargs: fake)
    client = kie.KieClient()

    client.create_video("x" * 4096, "https://files.test/start.jpg", 5)

    assert len(fake.calls) == 1
    assert len(fake.calls[0][2]["input"]["prompt"]) == 4096


def test_kie_status_polling_parses_task_and_credits(monkeypatch):
    fake = FakeKieHTTP()
    monkeypatch.setenv("KIE_API_KEY", "test-key")
    monkeypatch.setattr(kie.httpx, "Client", lambda **kwargs: fake)
    client = kie.KieClient()
    status = client.task("task-123")
    assert fake.calls == [("get", "/api/v1/jobs/recordInfo", {"taskId": "task-123"})]
    assert status["data"]["state"] == "success"
    assert status["data"]["creditsConsumed"] == 4.5


@pytest.mark.parametrize(("seconds", "images", "total"), [
    (12, 3, Decimal("0.38")), (15, 3, Decimal("0.4475")), (5, 0, Decimal("0.1325")),
])
def test_generation_estimates(seconds, images, total):
    assert estimate_generation(seconds, images).total_usd == total


def test_ffmpeg_probe_and_frame_sampling(tmp_path):
    source = tmp_path / "fixture.mp4"
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
        "testsrc=size=320x240:rate=12", "-t", "2", "-pix_fmt", "yuv420p", "-y", str(source),
    ], check=True, capture_output=True)
    info = probe_video(source)
    assert info["duration"] > 1.9
    assert (info["width"], info["height"]) == (320, 240)
    frames = sample_frames(source, tmp_path / "frames", count=3, long_edge=200)
    assert len(frames) == 3
    assert all(frame.is_file() and frame.stat().st_size > 0 for frame in frames)
    assert "1.800s" in frames[-1].name


def test_silent_export_removes_generated_audio(tmp_path):
    source = tmp_path / "with-audio.mp4"
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
        "testsrc=size=320x240:rate=12", "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
        "-t", "1", "-pix_fmt", "yuv420p", "-shortest", "-y", str(source),
    ], check=True, capture_output=True)
    assert probe_video(source)["has_audio"] is True
    silent = make_silent_mp4(source, tmp_path / "silent.mp4")
    assert silent.is_file()
    assert probe_video(silent)["has_audio"] is False


def test_audio_preserving_export_keeps_native_audio_stream(tmp_path):
    source = tmp_path / "with-audio.mp4"
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
        "testsrc=size=320x240:rate=12", "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
        "-t", "1", "-pix_fmt", "yuv420p", "-shortest", "-y", str(source),
    ], check=True, capture_output=True)
    preserved = make_mp4(source, tmp_path / "preserved.mp4", preserve_audio=True)
    assert probe_video(preserved)["has_audio"] is True


def test_audio_remux_preserves_relative_audio_video_timestamps(tmp_path):
    source = tmp_path / "offset-audio.mp4"
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
        "color=c=black:s=320x240:r=24:d=2", "-itsoffset", "0.12", "-f", "lavfi", "-i",
        "sine=frequency=440:sample_rate=48000:duration=1.8", "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "libx264", "-c:a", "aac", "-shortest", "-y", str(source),
    ], check=True, capture_output=True)

    source_info = probe_video(source)
    preserved = make_mp4(source, tmp_path / "offset-preserved.mp4", preserve_audio=True)
    output_info = probe_video(preserved)

    assert source_info["audio_video_start_offset"] == pytest.approx(output_info["audio_video_start_offset"], abs=0.02)
    assert output_info["audio_video_start_offset"] > 0.05


def test_preserving_requested_audio_fails_when_provider_returns_silent_clip(tmp_path):
    source = tmp_path / "silent-source.mp4"
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
        "color=c=black:s=320x240:r=24:d=1", "-c:v", "libx264", "-an", "-y", str(source),
    ], check=True, capture_output=True)

    with pytest.raises(RuntimeError, match="no audio stream"):
        make_mp4(source, tmp_path / "requested-audio.mp4", preserve_audio=True)
