from __future__ import annotations

import json
import subprocess
from decimal import Decimal
from pathlib import Path

import httpx
import pytest

from ecom_video.costs import estimate_generation
from ecom_video.media import make_silent_mp4, probe_video, sample_frames
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
        "continuity_group": "bottle-demo", "continuity_notes": ["Keep the bottle and hand consistent."],
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
    result = openai.analyze_reference([frame], "Show a different model", "Bottle is 20 cm tall", reference_duration_seconds=11.84)
    assert isinstance(result, AnalysisResult)
    assert result.shots[0].end_seconds == 6.8
    assert seen["url"] == openai.URL
    assert seen["headers"]["Authorization"] == "Bearer test-key"
    request = seen["json"]
    assert request["model"] == openai.MODEL
    assert request["text"]["format"]["type"] == "json_schema"
    shot_schema = request["text"]["format"]["schema"]["properties"]["shots"]["items"]
    assert shot_schema["properties"]["start_seconds"]["type"] == ["number", "null"]
    assert {"creative_role", "starting_state", "action", "ending_state", "camera", "continuity_group", "continuity_notes"} <= set(shot_schema["required"])
    image_part = next(part for part in request["input"][0]["content"] if part["type"] == "input_image")
    assert image_part["detail"] == "high"
    assert result.shots[0].start_seconds == 4.4
    analysis_prompt = request["input"][0]["content"][0]["text"]
    assert "11.84 seconds" in analysis_prompt
    assert "source-video timestamps" in analysis_prompt
    assert "concrete physical action" in analysis_prompt
    assert "same stable continuity_group" in analysis_prompt
    assert any("4.400 seconds" in part.get("text", "") for part in request["input"][0]["content"])


def test_openai_reference_analysis_sends_long_creator_brief_without_truncation(tmp_path, monkeypatch):
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
    brief = "Shared garment and character rules.\n" + ("Distinct request requirement. " * 700)

    openai.analyze_reference([frame], brief, "Product remains unchanged")

    analysis_text = seen["json"]["input"][0]["content"][0]["text"]
    assert len(brief) > 2000
    assert brief in analysis_text
    assert "complete creator instructions as authoritative" in analysis_text
    assert "shared requirements" in analysis_text


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
    assert body["input"]["resolution"] == "720p"
    with pytest.raises(ValueError, match="1 and 15"):
        client.create_video("too long", "https://files.test/start.jpg", 16)


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
