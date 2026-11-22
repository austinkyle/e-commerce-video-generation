from __future__ import annotations

import json
import time
from decimal import Decimal
from http.client import HTTPConnection
from pathlib import Path
from threading import Thread
from urllib.request import urlopen

import pytest

from ecom_video.models import AnalysisResult
from ecom_video.costs import DEFAULT_PROJECT_SPENDING_LIMIT_USD, estimate_within_approved_amount
from ecom_video.web_app import AppError, ConfirmationRequired, LocalJobService, _parse_multipart, create_server


def sample_analysis() -> AnalysisResult:
    return AnalysisResult.model_validate({
        "summary": "A person demonstrates a product in three distinct moments.",
        "questions": [],
        "shots": [{
            "index": index, "start_seconds": (index - 1) * 2, "end_seconds": index * 2,
            "visual_description": f"Distinct reference moment {index}.", "action": f"Action {index}",
            "camera": f"Handheld shot {index}", "product_interaction": "The product is held naturally.",
            "hook_or_payoff": "A clear product detail.", "uncertainties": [],
        } for index in range(1, 4)],
    })


def service_factory(tmp_path, *, usage=None, client_factory=lambda: object(), director=None):
    def analyzer(frames, instructions, product_facts, **kwargs):
        return sample_analysis()
    analyzer.last_usage = usage if usage is not None else {"input_tokens": 1000, "output_tokens": 50}

    def sampler(video_path, frame_dir, count=8):
        frame_dir.mkdir(parents=True, exist_ok=True)
        frames = []
        for index in range(count):
            frame = frame_dir / f"frame-{index:02d}.jpg"
            frame.write_bytes(b"frame")
            frames.append(frame)
        return frames

    service = LocalJobService(
        tmp_path / "app-data", analyzer=analyzer, client_factory=client_factory,
        director=director or __import__("ecom_video.providers.openai", fromlist=["direct_original"]).direct_original,
        frame_sampler=sampler,
        video_probe=lambda path: {"path": str(path.resolve()), "duration": 8, "width": 720, "height": 1280, "has_audio": False},
    )
    return service


def test_original_interview_persists_and_builds_generation_ready_plan(tmp_path):
    calls = []
    def director(instructions, answers, hook):
        calls.append((instructions, list(answers), hook))
        director.last_usage = {"input_tokens": 1000, "output_tokens": 100}
        if not answers:
            return AnalysisResult(summary="Clarify the product", questions=["How does the toy move, and what surface does it work on?"], shots=[])
        result = sample_analysis()
        for index, shot in enumerate(result.shots):
            shot.creative_role = ["Hook", "Demonstration", "Payoff"][index]
            shot.start_seconds = shot.end_seconds = None
            shot.continuity_group = "home"
        result.summary = "A coherent original product advertisement."
        return result
    director.last_usage = {}
    service = service_factory(tmp_path, director=director)
    service._spawn = lambda target, *args: target(*args)
    created = service.create_original_job("A small wall-crawling gecko toy ad", "countdown_reveal", 3, 5)
    assert created["status"] == "interviewing"
    assert service.get_job(created["id"])["director_questions"]
    restored = LocalJobService(tmp_path / "app-data", director=director)
    restored._spawn = lambda target, *args: target(*args)
    waiting = restored.get_job(created["id"])
    assert waiting["status"] == "interviewing"
    assert waiting["director_questions"] == ["How does the toy move, and what surface does it work on?"]
    ready = restored.answer_original_interview(created["id"], "It uses suction to crawl slowly on smooth glass; surprise me about the hook.")
    assert ready["status"] == "ready"
    assert ready["hook"] == "countdown_reveal"
    assert [clip["creative_role"] for clip in ready["clips"]] == ["Hook", "Demonstration", "Payoff"]
    assert all(clip["source_start_seconds"] is None for clip in ready["clips"])
    assert calls[-1][1] == ["It uses suction to crawl slowly on smooth glass; surprise me about the hook."]


def test_original_no_photo_generation_uses_text_image_and_video_requests(tmp_path, monkeypatch):
    from ecom_video.workflow import run_multiclip_generation

    class TextOnlyClient:
        def __init__(self): self.calls = []
        def upload(self, path): self.calls.append(("upload", path)); return "https://asset.test/ref.png"
        def create_text_image(self, prompt, aspect_ratio="9:16"):
            self.calls.append(("text_image", prompt)); return "image-task"
        def create_image(self, *args, **kwargs): raise AssertionError("No photo references should use text-to-image")
        def wait_for_task(self, task_id, on_state=None): return {"creditsConsumed": 6 if task_id == "image-task" else 22.5, "resultJson": '{"resultUrls":["https://asset.test/out"]}'}
        def create_video(self, prompt, image_url, duration_seconds, **kwargs):
            self.calls.append(("video", prompt, image_url, duration_seconds)); return "video-task"
        def download_result(self, data, destination): destination.parent.mkdir(parents=True, exist_ok=True); destination.write_bytes(b"mock"); return destination

    import ecom_video.workflow as workflow
    monkeypatch.setattr(workflow, "make_silent_mp4", lambda source, destination: (destination.write_bytes(b"silent"), destination)[1])
    service = service_factory(tmp_path)
    service._spawn = lambda target, *args: target(*args)
    def completed_director(instructions, answers, hook):
        result = sample_analysis()
        for shot in result.shots: shot.start_seconds = shot.end_seconds = None
        completed_director.last_usage = {"input_tokens": 100, "output_tokens": 10}
        return result
    completed_director.last_usage = {}
    service.director = completed_director
    job = service.create_original_job("A toy that crawls on glass", "recommend", 1, 5)
    assert job["status"] == "ready"
    client = TextOnlyClient()
    output = run_multiclip_generation(
        __import__("ecom_video.workflow", fromlist=["MultiClipPlan"]).MultiClipPlan.model_validate_json(service._read(job["id"])["plan_json"]),
        [], [], tmp_path / "outputs", client,
    )
    assert len(output) == 1
    assert [call[0] for call in client.calls if call[0] in {"text_image", "video"}] == ["text_image", "video"]


def test_original_mode_http_interview_survives_reload_and_displays_plan(tmp_path):
    def director(instructions, answers, hook):
        director.last_usage = {"input_tokens": 500, "output_tokens": 50}
        if not answers:
            return AnalysisResult(summary="Clarify the product", questions=["What physically moves, and how?"] , shots=[])
        return sample_analysis()
    director.last_usage = {}
    service = service_factory(tmp_path, director=director)
    service._spawn = lambda target, *args: target(*args)
    server = create_server(0, service)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        conn = HTTPConnection("127.0.0.1", server.server_port, timeout=3)
        boundary = "original-boundary"
        body = (
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"mode\"\r\n\r\noriginal\r\n"
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"hook\"\r\n\r\nwet_works\r\n"
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"instructions\"\r\n\r\nA gecko toy ad\r\n"
            f"--{boundary}--\r\n"
        ).encode()
        conn.request("POST", "/api/jobs", body, {"Content-Type": f"multipart/form-data; boundary={boundary}"})
        response = conn.getresponse(); created = json.loads(response.read())
        assert response.status == 202 and created["status"] == "interviewing"
        conn.request("GET", f"/api/jobs/{created['id']}")
        waiting = json.loads(conn.getresponse().read())
        assert waiting["director_questions"] == ["What physically moves, and how?"]
        assert waiting["review_images_before_video"] is True
        conn.request("POST", f"/api/jobs/{created['id']}/answer", json.dumps({"answer": "It crawls on smooth glass by suction."}), {"Content-Type": "application/json"})
        ready = json.loads(conn.getresponse().read())
        assert ready["status"] == "ready" and len(ready["clips"]) == 3
        html = (Path(__file__).parents[1] / "src/ecom_video/web/index.html").read_text()
        script = (Path(__file__).parents[1] / "src/ecom_video/web/app.js").read_text()
        assert 'value="original"' in html and 'id="answerForm"' in html
        assert "Creative Plan" in script
        assert "/answer" in script and "director_questions" in script
        conn.close()
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=2)


def test_generation_estimate_is_described_as_footage_and_intended_edit_length():
    from ecom_video.web_app import WEB_ROOT
    script = (WEB_ROOT / "app.js").read_text()
    assert "seconds generated" in script
    assert "intended final edit" in script


def test_refine_plan_preserves_context_and_unaffected_shots_without_reanalysis(tmp_path):
    director_calls = []
    revision_calls = []
    def director(instructions, answers, hook):
        director_calls.append((instructions, list(answers), hook))
        director.last_usage = {"input_tokens": 500, "output_tokens": 50}
        if not answers:
            return AnalysisResult(summary="Need product mechanics", questions=["How does it work?"], shots=[])
        return sample_analysis()
    director.last_usage = {}
    def reviser(instruction, plan, context):
        revision_calls.append((instruction, plan, context))
        reviser.last_usage = {"input_tokens": 400, "output_tokens": 40}
        shots = [shot.model_dump(mode="json") for shot in AnalysisResult.model_validate(service._read(job_id)["analysis"]).shots]
        shots[0]["action"] = "The gift recipient opens the box and lifts out the gecko toy."
        return {"summary": "Gift reveal followed by product demonstration and payoff.", "changed_shot_indices": [1], "shots": shots}
    reviser.last_usage = {}
    service = service_factory(tmp_path, director=director)
    service._spawn = lambda target, *args: target(*args)
    initial = service.create_original_job("A gecko toy gift ad", "gifting", 3, 5)
    job_id = initial["id"]
    service.answer_original_interview(job_id, "It crawls on smooth glass using suction cups.")
    before = service._read(job_id)
    old = __import__("ecom_video.workflow", fromlist=["MultiClipPlan"]).MultiClipPlan.model_validate_json(before["plan_json"])
    service.plan_reviser = reviser
    ready = service.refine_plan(job_id, "Make clip 1 a gift-opening hook; preserve the product specifications and remaining shots.")
    after = __import__("ecom_video.workflow", fromlist=["MultiClipPlan"]).MultiClipPlan.model_validate_json(service._read(job_id)["plan_json"])
    assert ready["status"] == "ready"
    assert after.clips[0].action.startswith("The gift recipient opens")
    assert after.clips[1:] == old.clips[1:]
    assert ready["estimate"]["total_usd"] != before["estimate"]["total_usd"]
    assert revision_calls[0][2]["interview_answers"] == ["It crawls on smooth glass using suction cups."]
    assert "A gecko toy gift ad" == revision_calls[0][2]["original_instructions"]
    assert "suction cups" in revision_calls[0][2]["confirmed_product_facts"] or "suction cups" in revision_calls[0][2]["original_instructions"]
    assert len(director_calls) == 2


def test_refine_plan_requires_ready_job_and_renders_controls():
    from ecom_video.web_app import WEB_ROOT
    html = (WEB_ROOT / "index.html").read_text()
    script = (WEB_ROOT / "app.js").read_text()
    assert 'id="refinePlanInput"' in html and 'id="updatePlanButton"' in html
    assert '"/refine"' not in script  # endpoint uses the stable per-job route template below
    assert "/refine`" in script


def create_ready_job(service: LocalJobService, tmp_path: Path) -> dict:
    image = b"\xff\xd8\xff" + b"fake-jpeg"
    created = service.create_job(
        "reference.mp4", b"not-real-but-the-probe-is-mocked", [("product.jpg", image)],
        "Keep the product appearance and change the setting.", 3, 5, review_images_before_video=False,
    )
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        job = service.get_job(created["id"])
        if job["status"] not in {"queued", "analyzing_reference"}:
            return job
        time.sleep(0.01)
    raise AssertionError("Mocked reference analysis did not finish")


def test_draft_persists_and_restores_inputs_without_analysis_or_duplicate_media(tmp_path):
    calls = []
    service = service_factory(tmp_path)
    service.analyzer = lambda *args, **kwargs: (calls.append("analysis") or sample_analysis())
    service._spawn = lambda target, *args: calls.append("spawned")
    fields = {
        "mode": "reference", "hook": "in_store_reaction", "instructions": "Surprise me with the product reveal.",
        "clip_count": "director", "seconds_per_clip": "director", "review_images_before_video": "true",
        "product_asset_ids": "[]",
    }
    created = service.save_draft(
        fields, {"reference_video": [("reference.mp4", b"reference-video")],
                 "product_images": [("product.png", b"\x89PNG\r\n\x1a\nproduct-image")]},
    )
    job_id = created["id"]
    assert created["status"] == "draft"
    assert calls == []
    assert created["mode"] == "reference"
    assert created["hook"] == "in_store_reaction"
    assert created["clip_count_selection"] == "director"
    assert created["duration_selection"] == "director"
    assert created["review_images_before_video"] is True
    assert len([asset for asset in created["inputs"] if asset["role"] == "reference_video"]) == 1
    assert len([asset for asset in created["inputs"] if asset["role"] == "product_reference"]) == 1

    restored_service = LocalJobService(
        tmp_path / "app-data", analyzer=service.analyzer, director=service.director,
        client_factory=service.client_factory, frame_sampler=service.frame_sampler,
        video_probe=service.video_probe,
    )
    restored = restored_service.get_job(job_id)
    assert restored["instructions"] == fields["instructions"]
    assert restored["requested_clip_count"] == 12
    assert restored["seconds_per_clip"] == 5
    for asset in restored["inputs"]:
        asset_path, _, _ = restored_service.get_asset(job_id, asset["asset_id"])
        assert asset_path.is_file()
    assert [asset["asset_id"] for asset in restored["inputs"]] == [asset["asset_id"] for asset in created["inputs"]]

    image_asset = next(asset for asset in restored["inputs"] if asset["role"] == "product_reference")
    updated = restored_service.save_draft({
        **fields, "mode": "original", "hook": "gifting", "instructions": "Updated concept.",
        "clip_count": "4", "seconds_per_clip": "8", "review_images_before_video": "false",
        "product_asset_ids": json.dumps([image_asset["asset_id"]]),
    }, {}, job_id)
    assert updated["mode"] == "original"
    assert updated["instructions"] == "Updated concept."
    assert updated["hook"] == "gifting"
    assert updated["requested_clip_count"] == 4
    assert updated["seconds_per_clip"] == 8
    assert updated["review_images_before_video"] is False
    assert len(updated["inputs"]) == 2
    assert len(list((tmp_path / "app-data" / "jobs" / job_id / "inputs").glob("*"))) == 2
    assert calls == []


def test_reference_form_exposes_separate_creation_mode_and_creative_format_choices():
    from ecom_video.web_app import WEB_ROOT

    html = (WEB_ROOT / "index.html").read_text()
    script = (WEB_ROOT / "app.js").read_text()
    assert '<label>Creation mode <select id="creationMode">' in html
    assert '<label id="hookField">Creative format <select id="hookType">' in html
    for value, label in (
        ("in_store_reaction", "In-Store Reaction"),
        ("countdown_reveal", "3-2-1 Reveal"),
        ("gifting", "Gifting"),
        ("versus", "Versus"),
    ):
        assert f'<option value="{value}"' in html and label in html
    assert "hookField.hidden = !original" not in script
    assert "form.append('mode', $('creationMode').value)" in script
    assert "form.append('hook', $('hookType').value)" in script
    assert "restoreSelectValue($('hookType'), job.hook" in script


@pytest.mark.parametrize("hook", ["recommend", "in_store_reaction", "countdown_reveal", "gifting", "versus"])
def test_reference_draft_selection_reaches_director_and_saved_plan(tmp_path, hook):
    seen = []

    def analyzer(frames, instructions, product_facts, **kwargs):
        seen.append(kwargs["hook"])
        analyzer.last_usage = {"input_tokens": 100, "output_tokens": 10}
        return sample_analysis()
    analyzer.last_usage = {}

    service = service_factory(tmp_path)
    service.analyzer = analyzer
    service._spawn = lambda target, *args: target(*args)
    draft = service.save_draft({
        "mode": "reference", "hook": hook, "instructions": "Follow the uploaded reference.",
        "clip_count": "3", "seconds_per_clip": "5", "review_images_before_video": "true",
        "product_asset_ids": "[]",
    }, {"reference_video": [("reference.mp4", b"reference")]})

    ready = service.analyze_draft(draft["id"])
    saved_plan = __import__("ecom_video.workflow", fromlist=["MultiClipPlan"]).MultiClipPlan.model_validate_json(
        service._read(draft["id"])["plan_json"]
    )
    assert ready["hook"] == hook
    assert seen == [hook]
    assert saved_plan.hook == hook


def test_explicit_saved_format_wins_over_legacy_instruction_inference(tmp_path):
    from ecom_video.web_app import LocalJobService
    from ecom_video.workflow import MultiClipPlan

    plan = MultiClipPlan.model_validate({
        "summary": "A gifting product ad.", "environment": "home", "instructions": "Mention In-Store Reaction in the creative notes.",
        "product_facts": "A product", "hook": "custom", "clips": [{
            "index": 1, "title": "Gift reveal", "source_shot_index": 1,
            "creative_role": "Gifting hook", "visual_description": "A recipient sees a wrapped gift.",
            "starting_state": "The wrapped product is on a table.", "action": "The recipient opens the gift.",
            "ending_state": "The product is revealed.", "camera": "Handheld medium shot.",
            "duration_seconds": 5, "starting_image_prompt": "A gift on a table.", "video_prompt": "",
        }],
    })
    service = LocalJobService(tmp_path / "app-data", analyzer=lambda *args, **kwargs: None,
                              client_factory=lambda: object(), director=lambda *args: None)
    loaded = service._load_plan({
        "plan_json": plan.model_dump_json(), "hook": "gifting",
        "instructions": "Mention In-Store Reaction in the creative notes.",
    })
    assert loaded.hook == "gifting"


def test_analyzing_saved_draft_requires_explicit_action_and_reuses_saved_reference(tmp_path):
    service = service_factory(tmp_path)
    service._spawn = lambda target, *args: target(*args)
    analyzer_settings = {}

    def analyzer(frames, instructions, product_facts, **kwargs):
        analyzer_settings.update(kwargs)
        return sample_analysis()
    analyzer.last_usage = {"input_tokens": 100, "output_tokens": 10}
    service.analyzer = analyzer
    draft = service.save_draft({
        "mode": "reference", "hook": "custom", "instructions": "Use the original footage.",
        "clip_count": "director", "seconds_per_clip": "director", "review_images_before_video": "false",
        "product_asset_ids": "[]",
    }, {"reference_video": [("input.mp4", b"saved-video")]})
    assert draft["status"] == "draft"
    ready = service.analyze_draft(draft["id"])
    assert ready["status"] == "ready"
    reference_path, _, _ = service.get_asset(ready["id"], ready["inputs"][0]["asset_id"])
    assert reference_path.read_bytes() == b"saved-video"
    assert ready["analysis"] is not None
    assert analyzer_settings["clip_count"] == 12
    assert analyzer_settings["duration_selection"] == "director"


def test_public_existing_project_restores_reference_metadata_and_saved_settings(tmp_path):
    service = service_factory(tmp_path)
    service._spawn = lambda target, *args: target(*args)
    ready = create_ready_job(service, tmp_path)
    restored = service.get_job(ready["id"])

    assert restored["mode"] == "reference"
    assert restored["requested_clip_count"] == 3
    assert restored["seconds_per_clip"] == 5
    assert any(item["role"] == "reference_video" and item["asset_id"] for item in restored["inputs"])
    assert any(item["role"] == "product_reference" and item["asset_id"] for item in restored["inputs"])
    assert restored["clips"] and restored["estimate"]


def test_draft_http_save_and_restore_serves_existing_media_without_running_workers(tmp_path):
    from http.client import HTTPConnection
    from threading import Thread
    from ecom_video.web_app import create_server

    service = service_factory(tmp_path)
    spawned = []
    service._spawn = lambda target, *args: spawned.append(target.__name__)
    server = create_server(0, service)
    thread = Thread(target=server.serve_forever, daemon=True); thread.start()
    try:
        boundary = "workspace-draft"
        parts = [
            ("mode", "reference"), ("hook", "gifting"), ("instructions", "Restore this complete product workspace."),
            ("clip_count", "director"), ("seconds_per_clip", "director"),
            ("review_images_before_video", "true"), ("product_asset_ids", "[]"),
            ("reference_video", "reference.mp4", b"saved-reference"),
            ("product_images", "product.png", b"\x89PNG\r\n\x1a\nthumb"),
        ]
        chunks = []
        for item in parts:
            if len(item) == 2:
                name, value = item
                chunks.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n".encode())
            else:
                name, filename, value = item
                chunks.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"; filename=\"{filename}\"\r\nContent-Type: application/octet-stream\r\n\r\n".encode() + value + b"\r\n")
        body = b"".join(chunks) + f"--{boundary}--\r\n".encode()
        conn = HTTPConnection("127.0.0.1", server.server_port, timeout=3)
        conn.request("POST", "/api/drafts", body, {"Content-Type": f"multipart/form-data; boundary={boundary}"})
        response = conn.getresponse(); draft = json.loads(response.read())
        assert response.status == 200 and draft["status"] == "draft" and spawned == []

        conn.request("GET", f"/api/jobs/{draft['id']}")
        reopened = json.loads(conn.getresponse().read())
        assert reopened["instructions"] == "Restore this complete product workspace."
        assert reopened["clip_count_selection"] == "director" and reopened["duration_selection"] == "director"
        video_asset = next(asset for asset in reopened["inputs"] if asset["role"] == "reference_video")
        conn.request("GET", video_asset["url"])
        assert conn.getresponse().read() == b"saved-reference"
        assert spawned == []

        conn.request("POST", f"/api/jobs/{draft['id']}/analyze", b"{}", {"Content-Type": "application/json"})
        submitted = json.loads(conn.getresponse().read())
        assert submitted["status"] == "queued" and spawned == ["_analyze_job"]
        conn.close()
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=2)


def test_job_analysis_returns_plan_estimate_and_redacts_local_paths(tmp_path):
    service = service_factory(tmp_path)
    job = create_ready_job(service, tmp_path)

    assert job["status"] == "ready"
    assert job["analysis"]["summary"].startswith("A person")
    assert len(job["clips"]) == 3
    assert Decimal(job["estimate"]["total_usd"]) < Decimal("1.00")
    assert all("path" not in asset for asset in job["inputs"])
    assert str(tmp_path) not in json.dumps(job)
    assert job["inputs"][0]["role"] in {"reference_video", "product_reference"}


def test_twenty_thousand_character_instruction_reaches_analysis_and_persists_intact(tmp_path):
    core = "Shared requirements: same character, product details, and lighting.\n\n"
    core += "Long creative detail. " * ((19_997 - len(core)) // len("Long creative detail. "))
    core += "x" * (19_997 - len(core))
    instructions = " " + core + " \n"
    observed = {}

    def analyzer(frames, received_instructions, product_facts, **kwargs):
        observed["instructions"] = received_instructions
        return sample_analysis()
    analyzer.last_usage = {"input_tokens": 2000, "output_tokens": 100}
    service = LocalJobService(
        tmp_path / "app-data", analyzer=analyzer,
        frame_sampler=lambda video, directory, count=8: [video],
        video_probe=lambda path: {"path": str(path), "duration": 5},
    )
    job = service.create_job("reference.mp4", b"video", [], instructions, 1, 5)
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        job = service.get_job(job["id"])
        if job["status"] not in {"queued", "analyzing_reference"}:
            break
        time.sleep(0.01)

    assert len(instructions) == 20_000
    assert observed["instructions"] == instructions
    assert service._read(job["id"])["instructions"] == instructions


def test_creative_instructions_over_twenty_thousand_characters_are_rejected(tmp_path):
    service = service_factory(tmp_path)
    with pytest.raises(AppError, match="20,000"):
        service.create_job("reference.mp4", b"video", [], "x" * 20_001)


def test_creative_instructions_ui_advertises_twenty_thousand_character_limit():
    from ecom_video.web_app import WEB_ROOT

    html = (WEB_ROOT / "index.html").read_text()
    script = (WEB_ROOT / "app.js").read_text()
    assert 'maxlength="20000"' in html
    assert "0 / 20,000" in html
    assert "MAX_INSTRUCTION_LENGTH = 20_000" in script


def test_project_names_have_mode_appropriate_defaults_and_rename_survives_restart(tmp_path):
    service = service_factory(tmp_path)
    service._spawn = lambda target, *args: None
    reference = service.create_job(
        "FurGlide shelf demo.mp4", b"video", [], "Make a product ad.", 1, 5,
    )
    original = service.create_original_job(
        "I want to create a 15-second ecommerce ad for FurGlide.", "wet_works", 1, 5,
    )

    assert reference["title"] == "FurGlide shelf demo"
    assert "FurGlide" in original["title"]
    assert original["title"] != "Untitled reference"

    path = service._job_path(original["id"])
    stored = json.loads(path.read_text())
    stored.update(
        title="Untitled reference", status="ready", stage="ready",
        interview_answers=["Confirmed the teal housing and washable silicone ridges."],
        plan_json='{"approved":"plan"}',
        clips=[{"index": 1, "status": "completed", "versions": [{
            "version": 1, "starting_image_asset": "saved-image", "video_asset": "saved-video",
            "duration_seconds": 5, "revision": "", "actual_kie_usd_equivalent": "0.03",
        }]}],
        assets={
            "saved-image": {"path": "/local/starting.png", "filename": "starting.png", "media_type": "image/png", "role": "starting_image"},
            "saved-video": {"path": "/local/clip.mp4", "filename": "clip.mp4", "media_type": "video/mp4", "role": "silent_video"},
        },
        charges={"analysis_usd": "0.02", "kie_credits": "6", "kie_usd_equivalent": "0.03", "total_usd": "0.05", "entries": [{"kind": "test"}]},
    )
    service._save(stored)
    before = service._read(original["id"])

    renamed = service.rename_project(original["id"], "  FurGlide   Home Demo  ")

    assert renamed["id"] == original["id"]
    assert renamed["title"] == "FurGlide Home Demo"
    after = service._read(original["id"])
    assert {key: value for key, value in after.items() if key not in {"title", "updated_at"}} == {
        key: value for key, value in before.items() if key not in {"title", "updated_at"}
    }
    restored = LocalJobService(tmp_path / "app-data")
    assert restored.get_job(original["id"])["title"] == "FurGlide Home Demo"


def test_project_rename_rejects_empty_or_overlong_names(tmp_path):
    service = service_factory(tmp_path)
    service._spawn = lambda target, *args: None
    job = service.create_original_job("A pet product video concept")

    with pytest.raises(AppError, match="cannot be empty"):
        service.rename_project(job["id"], "  \n ")
    with pytest.raises(AppError, match="100 characters"):
        service.rename_project(job["id"], "x" * 101)
    assert service.rename_project(job["id"], "x" * 100)["title"] == "x" * 100


def test_project_rename_is_available_through_local_http_api_and_recent_work_ui(tmp_path):
    from ecom_video.web_app import WEB_ROOT

    service = service_factory(tmp_path)
    service._spawn = lambda target, *args: None
    job = service.create_original_job("A FurGlide product concept")
    server = create_server(0, service)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        connection = HTTPConnection("127.0.0.1", server.server_port, timeout=3)
        connection.request(
            "POST", f"/api/jobs/{job['id']}/rename", json.dumps({"title": "FurGlide Launch"}),
            {"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        renamed = json.loads(response.read())
        assert response.status == 200
        assert renamed["title"] == "FurGlide Launch"
        connection.request("GET", "/api/jobs")
        listed = json.loads(connection.getresponse().read())
        assert next(item for item in listed["jobs"] if item["id"] == job["id"])["title"] == "FurGlide Launch"
        connection.close()
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()

    script = (WEB_ROOT / "app.js").read_text()
    css = (WEB_ROOT / "app.css").read_text()
    assert "history-rename" in script
    assert "keyEvent.key === 'Enter'" in script
    assert "keyEvent.key === 'Escape'" in script
    assert "input.maxLength = 100" in script
    assert "/rename`" in script
    assert ".history-rename" in css
    assert ".history-name{display:block;overflow:hidden;text-overflow:ellipsis" in css
    assert "$('topbarTitle').textContent = updated.title" in script


def test_delete_project_removes_only_its_files_and_preserves_neighbors(tmp_path):
    service = service_factory(tmp_path)
    service._spawn = lambda target, *args: None
    selected = service.create_original_job("Selected product concept")
    neighbor = service.create_original_job("Neighbor project")
    for job in (selected, neighbor):
        service._update(job["id"], lambda stored: stored.update(status="completed", stage="completed"))
    selected_root = service._job_path(selected["id"]).parent
    neighbor_root = service._job_path(neighbor["id"]).parent
    selected_asset = selected_root / "outputs" / "initial" / "clip-01.mp4"
    selected_asset.parent.mkdir(parents=True)
    selected_asset.write_bytes(b"selected generated video")
    shared_file = service.data_dir / "shared-config.json"
    shared_file.write_text('{"preserve":true}')
    neighbor_file = neighbor_root / "outputs" / "keep.txt"
    neighbor_file.parent.mkdir(parents=True)
    neighbor_file.write_text("neighbor data")

    result = service.delete_project(selected["id"])

    assert result["id"] == selected["id"] and result["deleted"] is True
    assert result["recovered_bytes"] >= len(b"selected generated video")
    assert not selected_root.exists()
    assert neighbor_root.is_dir() and neighbor_file.read_text() == "neighbor data"
    assert shared_file.read_text() == '{"preserve":true}'
    assert next(job for job in service.list_jobs() if job["id"] == neighbor["id"])
    assert all(job["id"] != selected["id"] for job in service.list_jobs())


def test_delete_project_refuses_invalid_ids_and_project_directory_symlinks(tmp_path):
    service = service_factory(tmp_path)
    with pytest.raises(AppError):
        service.delete_project("../../outside")

    project_id = "a1234567-1234-1234-1234-123456789abc"
    outside = tmp_path / "outside-project"
    outside.mkdir()
    outside_file = outside / "keep.txt"
    outside_file.write_text("outside")
    (service.jobs_dir / project_id).symlink_to(outside, target_is_directory=True)

    with pytest.raises(AppError):
        service.delete_project(project_id)

    assert outside_file.read_text() == "outside"


def test_delete_project_refuses_symlinked_jobs_storage_root(tmp_path):
    data_root = tmp_path / "app-data"
    outside_root = tmp_path / "outside-jobs"
    outside_root.mkdir(parents=True)
    project_id = "b1234567-1234-1234-1234-123456789abc"
    project_root = outside_root / project_id
    project_root.mkdir()
    (project_root / "job.json").write_text(json.dumps({"id": project_id, "status": "completed"}))
    (project_root / "keep.txt").write_text("outside")
    data_root.mkdir()
    (data_root / "jobs").symlink_to(outside_root, target_is_directory=True)
    service = LocalJobService(data_root)

    with pytest.raises(AppError, match="safe local directory"):
        service.delete_project(project_id)

    assert (project_root / "keep.txt").read_text() == "outside"


@pytest.mark.parametrize("status", ["interviewing", "generating_clips", "regenerating"])
def test_delete_project_refuses_while_background_work_is_active(tmp_path, status):
    service = service_factory(tmp_path)
    service._spawn = lambda target, *args: None
    job = service.create_original_job("Project with work in progress")
    service._update(job["id"], lambda stored: stored.update(status=status, stage=status))
    project_root = service._job_path(job["id"]).parent

    with pytest.raises(AppError, match="active"):
        service.delete_project(job["id"])

    assert project_root.is_dir()


def test_delete_project_reports_filesystem_errors_without_success(tmp_path, monkeypatch):
    import ecom_video.web_app as web_app

    service = service_factory(tmp_path)
    service._spawn = lambda target, *args: None
    job = service.create_original_job("Project with file cleanup error")
    service._update(job["id"], lambda stored: stored.update(status="completed", stage="completed"))
    project_root = service._job_path(job["id"]).parent
    monkeypatch.setattr(web_app.shutil, "rmtree", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk error")))

    with pytest.raises(AppError, match="could not be deleted"):
        service.delete_project(job["id"])

    assert project_root.is_dir()


def test_storage_estimates_project_and_total_bytes_without_following_symlinks(tmp_path):
    service = service_factory(tmp_path)
    service._spawn = lambda target, *args: None
    job = service.create_original_job("Storage estimate")
    root = service._job_path(job["id"]).parent
    payload = root / "outputs" / "video.mp4"
    payload.parent.mkdir(parents=True)
    payload.write_bytes(b"123456789")
    external = tmp_path / "external.bin"
    external.write_bytes(b"outside")
    (root / "external-link").symlink_to(external)

    listed = next(item for item in service.list_jobs() if item["id"] == job["id"])
    summary = service.storage_summary()

    assert listed["storage_bytes"] >= len(b"123456789")
    assert summary["project_count"] == 1
    assert summary["total_bytes"] >= listed["storage_bytes"]
    assert summary["total_bytes"] < listed["storage_bytes"] + external.stat().st_size


def test_delete_preserves_asset_still_referenced_by_another_project(tmp_path):
    service = service_factory(tmp_path)
    service._spawn = lambda target, *args: None
    owner = service.create_original_job("Owner")
    variation = service.create_original_job("Variation")
    shared = service._job_path(owner["id"]).parent / "inputs" / "shared.png"
    shared.parent.mkdir(parents=True, exist_ok=True)
    shared.write_bytes(b"shared image")
    for item in (owner, variation):
        service._update(item["id"], lambda job: job.update(status="completed", stage="completed"))
    service._update(owner["id"], lambda job: job["assets"].update(shared={"path": str(shared), "filename": "shared.png", "media_type": "image/png", "role": "product_reference"}))
    service._update(variation["id"], lambda job: job["assets"].update(shared={"path": str(shared), "filename": "shared.png", "media_type": "image/png", "role": "product_reference"}))

    service.delete_project(owner["id"])

    assert shared.read_bytes() == b"shared image"
    assert service._read(variation["id"])["assets"]["shared"]["path"] == str(shared)
    assert service.estimate_deletion([variation["id"]])["recoverable_bytes"] >= shared.stat().st_size
    service.delete_project(variation["id"])
    assert not shared.exists()


def test_bulk_delete_is_atomic_for_active_projects_and_deletes_selected_only(tmp_path):
    service = service_factory(tmp_path)
    service._spawn = lambda target, *args: None
    completed = service.create_original_job("Completed")
    active = service.create_original_job("Active")
    service._update(completed["id"], lambda job: job.update(status="completed", stage="completed"))
    service._update(active["id"], lambda job: job.update(status="generating_clips", stage="generating_clips"))

    with pytest.raises(AppError, match="active"):
        service.delete_projects([completed["id"], active["id"]])
    assert service._job_path(completed["id"]).exists()

    result = service.delete_projects([completed["id"]])
    assert result["deleted_count"] == 1
    assert result["recovered_bytes"] > 0
    assert service._job_path(active["id"]).exists()


def test_temporary_cleanup_only_removes_old_unreferenced_temp_files(tmp_path):
    import os
    import time

    service = service_factory(tmp_path)
    service._spawn = lambda target, *args: None
    job = service.create_original_job("Cleanup")
    service._update(job["id"], lambda stored: stored.update(status="completed", stage="completed"))
    root = service._job_path(job["id"]).parent
    stale = root / "job.json.tmp"
    stale.write_text("stale")
    recent = root / "recent.part"
    recent.write_text("recent")
    old_media = root / "outputs" / "approved.mp4"
    old_media.parent.mkdir()
    old_media.write_bytes(b"keep")
    old = time.time() - 49 * 60 * 60
    os.utime(stale, (old, old))
    os.utime(old_media, (old, old))

    result = service.cleanup_temporary_files()

    assert result["removed_count"] == 1
    assert not stale.exists()
    assert recent.exists() and old_media.exists()


def test_deletion_estimate_excludes_assets_used_by_nonselected_project(tmp_path):
    service = service_factory(tmp_path)
    service._spawn = lambda target, *args: None
    first = service.create_original_job("Selected")
    second = service.create_original_job("Reference")
    shared = service._job_path(first["id"]).parent / "inputs" / "shared.png"
    shared.parent.mkdir(exist_ok=True)
    shared.write_bytes(b"same source")
    for item in (first, second):
        service._update(item["id"], lambda job: job.update(status="completed", stage="completed"))
        service._update(item["id"], lambda job: job["assets"].update(shared={"path": str(shared), "filename": "shared.png"}))

    estimate = service.estimate_deletion([first["id"]])
    assert estimate["project_count"] == 1
    assert shared.stat().st_size > 0
    assert estimate["recoverable_bytes"] > 0


def test_delete_refuses_nested_symlinks_and_unreadable_neighbor_records(tmp_path):
    service = service_factory(tmp_path)
    service._spawn = lambda target, *args: None
    project = service.create_original_job("Nested symlink")
    service._update(project["id"], lambda job: job.update(status="completed", stage="completed"))
    outside = tmp_path / "outside"
    outside.write_text("preserve")
    (service._job_path(project["id"]).parent / "linked-file").symlink_to(outside)
    with pytest.raises(AppError, match="symbolic link"):
        service.delete_project(project["id"])
    assert outside.read_text() == "preserve"

    (service._job_path(project["id"]).parent / "linked-file").unlink()
    neighbor = service.create_original_job("Unreadable neighbor")
    service._update(neighbor["id"], lambda job: job.update(status="completed", stage="completed"))
    service._job_path(neighbor["id"]).write_text("broken")
    with pytest.raises(AppError, match="could not be verified"):
        service.delete_project(project["id"])
    assert service._job_path(project["id"]).exists()


def test_list_jobs_endpoint_includes_project_and_total_storage(tmp_path):
    service = service_factory(tmp_path)
    service._spawn = lambda target, *args: None
    job = service.create_original_job("Storage API")
    server = create_server(0, service)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        connection = HTTPConnection("127.0.0.1", server.server_port, timeout=3)
        connection.request("GET", "/api/jobs")
        response = connection.getresponse()
        payload = json.loads(response.read())
        assert response.status == 200
        assert payload["storage"]["project_count"] == 1
        assert next(item for item in payload["jobs"] if item["id"] == job["id"])["storage_bytes"] > 0
        connection.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_delete_confirmation_and_recent_work_controls_are_connected(tmp_path):
    from ecom_video.web_app import WEB_ROOT

    html = (WEB_ROOT / "index.html").read_text()
    script = (WEB_ROOT / "app.js").read_text()
    css = (WEB_ROOT / "app.css").read_text()
    assert 'id="deleteProjectDialog"' in html
    assert "saved creative plans, interview history, generated videos, starting images, and local project files" in html
    assert 'id="cancelDeleteProject"' in html and 'id="confirmDeleteProject"' in html
    assert "history-delete" in script
    assert "/delete`" in script
    assert "cancelDeleteProject" in script
    assert "newProject').click()" in script
    assert ".history-delete" in css
    cancel_handler = "$('cancelDeleteProject').addEventListener('click', () => $('deleteProjectDialog').close())"
    assert cancel_handler in script
    assert "$('deleteProjectDialog').addEventListener('close', () => { state.pendingDeleteProjectId = null" in script
    assert "request(`/api/jobs/${projectIds[0]}/delete`" in script
    assert "request('/api/jobs/delete-bulk'" in script
    assert 'id="selectCompletedProjects"' in html
    assert "delete-estimate" in script and "storage_bytes" in script


def test_cancel_delete_confirmation_closes_without_sending_delete_request():
    from ecom_video.web_app import WEB_ROOT

    script = (WEB_ROOT / "app.js").read_text()
    cancel_handler = "$('cancelDeleteProject').addEventListener('click', () => $('deleteProjectDialog').close())"
    assert cancel_handler in script
    assert "$('deleteProjectDialog').addEventListener('close', () => { state.pendingDeleteProjectId = null" in script
    assert 'id="cancelDeleteProject" class="button button-secondary" type="button">Cancel</button>' in (WEB_ROOT / "index.html").read_text()


def test_delete_project_local_http_api_requires_explicit_post_and_returns_errors(tmp_path):
    service = service_factory(tmp_path)
    service._spawn = lambda target, *args: None
    job = service.create_original_job("Project through delete API")
    service._update(job["id"], lambda stored: stored.update(status="completed", stage="completed"))
    server = create_server(0, service)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        connection = HTTPConnection("127.0.0.1", server.server_port, timeout=3)
        connection.request("GET", f"/api/jobs/{job['id']}/delete")
        assert connection.getresponse().status == 404
        connection.request("POST", f"/api/jobs/{job['id']}/delete", "{}", {"Content-Type": "application/json"})
        response = connection.getresponse()
        assert response.status == 200
        deleted = json.loads(response.read())
        assert deleted["id"] == job["id"] and deleted["deleted"] is True
        assert deleted["recovered_bytes"] > 0
        connection.request("GET", f"/api/jobs/{job['id']}")
        assert connection.getresponse().status == 404
        connection.close()
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


def test_browser_plan_labels_reference_time_separately_from_target_duration():
    from ecom_video.web_app import WEB_ROOT

    script = (WEB_ROOT / "app.js").read_text()
    assert "Reference segment" in script
    assert "Generated clip:" in script
    assert "Intended final edit:" in script
    assert "No exact reference segment" in script


def test_legacy_in_store_plan_uses_saved_job_hook_when_compiling_video_prompt(tmp_path):
    from ecom_video.web_app import LocalJobService
    from ecom_video.workflow import MultiClipPlan, compile_video_prompt

    plan = MultiClipPlan.model_validate({
        "summary": "In-Store Reaction: elderly woman reacts at a store; a younger woman demonstrates the product at home.",
        "environment": "store and home", "instructions": "In-Store Reaction reference recreation",
        "product_facts": "A retro telephone", "hook": "custom", "clips": [{
            "index": 2, "title": "Home demonstration", "source_shot_index": 2,
            "creative_role": "Product demonstration", "visual_description": "A younger woman lifts the phone receiver at home.",
            "starting_state": "The younger woman is already holding the receiver.",
            "action": "Lift the receiver and bring it to her ear.", "ending_state": "She uses the phone naturally.",
            "product_interaction": "The younger woman lifts the receiver from its base.",
            "scene_payoff": "Show the receiver working as intended.", "camera": "Handheld close-up", "continuity_group": "home_user",
            "continuity_notes": ["Younger woman at home."], "duration_seconds": 3,
            "starting_image_prompt": "The younger woman uses the phone.", "video_prompt": "",
        }],
    })
    service = LocalJobService(tmp_path / "legacy-app", analyzer=lambda *args, **kwargs: None,
                              client_factory=lambda: object(), director=lambda *args: None)
    loaded = service._load_plan({
        "plan_json": plan.model_dump_json(), "hook": "in_store_reaction", "instructions": "In-Store Reaction reference recreation",
    })
    prompt = compile_video_prompt(loaded, loaded.clips[0])
    assert loaded.hook == "in_store_reaction"
    assert "elderly woman" not in prompt.lower()
    assert "younger woman" in prompt.lower()
    assert "preserve the person already visible in the supplied starting image" in prompt.lower()


def test_uncertain_analysis_cost_requires_confirmation_before_client_creation(tmp_path):
    client_calls = []
    service = service_factory(tmp_path, usage={}, client_factory=lambda: client_calls.append(True))
    job = create_ready_job(service, tmp_path)

    assert job["estimate_reliable"] is False
    with pytest.raises(ConfirmationRequired):
        service.start_generation(job["id"])
    assert client_calls == []


def test_project_default_and_display_precision_approval(tmp_path):
    service = service_factory(tmp_path)
    service._spawn = lambda target, *args: None
    job = service.create_original_job("Spending policy concept")

    assert job["project_spending_limit_usd"] == str(DEFAULT_PROJECT_SPENDING_LIMIT_USD) == "6.00"
    # Provider arithmetic is 0.723712 while the approval dialog displays 0.7237.
    assert estimate_within_approved_amount(Decimal("0.723712"), Decimal("0.7237"))
    assert not estimate_within_approved_amount(Decimal("0.723712"), Decimal("0.7236"))


def test_resume_pending_starting_image_uses_operation_cost_and_project_ceiling(tmp_path):
    service = service_factory(tmp_path)
    ready = create_ready_job(service, tmp_path)
    service._spawn = lambda target, *args: None
    service._update(ready["id"], lambda job: job.update(
        status="failed", stage="failed", active_stage="regenerating_starting_image",
        pending_starting_image_regen={"clip_index": 1, "instructions": "", "output_dir": str(tmp_path / "retry")},
    ))

    resumed = service.start_generation(ready["id"], resume=True)

    assert resumed["status"] == "preparing_images"
    assert service._read(ready["id"])["project_spending_limit_usd"] == "6.00"


def test_project_ceiling_counts_history_before_starting_paid_generation(tmp_path):
    service = service_factory(tmp_path)
    ready = create_ready_job(service, tmp_path)
    service._spawn = lambda target, *args: None
    service._update(ready["id"], lambda job: job.update(project_spending_limit_usd="0.10"))

    with pytest.raises(AppError, match="project limit"):
        service.start_generation(ready["id"])


def test_project_spending_accumulates_initial_generation_regenerations_and_plan_revision(tmp_path):
    service = service_factory(tmp_path)
    service._spawn = lambda target, *args: None
    job = service.create_original_job("Spending ledger concept")
    root = service._job_path(job["id"]).parent / "outputs"
    manifests = {
        "initial": [
            {"clip_index": 1, "kind": "starting_image", "task_id": "img-initial", "credits_consumed": "6", "status": "success"},
            {"clip_index": 1, "kind": "video", "task_id": "vid-initial", "credits_consumed": "22.5", "status": "success"},
        ],
        "clip-01/version-2": [
            {"clip_index": 1, "kind": "starting_image", "task_id": "img-revision-1", "credits_consumed": "6", "status": "success"},
            {"clip_index": 1, "kind": "video", "task_id": "vid-revision-1", "credits_consumed": "22.5", "status": "success"},
        ],
        "clip-02/version-2": [
            {"clip_index": 2, "kind": "starting_image", "task_id": "img-revision-2", "credits_consumed": "6", "status": "success"},
            {"clip_index": 2, "kind": "video", "task_id": "vid-revision-2", "credits_consumed": "22.5", "status": "success"},
        ],
    }
    for relative, tasks in manifests.items():
        folder = root / relative
        folder.mkdir(parents=True)
        (folder / "generation-usage.json").write_text(json.dumps({"tasks": tasks, "all_task_charges_reported": True}))
    service._update(job["id"], lambda saved: saved.update(charges={
        "analysis_usd": "0.025000", "kie_credits": "0", "kie_usd_equivalent": "0", "total_usd": "0.025000",
        "entries": [
            {"kind": "creative_director", "usd_equivalent": "0.005000", "source": "OpenAI token usage", "recorded_at": "2026-05-01T10:00:00+00:00"},
            {"kind": "plan_revision", "usd_equivalent": "0.020000", "source": "OpenAI token usage", "recorded_at": "2026-05-01T10:05:00+00:00"},
            {"kind": "kie_generation", "label": "initial_generation", "credits": "28.5", "usd_equivalent": "0.1425", "complete": True},
        ],
    }))

    spending = service.get_job(job["id"])["spending"]

    assert spending["recorded_usd_equivalent"] == "0.452500"
    assert spending["unknown_charge_count"] == 0
    assert len([record for record in spending["records"] if record["provider"] == "Kie.ai"]) == 6
    assert any(record["operation"] == "Creative plan refinement" and record["recorded_at"] == "2026-05-01T10:05:00+00:00" for record in spending["records"])


def test_project_spending_deduplicates_task_ids_and_includes_failed_charged_attempts(tmp_path):
    service = service_factory(tmp_path)
    service._spawn = lambda target, *args: None
    job = service.create_original_job("Failed task accounting")
    outputs = service._job_path(job["id"]).parent / "outputs"
    initial = outputs / "initial"; initial.mkdir(parents=True)
    regen = outputs / "clip-02" / "version-2"; regen.mkdir(parents=True)
    shared_task = {"clip_index": 1, "kind": "video", "task_id": "duplicate-task", "credits_consumed": "22.5", "status": "success"}
    (initial / "generation-usage.json").write_text(json.dumps({"tasks": [shared_task]}))
    (regen / "generation-usage.json").write_text(json.dumps({"tasks": [
        shared_task,
        {"clip_index": 2, "kind": "video", "task_id": "failed-but-charged", "credits_consumed": "7.5", "status": "failed"},
    ]}))
    spending = service.get_job(job["id"])["spending"]

    assert spending["recorded_usd_equivalent"] == "0.150000"
    tasks = [record for record in spending["records"] if record.get("task_id") == "duplicate-task"]
    assert len(tasks) == 1
    failed = next(record for record in spending["records"] if record.get("task_id") == "failed-but-charged")
    assert failed["status"] == "usage_recorded_failed_task"
    assert failed["usd_equivalent"] == "0.0375"


def test_project_spending_updates_from_partial_task_to_resumed_usage_without_duplicate(tmp_path):
    service = service_factory(tmp_path)
    service._spawn = lambda target, *args: None
    job = service.create_original_job("Partial spending recovery")
    manifest_path = service._job_path(job["id"]).parent / "outputs" / "initial" / "generation-usage.json"
    manifest_path.parent.mkdir(parents=True)
    manifest = {"expected_generation_tasks": 8, "tasks": [
        {"clip_index": 1, "kind": "starting_image", "task_id": "resume-image", "credits_consumed": "6", "status": "success"},
        {"clip_index": 1, "kind": "video", "task_id": "resume-video", "credits_consumed": None, "status": "submitted"},
    ]}
    manifest_path.write_text(json.dumps(manifest))

    partial = service.get_job(job["id"])["spending"]
    assert partial["recorded_usd_equivalent"] == "0.030000"
    assert partial["unknown_charge_count"] == 1

    manifest["tasks"][1].update(credits_consumed="22.5", status="success", completed_at="2026-05-01T10:12:00+00:00")
    manifest_path.write_text(json.dumps(manifest))
    resumed = service.get_job(job["id"])["spending"]
    assert resumed["recorded_usd_equivalent"] == "0.142500"
    assert resumed["unknown_charge_count"] == 0
    assert len([record for record in resumed["records"] if record.get("task_id") == "resume-video"]) == 1


def test_duplicate_task_record_without_usage_does_not_erase_reported_credits(tmp_path):
    service = service_factory(tmp_path)
    service._spawn = lambda target, *args: None
    job = service.create_original_job("Duplicate task reconciliation")
    outputs = service._job_path(job["id"]).parent / "outputs"
    initial = outputs / "initial"; initial.mkdir(parents=True)
    regen = outputs / "clip-01" / "version-2"; regen.mkdir(parents=True)
    (initial / "generation-usage.json").write_text(json.dumps({"tasks": [
        {"clip_index": 1, "kind": "video", "task_id": "same-task", "credits_consumed": "22.5", "status": "success"},
    ]}))
    (regen / "generation-usage.json").write_text(json.dumps({"tasks": [
        {"clip_index": 1, "kind": "video", "task_id": "same-task", "credits_consumed": None, "status": "submitted"},
    ]}))

    spending = service.get_job(job["id"])["spending"]

    assert spending["recorded_usd_equivalent"] == "0.112500"
    assert spending["unknown_charge_count"] == 0


def test_duplicate_task_id_with_conflicting_usage_is_shown_as_unknown(tmp_path):
    service = service_factory(tmp_path)
    service._spawn = lambda target, *args: None
    job = service.create_original_job("Conflicting task usage")
    outputs = service._job_path(job["id"]).parent / "outputs"
    first = outputs / "initial"; first.mkdir(parents=True)
    second = outputs / "clip-01" / "version-2"; second.mkdir(parents=True)
    for folder, credits in ((first, "22.5"), (second, "6")):
        (folder / "generation-usage.json").write_text(json.dumps({"tasks": [
            {"clip_index": 1, "kind": "video", "task_id": "conflicting-task", "credits_consumed": credits, "status": "success"},
        ]}))

    spending = service.get_job(job["id"])["spending"]

    assert spending["recorded_usd_equivalent"] == "0.000000"
    assert spending["unknown_charge_count"] == 1
    conflict = next(record for record in spending["records"] if record.get("task_id") == "conflicting-task")
    assert conflict["status"] == "unknown_usage"


def test_duplicate_task_id_with_conflicting_usage_is_shown_as_unknown(tmp_path):
    service = service_factory(tmp_path)
    service._spawn = lambda target, *args: None
    job = service.create_original_job("Conflicting task usage")
    outputs = service._job_path(job["id"]).parent / "outputs"
    first = outputs / "initial"; first.mkdir(parents=True)
    second = outputs / "clip-01" / "version-2"; second.mkdir(parents=True)
    for folder, credits in ((first, "22.5"), (second, "6")):
        (folder / "generation-usage.json").write_text(json.dumps({"tasks": [
            {"clip_index": 1, "kind": "video", "task_id": "conflicting-task", "credits_consumed": credits, "status": "success"},
        ]}))

    spending = service.get_job(job["id"])["spending"]

    assert spending["recorded_usd_equivalent"] == "0.000000"
    assert spending["unknown_charge_count"] == 1
    conflict = next(record for record in spending["records"] if record.get("task_id") == "conflicting-task")
    assert conflict["status"] == "unknown_usage"


def test_project_spending_labels_missing_usage_and_unknown_openai_usage(tmp_path):
    service = service_factory(tmp_path)
    service._spawn = lambda target, *args: None
    job = service.create_original_job("Unknown usage accounting")
    initial = service._job_path(job["id"]).parent / "outputs" / "initial"
    initial.mkdir(parents=True)
    (initial / "generation-usage.json").write_text(json.dumps({"tasks": [
        {"clip_index": 1, "kind": "starting_image", "task_id": "unreported-image", "credits_consumed": None, "status": "submitted"},
    ]}))
    service._update(job["id"], lambda saved: saved.update(charges={
        "analysis_usd": "0.020000", "kie_credits": "0", "kie_usd_equivalent": "0", "total_usd": "0.020000",
        "entries": [{"kind": "plan_revision", "usd_equivalent": "0.020000", "source": "usage unavailable; allowance applied"}],
    }))

    spending = service.get_job(job["id"])["spending"]

    assert spending["recorded_usd_equivalent"] == "0.000000"
    assert spending["unknown_charge_count"] == 2
    assert {record["status"] for record in spending["records"]} >= {"unknown_usage"}
    assert all(record["usd_equivalent"] is None for record in spending["records"] if record["status"] == "unknown_usage")


def test_project_spending_supports_legacy_aggregate_charges_without_manifests(tmp_path):
    service = service_factory(tmp_path)
    service._spawn = lambda target, *args: None
    job = service.create_original_job("Legacy accounting project")
    service._update(job["id"], lambda saved: saved.update(charges={
        "analysis_usd": "0.01", "kie_credits": "28.5", "kie_usd_equivalent": "0.1425", "total_usd": "0.1525",
        "entries": [
            {"kind": "creative_director", "usd_equivalent": "0.01", "source": "OpenAI token usage"},
            {"kind": "kie_generation", "label": "clip_1_v2", "credits": "28.5", "usd_equivalent": "0.1425", "complete": True},
        ],
    }))

    spending = service.get_job(job["id"])["spending"]

    assert spending["recorded_usd_equivalent"] == "0.152500"
    assert any(record["operation"] == "Clip 1 regeneration · aggregate history" for record in spending["records"])
    assert all(record["recorded_at"] is None for record in spending["records"])


def test_spending_panel_is_distinct_from_estimate_and_renders_expandable_history():
    from ecom_video.web_app import WEB_ROOT

    html = (WEB_ROOT / "index.html").read_text()
    script = (WEB_ROOT / "app.js").read_text()
    assert 'id="projectSpending"' in html
    assert 'id="projectSpendingTotal"' in html
    assert '<details class="spending-history">' in html
    assert 'id="spendingRecords"' in html
    assert "function renderSpending(job)" in script
    assert "job.spending" in script
    assert "Time not recorded (legacy record)" in script
    assert "Allowance" in script and "Amount unknown" in script


def test_starting_image_review_interface_is_a_single_approval_screen_with_selective_regeneration():
    from ecom_video.web_app import WEB_ROOT

    html = (WEB_ROOT / "index.html").read_text()
    script = (WEB_ROOT / "app.js").read_text()
    assert 'id="imageReviewSection"' in html
    assert '<details class="starting-image-checklist">' in html
    assert "Frame zero" in html and "Physical realism" in html
    assert "does not run paid image analysis or block approval" in html
    assert 'id="startingImageReview"' in html
    assert 'id="approveImagesButton"' in html
    assert 'Approve images &amp; generate videos' in html
    assert 'Regenerate Image · ~$0.03' in script
    assert '/approve-images' in script
    assert '/starting-image/regenerate' in script
    assert 'id="reviewImagesBeforeVideo" type="checkbox"' in html
    assert 'Review starting images before generating videos' in html
    assert 'creative_role || clip.title' in script
    assert 'Regenerate Image · ~$0.03' in script
    assert 'imageChecks.length !== (job.clips || []).length' in script
    assert 'Optional: preserve a scene with masked editing' in script
    assert 'Generate Masked Candidate · ~$0.05' in script
    assert 'body instanceof FormData' in script
    assert 'job.scene_base_images' in script
    assert 'preferring approved anchors; provisional images are not approved automatically' in script


def _png_header(width: int, height: int, color_type: int = 6) -> bytes:
    import struct
    return b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)


def test_masked_edit_is_invokable_through_normal_image_regeneration_route(tmp_path):
    class FakeKie:
        def __init__(self): self.calls = []
        def upload(self, path): self.calls.append(("upload", Path(path).name)); return f"https://fake.test/{Path(path).name}"
        def create_masked_image_edit(self, prompt, inputs, mask, aspect_ratio="9:16"):
            self.calls.append(("edit", prompt, list(inputs), mask, aspect_ratio)); return "task-1"
        def wait_for_task(self, task_id, on_state=None):
            return {"state": "success", "creditsConsumed": "8.1", "resultJson": '{"resultUrls":["https://fake.test/output.png"]}'}
        def download_result(self, task_data, destination): destination.write_bytes(_png_header(8, 12)); return destination
        def close(self): pass

    client = FakeKie()
    service = service_factory(tmp_path, client_factory=lambda: client)
    service._spawn = lambda target, *args: target(*args)
    ready = create_ready_job(service, tmp_path)
    raw = service._read(ready["id"])
    plan = __import__("ecom_video.workflow", fromlist=["MultiClipPlan"]).MultiClipPlan.model_validate_json(raw["plan_json"])
    job_root = service._job_path(ready["id"]).parent
    images = []
    for index in range(1, len(plan.clips) + 1):
        image = job_root / f"scene-{index}.png"
        image.write_bytes(_png_header(8, 12))
        images.append(image)
    clips = [clip.model_copy(update={
        "character_groups": ["mother", "baby"],
        "environment_references": [str(images[index])] if index == 2 else [],
    }) for index, clip in enumerate(plan.clips)]
    plan = plan.model_copy(update={"clips": clips})
    service._update(ready["id"], lambda data: data.update(
        status="awaiting_image_review", stage="awaiting_image_review", estimate_reliable=True,
        plan_json=plan.model_dump_json(),
    ))
    service._save_starting_image_assets(ready["id"], plan, images)
    current = service._read(ready["id"])
    base_asset = current["clips"][2]["current_starting_image_asset"]
    public = service.get_job(ready["id"])
    assert any(item["value"] == f"asset:{base_asset}" for item in public["scene_base_images"])
    assert any(item["value"] == "environment:3:0" for item in public["scene_base_images"])

    server = create_server(0, service)
    thread = Thread(target=server.serve_forever, daemon=True); thread.start()
    try:
        boundary = "masked-boundary"
        mask = _png_header(8, 12)
        body = (
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"edit_mode\"\r\n\r\nmasked\r\n"
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"base_source\"\r\n\r\nenvironment:3:0\r\n"
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"instructions\"\r\n\r\nKeep the same mother and baby, now smiling softly.\r\n"
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"mask_image\"; filename=\"mask.png\"\r\nContent-Type: image/png\r\n\r\n"
        ).encode() + mask + f"\r\n--{boundary}--\r\n".encode()
        conn = HTTPConnection("127.0.0.1", server.server_port, timeout=3)
        conn.request("POST", f"/api/jobs/{ready['id']}/clips/3/starting-image/regenerate", body,
                     {"Content-Type": f"multipart/form-data; boundary={boundary}"})
        response = conn.getresponse(); result = json.loads(response.read()); conn.close()
        assert response.status == 202 and result["status"] == "awaiting_image_review"
        edit = next(call for call in client.calls if call[0] == "edit")
        assert edit[2][0] == f"https://fake.test/{images[2].name}"
        assert "character reference is for identity only" in edit[1].lower()
        assert "product reference is for product appearance only" in edit[1].lower()
        assert edit[2][-1] == f"https://fake.test/{Path(plan.product_references[0]).name}"
        updated = service.get_job(ready["id"])
        assert len(updated["clips"][2]["starting_image_history"]) == 2
        assert updated["charges"]["kie_credits"] == "8.1"
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=2)


def test_approved_character_anchor_survives_unapproved_image_revision(tmp_path):
    from ecom_video.workflow import MultiClipPlan

    service = service_factory(tmp_path)
    ready = create_ready_job(service, tmp_path)
    raw = service._read(ready["id"])
    plan = MultiClipPlan.model_validate_json(raw["plan_json"])
    plan = plan.model_copy(update={"clips": [clip.model_copy(update={"character_groups": ["mother", "baby"]}) for clip in plan.clips]})
    images = []
    for index in range(1, 4):
        image = tmp_path / f"approved-{index}.png"; image.write_bytes(_png_header(8, 12)); images.append(image)
    service._update(ready["id"], lambda data: data.update(plan_json=plan.model_dump_json()))
    service._save_starting_image_assets(ready["id"], plan, images)
    approved = service._read(ready["id"])
    first_id = approved["clips"][0]["current_starting_image_asset"]
    service._update(ready["id"], lambda data: data.update(
        images_approved_at="2026-11-21T00:00:00+00:00",
        approved_starting_image_assets=[item["current_starting_image_asset"] for item in data["clips"]],
    ))
    revised = tmp_path / "unapproved-revision.png"; revised.write_bytes(_png_header(8, 12))
    service._save_starting_image_assets(ready["id"], plan.model_copy(update={"clips": [plan.clips[0]]}), [revised])
    current = service._read(ready["id"])
    current["images_approved_at"] = None
    service._save(current)
    references = service._character_identity_reference_paths(
        current, plan, plan.clips[2],
    )
    assert references == [Path(current["assets"][first_id]["path"])]


def test_starting_image_review_defaults_checked_and_new_projects_store_the_default(tmp_path):
    from ecom_video.web_app import WEB_ROOT

    service = service_factory(tmp_path)
    service._spawn = lambda target, *args: None
    created = service.create_job("reference.mp4", b"video", [], "Make a product ad.")
    original = service.create_original_job("A retro telephone product ad")
    restored = service_factory(tmp_path).get_job(created["id"])
    html = (WEB_ROOT / "index.html").read_text()
    script = (WEB_ROOT / "app.js").read_text()

    assert created["review_images_before_video"] is True
    assert original["review_images_before_video"] is True
    assert restored["review_images_before_video"] is True
    assert 'id="reviewImagesBeforeVideo" type="checkbox" checked' in html
    assert "DEFAULT_REVIEW_IMAGES_BEFORE_VIDEO = true" in script


def test_starting_image_review_user_override_persists_and_legacy_jobs_keep_state(tmp_path):
    service = service_factory(tmp_path)
    ready = create_ready_job(service, tmp_path)
    assert ready["review_images_before_video"] is False
    assert service_factory(tmp_path).get_job(ready["id"])["review_images_before_video"] is False
    server = create_server(0, service)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        conn = HTTPConnection("127.0.0.1", server.server_port, timeout=3)
        conn.request("POST", f"/api/jobs/{ready['id']}/review-images", json.dumps({"enabled": True}),
                     {"Content-Type": "application/json"})
        changed_response = conn.getresponse()
        changed = json.loads(changed_response.read())
        assert changed_response.status == 200
        assert changed["review_images_before_video"] is True
        conn.close()
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=2)
    restarted = service_factory(tmp_path)
    assert restarted.get_job(ready["id"])["review_images_before_video"] is True

    legacy = service._read(ready["id"])
    legacy.pop("review_images_before_video")
    legacy["status"] = "awaiting_image_review"
    service._save(legacy)
    assert restarted.get_job(ready["id"])["review_images_before_video"] is True
    legacy["status"] = "ready"
    service._save(legacy)
    assert restarted.get_job(ready["id"])["review_images_before_video"] is False


def test_verified_under_threshold_generation_runs_and_preserves_clip_outputs(tmp_path, monkeypatch):
    service = service_factory(tmp_path, client_factory=lambda: type("Client", (), {"close": lambda self: None})())

    def fake_generate(plan, products, continuity, output_dir, client, **kwargs):
        output_dir.mkdir(parents=True, exist_ok=True)
        outputs = []
        tasks = []
        for clip in plan.clips:
            image = output_dir / f"clip-{clip.index:02d}-action-{clip.index}-starting.png"
            video = output_dir / f"clip-{clip.index:02d}-action-{clip.index}.mp4"
            if kwargs.get("images_only"):
                image.write_bytes(b"starting-image")
                outputs.append(image)
                tasks.append({"clip_index": clip.index, "kind": "starting_image", "credits_consumed": "6"})
            else:
                video.write_bytes(b"silent-video")
                outputs.append(video)
                tasks.append({"clip_index": clip.index, "kind": "starting_image", "credits_consumed": "6"})
                tasks.append({"clip_index": clip.index, "kind": "video", "credits_consumed": "22.5"})
        (output_dir / "generation-usage.json").write_text(json.dumps({
            "tasks": tasks, "all_task_charges_reported": True, "completed_generation_tasks": len(tasks),
        }))
        return outputs

    monkeypatch.setattr("ecom_video.web_app.run_multiclip_generation", fake_generate)
    service._spawn = lambda target, *args: target(*args)
    ready = create_ready_job(service, tmp_path)
    response = service.start_generation(ready["id"], review_images=True)
    assert response["status"] == "awaiting_image_review"
    final = service.get_job(ready["id"])
    assert final["status"] == "awaiting_image_review"
    assert all(clip["current_starting_image"] for clip in final["clips"])
    assert not any(clip["versions"] for clip in final["clips"])
    service.approve_starting_images(ready["id"])
    final = service.get_job(ready["id"])

    assert final["status"] == "completed"
    assert all(len(clip["versions"]) == 1 for clip in final["clips"])
    assert Decimal(final["charges"]["total_usd"]) > Decimal(ready["charges"]["total_usd"])
    assert len(service.list_jobs()) == 1
    starting = final["clips"][0]["versions"][0]["starting_image"]
    path, media_type, _ = service.get_asset(ready["id"], Path(starting["url"]).name)
    assert path.is_file() and media_type == "image/png"
    restarted = service_factory(tmp_path)
    restored = restarted.get_job(ready["id"])
    assert restored["status"] == "completed"
    restored_path, _, _ = restarted.get_asset(ready["id"], Path(starting["url"]).name)
    assert restored_path == path


def test_unchecked_review_setting_generates_images_then_videos_without_approval(tmp_path, monkeypatch):
    calls = []
    service = service_factory(tmp_path, client_factory=lambda: type("Client", (), {"close": lambda self: None})())
    service._spawn = lambda target, *args: target(*args)
    def fake_generate(plan, products, continuity, output_dir, client, **kwargs):
        output_dir.mkdir(parents=True, exist_ok=True)
        if kwargs.get("images_only"):
            calls.append("images")
            images=[]
            for clip in plan.clips:
                path=output_dir/f"clip-{clip.index:02d}-starting.png"; path.write_bytes(b"image"); images.append(path)
            return images
        calls.append("videos")
        videos=[]
        for clip in plan.clips:
            path=output_dir/f"clip-{clip.index:02d}.mp4"; path.write_bytes(b"video"); videos.append(path)
        (output_dir/"generation-usage.json").write_text(json.dumps({"tasks":[
            {"clip_index":clip.index,"kind":"starting_image","task_id":f"image-{clip.index}","credits_consumed":"6","status":"success"}
            for clip in plan.clips
        ]+[{
            "clip_index":clip.index,"kind":"video","task_id":f"video-{clip.index}","credits_consumed":"22.5","status":"success"
        } for clip in plan.clips],"all_task_charges_reported":True}))
        return videos
    monkeypatch.setattr("ecom_video.web_app.run_multiclip_generation", fake_generate)
    ready = create_ready_job(service, tmp_path)
    assert ready["review_images_before_video"] is False
    completed = service.start_generation(ready["id"], review_images=False)
    assert completed["status"] == "completed"
    assert calls == ["images", "videos"]
    assert all(len(clip["versions"]) == 1 for clip in completed["clips"])


def test_image_review_approval_is_rejected_when_any_image_file_is_missing(tmp_path):
    service = service_factory(tmp_path)
    ready = create_ready_job(service, tmp_path)
    service._update(ready["id"], lambda job: job.update(status="awaiting_image_review", review_images_before_video=True,
        clips=[{"index":1,"current_starting_image_asset":"missing-asset","versions":[]}]))
    with pytest.raises(Exception, match="Clip 1 is missing a readable starting image"):
        service.approve_starting_images(ready["id"])


def test_failed_later_clip_keeps_completed_clip_and_reported_charges(tmp_path, monkeypatch):
    service = service_factory(tmp_path, client_factory=lambda: type("Client", (), {"close": lambda self: None})())

    def partial_generation(plan, products, continuity, output_dir, client, **kwargs):
        output_dir.mkdir(parents=True, exist_ok=True)
        if kwargs.get("images_only"):
            images = []
            tasks = []
            for clip in plan.clips:
                image = output_dir / f"clip-{clip.index:02d}-action-{clip.index}-starting.png"
                image.write_bytes(b"starting-image")
                images.append(image)
                tasks.append({"clip_index": clip.index, "kind": "starting_image", "credits_consumed": "6"})
            (output_dir / "generation-usage.json").write_text(json.dumps({"tasks": tasks, "all_task_charges_reported": False, "completed_generation_tasks": len(tasks)}))
            return images
        (output_dir / "clip-01-action-1-starting.png").write_bytes(b"starting-image")
        (output_dir / "clip-01-action-1.mp4").write_bytes(b"silent-video")
        (output_dir / "generation-usage.json").write_text(json.dumps({
            "tasks": [
                {"clip_index": 1, "kind": "starting_image", "credits_consumed": "6"},
                {"clip_index": 1, "kind": "video", "credits_consumed": "22.5"},
            ], "all_task_charges_reported": False, "completed_generation_tasks": 2,
        }))
        raise RuntimeError("mock provider failure after first completed clip")

    monkeypatch.setattr("ecom_video.web_app.run_multiclip_generation", partial_generation)
    service._spawn = lambda target, *args: target(*args)
    ready = create_ready_job(service, tmp_path)
    service.start_generation(ready["id"], review_images=True)
    assert service.get_job(ready["id"])["status"] == "awaiting_image_review"
    service.approve_starting_images(ready["id"])
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        final = service.get_job(ready["id"])
        if final["status"] in {"completed", "failed"}:
            break
        time.sleep(0.01)

    assert final["status"] == "failed"
    assert len(final["clips"][0]["versions"]) == 1
    assert final["clips"][1]["versions"] == []
    assert Decimal(final["charges"]["kie_credits"]) == Decimal("28.5")


def test_resume_preserves_existing_clip_version_without_adding_a_duplicate(tmp_path, monkeypatch):
    service = service_factory(tmp_path, client_factory=lambda: type("Client", (), {"close": lambda self: None})())
    ready = create_ready_job(service, tmp_path)
    persisted = service._read(ready["id"])
    first_clip = __import__("ecom_video.workflow", fromlist=["MultiClipPlan"]).MultiClipPlan.model_validate_json(persisted["plan_json"]).clips[0]
    existing_image = tmp_path / "existing-starting.png"
    existing_video = tmp_path / "existing-video.mp4"
    existing_image.write_bytes(b"old-image")
    existing_video.write_bytes(b"old-video")
    service._save_clip_version(ready["id"], first_clip, existing_image, existing_video, Decimal("0.1425"), "")
    service._update(ready["id"], lambda job: job.update(status="failed", stage="failed", error="Previous attempt failed", review_images_before_video=True))

    def resume_all(plan, products, continuity, output_dir, client, **kwargs):
        assert kwargs["resume"] is True
        output_dir.mkdir(parents=True, exist_ok=True)
        outputs = []
        for clip in plan.clips:
            image = output_dir / f"clip-{clip.index:02d}-action-{clip.index}-starting.png"
            video = output_dir / f"clip-{clip.index:02d}-action-{clip.index}.mp4"
            if kwargs.get("images_only"):
                if not image.exists(): image.write_bytes(b"image")
                outputs.append(image)
            else:
                video.write_bytes(b"video")
                outputs.append(video)
        return outputs

    monkeypatch.setattr("ecom_video.web_app.run_multiclip_generation", resume_all)
    service._spawn = lambda target, *args: target(*args)
    assert service.get_job(ready["id"])["can_resume"] is True
    resumed = service.start_generation(ready["id"], resume=True)
    assert resumed["status"] == "awaiting_image_review"
    approved = service.approve_starting_images(ready["id"])

    assert approved["status"] == "completed"
    final = service.get_job(ready["id"])
    assert len(final["clips"][0]["versions"]) == 1
    assert all(len(clip["versions"]) == 1 for clip in final["clips"])


def test_starting_image_review_regenerates_only_selected_image_and_uses_it_for_video(tmp_path, monkeypatch):
    calls = []
    service = service_factory(tmp_path, client_factory=lambda: type("Client", (), {"close": lambda self: None})())
    service._spawn = lambda target, *args: target(*args)

    def fake_generate(plan, products, continuity, output_dir, client, **kwargs):
        output_dir.mkdir(parents=True, exist_ok=True)
        if kwargs.get("images_only"):
            images = []
            tasks = []
            for clip in plan.clips:
                calls.append(("image", clip.index, tuple(str(path) for path in continuity)))
                image = output_dir / f"clip-{clip.index:02d}-review-starting.png"
                image.write_bytes(f"image-{clip.index}-{len(calls)}".encode())
                images.append(image)
                tasks.append({"clip_index": clip.index, "kind": "starting_image", "task_id": f"image-task-{len(calls)}", "credits_consumed": "6", "status": "success"})
            (output_dir / "generation-usage.json").write_text(json.dumps({"tasks": [
                *tasks
            ], "all_task_charges_reported": False}))
            return images
        calls.append(("video", tuple(sorted(kwargs["starting_images"].items()))))
        outputs = []
        tasks = []
        for clip in plan.clips:
            path = output_dir / f"clip-{clip.index:02d}-approved.mp4"
            path.write_bytes(b"video")
            outputs.append(path)
            tasks.append({"clip_index": clip.index, "kind": "video", "task_id": f"video-task-{clip.index}", "credits_consumed": "22.5", "status": "success"})
        (output_dir / "generation-usage.json").write_text(json.dumps({"tasks": tasks, "all_task_charges_reported": False}))
        return outputs

    monkeypatch.setattr("ecom_video.web_app.run_multiclip_generation", fake_generate)
    ready = create_ready_job(service, tmp_path)
    service.start_generation(ready["id"], review_images=True)
    before = service.get_job(ready["id"])
    assert before["status"] == "awaiting_image_review"
    old_ids = [clip["current_starting_image"]["url"] for clip in before["clips"]]
    old_paths = [Path(service._read(ready["id"])["assets"][clip["current_starting_image_asset"]]["path"]) for clip in service._read(ready["id"])["clips"]]
    service.regenerate_starting_image(ready["id"], 2, "Place the tool flat against the sofa fabric.")
    after_regen = service.get_job(ready["id"])
    assert after_regen["status"] == "awaiting_image_review"
    assert after_regen["clips"][0]["current_starting_image"]["url"] == old_ids[0]
    assert after_regen["clips"][2]["current_starting_image"]["url"] == old_ids[2]
    assert len(after_regen["clips"][1]["starting_image_history"]) == 2
    assert old_paths[1].read_bytes() == b"image-2-2"
    assert [call[0] for call in calls] == ["image", "image", "image", "image"]
    service.approve_starting_images(ready["id"])
    finished = service.get_job(ready["id"])
    assert finished["status"] == "completed"
    assert calls[-1][0] == "video"
    submitted_images = dict(calls[-1][1])
    assert submitted_images[2] != str(old_paths[1])
    assert old_paths[0].read_bytes() == b"image-1-1" and old_paths[2].read_bytes() == b"image-3-3"
    assert len(finished["clips"][1]["versions"]) == 1


def test_in_store_opening_regeneration_uses_scene_identity_without_prior_pose_reference(tmp_path, monkeypatch):
    from ecom_video.workflow import MultiClipPlan

    service = service_factory(tmp_path, client_factory=lambda: type("Client", (), {"close": lambda self: None})())
    ready = create_ready_job(service, tmp_path)
    job = service._read(ready["id"])
    plan = MultiClipPlan.model_validate_json(job["plan_json"])
    approved_images = []
    for index in range(1, 4):
        image = tmp_path / f"approved-clip-{index}.png"
        image.write_bytes(f"approved-{index}".encode())
        approved_images.append(image)
    service._save_starting_image_assets(ready["id"], plan, approved_images)
    job = service._read(ready["id"])
    plan = plan.model_copy(update={"clips": [
        plan.clips[0].model_copy(update={
            "continuity_group": "store_husband_wife",
            "character_groups": ["mother", "baby"],
            "starting_state": "At frame zero the village display is mostly outside frame; the wife holds coffee upright.",
            "starting_image_prompt": (
                "Photorealistic warehouse aisle. Both look ahead with surprised expressions. "
                "Reference roles: approved Clip 2 defines the couple and warehouse; previous Clip 1 defines the coffee and hand pose; "
                "the product photo defines product identity. Show only this initial state."
            ),
        }),
        plan.clips[1].model_copy(update={"continuity_group": "store_husband_wife", "character_groups": ["mother", "baby"]}),
        plan.clips[2].model_copy(update={"continuity_group": "home_setup"}),
    ]})
    plan = plan.model_copy(update={"hook": "in_store_reaction"})
    service._update(ready["id"], lambda data: data.update(
        plan_json=plan.model_dump_json(),
        product_identity_reference_asset=data["clips"][2]["current_starting_image_asset"],
    ))
    saved = service._read(ready["id"])
    target_before = Path(saved["assets"][saved["clips"][0]["current_starting_image_asset"]]["path"])
    store_reference = Path(saved["assets"][saved["clips"][1]["current_starting_image_asset"]]["path"])
    unrelated_home = Path(saved["assets"][saved["clips"][2]["current_starting_image_asset"]]["path"])
    observed = {}

    def fake_generate(single_plan, products, continuity, output_dir, client, **kwargs):
        observed.update(
            prompt=single_plan.clips[0].starting_image_prompt,
            continuity=list(continuity),
            roles=list(kwargs.get("continuity_reference_roles", [])),
            character_references=list(kwargs.get("character_references", [])),
            products=list(products),
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        image = output_dir / "clip-01-revised-starting.png"
        image.write_bytes(b"revised-image")
        (output_dir / "generation-usage.json").write_text(json.dumps({"tasks": [], "all_task_charges_reported": False}))
        return [image]

    monkeypatch.setattr("ecom_video.web_app.run_multiclip_generation", fake_generate)
    service._regenerate_starting_image(ready["id"], plan.clips[0], "", tmp_path / "regen")

    assert observed["continuity"] == [store_reference]
    assert unrelated_home not in observed["continuity"]
    assert target_before not in observed["continuity"]
    assert observed["roles"] == ["same-scene identity, wardrobe, and environment only; ignore facial expression and pose"]
    assert observed["character_references"] == [store_reference]
    assert len(observed["products"]) == 1  # the hidden product reference is filtered before image submission
    assert "warehouse aisle" in observed["prompt"].lower()
    assert "understated, natural anticipation" in observed["prompt"]
    assert "unretouched skin texture" in observed["prompt"]
    assert "previous Clip 1 defines" not in observed["prompt"]
    assert "product photo defines" not in observed["prompt"]


def test_explicit_environment_reference_omits_stale_target_pose_on_regeneration(tmp_path, monkeypatch):
    from ecom_video.workflow import MultiClipPlan

    service = service_factory(tmp_path, client_factory=lambda: type("Client", (), {"close": lambda self: None})())
    ready = create_ready_job(service, tmp_path)
    job = service._read(ready["id"])
    plan = MultiClipPlan.model_validate_json(job["plan_json"])
    images = []
    for index in range(1, 4):
        image = tmp_path / f"initial-clip-{index}.png"
        image.write_bytes(f"image-{index}".encode())
        images.append(image)
    service._save_starting_image_assets(ready["id"], plan, images)
    environment = tmp_path / "target-environment.jpg"
    environment.write_bytes(b"target background")
    clips = [
        plan.clips[0].model_copy(update={
            "continuity_group": "target-store",
            "character_groups": ["mother", "baby"],
        }),
        plan.clips[1].model_copy(update={
            "continuity_group": "target-store",
            "character_groups": ["mother", "baby"],
            "environment_references": [str(environment)],
            "starting_image_prompt": "Reveal the hoodie; do not repeat the opening reaction.",
        }),
        plan.clips[2].model_copy(update={"continuity_group": "nursery"}),
    ]
    plan = plan.model_copy(update={"hook": "in_store_reaction", "clips": clips})
    service._update(ready["id"], lambda data: data.update(plan_json=plan.model_dump_json()))
    job = service._read(ready["id"])
    identity = Path(job["assets"][job["clips"][0]["current_starting_image_asset"]]["path"])
    old_target = Path(job["assets"][job["clips"][1]["current_starting_image_asset"]]["path"])
    observed = {}

    def fake_generate(single_plan, products, continuity, output_dir, client, **kwargs):
        observed.update(
            continuity=list(continuity),
            character_references=list(kwargs.get("character_references", [])),
            environment_references=list(single_plan.clips[0].environment_references),
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        image = output_dir / "clip-02-revised-starting.png"
        image.write_bytes(b"revised-image")
        (output_dir / "generation-usage.json").write_text(json.dumps({"tasks": [], "all_task_charges_reported": False}))
        return [image]

    monkeypatch.setattr("ecom_video.web_app.run_multiclip_generation", fake_generate)
    service._regenerate_starting_image(ready["id"], plan.clips[1], "Show the hoodie reveal.", tmp_path / "clip-02-regen")

    assert observed["continuity"] == []
    assert old_target not in observed["continuity"]
    assert observed["character_references"] == [identity]
    assert observed["environment_references"] == [str(environment)]


def test_failed_starting_image_regeneration_resumes_with_compact_prompt_and_fresh_task(tmp_path):
    from ecom_video.workflow import MultiClipPlan

    class DryRunKie:
        def __init__(self):
            self.calls = []

        def upload(self, path):
            self.calls.append(("upload", Path(path)))
            return f"https://dry-run.test/{Path(path).name}"

        def create_image(self, prompt, input_urls, aspect_ratio="9:16"):
            self.calls.append(("image", prompt, list(input_urls), aspect_ratio))
            return "fresh-dry-run-image-task"

        def create_text_image(self, prompt, aspect_ratio="9:16"):
            self.calls.append(("text_image", prompt, aspect_ratio))
            return "fresh-dry-run-image-task"

        def wait_for_task(self, task_id, on_state=None):
            self.calls.append(("poll", task_id))
            return {"state": "success", "creditsConsumed": 6,
                    "resultJson": json.dumps({"resultUrls": ["https://dry-run.test/result.png"]})}

        def download_result(self, task_data, destination):
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(b"dry-run-image")
            return destination

        def close(self):
            pass

    client = DryRunKie()
    service = service_factory(tmp_path, client_factory=lambda: client)
    service._spawn = lambda target, *args: target(*args)
    ready = create_ready_job(service, tmp_path)
    job = service._read(ready["id"])
    plan = MultiClipPlan.model_validate_json(job["plan_json"])
    plan = plan.model_copy(update={
        "summary": "video-only context " * 300,
        "clips": [clip.model_copy(update={"continuity_group": "warehouse"}) for clip in plan.clips],
    })
    approved = []
    for index in range(1, 4):
        image = tmp_path / f"approved-{index}.png"
        image.write_bytes(f"approved-{index}".encode())
        approved.append(image)
    service._update(ready["id"], lambda data: data.update(plan_json=plan.model_dump_json()))
    service._save_starting_image_assets(ready["id"], plan, approved)
    before = service._read(ready["id"])
    unchanged = [clip["current_starting_image_asset"] for clip in before["clips"][1:]]
    retry_dir = service._job_path(ready["id"]).parent / "outputs" / "clip-01" / "starting-image" / "version-2"
    service._update(ready["id"], lambda data: data.update(
        status="failed", stage="failed", active_clip=1, active_stage="regenerating_starting_image",
        pending_starting_image_regen={"clip_index": 1, "instructions": "", "output_dir": str(retry_dir)},
        review_images_before_video=True,
    ))

    retryable = service.get_job(ready["id"])
    assert retryable["can_resume"] is True
    assert service._read(ready["id"])["pending_starting_image_regen"]["clip_index"] == 1
    resumed = service.start_generation(ready["id"], resume=True)

    image_request = next(call for call in client.calls if call[0] == "image")
    assert len(image_request[1]) < 1_000
    before = service._read(ready["id"])
    product_path = next(Path(asset["path"]) for asset in before["assets"].values()
                        if asset.get("role") == "product_reference")
    assert image_request[2] == [
        f"https://dry-run.test/{product_path.name}",
        f"https://dry-run.test/{approved[1].name}",
        f"https://dry-run.test/{approved[0].name}",
    ]
    assert [call[0] for call in client.calls].count("image") == 1
    assert resumed["status"] == "awaiting_image_review"
    after = service._read(ready["id"])
    assert [clip["current_starting_image_asset"] for clip in after["clips"][1:]] == unchanged
    assert len(after["clips"][0]["starting_image_history"]) == 2


def test_selective_image_regeneration_avoids_unrelated_scene_anchor(tmp_path, monkeypatch):
    from ecom_video.workflow import MultiClipPlan

    service = service_factory(tmp_path, client_factory=lambda: type("Client", (), {"close": lambda self: None})())
    ready = create_ready_job(service, tmp_path)
    job = service._read(ready["id"])
    plan = MultiClipPlan.model_validate_json(job["plan_json"])
    roles = ["Hook", "Product reveal", "Water-fill demonstration"]
    plan = plan.model_copy(update={"clips": [clip.model_copy(update={"creative_role": role}) for clip, role in zip(plan.clips, roles)]})
    service._update(ready["id"], lambda data: data.update(plan_json=plan.model_dump_json()))
    images = []
    videos = []
    for clip in plan.clips:
        image = tmp_path / f"identity-test-{clip.index}.png"; image.write_bytes(f"image-{clip.index}".encode()); images.append(image)
        video = tmp_path / f"identity-test-{clip.index}.mp4"; video.write_bytes(b"video"); videos.append(video)
    service._save_starting_image_assets(ready["id"], plan, images)
    for clip, image, video in zip(plan.clips, images, videos):
        service._save_clip_version(ready["id"], clip, image, video, Decimal("0.14"), "")
    saved = service._read(ready["id"])
    old_target_path = Path(saved["assets"][saved["clips"][2]["current_starting_image_asset"]]["path"])
    unrelated_product_scene = Path(saved["assets"][saved["clips"][1]["current_starting_image_asset"]]["path"])
    output_dir = tmp_path / "selective-regen"
    service._update(ready["id"], lambda data: data.update(pending_clip_regeneration={"clip_index": 3, "version": 2}))
    observed = {}

    def fake_generate(single_plan, products, continuity, target_dir, client, **kwargs):
        observed.update(clip=single_plan.clips[0].index, continuity=[Path(path) for path in continuity])
        image = target_dir / "clip-03-revised-starting.png"; target_dir.mkdir(parents=True, exist_ok=True); image.write_bytes(b"new-clip-3")
        return [image]

    monkeypatch.setattr("ecom_video.web_app.run_multiclip_generation", fake_generate)
    service._prepare_regenerated_clip_image(ready["id"], 3, "keep the bowl forest green", output_dir, False)

    assert observed["clip"] == 3
    assert observed["continuity"] == [old_target_path]
    assert unrelated_product_scene not in observed["continuity"]
    after = service._read(ready["id"])
    assert after["product_identity_reference_asset"] == saved["product_identity_reference_asset"]
    assert len(after["clips"][1]["starting_image_history"]) == 1
    assert len(after["clips"][2]["starting_image_history"]) == 2


def test_interrupted_starting_image_regeneration_resumes_saved_provider_task(tmp_path, monkeypatch):
    service = service_factory(tmp_path, client_factory=lambda: type("Client", (), {"close": lambda self: None})())
    service._spawn = lambda target, *args: target(*args)
    ready = create_ready_job(service, tmp_path)
    initial = tmp_path / "existing-clip-1.png"
    initial.write_bytes(b"existing-image")
    persisted = service._read(ready["id"])
    plan = __import__("ecom_video.workflow", fromlist=["MultiClipPlan"]).MultiClipPlan.model_validate_json(persisted["plan_json"])
    service._save_starting_image_assets(ready["id"], plan.model_copy(update={"clips": [plan.clips[0]]}), [initial])
    job_dir = service._job_path(ready["id"]).parent
    output_dir = job_dir / "outputs" / "clip-01" / "starting-image" / "version-2"
    output_dir.mkdir(parents=True)
    saved = output_dir / "clip-01-hook-starting.png"
    saved.write_bytes(b"already-generated")
    (output_dir / "generation-usage.json").write_text(json.dumps({"tasks": [
        {"clip_index": 1, "kind": "starting_image", "task_id": "pending-image-task", "credits_consumed": None, "status": "submitted"}
    ], "all_task_charges_reported": False}))
    service._update(ready["id"], lambda job: job.update(
        status="interrupted", stage="interrupted", pending_starting_image_regen={
            "clip_index": 1, "instructions": "show a proper contact angle", "output_dir": str(output_dir)
        }))
    calls = []
    def resume_image(plan, products, continuity, target_dir, client, **kwargs):
        calls.append(kwargs)
        assert kwargs["resume"] is True and kwargs["images_only"] is True
        return [saved]
    monkeypatch.setattr("ecom_video.web_app.run_multiclip_generation", resume_image)
    resumed_service = service_factory(tmp_path, client_factory=lambda: type("Client", (), {"close": lambda self: None})())
    resumed_service._spawn = lambda target, *args: target(*args)
    resumed = resumed_service.start_generation(ready["id"], resume=True)
    assert resumed["status"] == "awaiting_image_review"
    assert calls and saved.read_bytes() == b"already-generated"
    assert len(resumed["clips"][0]["starting_image_history"]) == 2


def test_clip_video_regeneration_waits_for_image_approval(tmp_path, monkeypatch):
    calls = []
    service = service_factory(tmp_path, client_factory=lambda: type("Client", (), {"close": lambda self: None})())
    service._spawn = lambda target, *args: target(*args)
    ready = create_ready_job(service, tmp_path)
    plan = __import__("ecom_video.workflow", fromlist=["MultiClipPlan"]).MultiClipPlan.model_validate_json(service._read(ready["id"])["plan_json"])
    for clip in plan.clips:
        image = tmp_path / f"old-{clip.index}.png"; image.write_bytes(f"old-{clip.index}".encode())
        video = tmp_path / f"old-{clip.index}.mp4"; video.write_bytes(b"old-video")
        service._save_clip_version(ready["id"], clip, image, video, Decimal("0.14"), "")
    service._update(ready["id"], lambda job: job.update(status="completed", stage="completed"))

    def fake_generate(single_plan, products, continuity, output_dir, client, **kwargs):
        output_dir.mkdir(parents=True, exist_ok=True)
        clip = single_plan.clips[0]
        if kwargs.get("images_only"):
            calls.append(("image", clip.index))
            image = output_dir / f"clip-{clip.index:02d}-revised-starting.png"; image.write_bytes(b"revised-image")
            (output_dir / "generation-usage.json").write_text(json.dumps({"tasks": [
                {"clip_index": clip.index, "kind": "starting_image", "task_id": "regen-image-task", "credits_consumed": "6", "status": "success"}
            ], "all_task_charges_reported": False}))
            return [image]
        calls.append(("video", kwargs["starting_images"][clip.index].read_bytes()))
        video = output_dir / f"clip-{clip.index:02d}-revised.mp4"; video.write_bytes(b"revised-video")
        (output_dir / "generation-usage.json").write_text(json.dumps({"tasks": [
            {"clip_index": clip.index, "kind": "starting_image", "task_id": "regen-image-task", "credits_consumed": "6", "status": "success"},
            {"clip_index": clip.index, "kind": "video", "task_id": "regen-video-task", "credits_consumed": "22.5", "status": "success"},
        ], "all_task_charges_reported": True}))
        return [video]

    monkeypatch.setattr("ecom_video.web_app.run_multiclip_generation", fake_generate)
    response = service.start_regeneration(ready["id"], 2, "show the tool contacting the sofa")
    assert response["status"] == "awaiting_image_review"
    assert calls == [("image", 2)]
    review = service.get_job(ready["id"])
    assert review["clip_regeneration_pending"] is True
    assert all(clip["current_starting_image"] for clip in review["clips"])
    service.approve_starting_images(ready["id"])
    final = service.get_job(ready["id"])
    assert final["status"] == "completed"
    assert calls[-1] == ("video", b"revised-image")
    assert len(final["clips"][1]["versions"]) == 2
    assert final["clips"][1]["versions"][0]["starting_image"]["url"] != final["clips"][1]["versions"][1]["starting_image"]["url"]


def test_native_audio_clip_regeneration_reuses_approved_image_and_only_updates_selected_clip(tmp_path, monkeypatch):
    from ecom_video.workflow import MultiClipPlan

    calls = []
    service = service_factory(tmp_path, client_factory=lambda: type("Client", (), {"close": lambda self: None})())
    service._spawn = lambda target, *args: target(*args)
    ready = create_ready_job(service, tmp_path)
    plan = MultiClipPlan.model_validate_json(service._read(ready["id"])["plan_json"])
    plan.clips[0].audio_mode = "enabled"
    plan.clips[0].audio_direction = "Natural reaction and splash only; no music."
    service._update(ready["id"], lambda job: job.update(plan_json=plan.model_dump_json()))
    image = tmp_path / "approved-clip-1.png"; image.write_bytes(b"approved-image")
    video = tmp_path / "old-clip-1.mp4"; video.write_bytes(b"old-video")
    service._save_clip_version(ready["id"], plan.clips[0], image, video, Decimal("0.14"), "original")
    saved = service._read(ready["id"])
    approved_image_id = saved["clips"][0]["current_starting_image_asset"]
    service._update(ready["id"], lambda job: job.update(
        status="completed", stage="completed", images_approved_at="2026-11-19T12:00:00Z",
    ))

    def fake_generate(single_plan, products, continuity, output_dir, client, **kwargs):
        calls.append((kwargs.get("images_only", False), single_plan.clips[0].index,
                      kwargs["starting_images"][1].read_bytes(), single_plan.clips[0].audio_enabled))
        output_dir.mkdir(parents=True, exist_ok=True)
        result = output_dir / "clip-01-revised.mp4"; result.write_bytes(b"audio-video")
        (output_dir / "generation-usage.json").write_text(json.dumps({
            "expected_generation_tasks": 1, "all_task_charges_reported": True,
            "tasks": [{"clip_index": 1, "kind": "video", "task_id": "one-video-task",
                       "credits_consumed": "22.5", "status": "success"}],
        }))
        return [result]

    monkeypatch.setattr("ecom_video.web_app.run_multiclip_generation", fake_generate)
    result = service.start_regeneration(ready["id"], 1, "Drop the iced coffee and reveal the display.", Decimal("1.00"))

    assert result["status"] == "completed"
    assert calls == [(False, 1, b"approved-image", True)]
    final = service._read(ready["id"])
    assert final["clips"][0]["current_starting_image_asset"] == approved_image_id
    assert len(final["clips"][0]["versions"]) == 2
    assert all(not clip["versions"] for clip in final["clips"][1:])
    latest_video_id = final["clips"][0]["versions"][-1]["video_asset"]
    assert final["assets"][latest_video_id]["role"] == "video_with_native_audio"


def test_per_clip_audio_mode_is_editable_visible_and_survives_restart(tmp_path):
    service = service_factory(tmp_path)
    service._spawn = lambda target, *args: target(*args)
    ready = create_ready_job(service, tmp_path)
    plan = __import__("ecom_video.workflow", fromlist=["MultiClipPlan"]).MultiClipPlan.model_validate_json(
        service._read(ready["id"])["plan_json"]
    )
    plan.clips[2].audio_requested = True
    plan.clips[2].audio_direction = "A synchronized product click and natural laughter."
    service._update(ready["id"], lambda job: job.update(plan_json=plan.model_dump_json()))

    enabled = service.set_clip_audio_mode(ready["id"], 3, "enabled")
    assert enabled["clips"][2]["audio_mode"] == "enabled"
    assert enabled["clips"][2]["audio_enabled"] is True
    assert "synchronized product click" in enabled["clips"][2]["audio_direction"].lower()

    silent = service.set_clip_audio_mode(ready["id"], 3, "silent")
    auto = service.set_clip_audio_mode(ready["id"], 3, "auto")
    assert silent["clips"][2]["audio_enabled"] is False
    assert silent["clips"][2]["audio_mode"] == "silent"
    assert auto["clips"][2]["audio_enabled"] is True
    assert service._read(ready["id"])["plan_json"]
    restarted = LocalJobService(tmp_path / "app-data")
    restored = restarted.get_job(ready["id"])
    assert restored["clips"][2]["audio_mode"] == "auto"
    assert restored["clips"][2]["audio_enabled"] is True


def test_audio_mode_rejects_active_job_and_unknown_clip(tmp_path):
    service = service_factory(tmp_path)
    service._spawn = lambda target, *args: target(*args)
    ready = create_ready_job(service, tmp_path)
    service._update(ready["id"], lambda job: job.update(status="generating", stage="generating_clips"))

    from ecom_video.web_app import AppError, JobConflict
    with pytest.raises(JobConflict, match="before generation or between runs"):
        service.set_clip_audio_mode(ready["id"], 1, "silent")

    service._update(ready["id"], lambda job: job.update(status="ready", stage="ready"))
    with pytest.raises(AppError, match="clip does not exist"):
        service.set_clip_audio_mode(ready["id"], 99, "silent")


def test_audio_controls_are_in_clip_details_and_use_local_endpoint():
    from pathlib import Path
    html = Path("src/ecom_video/web/index.html").read_text()
    js = Path("src/ecom_video/web/app.js").read_text()

    assert "audio mode" in js.lower()
    assert "audio direction" in js.lower()
    assert "clips/${clip.index}/audio" in js
    assert 'value="versus"' in html


def test_audio_controls_wrap_inside_responsive_clip_cards():
    from pathlib import Path
    css = Path("src/ecom_video/web/app.css").read_text()
    js = Path("src/ecom_video/web/app.js").read_text()

    assert ".planned-shot{grid-template-columns:35px minmax(0,1fr) minmax(110px,220px);min-width:0}" in css
    assert ".planned-shot .audio-controls{grid-column:2/-1;width:100%;min-width:0" in css
    assert ".audio-controls{grid-template-columns:minmax(0,1fr)}" in css
    assert ".audio-controls select{width:min(100%,320px);max-width:100%;min-width:0}" in css
    assert "overflow-wrap:anywhere" in css
    assert "@media(max-width:760px){.planned-shot{grid-template-columns:24px minmax(0,1fr)}" in css
    assert "['auto', 'Auto · Creative Director'], ['enabled', 'Enabled · Request native audio'], ['silent', 'Silent · No audio']" in js


def test_workspace_restores_saved_inputs_settings_and_autosaved_drafts():
    from pathlib import Path
    html = Path("src/ecom_video/web/index.html").read_text()
    js = Path("src/ecom_video/web/app.js").read_text()

    assert 'value="director">Director decides' in html
    assert "ecomVideoCurrentJobId" in js
    assert "job.inputs" in js
    assert "job.requested_clip_count" in js
    assert "job.duration_selection" in js
    assert "scheduleDraftSave" in js
    assert "'/api/drafts'" in js
    assert "/analyze`" in js
    assert "Draft saved locally. Analysis starts only when you submit it." in js


def test_generation_error_identifies_clip_stage_and_safe_resume(tmp_path):
    service = service_factory(tmp_path)
    ready = create_ready_job(service, tmp_path)
    service._update(ready["id"], lambda job: job.update(active_clip=2, active_stage="generating_clips"))

    service._set_failure(ready["id"], "Kie task failed: provider returned an error")

    failed = service.get_job(ready["id"])
    assert failed["status"] == "failed"
    assert failed["can_resume"] is True
    assert "Clip 2 failed while submitting or generating its video" in failed["error"]
    assert "provider returned an error" in failed["message"]
    assert "Resume reuses saved starting images and provider tasks" in failed["message"]


def test_failed_starting_image_resume_targets_only_missing_image_and_shows_cost(tmp_path, monkeypatch):
    service = service_factory(tmp_path, client_factory=lambda: type("Client", (), {"close": lambda self: None})())
    ready = create_ready_job(service, tmp_path)
    job_path = service._job_path(ready["id"])
    output_dir = job_path.parent / "outputs" / "initial"
    output_dir.mkdir(parents=True)
    for index in (1, 2):
        (output_dir / f"clip-{index:02d}-saved-starting.png").write_bytes(f"image-{index}".encode())
    (output_dir / "generation-usage.json").write_text(json.dumps({
        "expected_generation_tasks": 6,
        "tasks": [
            {"clip_index": 1, "kind": "starting_image", "task_id": "image-1", "credits_consumed": "6", "status": "success"},
            {"clip_index": 2, "kind": "starting_image", "task_id": "image-2", "credits_consumed": "6", "status": "success"},
            {"clip_index": 3, "kind": "starting_image", "task_id": "image-3-failed", "credits_consumed": "0.0", "status": "failed", "provider_state": "fail", "provider_failure": {"code": "500", "message": "Internal Error, Please try again later."}},
        ],
    }))
    service._update(ready["id"], lambda job: job.update(
        status="failed", stage="failed", active_clip=3, active_stage="preparing_images",
        review_images_before_video=True,
    ))
    observed = {}

    def prepare_images(plan, products, continuity, destination, client, **kwargs):
        observed.update(kwargs)
        outputs = []
        for clip in plan.clips:
            image = next(iter(destination.glob(f"clip-{clip.index:02d}-*-starting.png")), None)
            if image is None:
                image = destination / f"clip-{clip.index:02d}-retry-starting.png"
                image.write_bytes(f"image-{clip.index}".encode())
            outputs.append(image)
        return outputs

    monkeypatch.setattr("ecom_video.web_app.run_multiclip_generation", prepare_images)
    service._spawn = lambda target, *args: target(*args)
    before = service.get_job(ready["id"])
    assert before["resume_operation"]["clip_indices"] == [3]
    assert before["resume_operation"]["estimated_usd"] == "0.0300"
    assert "reported 0.0 Kie credits" in before["resume_operation"]["description"]
    assert "code 500" in before["resume_operation"]["description"]
    assert "Internal Error, Please try again later." in before["resume_operation"]["description"]

    resumed = service.start_generation(ready["id"], resume=True)

    assert observed["resume"] is True and observed["images_only"] is True
    assert observed["approved_max_usd"] is None
    assert resumed["status"] == "awaiting_image_review"
    assert all(clip["current_starting_image"] for clip in resumed["clips"])
    assert (output_dir / "clip-01-saved-starting.png").read_bytes() == b"image-1"
    assert (output_dir / "clip-02-saved-starting.png").read_bytes() == b"image-2"


def test_resume_button_and_error_are_visible_in_browser_script():
    from ecom_video.web_app import WEB_ROOT

    script = (WEB_ROOT / "app.js").read_text()
    assert "job.error || job.message" in script
    assert "Resume clips" in script
    assert "Boolean(state.job.can_resume)" in script
    assert "job.resume_operation?.label" in script
    assert "job.resume_operation?.description" in script
    assert "performPaidAction(`/api/jobs/${state.job.id}/approve-images`" in script


def test_restart_marks_running_job_interrupted_and_recovers_written_usage(tmp_path):
    service = service_factory(tmp_path)
    ready = create_ready_job(service, tmp_path)
    job_path = tmp_path / "app-data" / "jobs" / ready["id"] / "job.json"
    persisted = json.loads(job_path.read_text())
    persisted.update(status="generating_clips", stage="generating_clips")
    job_path.write_text(json.dumps(persisted))
    output_dir = job_path.parent / "outputs" / "initial"
    output_dir.mkdir(parents=True)
    (output_dir / "generation-usage.json").write_text(json.dumps({
        "tasks": [
            {"clip_index": 1, "kind": "starting_image", "credits_consumed": "6"},
            {"clip_index": 1, "kind": "video", "credits_consumed": "22.5"},
            {"clip_index": 2, "kind": "starting_image", "task_id": "still-running", "credits_consumed": "100", "status": "submitted"},
        ], "all_task_charges_reported": False, "completed_generation_tasks": 2,
    }))

    restarted = service_factory(tmp_path)
    restored = restarted.get_job(ready["id"])
    assert restored["status"] == "interrupted"
    assert Decimal(restored["charges"]["kie_credits"]) == Decimal("128.5")
    assert restored["charges"]["entries"][-1]["label"] == "initial_generation"


def test_restart_reconciliation_does_not_charge_already_recorded_provider_tasks_twice(tmp_path):
    service = service_factory(tmp_path)
    ready = create_ready_job(service, tmp_path)
    job_path = tmp_path / "app-data" / "jobs" / ready["id"] / "job.json"
    persisted = json.loads(job_path.read_text())
    persisted["charges"].update(kie_credits="34.5", kie_usd_equivalent="0.1725", total_usd="0.192953")
    persisted["charges"]["entries"].append({
        "kind": "kie_generation", "label": "initial_generation", "credits": "34.5",
        "usd_equivalent": "0.1725", "complete": False, "task_count": 3,
    })
    job_path.write_text(json.dumps(persisted))
    output_dir = job_path.parent / "outputs" / "initial"
    output_dir.mkdir(parents=True)
    (output_dir / "generation-usage.json").write_text(json.dumps({
        "tasks": [
            {"clip_index": 1, "kind": "starting_image", "credits_consumed": "6", "status": "success"},
            {"clip_index": 1, "kind": "video", "credits_consumed": "22.5", "status": "success"},
            {"clip_index": 2, "kind": "starting_image", "credits_consumed": "6", "status": "success"},
        ], "all_task_charges_reported": False, "completed_generation_tasks": 3,
    }))

    restored = service_factory(tmp_path).get_job(ready["id"])

    assert Decimal(restored["charges"]["kie_credits"]) == Decimal("34.5")
    assert Decimal(restored["charges"]["kie_usd_equivalent"]) == Decimal("0.1725")
    assert len([entry for entry in restored["charges"]["entries"] if entry.get("label") == "initial_generation"]) == 1


def test_upload_validation_rejects_fake_product_image_without_creating_job(tmp_path):
    service = service_factory(tmp_path)
    with pytest.raises(AppError, match="not a supported"):
        service.create_job("reference.mp4", b"video", [("bad.jpg", b"not jpeg")], "instructions")
    assert service.list_jobs() == []


def test_multipart_upload_parser_keeps_text_and_binary_file_payloads():
    boundary = "form-boundary"
    body = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"instructions\"\r\n\r\n"
        "Keep the product details.\r\n"
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"reference_video\"; filename=\"demo.mp4\"\r\n"
        "Content-Type: video/mp4\r\n\r\n"
    ).encode() + b"\x00\x01video\r\n" + f"--{boundary}--\r\n".encode()
    fields, files = _parse_multipart(f"multipart/form-data; boundary={boundary}", body)

    assert fields["instructions"] == "Keep the product details."
    assert files["reference_video"] == [("demo.mp4", b"\x00\x01video")]


def test_local_http_server_serves_ui_and_rejects_cross_origin_mutations(tmp_path):
    service = service_factory(tmp_path)
    server = create_server(0, service)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urlopen(f"http://127.0.0.1:{server.server_port}/", timeout=2) as response:
            html = response.read().decode()
            assert response.status == 200
            assert "E-Commerce Video Generator" in html
        connection = HTTPConnection("127.0.0.1", server.server_port, timeout=2)
        connection.request("POST", "/api/jobs", body=b"{}", headers={
            "Host": f"127.0.0.1:{server.server_port}", "Origin": "http://attacker.example",
        })
        response = connection.getresponse()
        assert response.status == 403
        assert "Cross-origin" in response.read().decode()
        connection.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_http_upload_endpoint_creates_job_and_runs_mocked_analysis(tmp_path):
    server = create_server(0, service_factory(tmp_path))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        boundary = "browser-upload-fixture"
        fields = [
            ("instructions", "Preserve the visible product details."),
            ("clip_count", "1"),
            ("seconds_per_clip", "5"),
        ]
        pieces = []
        for name, value in fields:
            pieces.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n".encode())
        pieces.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"reference_video\"; filename=\"reference.mp4\"\r\n"
            "Content-Type: video/mp4\r\n\r\n".encode() + b"mock-video-bytes\r\n"
        )
        pieces.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"product_images\"; filename=\"product.jpg\"\r\n"
            "Content-Type: image/jpeg\r\n\r\n".encode() + b"\xff\xd8\xffmock-jpeg\r\n"
        )
        pieces.append(f"--{boundary}--\r\n".encode())
        body = b"".join(pieces)
        connection = HTTPConnection("127.0.0.1", server.server_port, timeout=3)
        connection.request("POST", "/api/jobs", body=body, headers={
            "Host": f"127.0.0.1:{server.server_port}",
            "Origin": f"http://127.0.0.1:{server.server_port}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        })
        response = connection.getresponse()
        assert response.status == 202
        job = json.loads(response.read())
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline and job["status"] in {"queued", "analyzing_reference"}:
            connection.request("GET", f"/api/jobs/{job['id']}", headers={"Host": f"127.0.0.1:{server.server_port}"})
            response = connection.getresponse()
            job = json.loads(response.read())
            time.sleep(0.01)
        assert job["status"] == "ready"
        assert {item["role"] for item in job["inputs"]} == {"reference_video", "product_reference"}
        assert job["instructions"] == "Preserve the visible product details."
        connection.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_video_asset_route_supports_byte_ranges_for_preview_and_seek(tmp_path):
    service = service_factory(tmp_path)
    job = create_ready_job(service, tmp_path)
    raw = service._read(job["id"])
    file_path = service._job_path(job["id"]).parent / "outputs" / "asset.mp4"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_bytes(b"abcdefghij")
    asset_id = service._add_asset(raw, file_path, "clip.mp4", "video", "silent_video")
    service._save(raw)
    server = create_server(0, service)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        connection = HTTPConnection("127.0.0.1", server.server_port, timeout=3)
        connection.request("GET", f"/api/jobs/{job['id']}/assets/{asset_id}", headers={
            "Host": f"127.0.0.1:{server.server_port}", "Range": "bytes=4-6",
        })
        response = connection.getresponse()
        assert response.status == 206
        assert response.getheader("Content-Range") == "bytes 4-6/10"
        assert response.getheader("Content-Disposition").startswith("inline;")
        assert response.read() == b"efg"
        connection.request("GET", f"/api/jobs/{job['id']}/assets/{asset_id}", headers={
            "Host": f"127.0.0.1:{server.server_port}", "Range": "bytes=-3",
        })
        response = connection.getresponse()
        assert response.status == 206
        assert response.read() == b"hij"
        connection.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_asset_downloads_use_real_filename_mime_type_and_unmodified_bytes(tmp_path):
    service = service_factory(tmp_path)
    job = create_ready_job(service, tmp_path)
    raw = service._read(job["id"])
    directory = service._job_path(job["id"]).parent / "outputs" / "download-test"
    directory.mkdir(parents=True, exist_ok=True)
    video_bytes = b"\x00\x00\x00\x18ftypmp42\x00video-original"
    image_bytes = b"\x89PNG\r\n\x1a\noriginal-image-bytes"
    video_path = directory / "clip-02-gecko-demo.mp4"
    image_path = directory / "clip-02-starting.png"
    video_path.write_bytes(video_bytes)
    image_path.write_bytes(image_bytes)
    video_id = service._add_asset(raw, video_path, video_path.name, "video", "silent_video")
    image_id = service._add_asset(raw, image_path, image_path.name, "image", "starting_image")
    service._save(raw)
    server = create_server(0, service)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        connection = HTTPConnection("127.0.0.1", server.server_port, timeout=3)
        for asset_id, filename, media_type, expected_bytes in [
            (video_id, video_path.name, "video/mp4", video_bytes),
            (image_id, image_path.name, "image/png", image_bytes),
        ]:
            connection.request("GET", f"/api/jobs/{job['id']}/assets/{asset_id}?download=1", headers={
                "Host": f"127.0.0.1:{server.server_port}",
            })
            response = connection.getresponse()
            assert response.status == 200
            assert response.getheader("Content-Type") == media_type
            disposition = response.getheader("Content-Disposition")
            assert disposition.startswith("attachment;")
            assert f'filename="{filename}"' in disposition
            assert response.read() == expected_bytes
        connection.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_download_links_request_attachment_mode():
    script = (Path(__file__).parents[1] / "src/ecom_video/web/app.js").read_text()
    assert "function assetDownloadUrl(asset)" in script
    assert "download=1" in script
    assert "assetDownloadUrl(selected.video)" in script
    assert "assetDownloadUrl(selected.starting_image)" in script
