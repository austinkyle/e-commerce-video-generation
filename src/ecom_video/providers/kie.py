from __future__ import annotations

from pathlib import Path
import json
import time
from typing import Any, Callable
from urllib.parse import urlparse

import httpx

from ecom_video.config import require_key
from ecom_video.prompt_limits import KIE_VIDEO_PROMPT_MAX_CHARACTERS, kie_video_prompt_length

API = "https://api.kie.ai"
UPLOAD_API = "https://kieai.redpandaai.co"
VIDEO_MODEL = "grok-imagine-video-1-5-preview"
IMAGE_MODEL = "gpt-image-2-5-sunburst-image-to-image"
MASKED_EDIT_MODEL = "gpt-image-2-6/sunburst-image-to-image"


class KieTaskFailed(RuntimeError):
    """A terminal provider failure, including any usage Kie reported for it."""

    def __init__(self, task_id: str, task_data: dict[str, Any]):
        self.task_id = task_id
        self.task_data = task_data
        details = task_data.get("failMsg") or "unknown error"
        code = task_data.get("failCode")
        code_text = f" (code {code})" if code not in (None, "") else ""
        super().__init__(f"Kie task {task_id} failed{code_text}: {details}")


class KieTaskSubmissionUnknown(RuntimeError):
    """The create request may have been accepted, but no task ID was saved."""

    outcome_uncertain = True


class KieClient:
    """Thin Kie transport. Generation methods only submit when explicitly called."""

    def __init__(self, timeout: float = 60):
        self._client = httpx.Client(
            base_url=API,
            headers={"Authorization": f"Bearer {require_key('KIE_API_KEY')}", "Content-Type": "application/json"},
            timeout=timeout,
        )

    def upload(self, path: Path, upload_path: str = "ecommerce-video") -> str:
        with path.open("rb") as media:
            response = httpx.post(
                f"{UPLOAD_API}/api/file-stream-upload",
                headers={"Authorization": f"Bearer {require_key('KIE_API_KEY')}"},
                data={"uploadPath": upload_path}, files={"file": (path.name, media)}, timeout=180,
            )
        response.raise_for_status()
        body = response.json()
        if not body.get("success") or not body.get("data", {}).get("downloadUrl"):
            raise RuntimeError("Kie upload failed")
        return body["data"]["downloadUrl"]

    def create_image(self, prompt: str, input_urls: list[str], aspect_ratio: str = "9:16") -> str:
        if not input_urls or len(input_urls) > 16:
            raise ValueError("Image generation requires 1 to 16 reference images")
        return self._create_task(IMAGE_MODEL, {"prompt": prompt, "input_urls": input_urls,
                                               "aspect_ratio": aspect_ratio, "resolution": "1K",
                                               "background": "opaque"})

    def create_masked_image_edit(self, prompt: str, input_urls: list[str], mask_url: str,
                                 aspect_ratio: str = "9:16") -> str:
        """Submit an explicitly requested, masked GPT Image 2.6 Sunburst edit.

        This separate method intentionally leaves the default 2.5 generation
        path untouched. Kie's endpoint applies ``mask_url`` to the first input.
        """
        if not input_urls or len(input_urls) > 16:
            raise ValueError("Masked image editing requires 1 to 16 reference images")
        if not mask_url.startswith("https://"):
            raise ValueError("Masked image editing requires an HTTPS mask URL")
        return self._create_task(MASKED_EDIT_MODEL, {
            "prompt": prompt,
            "input_urls": input_urls,
            "mask_url": mask_url,
            "aspect_ratio": aspect_ratio,
            "resolution": "1K",
            "quality": "medium",
            "output_format": "png",
            "background": "opaque",
        })

    def create_text_image(self, prompt: str, aspect_ratio: str = "9:16") -> str:
        return self._create_task("gpt-image-2-5-sunburst-text-to-image", {
            "prompt": prompt, "aspect_ratio": aspect_ratio,
            "resolution": "1K", "background": "opaque",
        })

    def create_video(self, prompt: str, image_url: str, duration_seconds: int,
                     first_frame_url: str | None = None, last_frame_url: str | None = None,
                     keyframes: list[dict[str, Any]] | None = None,
                     reference_image_urls: list[str] | None = None,
                     aspect_ratio: str = "9:16", resolution: str = "720p",
                     enable_audio: bool | None = False) -> str:
        prompt_length = kie_video_prompt_length(prompt)
        if prompt_length > KIE_VIDEO_PROMPT_MAX_CHARACTERS:
            raise ValueError(
                f"Grok video prompt is {prompt_length} characters; Kie allows at most "
                f"{KIE_VIDEO_PROMPT_MAX_CHARACTERS} characters. Shorten the prompt before submission."
            )
        if not 1 <= duration_seconds <= 15:
            raise ValueError("Grok preview duration must be between 1 and 15 seconds")
        if resolution not in {"480p", "720p", "1080p"}:
            raise ValueError("Unsupported Grok preview resolution")
        if keyframes and len(keyframes) > 4:
            raise ValueError("Grok preview accepts at most four intermediate keyframes")
        reference_image_urls = list(reference_image_urls or [])
        if len(reference_image_urls) > 6 or image_url in reference_image_urls or len(set(reference_image_urls)) != len(reference_image_urls):
            raise ValueError("Grok preview accepts at most 7 total images: the starting frame and six additional, distinct references")
        if reference_image_urls and resolution == "1080p":
            raise ValueError("Grok preview supports only one image reference at 1080p")
        inputs: dict[str, Any] = {"prompt": prompt, "image_urls": [image_url, *reference_image_urls], "duration": duration_seconds,
                                  "resolution": resolution, "aspect_ratio": aspect_ratio}
        if enable_audio is not None:
            inputs["enable_audio"] = enable_audio
        if first_frame_url:
            inputs["first_frame_url"] = first_frame_url
        if last_frame_url:
            inputs["last_frame_url"] = last_frame_url
        if keyframes:
            inputs["keyframes"] = keyframes
        return self._create_task(VIDEO_MODEL, inputs)

    def task(self, task_id: str) -> dict[str, Any]:
        response = self._client.get("/api/v1/jobs/recordInfo", params={"taskId": task_id})
        return self._checked_json(response)

    def wait_for_task(
        self,
        task_id: str,
        timeout_seconds: int = 900,
        on_state: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        """Poll an existing task only; this never creates another paid task."""
        deadline = time.monotonic() + timeout_seconds
        delay = 2.0
        while time.monotonic() < deadline:
            result = self.task(task_id)
            data = result.get("data", {})
            state = data.get("state")
            if on_state:
                on_state(str(state))
            if state == "success":
                return data
            if state == "fail":
                raise KieTaskFailed(task_id, data)
            if state not in {"waiting", "queuing", "generating"}:
                raise RuntimeError(f"Kie returned unexpected task state: {state!r}")
            time.sleep(delay)
            delay = min(delay * 1.5, 15)
        raise TimeoutError(f"Kie task {task_id} is still pending; check its status before taking further action")

    def download_result(self, task_data: dict[str, Any], destination: Path) -> Path:
        """Download the first generated result from an already completed task."""
        raw = task_data.get("resultJson")
        try:
            result = json.loads(raw) if isinstance(raw, str) else raw or {}
            urls = result.get("resultUrls", [])
            url = urls[0] if urls else None
        except (TypeError, json.JSONDecodeError):
            url = None
        if not url or urlparse(url).scheme != "https":
            raise RuntimeError("Completed Kie task did not contain a valid HTTPS result URL")
        response = httpx.get(url, timeout=180, follow_redirects=True)
        response.raise_for_status()
        destination = destination.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(response.content)
        return destination

    def _create_task(self, model: str, inputs: dict[str, Any]) -> str:
        try:
            response = self._client.post("/api/v1/jobs/createTask", json={"model": model, "input": inputs})
        except httpx.RequestError as exc:
            raise KieTaskSubmissionUnknown("Kie task submission outcome is uncertain after a network error; reconcile the provider task before retrying.") from exc
        body = self._checked_json(response)
        task_id = body.get("data", {}).get("taskId")
        if not task_id:
            raise KieTaskSubmissionUnknown("Kie accepted no task ID; submission outcome may be uncertain. Do not resubmit automatically.")
        return task_id

    @staticmethod
    def _checked_json(response: httpx.Response) -> dict[str, Any]:
        response.raise_for_status()
        body = response.json()
        if body.get("code") != 200:
            raise RuntimeError(f"Kie request failed ({body.get('code', 'unknown')}): {body.get('msg', 'unknown error')}")
        return body

    def close(self) -> None:
        self._client.close()
