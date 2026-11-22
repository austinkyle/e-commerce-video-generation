from __future__ import annotations

import json
import hashlib
import mimetypes
import os
import re
import shutil
import threading
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from email import policy
from email.parser import BytesParser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable
from urllib.parse import parse_qs, quote, unquote, urlsplit
import struct

from ecom_video.costs import (
    ANALYSIS_ALLOWANCE_USD,
    DEFAULT_PROJECT_SPENDING_LIMIT_USD,
    IMAGE_1K_USD,
    MASKED_EDIT_1K_USD,
    KIE_USD_PER_CREDIT,
    price_openai_analysis_usage,
    estimate_within_approved_amount,
    reserve_estimate_usd,
)
from ecom_video.media import probe_video, sample_frames
from ecom_video.models import AnalysisResult, AudioMode
from ecom_video.providers.kie import KieClient
from ecom_video.providers.openai import analyze_reference, direct_original, revise_creative_plan
from ecom_video.workflow import (
    AUTO_GENERATION_APPROVAL_LIMIT_USD,
    MultiClipPlan,
    MAX_CREATIVE_INSTRUCTIONS,
    PlannedClip,
    build_multiclip_plan,
    creative_instruction_length,
    estimate_multiclip_generation,
    fit_reference_analysis_to_clip_count,
    generation_approval_required,
    normalize_reference_segments,
    product_identity_clip_index,
    run_multiclip_generation,
    find_saved_mask_for_scene,
    run_masked_image_edit,
    starting_image_revision_prompt,
)

MAX_UPLOAD_BYTES = 200 * 1024 * 1024
MAX_PRODUCT_IMAGES = 16
MAX_PRODUCT_IMAGE_BYTES = 20 * 1024 * 1024
MAX_MASK_IMAGE_BYTES = 20 * 1024 * 1024
MAX_PROJECT_TITLE_LENGTH = 100
DEFAULT_REVIEW_IMAGES_BEFORE_VIDEO = True
MAX_REQUEST_BYTES = MAX_UPLOAD_BYTES + MAX_PRODUCT_IMAGES * MAX_PRODUCT_IMAGE_BYTES + 1024 * 1024
ACTIVE_STATUSES = {"queued", "analyzing_reference", "refining_plan", "preparing_images", "generating_clips", "processing_outputs", "regenerating", "regenerating_starting_image"}
DELETION_BLOCKED_STATUSES = ACTIVE_STATUSES | {"interviewing"}
WEB_ROOT = Path(__file__).with_name("web")


class AppError(Exception):
    status = HTTPStatus.BAD_REQUEST


class JobNotFound(AppError):
    status = HTTPStatus.NOT_FOUND


class JobConflict(AppError):
    status = HTTPStatus.CONFLICT


class ProjectStorageError(AppError):
    status = HTTPStatus.INTERNAL_SERVER_ERROR


class ConfirmationRequired(AppError):
    status = HTTPStatus.CONFLICT

    def __init__(self, estimate_usd: Decimal, reason: str):
        super().__init__(reason)
        self.estimate_usd = estimate_usd


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _image_extension(data: bytes) -> str | None:
    if data.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    return None


def _image_dimensions(path: Path) -> tuple[int, int] | None:
    """Read PNG/JPEG dimensions without adding an image-processing dependency."""
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) >= 26:
        return struct.unpack(">II", data[16:24])
    if not data.startswith(b"\xff\xd8"):
        return None
    offset = 2
    while offset + 4 <= len(data):
        if data[offset] != 0xFF:
            offset += 1
            continue
        marker = data[offset + 1]
        offset += 2
        if marker in {0xD8, 0xD9} or 0xD0 <= marker <= 0xD7:
            continue
        if offset + 2 > len(data):
            break
        length = int.from_bytes(data[offset:offset + 2], "big")
        if length < 2 or offset + length > len(data):
            break
        if marker in {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}:
            if length < 7:
                break
            height = int.from_bytes(data[offset + 3:offset + 5], "big")
            width = int.from_bytes(data[offset + 5:offset + 7], "big")
            return width, height
        offset += length
    return None


def _validate_mask_png(mask_path: Path, source_path: Path) -> tuple[int, int]:
    try:
        header = mask_path.read_bytes()[:29]
    except OSError as exc:
        raise AppError("The selected mask file could not be read.") from exc
    if (len(header) < 29 or header[:8] != b"\x89PNG\r\n\x1a\n"
            or header[12:16] != b"IHDR" or header[25] not in {4, 6}):
        raise AppError("Use a transparent PNG mask (grayscale-alpha or RGBA). Transparent pixels are edited; opaque pixels are protected.")
    dimensions = struct.unpack(">II", header[16:24])
    if dimensions != _image_dimensions(source_path):
        raise AppError("The mask dimensions must exactly match the selected scene image.")
    return dimensions


def _credit_value(task: dict) -> Decimal:
    for key in ("credits_consumed", "creditsConsumed", "consumeCredits", "consumedCredits"):
        if task.get(key) is not None:
            return Decimal(str(task[key]))
    return Decimal("0")


def _reported_credits(task: dict) -> Decimal | None:
    for key in ("credits_consumed", "creditsConsumed", "consumeCredits", "consumedCredits"):
        if task.get(key) is not None:
            try:
                return Decimal(str(task[key]))
            except Exception:
                return None
    return None


def _default_project_title(mode: str, instructions: str, reference_filename: str | None = None) -> str:
    if mode != "original":
        title = Path(reference_filename or "").stem.strip()
        return title or "Reference project"

    first_line = next((line.strip() for line in instructions.splitlines() if line.strip()), "")
    first_sentence = re.split(r"(?<=[.!?])\s+", first_line, maxsplit=1)[0]
    title = re.sub(
        r"^(?:I want to create|I would like to create|I'd like to create|Create|Make)\s+",
        "", first_sentence, flags=re.IGNORECASE,
    ).strip(" .:;-")
    if not title:
        return "Original Concept"
    if creative_instruction_length(title) > 80:
        title = title[:77].rsplit(" ", 1)[0].rstrip(" .:;-") + "…"
    return title or "Original Concept"


class LocalJobService:
    """Small file-backed job service; provider credentials remain server-side."""

    def __init__(
        self,
        data_dir: Path | None = None,
        *,
        analyzer: Callable = analyze_reference,
        director: Callable = direct_original,
        plan_reviser: Callable = revise_creative_plan,
        client_factory: Callable = KieClient,
        frame_sampler: Callable = sample_frames,
        video_probe: Callable = probe_video,
    ):
        root = data_dir or Path(os.getenv("APP_DATA_DIR", "data"))
        self.data_dir = root.expanduser().resolve()
        self.jobs_dir = self.data_dir / "jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.analyzer = analyzer
        self.director = director
        self.plan_reviser = plan_reviser
        self.client_factory = client_factory
        self.frame_sampler = frame_sampler
        self.video_probe = video_probe
        self.lock = threading.RLock()
        self.workers: set[threading.Thread] = set()
        self._mark_interrupted_jobs()
        self.cleanup_temporary_files()

    def _job_path(self, job_id: str) -> Path:
        if not re.fullmatch(r"[0-9a-f-]{36}", job_id):
            raise JobNotFound("Job not found")
        return self.jobs_dir / job_id / "job.json"

    def _read(self, job_id: str) -> dict:
        path = self._job_path(job_id)
        if not path.is_file():
            raise JobNotFound("Job not found")
        return json.loads(path.read_text())

    def _save(self, job: dict) -> None:
        path = self._job_path(job["id"])
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(job, indent=2) + "\n")
        temporary.replace(path)

    def _update(self, job_id: str, change: Callable[[dict], None]) -> dict:
        with self.lock:
            job = self._read(job_id)
            change(job)
            job["updated_at"] = _now()
            self._save(job)
            return job

    def _mark_interrupted_jobs(self) -> None:
        for path in self.jobs_dir.glob("*/job.json"):
            try:
                job = json.loads(path.read_text())
                self._reconcile_usage(job, path.parent)
                if job.get("status") in ACTIVE_STATUSES:
                    job["status"] = "interrupted"
                    job["stage"] = "interrupted"
                    job["message"] = "The app stopped before this job finished. Completed files and recorded charges are preserved."
                    job["updated_at"] = _now()
                temporary = path.with_suffix(".json.tmp")
                temporary.write_text(json.dumps(job, indent=2) + "\n")
                temporary.replace(path)
            except (OSError, ValueError, json.JSONDecodeError):
                continue

    @staticmethod
    def _reconcile_usage(job: dict, job_dir: Path) -> None:
        """Recover already written Kie usage manifests after an app restart."""
        outputs = job_dir / "outputs"
        for manifest_path in outputs.glob("**/generation-usage.json"):
            relative = manifest_path.parent.relative_to(outputs).parts
            if relative == ("initial",):
                label = "initial_generation"
            elif len(relative) == 2:
                match = re.fullmatch(r"clip-(\d+)", relative[0])
                version = re.fullmatch(r"version-(\d+)", relative[1])
                if not match or not version:
                    continue
                label = f"clip_{int(match.group(1))}_v{int(version.group(1))}"
            elif len(relative) == 3 and relative[1] == "starting-image":
                match = re.fullmatch(r"clip-(\d+)", relative[0])
                version = re.fullmatch(r"version-(\d+)", relative[2])
                if not match or not version:
                    continue
                label = f"clip_{int(match.group(1))}_starting_image_v{int(version.group(1))}"
            else:
                continue
            entries = job.get("charges", {}).get("entries", [])
            previous = next((entry for entry in entries if entry.get("label") == label), None)
            try:
                manifest = json.loads(manifest_path.read_text())
                tasks = manifest.get("tasks", [])
                if not tasks:
                    continue
                unique_tasks = {}
                for index, task in enumerate(tasks):
                    task_id = task.get("task_id") or task.get("taskId") or f"{label}:task-{index}"
                    unique_tasks.setdefault(str(task_id), task)
                chargeable_tasks = [task for task in unique_tasks.values() if _reported_credits(task) is not None]
                credits = sum((_reported_credits(task) for task in chargeable_tasks), Decimal("0"))
                kie_usd = credits * KIE_USD_PER_CREDIT
                previous_credits = Decimal(previous.get("credits", "0")) if previous else Decimal("0")
                delta_credits = credits - previous_credits
                job["charges"]["kie_credits"] = f"{Decimal(job['charges']['kie_credits']) + delta_credits:.1f}"
                job["charges"]["kie_usd_equivalent"] = f"{Decimal(job['charges']['kie_usd_equivalent']) + delta_credits * KIE_USD_PER_CREDIT:.4f}"
                total = Decimal(job["charges"]["analysis_usd"]) + Decimal(job["charges"]["kie_usd_equivalent"])
                job["charges"]["total_usd"] = f"{total:.6f}"
                entry = {
                    "kind": "kie_generation", "label": label, "credits": str(credits),
                    "usd_equivalent": f"{kie_usd:.4f}",
                    "complete": manifest.get("all_task_charges_reported", False),
                    "task_count": len(chargeable_tasks),
                    "recorded_at": previous.get("recorded_at") if previous else _now(),
                }
                if previous:
                    previous.update(entry)
                else:
                    entries.append(entry)
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                continue

    def _spawn(self, target: Callable, *args) -> None:
        def run() -> None:
            try:
                target(*args)
            finally:
                with self.lock:
                    self.workers.discard(threading.current_thread())

        thread = threading.Thread(target=run, daemon=True)
        with self.lock:
            self.workers.add(thread)
        thread.start()

    @staticmethod
    def _draft_selection(fields: dict[str, str]) -> tuple[str, str, int, int]:
        count_choice = fields.get("clip_count", "3") or "3"
        duration_choice = fields.get("seconds_per_clip", "5") or "5"
        count_mode = "director" if count_choice == "director" else "fixed"
        duration_mode = "director" if duration_choice == "director" else "fixed"
        try:
            clip_count = 12 if count_mode == "director" else int(count_choice)
            seconds = 5 if duration_mode == "director" else int(duration_choice)
        except ValueError:
            raise AppError("Choose valid clip count and duration settings.") from None
        if not 1 <= clip_count <= 12 or not 1 <= seconds <= 15:
            raise AppError("Choose 1–12 clips and 1–15 seconds per clip.")
        return count_mode, duration_mode, clip_count, seconds

    def save_draft(
        self,
        fields: dict[str, str],
        files: dict[str, list[tuple[str, bytes]]],
        job_id: str | None = None,
    ) -> dict:
        """Persist an editable workspace without starting any paid work."""
        mode = fields.get("mode", "reference")
        hook = fields.get("hook", "recommend")
        if mode not in {"reference", "original"}:
            raise AppError("Choose Reference Recreation or Original Concept.")
        if hook not in {"recommend", "in_store_reaction", "wet_works", "countdown_reveal", "gifting", "versus", "custom"}:
            raise AppError("Choose a supported creative hook.")
        instructions = fields.get("instructions", "")
        if creative_instruction_length(instructions) > MAX_CREATIVE_INSTRUCTIONS:
            raise AppError("Keep creative instructions to 20,000 characters or fewer.")
        count_mode, duration_mode, clip_count, seconds = self._draft_selection(fields)
        review = fields.get("review_images_before_video", "true").lower() in {"1", "true", "on"}
        video_files = files.get("reference_video", [])
        product_files = files.get("product_images", [])
        if len(video_files) > 1:
            raise AppError("Choose one reference video.")
        validated_images: list[tuple[str, bytes, str]] = []
        for name, content in product_files:
            extension = _image_extension(content)
            if not extension or len(content) > MAX_PRODUCT_IMAGE_BYTES:
                raise AppError("Each product image must be a supported JPEG, PNG, or WebP image up to 20 MB.")
            validated_images.append((Path(name).name, content, extension))
        if len(validated_images) > MAX_PRODUCT_IMAGES:
            raise AppError(f"Upload no more than {MAX_PRODUCT_IMAGES} product images.")

        with self.lock:
            if job_id:
                job = self._read(job_id)
                if job.get("status") != "draft":
                    raise JobConflict("Only an unanalyzed draft can be edited from the creation form.")
            else:
                job_id = str(uuid.uuid4())
                job = {
                    "id": job_id, "title": "New creation", "mode": mode, "hook": hook,
                    "project_spending_limit_usd": str(DEFAULT_PROJECT_SPENDING_LIMIT_USD),
                    "status": "draft", "stage": "draft", "message": "Draft saved locally.",
                    "provider_state": None, "created_at": _now(), "updated_at": _now(),
                    "reference_video": None, "reference_video_info": {}, "reference_asset_id": None,
                    "product_references": [], "product_asset_ids": [], "instructions": "",
                    "requested_clip_count": 3, "clip_count_selection": "fixed", "seconds_per_clip": 5,
                    "duration_selection": "fixed", "review_images_before_video": DEFAULT_REVIEW_IMAGES_BEFORE_VIDEO,
                    "assets": {}, "analysis": None, "plan_json": None, "estimate": None,
                    "estimate_reliable": True, "clips": [],
                    "charges": {"analysis_usd": "0", "kie_credits": "0", "kie_usd_equivalent": "0", "total_usd": "0", "entries": []},
                    "error": None,
                }
                (self._job_path(job_id).parent / "inputs").mkdir(parents=True, exist_ok=True)

            job.update(
                mode=mode, hook=hook, instructions=instructions,
                requested_clip_count=clip_count, clip_count_selection=count_mode,
                seconds_per_clip=seconds, duration_selection=duration_mode,
                review_images_before_video=review,
                title=_default_project_title(mode, instructions, Path(video_files[0][0]).name if video_files else None),
            )

            retained_raw = fields.get("product_asset_ids")
            if retained_raw is not None:
                try:
                    retained_ids = json.loads(retained_raw)
                except (TypeError, json.JSONDecodeError):
                    raise AppError("The saved product image selection is invalid.") from None
                if not isinstance(retained_ids, list) or any(not isinstance(item, str) for item in retained_ids):
                    raise AppError("The saved product image selection is invalid.")
                for asset_id in retained_ids:
                    asset = job["assets"].get(asset_id)
                    if not asset or asset.get("role") != "product_reference" or not Path(asset["path"]).is_file():
                        raise AppError("A saved product image is no longer available. Choose it again to continue.")
                product_asset_ids = list(dict.fromkeys(retained_ids))
            else:
                product_asset_ids = list(job.get("product_asset_ids", []))
            if len(product_asset_ids) + len(validated_images) > MAX_PRODUCT_IMAGES:
                raise AppError(f"Upload no more than {MAX_PRODUCT_IMAGES} product images.")

            if video_files:
                name, content = video_files[0]
                suffix = Path(name).suffix.lower()
                if suffix not in {".mp4", ".mov"} or not content or len(content) > MAX_UPLOAD_BYTES:
                    raise AppError("Choose a readable MP4 or MOV reference video up to 200 MB.")
                input_dir = self._job_path(job_id).parent / "inputs"
                version = 1
                while True:
                    candidate = input_dir / (f"reference{suffix}" if version == 1 else f"reference-{version:02d}{suffix}")
                    if not candidate.exists():
                        break
                    version += 1
                candidate.write_bytes(content)
                try:
                    video_info = self.video_probe(candidate)
                except Exception:
                    candidate.unlink(missing_ok=True)
                    raise AppError("The selected file could not be read as a video. Check that it opens and try again.") from None
                job["reference_video"] = str(candidate)
                job["reference_video_info"] = video_info
                job["reference_asset_id"] = self._add_asset(job, candidate, Path(name).name, "video", "reference_video")
            elif fields.get("remove_reference_video", "false").lower() in {"1", "true", "on"}:
                job["reference_video"] = None
                job["reference_video_info"] = {}
                job["reference_asset_id"] = None

            input_dir = self._job_path(job_id).parent / "inputs"
            for name, content, extension in validated_images:
                version = 1
                while True:
                    candidate = input_dir / f"product-{len(job['assets']) + version:03d}{extension}"
                    if not candidate.exists():
                        break
                    version += 1
                candidate.write_bytes(content)
                product_asset_ids.append(self._add_asset(job, candidate, name, "image", "product_reference"))
            job["product_asset_ids"] = list(dict.fromkeys(product_asset_ids))
            job["product_references"] = [job["assets"][asset_id]["path"] for asset_id in job["product_asset_ids"]]
            job["message"] = "Draft saved locally. No analysis or generation has started."
            job["updated_at"] = _now()
            self._save(job)
        return self.get_job(job_id)

    def analyze_draft(self, job_id: str) -> dict:
        with self.lock:
            job = self._read(job_id)
            if job.get("status") != "draft":
                raise JobConflict("Only an unanalyzed draft can be submitted for analysis.")
            if not job.get("instructions", "").strip():
                raise AppError("Add creative instructions before analyzing this project.")
            if job.get("mode", "reference") == "reference":
                reference = Path(job.get("reference_video") or "")
                if not reference.is_file():
                    raise AppError("Choose a reference video before analyzing this project.")
                job.update(status="queued", stage="queued", message="Reference received. Analysis is starting.")
                worker = self._analyze_job
            else:
                job.update(status="analyzing_reference", stage="analyzing_reference", message="The Creative Director is reviewing your idea.")
                job.setdefault("interview_answers", [])
                job.setdefault("director_questions", [])
                worker = self._director_turn
            job["updated_at"] = _now()
            self._save(job)
        self._spawn(worker, job_id)
        return self.get_job(job_id)

    def create_job(
        self,
        video_name: str,
        video_data: bytes,
        product_images: list[tuple[str, bytes]],
        instructions: str,
        clip_count: int = 3,
        seconds_per_clip: int = 5,
        review_images_before_video: bool = DEFAULT_REVIEW_IMAGES_BEFORE_VIDEO,
        hook: str = "recommend",
    ) -> dict:
        if hook not in {"recommend", "in_store_reaction", "wet_works", "countdown_reveal", "gifting", "versus", "custom"}:
            raise AppError("Choose a supported creative hook.")
        suffix = Path(video_name).suffix.lower()
        if suffix not in {".mp4", ".mov"}:
            raise AppError("Choose an MP4 or MOV reference video.")
        if not video_data or len(video_data) > MAX_UPLOAD_BYTES:
            raise AppError("The reference video is empty or exceeds the 200 MB upload limit.")
        if len(product_images) > MAX_PRODUCT_IMAGES:
            raise AppError(f"Upload no more than {MAX_PRODUCT_IMAGES} product images.")
        if not instructions.strip():
            raise AppError("Add creative instructions before analyzing the reference.")
        if creative_instruction_length(instructions) > MAX_CREATIVE_INSTRUCTIONS:
            raise AppError("Keep creative instructions to 20,000 characters or fewer.")
        if clip_count < 1 or clip_count > 12 or seconds_per_clip < 1 or seconds_per_clip > 15:
            raise AppError("Choose 1–12 clips and 1–15 seconds per clip.")
        validated_images: list[tuple[str, bytes, str]] = []
        for name, data in product_images:
            extension = _image_extension(data)
            if not extension:
                raise AppError(f"Product image {Path(name).name} is not a supported JPEG, PNG, or WebP image.")
            if len(data) > MAX_PRODUCT_IMAGE_BYTES:
                raise AppError("Each product image must be 20 MB or smaller.")
            validated_images.append((name, data, extension))

        job_id = str(uuid.uuid4())
        job_dir = self.jobs_dir / job_id
        input_dir = job_dir / "inputs"
        input_dir.mkdir(parents=True)
        video_path = input_dir / f"reference{suffix}"
        video_path.write_bytes(video_data)
        try:
            video_info = self.video_probe(video_path)
        except Exception as exc:
            shutil.rmtree(job_dir, ignore_errors=True)
            raise AppError("The selected file could not be read as a video. Check that it opens and try again.") from None

        job = {
            "id": job_id,
            "project_spending_limit_usd": str(DEFAULT_PROJECT_SPENDING_LIMIT_USD),
            "title": _default_project_title("reference", instructions, Path(video_name).name),
            "mode": "reference",
            "hook": hook,
            "status": "queued",
            "stage": "queued",
            "message": "Reference received. Analysis is starting.",
            "provider_state": None,
            "created_at": _now(),
            "updated_at": _now(),
            "reference_video": str(video_path),
            "reference_video_info": video_info,
            "product_references": [],
            "instructions": instructions,
            "requested_clip_count": clip_count,
            "clip_count_selection": "fixed",
            "seconds_per_clip": seconds_per_clip,
            "duration_selection": "fixed",
            "review_images_before_video": bool(review_images_before_video),
            "assets": {},
            "analysis": None,
            "plan_json": None,
            "estimate": None,
            "estimate_reliable": True,
            "clips": [],
            "charges": {"analysis_usd": "0", "kie_credits": "0", "kie_usd_equivalent": "0", "total_usd": "0", "entries": []},
            "error": None,
        }
        self._add_asset(job, video_path, Path(video_name).name, "video", "reference_video")
        for index, (name, data, extension) in enumerate(validated_images, start=1):
            image_path = input_dir / f"product-{index:02d}{extension}"
            image_path.write_bytes(data)
            job["product_references"].append(str(image_path))
            self._add_asset(job, image_path, Path(name).name, "image", "product_reference")
        self._save(job)
        self._spawn(self._analyze_job, job_id)
        return self.get_job(job_id)

    def create_original_job(self, instructions: str, hook: str = "recommend", clip_count: int = 3, seconds_per_clip: int = 5, product_images: list[tuple[str, bytes]] | None = None, review_images_before_video: bool = DEFAULT_REVIEW_IMAGES_BEFORE_VIDEO) -> dict:
        if not instructions.strip() or creative_instruction_length(instructions) > MAX_CREATIVE_INSTRUCTIONS:
            raise AppError("Describe your original video idea in 1–20,000 characters.")
        if hook not in {"recommend", "in_store_reaction", "wet_works", "countdown_reveal", "gifting", "versus", "custom"}:
            raise AppError("Choose a supported creative hook.")
        if not 1 <= clip_count <= 12 or not 1 <= seconds_per_clip <= 15:
            raise AppError("Choose 1–12 clips and 1–15 seconds per clip.")
        product_images = product_images or []
        if len(product_images) > MAX_PRODUCT_IMAGES:
            raise AppError(f"Upload no more than {MAX_PRODUCT_IMAGES} product images.")
        validated = []
        for name, content in product_images:
            extension = _image_extension(content)
            if not extension or len(content) > MAX_PRODUCT_IMAGE_BYTES:
                raise AppError("Each product image must be a supported JPEG, PNG, or WebP image up to 20 MB.")
            validated.append((name, content, extension))
        job_id = str(uuid.uuid4())
        job = {
            "id": job_id, "title": _default_project_title("original", instructions), "mode": "original", "hook": hook,
            "project_spending_limit_usd": str(DEFAULT_PROJECT_SPENDING_LIMIT_USD),
            "status": "interviewing", "stage": "interviewing",
            "message": "The Creative Director is reviewing your idea.", "provider_state": None,
            "created_at": _now(), "updated_at": _now(), "reference_video": None,
            "reference_video_info": {}, "product_references": [], "instructions": instructions,
            "interview_answers": [], "director_questions": [], "requested_clip_count": clip_count,
            "seconds_per_clip": seconds_per_clip, "assets": {}, "analysis": None,
            "clip_count_selection": "fixed", "duration_selection": "fixed",
            "review_images_before_video": bool(review_images_before_video),
            "plan_json": None, "estimate": None, "estimate_reliable": True, "clips": [],
            "charges": {"analysis_usd": "0", "kie_credits": "0", "kie_usd_equivalent": "0", "total_usd": "0", "entries": []},
            "error": None,
        }
        input_dir = self._job_path(job_id).parent / "inputs"
        input_dir.mkdir(parents=True)
        for index, (name, content, extension) in enumerate(validated, start=1):
            image_path = input_dir / f"product-{index:02d}{extension}"
            image_path.write_bytes(content)
            job["product_references"].append(str(image_path))
            self._add_asset(job, image_path, Path(name).name, "image", "product_reference")
        self._save(job)
        self._spawn(self._director_turn, job_id)
        return self.get_job(job_id)

    def set_review_images_before_video(self, job_id: str, enabled: bool) -> dict:
        if not isinstance(enabled, bool):
            raise AppError("Choose whether to review starting images before video generation.")
        job = self._read(job_id)
        if job.get("status") not in {"ready", "failed", "interrupted"}:
            raise JobConflict("Starting-image review can only be changed before generation starts.")
        updated = self._update(job_id, lambda current: current.update(review_images_before_video=enabled))
        return self._public_job(updated)

    def set_clip_audio_mode(self, job_id: str, clip_index: int, mode: str) -> dict:
        if mode not in {"auto", "enabled", "silent"}:
            raise AppError("Choose Auto, Enabled, or Silent for this clip's audio.")
        with self.lock:
            job = self._read(job_id)
            if job.get("status") not in {"ready", "awaiting_image_review", "failed", "interrupted", "completed"}:
                raise JobConflict("Audio settings can be changed before generation or between runs.")
            plan = self._load_plan(job)
            if not 1 <= clip_index <= len(plan.clips):
                raise AppError("That clip does not exist in this job.")
            clip = plan.clips[clip_index - 1]
            clip.audio_mode = mode  # type: ignore[assignment]
            job["plan_json"] = plan.model_dump_json()
            job["clips"][clip_index - 1].update(
                audio_mode=clip.audio_mode,
                audio_requested=clip.audio_requested,
                audio_enabled=clip.audio_enabled,
                audio_direction=clip.audio_direction,
            )
            job["message"] = f"Clip {clip_index} audio preference saved; it applies to its next generation or regeneration."
            job["updated_at"] = _now()
            self._save(job)
        return self._public_job(job)

    def answer_original_interview(self, job_id: str, answer: str) -> dict:
        if not answer.strip() or creative_instruction_length(answer) > MAX_CREATIVE_INSTRUCTIONS:
            raise AppError("Enter an answer of 1–20,000 characters.")
        with self.lock:
            job = self._read(job_id)
            if job.get("mode") != "original" or job["status"] != "interviewing":
                raise JobConflict("This project is not waiting for an interview answer.")
            job["interview_answers"].append(answer.strip())
            job["status"] = "analyzing_reference"
            job["stage"] = "analyzing_reference"
            job["message"] = "The Creative Director is updating the plan."
            self._save(job)
        self._spawn(self._director_turn, job_id)
        return self.get_job(job_id)

    def refine_plan(self, job_id: str, instruction: str) -> dict:
        if not instruction.strip() or creative_instruction_length(instruction) > MAX_CREATIVE_INSTRUCTIONS:
            raise AppError("Describe the plan change in 1–20,000 characters.")
        revision_attempted = False
        revision_returned = False
        with self.lock:
            job = self._read(job_id)
            if job["status"] != "ready" or not job.get("plan_json"):
                raise JobConflict("The plan can only be refined before generation while the job is ready.")
            self._check_project_spending_limit(job, ANALYSIS_ALLOWANCE_USD)
            current_plan = self._load_plan(job)
            current_analysis = AnalysisResult.model_validate(job["analysis"])
            context = {
                "mode": job.get("mode", "reference"), "hook": job.get("hook", ""),
                "original_instructions": job["instructions"],
                "interview_answers": job.get("interview_answers", []),
                "confirmed_product_facts": current_plan.product_facts + "\nInterview answers: " + "\n".join(job.get("interview_answers", [])),
                "analysis_summary": current_analysis.summary,
            }
            # Reserve this job while the model revises so concurrent edits or generation cannot race it.
            job["status"] = "refining_plan"
            job["stage"] = "refining_plan"
            job["message"] = "Updating the creative plan."
            self._save(job)
        try:
            revision_attempted = True
            revision = self.plan_reviser(instruction.strip(), current_plan.model_dump(mode="json"), context)
            revision_returned = True
            returned = AnalysisResult.model_validate({"summary": revision["summary"], "questions": [], "shots": revision["shots"]})
            expected = {clip.source_shot_index for clip in current_plan.clips}
            actual = {shot.index for shot in returned.shots}
            changed = set(revision["changed_shot_indices"])
            if actual != expected or not changed <= expected:
                raise AppError("The revised plan did not preserve the existing shot list.")
            revised_by_id = {shot.index: shot for shot in returned.shots}
            old_shots = {shot.index: shot for shot in current_analysis.shots}
            merged_analysis = current_analysis.model_copy(update={"summary": revision["summary"], "shots": [revised_by_id.get(shot.index, shot) if shot.index in changed else shot for shot in current_analysis.shots]})
            planned_analysis = merged_analysis.model_copy(update={
                "shots": [revised_by_id[clip.source_shot_index] if clip.source_shot_index in changed else old_shots[clip.source_shot_index] for clip in current_plan.clips],
            })
            new_plan = build_multiclip_plan(
                planned_analysis, current_plan.instructions, current_plan.product_facts,
                current_plan.environment, len(current_plan.clips), current_plan.clips[0].duration_seconds,
                product_references=[Path(path) for path in current_plan.product_references],
                continuity_references=[Path(path) for path in current_plan.continuity_references],
            )
            # Keep duration, source mapping, and unrequested shot prompts exactly when those shots were not changed.
            new_clips = []
            by_source = {clip.source_shot_index: clip for clip in new_plan.clips}
            for old in current_plan.clips:
                if old.source_shot_index not in changed:
                    new_clips.append(old)
                else:
                    new_clips.append(by_source[old.source_shot_index].model_copy(update={
                        "index": old.index, "duration_seconds": old.duration_seconds,
                        "source_start_seconds": old.source_start_seconds,
                        "source_end_seconds": old.source_end_seconds,
                        "audio_mode": old.audio_mode,
                    }))
            new_plan = current_plan.model_copy(update={"summary": revision["summary"], "clips": new_clips})
            usage = getattr(self.plan_reviser, "last_usage", {}) or {}
            cost = price_openai_analysis_usage(usage)
            reliable = cost is not None
            if cost is None:
                cost = ANALYSIS_ALLOWANCE_USD
            estimate = estimate_multiclip_generation(new_plan, analysis_usd=Decimal(job["charges"]["analysis_usd"]) + cost)
            def finish(data: dict) -> None:
                data["charges"]["analysis_usd"] = f"{Decimal(data['charges']['analysis_usd']) + cost:.6f}"
                data["charges"]["total_usd"] = f"{Decimal(data['charges']['analysis_usd']) + Decimal(data['charges']['kie_usd_equivalent']):.6f}"
                data["charges"]["entries"].append({"operation_id": uuid.uuid4().hex, "recorded_at": _now(), "kind": "plan_revision", "usd_equivalent": f"{cost:.6f}", "source": "OpenAI token usage" if reliable else "usage unavailable; allowance applied"})
                data["estimate_reliable"] = bool(data.get("estimate_reliable", True) and reliable)
                data["analysis"] = merged_analysis.model_dump(mode="json")
                data["plan_json"] = new_plan.model_dump_json()
                edit_total = sum(clip.edit_duration_seconds or 0 for clip in new_clips)
                data["estimate"] = {"analysis_usd": data["charges"]["analysis_usd"], "images_usd": f"{estimate.image_usd:.4f}", "video_usd": f"{estimate.video_usd:.4f}", "total_usd": f"{estimate.total_usd:.4f}", "clip_count": len(new_clips), "seconds_per_clip": new_clips[0].duration_seconds, "total_video_seconds": estimate.generated_seconds, "intended_edit_seconds": edit_total or None}
                for clip, planned in zip(data["clips"], new_clips):
                    clip.update(title=planned.title, creative_role=planned.creative_role, action=planned.action, camera=planned.camera, duration_seconds=planned.duration_seconds, edit_duration_seconds=planned.edit_duration_seconds, audio_mode=planned.audio_mode, audio_requested=planned.audio_requested, audio_enabled=planned.audio_enabled, audio_direction=planned.audio_direction)
                data.update(status="ready", stage="ready", message="Creative plan updated. Review the clips and estimate before generation.", error=None)
            return self._public_job(self._update(job_id, finish))
        except Exception as exc:
            if revision_attempted:
                revision_cost = price_openai_analysis_usage(getattr(self.plan_reviser, "last_usage", {}) or {}) if revision_returned else None
                if revision_cost is not None:
                    self._record_openai_cost(job_id, "plan_revision", revision_cost)
                else:
                    self._record_unknown_openai_operation(job_id, "plan_revision")
            self._update(job_id, lambda data: data.update(status="ready", stage="ready", message="Plan update failed; the previous plan is preserved.", error=str(exc)))
            raise

    def _director_turn(self, job_id: str) -> None:
        director_attempted = False
        director_returned = False
        director_usage_recorded = False
        try:
            job = self._read(job_id)
            self._check_project_spending_limit(job, ANALYSIS_ALLOWANCE_USD)
            director_attempted = True
            director_instructions = job["instructions"]
            if job.get("clip_count_selection") == "director":
                director_instructions += "\nClip count preference: Creative Director decides the purposeful shot count, up to the supported maximum."
            if job.get("duration_selection") == "director":
                director_instructions += "\nDuration preference: Creative Director chooses an appropriate generated duration for each planned shot."
            analysis = self.director(director_instructions, job["interview_answers"], job["hook"])
            director_returned = True
            usage = getattr(self.director, "last_usage", {}) or {}
            cost = price_openai_analysis_usage(usage)
            reliable = cost is not None
            if cost is None:
                cost = ANALYSIS_ALLOWANCE_USD
            def persist_usage(data: dict) -> None:
                data["charges"]["analysis_usd"] = f"{Decimal(data['charges']['analysis_usd']) + cost:.6f}"
                data["charges"]["total_usd"] = data["charges"]["analysis_usd"]
                data["estimate_reliable"] = bool(data.get("estimate_reliable", True) and reliable)
                data["charges"]["entries"].append({"operation_id": uuid.uuid4().hex, "recorded_at": _now(), "kind": "creative_director", "usd_equivalent": f"{cost:.6f}", "source": "OpenAI token usage" if reliable else "usage unavailable; allowance applied"})
            self._update(job_id, persist_usage)
            director_usage_recorded = True
            job = self._read(job_id)
            if analysis.questions or not analysis.shots:
                self._update(job_id, lambda data: data.update(
                    status="interviewing", stage="interviewing", director_questions=analysis.questions or ["Tell me more about the product's real function so I can plan a believable demonstration."],
                    analysis=analysis.model_dump(mode="json"), message="Answer the Creative Director's questions to continue."))
                return
            seconds = job["seconds_per_clip"]
            requested_hook = job.get("hook", "recommend")
            if requested_hook in {"in_store_reaction", "wet_works"}:
                plan_analysis = fit_reference_analysis_to_clip_count(
                    analysis, job["requested_clip_count"], hook=requested_hook, seconds_per_clip=job["seconds_per_clip"],
                )
            else:
                plan_analysis = analysis.model_copy(deep=True)
                plan_analysis.shots = plan_analysis.shots[:job["requested_clip_count"]]
            shots = plan_analysis.shots
            if not shots:
                raise AppError("The Creative Director did not return a usable shot plan.")
            image_refs = [Path(path) for path in job["product_references"]]
            plan = build_multiclip_plan(
                plan_analysis, job["instructions"],
                "Preserve confirmed product facts and supplied visual references. Do not invent product capabilities.",
                "the environment specified for this individual scene", len(shots), seconds,
                product_references=image_refs, hook=requested_hook,
            )
            estimate = estimate_multiclip_generation(plan, analysis_usd=Decimal(job["charges"]["analysis_usd"]))
            clips = [{"index": clip.index, "title": clip.title, "creative_role": clip.creative_role, "action": clip.action, "camera": clip.camera, "source_start_seconds": None, "source_end_seconds": None, "duration_seconds": clip.duration_seconds, "edit_duration_seconds": clip.edit_duration_seconds, "audio_mode": clip.audio_mode, "audio_requested": clip.audio_requested, "audio_enabled": clip.audio_enabled, "audio_direction": clip.audio_direction, "status": "planned", "versions": []} for clip in plan.clips]
            edit_total = sum(clip.edit_duration_seconds or 0 for clip in plan.clips)
            self._update(job_id, lambda data: data.update(
                status="ready", stage="ready", message=f"Creative plan ready: {len(clips)} distinct clips.",
                analysis=analysis.model_dump(mode="json"), plan_json=plan.model_dump_json(), clips=clips,
                estimate={"analysis_usd": data["charges"]["analysis_usd"], "images_usd": f"{estimate.image_usd:.4f}", "video_usd": f"{estimate.video_usd:.4f}", "total_usd": f"{estimate.total_usd:.4f}", "clip_count": len(clips), "seconds_per_clip": seconds, "total_video_seconds": estimate.generated_seconds, "intended_edit_seconds": edit_total or None},
                estimate_reliable=data["estimate_reliable"], director_questions=[]))
        except Exception as exc:
            if director_attempted and not director_usage_recorded:
                director_cost = price_openai_analysis_usage(getattr(self.director, "last_usage", {}) or {}) if director_returned else None
                if director_cost is not None:
                    self._record_openai_cost(job_id, "creative_director", director_cost)
                else:
                    self._record_unknown_openai_operation(job_id, "creative_director")
            self._set_failure(job_id, str(exc))

    @staticmethod
    def _add_asset(job: dict, path: Path, filename: str, media_type: str, role: str) -> str:
        asset_id = str(uuid.uuid4())
        content_type = mimetypes.guess_type(path.name)[0]
        if media_type == "video":
            content_type = content_type or "video/mp4"
        else:
            content_type = content_type or "application/octet-stream"
        job["assets"][asset_id] = {
            "path": str(path.resolve()),
            "filename": filename,
            "media_type": content_type,
            "role": role,
        }
        return asset_id

    def _analyze_job(self, job_id: str) -> None:
        analysis_attempted = False
        analysis_returned = False
        analysis_usage_recorded = False
        try:
            job = self._update(job_id, lambda data: data.update(
                status="analyzing_reference", stage="analyzing_reference",
                message="Extracting local frames and analyzing the reference video.",
            ))
            video_path = Path(job["reference_video"])
            job_dir = self._job_path(job_id).parent
            frame_dir = job_dir / "frames"
            frames = self.frame_sampler(video_path, frame_dir, count=8)
            self._check_project_spending_limit(job, ANALYSIS_ALLOWANCE_USD)
            analysis_attempted = True
            analysis = self.analyzer(
                frames,
                job["instructions"],
                "Use supplied product images as authoritative appearance references. Do not invent product construction or capabilities.",
                reference_duration_seconds=job["reference_video_info"]["duration"],
                hook=job.get("hook") or ("in_store_reaction" if re.search(r"\b(?:in[- ]store reaction|wet works)\b", job["instructions"], re.I) else "recommend"),
                clip_count=job["requested_clip_count"],
                seconds_per_clip=job["seconds_per_clip"],
                duration_selection=job.get("duration_selection", "fixed"),
                reference_has_audio=job["reference_video_info"].get("has_audio"),
            )
            analysis_returned = True
            analysis = normalize_reference_segments(analysis, job["reference_video_info"]["duration"])
            usage = getattr(self.analyzer, "last_usage", {}) or {}
            analysis_usd = price_openai_analysis_usage(usage)
            reliable = analysis_usd is not None
            if analysis_usd is None:
                analysis_usd = ANALYSIS_ALLOWANCE_USD
            if not analysis.shots:
                raise AppError("The reference analysis did not identify any usable shots.")

            clip_count = min(job["requested_clip_count"], len(analysis.shots))
            selected_hook = job.get("hook") or ("in_store_reaction" if re.search(r"\b(?:in[- ]store reaction|wet works)\b", job["instructions"], re.I) else "recommend")
            if selected_hook in {"in_store_reaction", "wet_works"}:
                plan_analysis = fit_reference_analysis_to_clip_count(
                    analysis, clip_count, hook=selected_hook, seconds_per_clip=job["seconds_per_clip"],
                )
            else:
                plan_analysis = analysis.model_copy(deep=True)
                plan_analysis.shots = plan_analysis.shots[:clip_count]
            if job["product_references"]:
                image_refs = [Path(path) for path in job["product_references"]]
            else:
                image_refs = frames[: min(3, len(frames))]
            environment = (
                "the scene and lighting specified in the user's creative instructions; "
                "if no replacement setting is requested, match the analyzed reference video's environment"
            )
            product_facts = (
                "Match visible product appearance and construction in supplied product image references. "
                "When no product photos are supplied, use the reference video frames. Do not infer unseen capabilities."
            )
            plan = build_multiclip_plan(
                plan_analysis,
                job["instructions"],
                product_facts,
                environment,
                clip_count,
                job["seconds_per_clip"],
                product_references=image_refs,
                hook=selected_hook,
            )
            estimate = estimate_multiclip_generation(plan, analysis_usd=analysis_usd)
            edit_total = sum(clip.edit_duration_seconds or 0 for clip in plan.clips)
            public_analysis = analysis.model_dump(mode="json")
            clips = [{
                "index": clip.index,
                "title": clip.title,
                "creative_role": clip.creative_role,
                "action": clip.action,
                "camera": clip.camera,
                "source_start_seconds": clip.source_start_seconds,
                "source_end_seconds": clip.source_end_seconds,
                "duration_seconds": clip.duration_seconds,
                "edit_duration_seconds": clip.edit_duration_seconds,
                "audio_mode": clip.audio_mode,
                "audio_requested": clip.audio_requested,
                "audio_enabled": clip.audio_enabled,
                "audio_direction": clip.audio_direction,
                "status": "planned",
                "versions": [],
            } for clip in plan.clips]

            def finish(data: dict) -> None:
                data.update(
                    status="ready",
                    stage="ready",
                    message=f"Analysis complete. {len(clips)} distinct clips are ready to generate.",
                    analysis=public_analysis,
                    plan_json=plan.model_dump_json(),
                    estimate={
                        "analysis_usd": f"{analysis_usd:.6f}",
                        "images_usd": f"{estimate.image_usd:.4f}",
                        "video_usd": f"{estimate.video_usd:.4f}",
                        "total_usd": f"{estimate.total_usd:.4f}",
                        "clip_count": len(clips),
                        "seconds_per_clip": job["seconds_per_clip"],
                        "total_video_seconds": estimate.generated_seconds,
                        "intended_edit_seconds": edit_total or None,
                    },
                    estimate_reliable=reliable,
                    clips=clips,
                )
                data["charges"]["analysis_usd"] = f"{analysis_usd:.6f}"
                data["charges"]["total_usd"] = f"{analysis_usd:.6f}"
                data["charges"]["entries"].append({
                    "operation_id": uuid.uuid4().hex,
                    "recorded_at": _now(),
                    "kind": "reference_analysis",
                    "usd_equivalent": f"{analysis_usd:.6f}",
                    "source": "OpenAI token usage" if reliable else "analysis allowance; usage unavailable",
                })

            self._update(job_id, finish)
            analysis_usage_recorded = True
        except Exception as exc:
            if analysis_attempted and not analysis_usage_recorded:
                analysis_cost = price_openai_analysis_usage(getattr(self.analyzer, "last_usage", {}) or {}) if analysis_returned else None
                if analysis_cost is not None:
                    self._record_openai_cost(job_id, "reference_analysis", analysis_cost)
                else:
                    self._record_unknown_openai_operation(job_id, "reference_analysis")
            self._set_failure(job_id, str(exc))

    def _set_failure(self, job_id: str, message: str) -> None:
        try:
            def fail(job: dict) -> None:
                if job.get("stage") in {"queued", "analyzing_reference"}:
                    safe_message = "Reference analysis could not complete. Check the OpenAI key, connection, and reference file."
                else:
                    clip = job.get("active_clip")
                    stage = job.get("active_stage") or job.get("stage")
                    stage = {
                        "preparing_images": "preparing its starting image",
                        "regenerating_starting_image": "regenerating its starting image",
                        "generating_clips": "submitting or generating its video",
                        "processing_outputs": "saving its video output",
                    }.get(stage, stage)
                    detail = re.sub(r"Bearer\s+[^\s]+", "Bearer [redacted]", str(message))[:500]
                    if clip and stage == "preparing its starting image":
                        safe_message = (f"Clip {clip} failed while preparing its starting image: {detail}. Saved starting images and provider task history are preserved. "
                                        "Retrying will reuse completed images and submit a new task only for this failed image; check provider-reported usage before retrying.")
                    else:
                        safe_message = (f"Clip {clip} failed while {stage}: {detail}. Finished clips and reported charges are preserved. Resume reuses saved starting images and provider tasks.") if clip else f"Generation failed: {detail}. Finished clips and reported charges are preserved."
                job.update(status="failed", stage="failed", message=safe_message, error=safe_message)
            self._update(job_id, fail)
        except JobNotFound:
            return

    def _estimate(self, job: dict):
        plan = self._load_plan(job)
        analysis_cost = Decimal(job.get("charges", {}).get("analysis_usd", "0"))
        return estimate_multiclip_generation(plan, analysis_usd=analysis_cost)

    def _video_operation(self, job: dict) -> tuple[str, Decimal]:
        plan = self._load_plan(job)
        image_ids = [clip.get("current_starting_image_asset") for clip in sorted(job.get("clips", []), key=lambda item: item.get("index", 0))]
        operation = hashlib.sha256((plan.model_dump_json() + json.dumps(image_ids, separators=(",", ":"))).encode()).hexdigest()
        amount = sum((Decimal(str(clip.duration_seconds)) * Decimal("0.0225") for clip in plan.clips), Decimal("0"))
        return operation, amount

    def _bind_video_authorization(self, job_id: str) -> None:
        def bind(job: dict) -> None:
            approved = job.get("approved_max_usd")
            if not approved:
                return
            operation, amount = self._video_operation(job)
            if estimate_within_approved_amount(amount, Decimal(approved)):
                job["video_generation_authorization"] = {
                    "operation": operation,
                    "approved_max_usd": approved,
                    "authorized_at": _now(),
                }
        self._update(job_id, bind)

    @staticmethod
    def _load_plan(job: dict) -> MultiClipPlan:
        plan = MultiClipPlan.model_validate_json(job["plan_json"])
        selected_hook = job.get("hook")
        automatic_hook = selected_hook in {None, "", "recommend"}
        if automatic_hook and plan.hook in {"custom", "recommend"} and re.search(
            r"\b(?:in[- ]store reaction|wet works)\b", job.get("instructions", ""), re.I,
        ):
            selected_hook = "in_store_reaction"
        if selected_hook in {"in_store_reaction", "wet_works", "countdown_reveal", "gifting", "versus", "custom"}:
            plan = plan.model_copy(update={"hook": selected_hook})
        return plan

    def start_generation(self, job_id: str, approved_max_usd: Decimal | None = None, resume: bool = False, review_images: bool | None = None) -> dict:
        with self.lock:
            job = self._read(job_id)
            recoverable = resume and job["status"] in {"failed", "interrupted"} and bool(job.get("plan_json"))
            if job["status"] != "ready" and not recoverable:
                raise JobConflict("This job is not ready to generate. Check its current status.")
            retry_image_info = self._resume_image_retry_info(job) if recoverable else None
            pending_image_regen = job.get("pending_starting_image_regen") if recoverable else None
            pending_clip_regen = job.get("pending_clip_regeneration") if recoverable else None
            estimate = self._estimate(job)
            saved_images_ready = bool(job.get("clips")) and all(
                clip.get("current_starting_image_asset") and
                Path(job.get("assets", {}).get(clip["current_starting_image_asset"], {}).get("path", "")).is_file()
                for clip in job.get("clips", [])
            )
            videos_ready = (bool(job.get("images_approved_at")) or
                            (recoverable and not job.get("review_images_before_video", False) and saved_images_ready)) and not pending_image_regen
            video_operation = None
            video_estimate = None
            if videos_ready:
                video_operation, video_estimate = self._video_operation(job)
                authorization = job.get("video_generation_authorization") or {}
                authorized = authorization.get("operation") == video_operation
                if authorized and approved_max_usd is None:
                    approved_max_usd = Decimal(authorization["approved_max_usd"])
            if pending_image_regen:
                pending_clip = bool(pending_image_regen.get("clip_regeneration"))
                pending_duration = self._load_plan(job).clips[int(pending_image_regen["clip_index"]) - 1].duration_seconds
                operation_estimate = (Decimal(str(pending_duration)) * Decimal("0.0225")
                                      if pending_clip and pending_image_regen.get("image_ready")
                                      else IMAGE_1K_USD)
            elif videos_ready:
                operation_estimate = video_estimate
            else:
                operation_estimate = Decimal(retry_image_info["estimated_usd"]) if retry_image_info else estimate.total_usd
            confirmation_needed = (
                operation_estimate > AUTO_GENERATION_APPROVAL_LIMIT_USD or not job["estimate_reliable"]
                if retry_image_info or pending_image_regen or videos_ready else generation_approval_required(estimate, pricing_verified=True, estimate_reliable=job["estimate_reliable"])
            )
            if confirmation_needed and approved_max_usd is None:
                raise ConfirmationRequired(operation_estimate, "This job exceeds the automatic-spend threshold or its estimate is uncertain.")
            if approved_max_usd is not None and not estimate_within_approved_amount(operation_estimate, approved_max_usd):
                raise AppError(f"Estimated total ${operation_estimate:.4f} exceeds the approved maximum ${approved_max_usd:.4f}.")
            project_operation_estimate = operation_estimate if (retry_image_info or pending_image_regen or videos_ready) else estimate.image_usd + estimate.video_usd
            self._check_project_spending_limit(job, project_operation_estimate)
            if pending_clip_regen and not pending_clip_regen.get("image_ready"):
                pending_image_regen = {"clip_index": pending_clip_regen["clip_index"],
                    "instructions": pending_clip_regen["instructions"], "output_dir": pending_clip_regen["output_dir"],
                    "clip_regeneration": True}
                job["pending_starting_image_regen"] = pending_image_regen
            if pending_clip_regen and pending_clip_regen.get("image_ready") and not job.get("images_approved_at"):
                job.update(status="awaiting_image_review", stage="awaiting_image_review",
                           message="Review the revised starting image, then approve it to generate the revised clip.",
                           updated_at=_now())
                self._save(job)
                return self.get_job(job_id)
            job["status"] = "generating_clips" if videos_ready else "preparing_images"
            job["stage"] = job["status"]
            job["message"] = "Resuming from saved tasks and outputs." if recoverable else "Preparing starting images for review."
            job["resume_generation"] = recoverable
            if review_images is not None and not recoverable:
                job["review_images_before_video"] = bool(review_images)
            job["approved_max_usd"] = str(approved_max_usd) if approved_max_usd is not None else None
            if videos_ready and approved_max_usd is not None:
                job["video_generation_authorization"] = {
                    "operation": video_operation,
                    "approved_max_usd": str(approved_max_usd),
                    "authorized_at": _now(),
                }
            job["resume_operation"] = retry_image_info
            job["provider_state"] = None
            job["updated_at"] = _now()
            self._save(job)
        if pending_image_regen:
            self._spawn(self._resume_starting_image_regen, job_id)
        elif pending_clip_regen and videos_ready:
            self._spawn(self._generate_regenerated_clip, job_id)
        else:
            self._spawn(self._generate_videos if videos_ready else self._generate_job, job_id)
        return self.get_job(job_id)

    def _generate_job(self, job_id: str) -> None:
        job = self._read(job_id)
        plan = self._load_plan(job)
        output_dir = self._job_path(job_id).parent / "outputs" / "initial"
        analysis_usd = Decimal(job["charges"]["analysis_usd"])

        def progress(event: dict) -> None:
            def change(data: dict) -> None:
                data["status"] = event["stage"]
                data["stage"] = event["stage"]
                data["provider_state"] = event.get("provider_state")
                clip_index = event.get("clip_index")
                data["active_clip"] = clip_index
                data["active_stage"] = event["stage"]
                count = event.get("clip_count", len(plan.clips))
                stage_text = {
                    "preparing_images": "Preparing starting images",
                    "generating_clips": "Generating separate clips",
                    "processing_outputs": "Making silent video files",
                }.get(event["stage"], event["stage"])
                data["message"] = f"{stage_text} · clip {clip_index} of {count}"
            self._update(job_id, change)

        client = None
        try:
            client = self.client_factory()
            outputs = run_multiclip_generation(
                plan,
                [Path(path) for path in plan.product_references],
                [Path(path) for path in plan.continuity_references],
                output_dir,
                client,
                approved_max_usd=Decimal(job["approved_max_usd"]) if job.get("approved_max_usd") else None,
                pricing_verified=True,
                estimate_reliable=job["estimate_reliable"],
                analysis_usd=analysis_usd,
                on_progress=progress,
                resume=bool(job.get("resume_generation")),
                images_only=True,
            )
        except Exception as exc:
            self._collect_charges(job_id, output_dir, "initial_generation")
            self._set_failure(job_id, str(exc))
            return
        finally:
            if client is not None:
                client.close()

        self._collect_charges(job_id, output_dir, "initial_generation")
        self._save_starting_image_assets(job_id, plan, outputs)
        self._bind_video_authorization(job_id)
        self._update(job_id, lambda data: data.update(resume_operation=None))
        current = self._read(job_id)
        if not current.get("review_images_before_video", False):
            self._update(job_id, lambda data: data.update(status="generating_clips", stage="generating_clips",
                message="Starting images are ready. Generating planned clips.", resume_generation=True))
            self._generate_videos(job_id)
            return
        self._update(job_id, lambda data: data.update(
            status="awaiting_image_review", stage="awaiting_image_review", provider_state="success",
            message="Review every starting image. Approve the set to generate videos.", error=None,
            active_clip=None, active_stage=None,
        ))

    def _save_starting_image_assets(self, job_id: str, plan: MultiClipPlan, images: list[Path]) -> None:
        saved_job = self._read(job_id)
        full_plan = self._load_plan(saved_job)
        identity_index = product_identity_clip_index(full_plan)
        def change(job: dict) -> None:
            for clip, image_path in zip(plan.clips, images):
                target = job["clips"][clip.index - 1]
                resolved = str(Path(image_path).resolve())
                current_id = target.get("current_starting_image_asset")
                if current_id and Path(job["assets"][current_id]["path"]).resolve() == Path(resolved):
                    continue
                asset_id = self._add_asset(job, Path(resolved), f"clip-{clip.index:02d}-starting.png", "image", "starting_image")
                history = target.setdefault("starting_image_history", [])
                entry = {"version": len(history) + 1, "asset_id": asset_id, "created_at": _now()}
                history.append(entry)
                target["current_starting_image_asset"] = asset_id
                target["status"] = "image_ready"
            identity_clip = next((clip for clip in job.get("clips", []) if clip.get("index") == identity_index), None)
            if identity_clip and identity_clip.get("current_starting_image_asset"):
                job["product_identity_reference_asset"] = identity_clip["current_starting_image_asset"]
        self._update(job_id, change)

    def _product_identity_reference_path(self, job: dict) -> Path | None:
        asset_id = job.get("product_identity_reference_asset")
        if not asset_id:
            try:
                plan = self._load_plan(job)
                identity_index = product_identity_clip_index(plan)
                clip = next((item for item in job.get("clips", []) if item.get("index") == identity_index), None)
                asset_id = clip.get("current_starting_image_asset") if clip else None
            except (KeyError, ValueError):
                asset_id = None
        asset = job.get("assets", {}).get(asset_id) if asset_id else None
        path = Path(asset["path"]) if asset and asset.get("path") else None
        return path if path and path.is_file() else None

    def _character_identity_reference_paths(self, job: dict, plan: MultiClipPlan, clip: PlannedClip) -> list[Path]:
        """Return one same-group identity anchor, preferring an approved image."""
        groups = {group.strip() for group in clip.character_groups if group.strip()}
        if not groups:
            return []
        planned = {item.index: item for item in plan.clips}
        candidates: dict[str, list[tuple[int, Path, bool]]] = {group: [] for group in groups}
        approved_ids = set(job.get("approved_starting_image_assets", []))
        if job.get("images_approved_at"):
            approved_ids.update({
                record.get("current_starting_image_asset")
                for record in job.get("clips", [])
                if record.get("current_starting_image_asset")
            })
        for record in job.get("clips", []):
            index = record.get("index")
            if index == clip.index or index not in planned:
                continue
            history = record.get("starting_image_history", [])
            asset_ids = [item.get("asset_id") for item in history if item.get("asset_id")]
            current_id = record.get("current_starting_image_asset")
            if current_id and current_id not in asset_ids:
                asset_ids.append(current_id)
            approved_entries = []
            for asset_id in asset_ids:
                asset = job.get("assets", {}).get(asset_id, {})
                path = Path(asset.get("path", ""))
                if path.is_file() and asset_id in approved_ids:
                    approved_entries.append((index, path, True))
            if approved_entries:
                version_by_path = {
                    Path(job.get("assets", {}).get(item.get("asset_id"), {}).get("path", "")).resolve(): item.get("version", 0)
                    for item in history if item.get("asset_id")
                }
                record_candidates = [max(approved_entries, key=lambda item: version_by_path.get(item[1].resolve(), 0))]
            else:
                current_asset = job.get("assets", {}).get(current_id, {}) if current_id else {}
                current_path = Path(current_asset.get("path", ""))
                record_candidates = [(index, current_path, False)] if current_path.is_file() else []
            if not record_candidates:
                continue
            for group in groups.intersection(g.strip() for g in planned[index].character_groups):
                candidates[group].extend(record_candidates)
        selected = []
        seen = set()
        for group in sorted(groups):
            if candidates[group]:
                # When the set has been approved, do not let a newer unapproved
                # edit silently replace the established identity anchor.
                same_group = candidates[group]
                approved = [item for item in same_group if item[2]]
                path = min(approved or same_group, key=lambda item: item[0])[1]
                if path not in seen:
                    selected.append(path)
                    seen.add(path)
        return selected

    def get_scene_reference(self, job_id: str, clip_index: int, reference_index: int) -> tuple[Path, str, str]:
        job = self._read(job_id)
        plan = self._load_plan(job)
        if not 1 <= clip_index <= len(plan.clips):
            raise JobNotFound("Scene reference not found")
        references = plan.clips[clip_index - 1].environment_references
        if not 0 <= reference_index < len(references):
            raise JobNotFound("Scene reference not found")
        path = Path(references[reference_index]).expanduser().resolve()
        root = self._job_path(job_id).parent.resolve()
        media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if root not in path.parents or not path.is_file() or not media_type.startswith("image/"):
            raise JobNotFound("Scene reference not found")
        return path, media_type, path.name

    def _resolve_scene_base(self, job_id: str, job: dict, source: str) -> Path:
        if source.startswith("asset:"):
            asset_id = source.removeprefix("asset:")
            asset = job.get("assets", {}).get(asset_id)
            if not asset or asset.get("role") != "starting_image" or not str(asset.get("media_type", "")).startswith("image/"):
                raise AppError("Choose a saved starting image or a planned scene environment reference as the edit base.")
            path = Path(asset["path"]).expanduser().resolve()
            root = self._job_path(job_id).parent.resolve()
            if root not in path.parents or not path.is_file():
                raise AppError("The selected scene image is unavailable.")
            return path
        match = re.fullmatch(r"environment:(\d+):(\d+)", source)
        if match:
            path, _, _ = self.get_scene_reference(job_id, int(match.group(1)), int(match.group(2)))
            return path
        raise AppError("Choose a saved starting image or a planned scene environment reference as the edit base.")

    def scene_base_images(self, job_id: str, job: dict) -> list[dict]:
        outputs = self._job_path(job_id).parent / "outputs"
        candidates = []
        for clip in job.get("clips", []):
            for history in clip.get("starting_image_history", []):
                asset_id = history.get("asset_id")
                asset = job.get("assets", {}).get(asset_id, {})
                path = Path(asset.get("path", "")).expanduser().resolve()
                if asset.get("role") != "starting_image" or not path.is_file():
                    continue
                candidates.append({
                    "value": f"asset:{asset_id}",
                    "label": f"Clip {clip['index']} image v{history.get('version', '?')} · {asset.get('filename', path.name)}",
                    "url": f"/api/jobs/{job_id}/assets/{asset_id}",
                    "mask_available": bool(find_saved_mask_for_scene(outputs, path)),
                })
        try:
            plan = self._load_plan(job)
        except (KeyError, ValueError):
            plan = None
        if plan:
            for clip in plan.clips:
                for ref_index, raw_path in enumerate(clip.environment_references):
                    path = Path(raw_path).expanduser().resolve()
                    root = self._job_path(job_id).parent.resolve()
                    if root not in path.parents or not path.is_file() or not (mimetypes.guess_type(path.name)[0] or "").startswith("image/"):
                        continue
                    candidates.append({
                        "value": f"environment:{clip.index}:{ref_index}",
                        "label": f"Clip {clip.index} scene reference · {path.name}",
                        "url": f"/api/jobs/{job_id}/scene-references/{clip.index}/{ref_index}",
                        "mask_available": bool(find_saved_mask_for_scene(outputs, path)),
                    })
        return candidates

    def approve_starting_images(self, job_id: str, approved_max_usd: Decimal | None = None) -> dict:
        with self.lock:
            job = self._read(job_id)
            if job.get("status") != "awaiting_image_review":
                raise JobConflict("Starting images are not awaiting approval.")
            for clip in job.get("clips", []):
                asset_id = clip.get("current_starting_image_asset")
                asset = job.get("assets", {}).get(asset_id) if asset_id else None
                if not asset or not Path(asset.get("path", "")).is_file():
                    raise JobConflict(f"Clip {clip.get('index')} is missing a readable starting image. Resume image preparation before approval.")
            pending_clip = job.get("pending_clip_regeneration")
            if pending_clip:
                clip = self._load_plan(job).clips[int(pending_clip["clip_index"]) - 1]
                operation = None
                operation_estimate = Decimal(str(clip.duration_seconds)) * Decimal("0.0225")
                if approved_max_usd is None and pending_clip.get("approved_max_usd"):
                    approved_max_usd = Decimal(pending_clip["approved_max_usd"])
            else:
                operation, operation_estimate = self._video_operation(job)
                authorization = job.get("video_generation_authorization") or {}
                if authorization.get("operation") == operation and approved_max_usd is None:
                    approved_max_usd = Decimal(authorization["approved_max_usd"])
            confirmation_needed = operation_estimate > AUTO_GENERATION_APPROVAL_LIMIT_USD or not job.get("estimate_reliable", True)
            if confirmation_needed and approved_max_usd is None:
                raise ConfirmationRequired(operation_estimate, "Video generation exceeds the automatic-spend threshold or its estimate is uncertain.")
            if approved_max_usd is not None and not estimate_within_approved_amount(operation_estimate, approved_max_usd):
                raise AppError(f"Estimated video cost ${operation_estimate:.4f} exceeds the approved maximum ${approved_max_usd:.4f}.")
            self._check_project_spending_limit(job, operation_estimate)
            if approved_max_usd is not None and not pending_clip:
                job["video_generation_authorization"] = {
                    "operation": operation,
                    "approved_max_usd": str(approved_max_usd),
                    "authorized_at": _now(),
                }
            elif not pending_clip:
                job.pop("video_generation_authorization", None)
            job.update(images_approved_at=_now(), status="generating_clips", stage="generating_clips",
                       message="Starting images approved. Generating planned clips.", error=None,
                       resume_generation=True, resume_operation=None, updated_at=_now())
            job["approved_starting_image_assets"] = [
                clip.get("current_starting_image_asset") for clip in job.get("clips", [])
                if clip.get("current_starting_image_asset")
            ]
            self._save(job)
            is_clip_regeneration = bool(job.get("pending_clip_regeneration"))
        self._spawn(self._generate_regenerated_clip if is_clip_regeneration else self._generate_videos, job_id)
        return self.get_job(job_id)

    def regenerate_starting_image(
        self, job_id: str, clip_index: int, instructions: str = "",
        approved_max_usd: Decimal | None = None, *, masked_edit: bool = False,
        base_source: str | None = None, mask_upload: tuple[str, bytes] | None = None,
    ) -> dict:
        with self.lock:
            job = self._read(job_id)
            if job.get("status") != "awaiting_image_review":
                raise JobConflict("Starting images can only be revised during image review.")
            if not 1 <= clip_index <= len(job.get("clips", [])):
                raise AppError("That clip does not exist in this job.")
            clip_record = job["clips"][clip_index - 1]
            if not clip_record.get("current_starting_image_asset"):
                raise JobConflict("This clip does not have a starting image to revise.")
            if len(instructions) > 1200:
                raise AppError("Image revision instructions must be 1,200 characters or fewer.")
            estimate = MASKED_EDIT_1K_USD if masked_edit else IMAGE_1K_USD
            scene_base = None
            mask_path = None
            if masked_edit:
                if not base_source:
                    raise AppError("Select an appropriate saved scene image before using masked editing.")
                scene_base = self._resolve_scene_base(job_id, job, base_source)
                mask_path = find_saved_mask_for_scene(self._job_path(job_id).parent / "outputs", scene_base)
                if mask_upload:
                    filename, data = mask_upload
                    if len(data) > MAX_MASK_IMAGE_BYTES:
                        raise AppError("Mask images must be 20 MB or smaller.")
                    if not filename.lower().endswith(".png") or not data.startswith(b"\x89PNG\r\n\x1a\n"):
                        raise AppError("Upload the localized edit mask as a PNG.")
                    output_dir = self._job_path(job_id).parent / "outputs" / f"clip-{clip_index:02d}" / "starting-image" / f"version-{len(clip_record.get('starting_image_history', [])) + 1}"
                    output_dir.mkdir(parents=True, exist_ok=True)
                    mask_path = output_dir / "scene-inpaint-mask.png"
                    mask_path.write_bytes(data)
                if not mask_path or not mask_path.is_file():
                    raise AppError("This scene image has no matching saved mask. Upload a transparent PNG mask, or choose a scene image with a saved mask.")
                _validate_mask_png(mask_path, scene_base)
            if job.get("estimate_reliable") is False:
                if approved_max_usd is None:
                    raise ConfirmationRequired(estimate, "Image pricing or this project's estimate is uncertain; approval is required before regenerating this image.")
            if approved_max_usd is not None and not estimate_within_approved_amount(estimate, approved_max_usd):
                raise AppError(f"Estimated image cost ${estimate:.2f} exceeds the approved maximum ${approved_max_usd:.2f}.")
            self._check_project_spending_limit(job, estimate)
            clip = self._load_plan(job).clips[clip_index - 1]
            output_dir = self._job_path(job_id).parent / "outputs" / f"clip-{clip_index:02d}" / "starting-image" / f"version-{len(clip_record.get('starting_image_history', [])) + 1}"
            if job.get("images_approved_at") and not job.get("approved_starting_image_assets"):
                job["approved_starting_image_assets"] = [
                    record.get("current_starting_image_asset") for record in job.get("clips", [])
                    if record.get("current_starting_image_asset")
                ]
            job.update(status="regenerating_starting_image", stage="regenerating_starting_image",
                       message=f"Regenerating the starting image for clip {clip_index} (~${estimate:.2f}).",
                       active_clip=clip_index, active_stage="regenerating_starting_image", error=None,
                       images_approved_at=None, updated_at=_now())
            job["pending_starting_image_regen"] = {
                "clip_index": clip_index, "instructions": instructions.strip(), "output_dir": str(output_dir),
                "masked_edit": masked_edit, "base_source": base_source if masked_edit else None,
                "mask_path": str(mask_path) if mask_path else None,
            }
            self._save(job)
        self._spawn(self._regenerate_starting_image, job_id, clip, instructions.strip(), output_dir,
                    False, masked_edit, scene_base, mask_path)
        return self.get_job(job_id)

    def _resume_starting_image_regen(self, job_id: str) -> None:
        job = self._read(job_id)
        pending = job.get("pending_starting_image_regen") or {}
        plan = self._load_plan(job)
        index = int(pending["clip_index"])
        if pending.get("clip_regeneration"):
            self._prepare_regenerated_clip_image(job_id, index, pending.get("instructions", ""), Path(pending["output_dir"]), resume=True)
            return
        base = self._resolve_scene_base(job_id, job, pending["base_source"]) if pending.get("masked_edit") else None
        self._regenerate_starting_image(
            job_id, plan.clips[index - 1], pending.get("instructions", ""), Path(pending["output_dir"]),
            resume=True, masked_edit=bool(pending.get("masked_edit")), scene_base=base,
            mask_path=Path(pending["mask_path"]) if pending.get("mask_path") else None,
        )

    def _regenerate_starting_image(
        self, job_id: str, clip: PlannedClip, instructions: str, output_dir: Path,
        resume: bool = False, masked_edit: bool = False, scene_base: Path | None = None,
        mask_path: Path | None = None,
    ) -> None:
        job = self._read(job_id)
        plan = self._load_plan(job)
        planned_clips = {candidate.index: candidate for candidate in plan.clips}
        current_images = {}
        for record in job["clips"]:
            asset_id = record.get("current_starting_image_asset")
            if asset_id and record["index"] != clip.index:
                current_images[record["index"]] = Path(job["assets"][asset_id]["path"])
        in_store_opening = plan.hook in {"in_store_reaction", "wet_works"} and clip.index == 1
        prompt_clip = clip.model_copy(update={
            "starting_image_prompt": starting_image_revision_prompt(plan, clip, instructions),
        })
        single_plan = plan.model_copy(update={"clips": [prompt_clip]})
        old_asset = job["clips"][clip.index - 1]["current_starting_image_asset"]
        old_path = Path(job["assets"][old_asset]["path"])
        target_group = clip.continuity_group.strip()
        continuity = []
        continuity_roles = []
        character_references = self._character_identity_reference_paths(job, plan, clip)
        has_scene_environment_reference = bool(clip.environment_references)
        if target_group and not has_scene_environment_reference:
            peers = [
                (abs(index - clip.index), index, path)
                for index, path in current_images.items()
                if path.is_file()
                and index in planned_clips
                and planned_clips[index].continuity_group.strip() == target_group
            ]
            # Without an explicit environment image, one nearby same-group image
            # can supply the scene. A dedicated environment input takes priority.
            if peers:
                continuity.append(min(peers)[2])
                continuity_roles.append(
                    "same-scene identity, wardrobe, and environment only; ignore facial expression and pose"
                    if in_store_opening else
                    "same-scene environment only; preserve setting and lighting, ignore people, faces, wardrobe, and pose"
                )
        # Do not feed a stale target pose back when an explicit environment
        # reference is available; character and product inputs carry those roles.
        if (old_path.is_file() and old_path not in continuity and not in_store_opening
                and not has_scene_environment_reference):
            continuity.append(old_path)
            continuity_roles.append("pose and held objects only; ignore background")
        client = None
        try:
            client = self.client_factory()
            if masked_edit:
                if not scene_base or not mask_path:
                    raise AppError("The saved masked-edit inputs are incomplete; no request was submitted.")
                image_path = run_masked_image_edit(
                    single_plan, prompt_clip, scene_base, mask_path,
                    [Path(p) for p in plan.product_references], character_references,
                    output_dir, client, scene_prompt=prompt_clip.starting_image_prompt, resume=resume,
                )
                images = [image_path]
            else:
                image_estimate = estimate_multiclip_generation(single_plan, analysis_usd=Decimal("0"))
                images = run_multiclip_generation(single_plan, [Path(p) for p in plan.product_references], continuity,
                    output_dir, client, pricing_verified=True, estimate_reliable=True,
                    approved_max_usd=image_estimate.total_usd, resume=resume, images_only=True,
                    continuity_reference_roles=continuity_roles,
                    character_references=character_references)
        except Exception as exc:
            self._collect_charges(job_id, output_dir, f"clip_{clip.index}_starting_image_v{len(job['clips'][clip.index - 1].get('starting_image_history', [])) + 1}")
            try:
                manifest = json.loads((output_dir / "generation-usage.json").read_text())
                image_task = next((task for task in manifest.get("tasks", []) if task.get("kind") == "starting_image"), {})
            except (OSError, ValueError, json.JSONDecodeError):
                image_task = {}
            if image_task.get("status") == "failed":
                self._update(job_id, lambda data: data.update(status="awaiting_image_review", stage="awaiting_image_review",
                    message=f"Clip {clip.index} image regeneration failed. The previous image remains available for approval; choose another adjustment only if you want a new paid request.",
                    error=str(exc), pending_starting_image_regen=None, active_clip=None, active_stage=None))
            else:
                self._set_failure(job_id, str(exc))
            return
        finally:
            if client is not None:
                client.close()
        self._collect_charges(job_id, output_dir, f"clip_{clip.index}_starting_image_v{len(job['clips'][clip.index - 1].get('starting_image_history', [])) + 1}")
        self._save_starting_image_assets(job_id, plan.model_copy(update={"clips": [clip]}), images)
        self._update(job_id, lambda data: data.update(status="awaiting_image_review", stage="awaiting_image_review",
            message=f"Clip {clip.index} image updated. Review the set or approve it to generate videos.",
            active_clip=None, active_stage=None, error=None, pending_starting_image_regen=None))

    def _generate_videos(self, job_id: str) -> None:
        job = self._read(job_id)
        plan = self._load_plan(job)
        output_dir = self._job_path(job_id).parent / "outputs" / "initial"
        image_paths = {clip["index"]: Path(job["assets"][clip["current_starting_image_asset"]]["path"])
                       for clip in job["clips"] if clip.get("current_starting_image_asset")}
        def progress(event: dict) -> None:
            self._update(job_id, lambda data: data.update(status=event["stage"], stage=event["stage"],
                provider_state=event.get("provider_state"), active_clip=event.get("clip_index"),
                active_stage=event["stage"], message=f"Generating approved images · clip {event.get('clip_index')} of {event.get('clip_count')}"))
        client = None
        try:
            client = self.client_factory()
            outputs = run_multiclip_generation(plan, [Path(path) for path in plan.product_references],
                [Path(path) for path in plan.continuity_references], output_dir, client,
                approved_max_usd=(Decimal(job["video_generation_authorization"]["approved_max_usd"])
                                  if job.get("video_generation_authorization", {}).get("operation") == self._video_operation(job)[0]
                                  else None),
                pricing_verified=True, estimate_reliable=job["estimate_reliable"],
                analysis_usd=Decimal(job["charges"]["analysis_usd"]), on_progress=progress,
                resume=True, starting_images=image_paths)
        except Exception as exc:
            self._collect_charges(job_id, output_dir, "initial_generation")
            self._harvest_completed_clips(job_id, plan, output_dir)
            self._set_failure(job_id, str(exc))
            return
        finally:
            if client is not None:
                client.close()
        self._collect_charges(job_id, output_dir, "initial_generation")
        for clip, video_path in zip(plan.clips, outputs):
            current = self._read(job_id)
            if current["clips"][clip.index - 1]["versions"]:
                continue
            image = current["clips"][clip.index - 1]["current_starting_image_asset"]
            image_path = Path(current["assets"][image]["path"])
            self._save_clip_version(job_id, clip, image_path, video_path, self._clip_cost(output_dir / "generation-usage.json", clip.index), "")
        self._update(job_id, lambda data: data.update(status="completed", stage="completed", provider_state="success",
            message="All planned clips are complete.", error=None, active_clip=None, active_stage=None))

    def _harvest_completed_clips(self, job_id: str, plan: MultiClipPlan, output_dir: Path) -> None:
        """Expose finished clip files even when a later clip fails."""
        for clip in plan.clips:
            videos = [path for path in output_dir.glob(f"clip-{clip.index:02d}-*.mp4") if not path.name.endswith("-provider.mp4")]
            images = list(output_dir.glob(f"clip-{clip.index:02d}-*-starting.png"))
            if not videos or not images:
                continue
            job = self._read(job_id)
            record = job["clips"][clip.index - 1]
            if record["versions"]:
                continue
            actual = self._clip_cost(output_dir / "generation-usage.json", clip.index)
            asset_id = record.get("current_starting_image_asset")
            approved_image = Path(job["assets"][asset_id]["path"]) if asset_id else images[0]
            self._save_clip_version(job_id, clip, approved_image, videos[0], actual, "")

    @staticmethod
    def _clip_cost(manifest_path: Path, clip_index: int) -> Decimal | None:
        try:
            manifest = json.loads(manifest_path.read_text())
            credits = sum((_credit_value(task) for task in manifest["tasks"] if task["clip_index"] == clip_index), Decimal("0"))
            return credits * KIE_USD_PER_CREDIT
        except (OSError, KeyError, ValueError, json.JSONDecodeError):
            return None

    def _record_openai_cost(self, job_id: str, kind: str, cost: Decimal) -> None:
        def record(job: dict) -> None:
            charges = job.setdefault("charges", {})
            charges["analysis_usd"] = f"{Decimal(charges.get('analysis_usd', '0')) + cost:.6f}"
            charges["total_usd"] = f"{Decimal(charges['analysis_usd']) + Decimal(charges.get('kie_usd_equivalent', '0')):.6f}"
            charges.setdefault("entries", []).append({
                "operation_id": uuid.uuid4().hex, "recorded_at": _now(), "kind": kind,
                "usd_equivalent": f"{cost:.6f}", "source": "OpenAI token usage",
            })
            estimate = job.get("estimate")
            if estimate:
                estimate["analysis_usd"] = charges["analysis_usd"]
                estimate["total_usd"] = f"{Decimal(estimate.get('images_usd', '0')) + Decimal(estimate.get('video_usd', '0')) + Decimal(charges['analysis_usd']):.4f}"
        self._update(job_id, record)

    def _record_unknown_openai_operation(self, job_id: str, kind: str) -> None:
        labels = {
            "reference_analysis": "Reference analysis",
            "creative_director": "Creative Director interview",
            "plan_revision": "Creative plan refinement",
        }
        def record(job: dict) -> None:
            job.setdefault("charges", {}).setdefault("entries", []).append({
                "operation_id": uuid.uuid4().hex,
                "recorded_at": _now(),
                "kind": kind,
                "usd_equivalent": None,
                "source": "request failed; provider usage unavailable",
                "operation": labels.get(kind, kind.replace("_", " ").title()),
            })
            job["estimate_reliable"] = False
        try:
            self._update(job_id, record)
        except JobNotFound:
            return

    def _check_project_spending_limit(self, job: dict, additional_estimate: Decimal) -> None:
        spending = self._project_spending(job)
        unresolved = [record for record in spending["records"]
                      if record.get("status") == "unknown_usage"
                      and not re.search(r"Kie task (?:submitted|processing|queued|pending);", str(record.get("basis", "")))]
        if unresolved:
            raise AppError("This project has provider charges with unknown amounts, so its spending ceiling cannot be verified safely.")
        limit = Decimal(str(job.get("project_spending_limit_usd", DEFAULT_PROJECT_SPENDING_LIMIT_USD)))
        recorded = Decimal(spending["recorded_usd_equivalent"])
        projected = recorded + reserve_estimate_usd(additional_estimate)
        if projected > limit:
            raise AppError(
                f"This operation would bring project spending to about ${projected:.4f}, "
                f"above the ${limit:.2f} project limit. Recorded usage: ${recorded:.6f}."
            )

    def _project_spending(self, job: dict) -> dict:
        """Build a cumulative usage ledger from persisted OpenAI and Kie evidence."""
        records: list[dict] = []
        recorded_total = Decimal("0")
        unknown_count = 0
        charges = job.get("charges", {})
        entries = charges.get("entries", [])

        def add_record(record: dict) -> None:
            nonlocal recorded_total, unknown_count
            amount = record.get("usd_equivalent")
            if amount is None:
                unknown_count += 1
            else:
                recorded_total += Decimal(str(amount))
            records.append(record)

        openai_operations = {
            "reference_analysis": "Reference analysis",
            "creative_director": "Creative Director interview",
            "plan_revision": "Creative plan refinement",
        }
        for index, entry in enumerate(entries):
            if entry.get("kind") == "kie_generation":
                continue
            source = str(entry.get("source", ""))
            known = source == "OpenAI token usage" and entry.get("usd_equivalent") is not None
            estimated = entry.get("usd_equivalent")
            add_record({
                "id": entry.get("operation_id") or f"openai-legacy-{index}",
                "provider": "OpenAI",
                "operation": openai_operations.get(entry.get("kind"), entry.get("kind", "OpenAI operation").replace("_", " ").title()),
                "recorded_at": entry.get("recorded_at"),
                "status": "usage_recorded" if known else "unknown_usage",
                "usd_equivalent": str(estimated) if known else None,
                "estimated_usd_equivalent": None if known or estimated is None else str(estimated),
                "basis": "Token usage priced at configured rates; not a verified invoice charge" if known else "Usage data unavailable; amount is unknown",
            })

        usage_by_label: dict[str, tuple[Path, dict]] = {}
        usage_records: list[dict] = []
        usage_task_counts: dict[str, set[str]] = {}
        outputs = self._job_path(job["id"]).parent / "outputs"
        if outputs.is_dir():
            for manifest_path in sorted(outputs.glob("**/generation-usage.json")):
                relative = manifest_path.parent.relative_to(outputs).parts
                if relative == ("initial",):
                    label = "initial_generation"
                elif len(relative) == 2:
                    match = re.fullmatch(r"clip-(\d+)", relative[0])
                    version = re.fullmatch(r"version-(\d+)", relative[1])
                    if not match or not version:
                        continue
                    label = f"clip_{int(match.group(1))}_v{int(version.group(1))}"
                elif len(relative) == 3 and relative[1] == "starting-image":
                    match = re.fullmatch(r"clip-(\d+)", relative[0])
                    version = re.fullmatch(r"version-(\d+)", relative[2])
                    if not match or not version:
                        continue
                    label = f"clip_{int(match.group(1))}_starting_image_v{int(version.group(1))}"
                else:
                    continue
                try:
                    manifest = json.loads(manifest_path.read_text())
                except (OSError, ValueError, json.JSONDecodeError):
                    usage_by_label[label] = (manifest_path, {"_read_error": True, "tasks": []})
                    continue
                usage_by_label[label] = (manifest_path, manifest)
                for task_index, task in enumerate(manifest.get("tasks", [])):
                    usage_records.append({"label": label, "path": manifest_path, "task_index": task_index, "task": task})

        task_records_by_label: dict[str, int] = {}
        seen_task_ids: dict[str, dict] = {}
        for item in usage_records:
            label, manifest_path, task_index, task = item["label"], item["path"], item["task_index"], item["task"]
            task_id = task.get("task_id") or task.get("taskId")
            dedupe_key = str(task_id) if task_id else f"{label}:task-{task_index}"
            task_records_by_label.setdefault(label, 0)
            usage_task_counts.setdefault(label, set()).add(str(task_id) if task_id else f"{label}:task-{task_index}")
            task_records_by_label[label] = len(usage_task_counts[label])
            raw_credits = task.get("credits_consumed", task.get("creditsConsumed", task.get("consumeCredits", task.get("consumedCredits"))))
            try:
                credits = Decimal(str(raw_credits)) if raw_credits is not None else None
            except Exception:
                credits = None
            task_status = str(task.get("status", "unknown")).lower()
            clip_index = task.get("clip_index")
            kind = task.get("kind", "generation")
            operation = f"{'Starting image' if kind == 'starting_image' else 'Video generation'}"
            if clip_index is not None:
                operation += f" · Clip {clip_index}"
            if label != "initial_generation":
                operation += f" · v{label.rsplit('_v', 1)[-1]} regeneration" if "_v" in label else " · regeneration"
            known = credits is not None
            record = {
                "id": f"kie:{dedupe_key}", "task_id": str(task_id) if task_id else None,
                "provider": "Kie.ai", "operation": operation,
                "recorded_at": task.get("completed_at") or task.get("recorded_at") or task.get("submitted_at"),
                "status": "usage_recorded_failed_task" if known and task_status in {"failed", "fail"} else "usage_recorded" if known else "unknown_usage",
                "usd_equivalent": str(credits * KIE_USD_PER_CREDIT) if known else None,
                "credits": str(credits) if known else None,
                "estimated_usd_equivalent": None,
                "basis": "Provider-reported credits converted at configured rate; not a verified invoice charge" if known else f"Kie task {task_status}; usage not reported, possible charge unknown",
            }
            previous = seen_task_ids.get(dedupe_key)
            if previous:
                if previous.get("duplicate_conflict"):
                    continue
                if previous.get("credits") is not None and record.get("credits") is not None and previous.get("credits") != record.get("credits"):
                    if previous.get("usd_equivalent") is not None:
                        recorded_total -= Decimal(previous["usd_equivalent"])
                        unknown_count += 1
                    previous.update(status="unknown_usage", usd_equivalent=None, credits=None,
                                    basis="Duplicate task ID has conflicting usage records; charge amount is unknown",
                                    duplicate_conflict=True)
                elif previous.get("credits") is None and record.get("credits") is not None:
                    unknown_count -= 1
                    recorded_total += Decimal(record["usd_equivalent"])
                    previous.update(record)
                continue
            seen_task_ids[dedupe_key] = record
            add_record(record)

        kie_entries = {str(entry.get("label")): entry for entry in entries if entry.get("kind") == "kie_generation" and entry.get("label")}
        for label, (manifest_path, manifest) in usage_by_label.items():
            if manifest.get("_read_error"):
                add_record({"id": f"kie-manifest:{label}", "provider": "Kie.ai", "operation": f"Generation usage · {label}", "recorded_at": None, "status": "unknown_usage", "usd_equivalent": None, "estimated_usd_equivalent": None, "basis": "Usage manifest could not be read; charges are unknown"})
                continue

        # Older projects may have charge aggregates but no retained task manifest.
        for label, entry in kie_entries.items():
            manifest_data = usage_by_label.get(label)
            if manifest_data and (manifest_data[1].get("_read_error") or task_records_by_label.get(label, 0)):
                continue
            raw_credits = entry.get("credits")
            try:
                credits = Decimal(str(raw_credits)) if raw_credits is not None else None
            except Exception:
                credits = None
            version_label = re.fullmatch(r"clip_(\d+)_v(\d+)", label)
            operation = (
                f"Clip {version_label.group(1)} regeneration · aggregate history"
                if version_label else f"{label.replace('_', ' ').title()} · aggregate history"
            )
            add_record({
                "id": f"kie-aggregate:{label}", "provider": "Kie.ai", "operation": operation,
                "recorded_at": entry.get("recorded_at"),
                "status": "usage_recorded" if credits is not None else "unknown_usage",
                "usd_equivalent": str(credits * KIE_USD_PER_CREDIT) if credits is not None else None,
                "credits": str(credits) if credits is not None else None,
                "estimated_usd_equivalent": None,
                "basis": "Historical provider-reported credit aggregate; per-task detail unavailable" if credits is not None else "Historical charge aggregate lacks reported credits",
            })
            if not entry.get("complete", False):
                add_record({
                    "id": f"kie-aggregate-unknown:{label}", "provider": "Kie.ai",
                    "operation": f"Unreported usage · {label}", "recorded_at": None,
                    "status": "unknown_usage", "usd_equivalent": None,
                    "estimated_usd_equivalent": None,
                    "basis": "Historical aggregate is incomplete; additional charges cannot be determined",
                })

        records.sort(key=lambda record: (record.get("recorded_at") is None, record.get("recorded_at") or ""))
        return {
            "recorded_usd_equivalent": f"{recorded_total:.6f}",
            "unknown_charge_count": unknown_count,
            "basis": "Recorded provider usage equivalents; actual invoices may differ.",
            "records": records,
        }

    def _collect_charges(self, job_id: str, output_dir: Path, label: str) -> None:
        try:
            manifest = json.loads((output_dir / "generation-usage.json").read_text())
        except (OSError, ValueError, json.JSONDecodeError):
            return

        def change(job: dict) -> None:
            entries = job["charges"]["entries"]
            previous = next((entry for entry in entries if entry.get("label") == label), None)
            unique_tasks = {}
            for index, task in enumerate(manifest.get("tasks", [])):
                task_id = task.get("task_id") or task.get("taskId") or f"{label}:task-{index}"
                unique_tasks.setdefault(str(task_id), task)
            chargeable_tasks = [task for task in unique_tasks.values() if _reported_credits(task) is not None]
            credits = sum((_reported_credits(task) for task in chargeable_tasks), Decimal("0"))
            kie_usd = credits * KIE_USD_PER_CREDIT
            previous_credits = Decimal(previous.get("credits", "0")) if previous else Decimal("0")
            delta_credits = credits - previous_credits
            job["charges"]["kie_credits"] = f"{Decimal(job['charges']['kie_credits']) + delta_credits:.1f}"
            job["charges"]["kie_usd_equivalent"] = f"{Decimal(job['charges']['kie_usd_equivalent']) + delta_credits * KIE_USD_PER_CREDIT:.4f}"
            total = Decimal(job["charges"]["analysis_usd"]) + Decimal(job["charges"]["kie_usd_equivalent"])
            job["charges"]["total_usd"] = f"{total:.6f}"
            entry = {
                "kind": "kie_generation",
                "label": label,
                "credits": str(credits),
                "usd_equivalent": f"{kie_usd:.4f}",
                "complete": manifest.get("all_task_charges_reported", False),
                "task_count": len(chargeable_tasks),
                "recorded_at": previous.get("recorded_at") if previous else _now(),
            }
            if previous:
                previous.update(entry)
            else:
                entries.append(entry)
        self._update(job_id, change)

    def _save_clip_version(
        self,
        job_id: str,
        clip: PlannedClip,
        image_path: Path,
        video_path: Path,
        actual_cost: Decimal | None,
        revision: str,
    ) -> None:
        def change(job: dict) -> None:
            target = next(item for item in job["clips"] if item["index"] == clip.index)
            current_id = target.get("current_starting_image_asset")
            if current_id and Path(job["assets"][current_id]["path"]).resolve() == Path(image_path).resolve():
                image_id = current_id
            else:
                image_id = self._add_asset(job, image_path, f"clip-{clip.index:02d}-starting.png", "image", "starting_image")
                history = target.setdefault("starting_image_history", [])
                history.append({"version": len(history) + 1, "asset_id": image_id, "created_at": _now()})
                target["current_starting_image_asset"] = image_id
            audio_enabled = clip.audio_enabled
            # run_multiclip_generation validates stream presence and relative
            # timestamps before returning; preserve that verified result here.
            audio_present = audio_enabled
            video_role = "video_with_native_audio" if audio_present else "silent_video"
            video_id = self._add_asset(job, video_path, f"clip-{clip.index:02d}.mp4", "video", video_role)
            target["status"] = "completed"
            target["versions"].append({
                "version": len(target["versions"]) + 1,
                "starting_image_asset": image_id,
                "video_asset": video_id,
                "duration_seconds": clip.duration_seconds,
                "revision": revision,
                "audio_mode": clip.audio_mode,
                "audio_enabled": audio_enabled,
                "audio_direction": clip.audio_direction,
                "audio_present": audio_present,
                "actual_kie_usd_equivalent": f"{actual_cost:.4f}" if actual_cost is not None else None,
            })
        self._update(job_id, change)

    def start_regeneration(
        self,
        job_id: str,
        clip_index: int,
        instructions: str,
        approved_max_usd: Decimal | None = None,
    ) -> dict:
        if not instructions.strip():
            raise AppError("Add a change request for this clip before regenerating it.")
        with self.lock:
            job = self._read(job_id)
            if job["status"] not in {"completed", "failed"}:
                raise JobConflict("Wait for the current job to finish before regenerating a clip.")
            if not 1 <= clip_index <= len(job["clips"]):
                raise AppError("That clip does not exist in this job.")
            if not job["clips"][clip_index - 1]["versions"]:
                raise JobConflict("This clip has no completed version to revise yet.")
            for record in job["clips"]:
                if record.get("current_starting_image_asset") or not record.get("versions"):
                    continue
                legacy_asset = record["versions"][-1].get("starting_image_asset")
                if legacy_asset:
                    record["current_starting_image_asset"] = legacy_asset
                    record.setdefault("starting_image_history", [
                        {"version": version.get("version", index + 1), "asset_id": version.get("starting_image_asset"), "created_at": None}
                        for index, version in enumerate(record["versions"]) if version.get("starting_image_asset")
                    ])
            clip = self._load_plan(job).clips[clip_index - 1]
            single_plan = MultiClipPlan.model_validate_json(
                json.dumps({
                    "summary": job["analysis"]["summary"],
                    "environment": "the scene and lighting specified in the user's creative instructions",
                    "instructions": job["instructions"],
                    "product_facts": "Preserve visible product details from the supplied references.",
                    "product_references": [],
                    "continuity_references": [],
                    "hook": self._load_plan(job).hook,
                    "clips": [clip.model_dump(mode="json")],
                })
            )
            estimate = estimate_multiclip_generation(single_plan, analysis_usd=Decimal("0"))
            if generation_approval_required(estimate) and approved_max_usd is None:
                raise ConfirmationRequired(estimate.total_usd, "This regeneration needs spending confirmation.")
            if approved_max_usd is not None and not estimate_within_approved_amount(estimate.total_usd, approved_max_usd):
                raise AppError(f"Estimated total ${estimate.total_usd:.4f} exceeds the approved maximum ${approved_max_usd:.4f}.")
            self._check_project_spending_limit(job, estimate.image_usd + estimate.video_usd)
            version_number = len(job["clips"][clip_index - 1]["versions"]) + 1
            output_dir = self._job_path(job_id).parent / "outputs" / f"clip-{clip_index:02d}" / f"version-{version_number}"
            current_image_id = job["clips"][clip_index - 1].get("current_starting_image_asset")
            current_image_path = job.get("assets", {}).get(current_image_id, {}).get("path")
            reuse_approved_image = bool(
                clip.audio_enabled and job.get("images_approved_at") and current_image_path
                and Path(current_image_path).is_file()
            )
            job["status"] = "generating_clips" if reuse_approved_image else "regenerating_starting_image"
            job["stage"] = job["status"]
            job["message"] = (
                f"Reusing Clip {clip_index}'s approved starting image and preparing its revised video."
                if reuse_approved_image else
                f"Preparing a revised starting image for clip {clip_index} (~${IMAGE_1K_USD:.2f})."
            )
            job["active_clip"] = clip_index
            job["active_stage"] = "regenerating_starting_image"
            job["images_approved_at"] = None
            job["pending_clip_regeneration"] = {
                "clip_index": clip_index, "instructions": instructions.strip(),
                "output_dir": str(output_dir), "version": version_number, "image_ready": False,
                "approved_max_usd": str(approved_max_usd) if approved_max_usd is not None else None,
            }
            if reuse_approved_image:
                job["pending_clip_regeneration"].update(image_ready=True, reuse_approved_starting_image=True)
            job["updated_at"] = _now()
            self._save(job)
        if reuse_approved_image:
            self._spawn(self._generate_regenerated_clip, job_id)
        else:
            self._spawn(self._prepare_regenerated_clip_image, job_id, clip_index, instructions.strip(), output_dir, False)
        return self.get_job(job_id)

    def _prepare_regenerated_clip_image(self, job_id: str, clip_index: int, instructions: str, output_dir: Path, resume: bool) -> None:
        job = self._read(job_id)
        full_plan = self._load_plan(job)
        original = full_plan.clips[clip_index - 1]
        clip = original.model_copy(update={
            "starting_image_prompt": original.starting_image_prompt + f"\nClip-specific revision: {instructions}",
            "video_prompt": original.video_prompt + f"\nClip-specific revision: {instructions}",
        })
        old_version = job["clips"][clip_index - 1]["versions"][-1]
        old_image = Path(job["assets"][old_version["starting_image_asset"]]["path"])
        planned_clips = {candidate.index: candidate for candidate in full_plan.clips}
        target_group = original.continuity_group.strip()
        image_continuity = []
        continuity_roles = []
        character_references = self._character_identity_reference_paths(job, full_plan, original)
        if target_group:
            peers = []
            for record in job["clips"]:
                peer_index = record["index"]
                asset_id = record.get("current_starting_image_asset")
                if peer_index == clip_index or peer_index not in planned_clips or not asset_id:
                    continue
                if planned_clips[peer_index].continuity_group.strip() != target_group:
                    continue
                peer_path = Path(job["assets"][asset_id]["path"])
                if peer_path.is_file():
                    peers.append((abs(peer_index - clip_index), peer_index, peer_path))
            if peers:
                image_continuity.append(min(peers)[2])
                continuity_roles.append("same-scene faces/clothes/location; use background")
        if old_image.is_file() and old_image not in image_continuity:
            image_continuity.append(old_image)
            continuity_roles.append("pose and held objects only; ignore background")
        single_plan = full_plan.model_copy(update={"clips": [clip]})
        client = None
        try:
            client = self.client_factory()
            image_estimate = estimate_multiclip_generation(single_plan, analysis_usd=Decimal("0"))
            images = run_multiclip_generation(single_plan, [Path(path) for path in full_plan.product_references],
                image_continuity, output_dir, client, pricing_verified=True, estimate_reliable=True,
                approved_max_usd=image_estimate.total_usd, resume=resume, images_only=True,
                continuity_reference_roles=continuity_roles,
                character_references=character_references)
        except Exception as exc:
            label = f"clip_{clip_index}_v{job['pending_clip_regeneration']['version']}"
            self._collect_charges(job_id, output_dir, label)
            self._set_failure(job_id, str(exc))
            return
        finally:
            if client is not None:
                client.close()
        label = f"clip_{clip_index}_v{job['pending_clip_regeneration']['version']}"
        self._collect_charges(job_id, output_dir, label)
        self._save_starting_image_assets(job_id, single_plan, images)
        self._update(job_id, lambda data: data.update(status="awaiting_image_review", stage="awaiting_image_review",
            message=f"Review clip {clip_index}'s revised image. Approve it to generate this clip version.",
            active_clip=None, active_stage=None, pending_clip_regeneration={**data["pending_clip_regeneration"], "image_ready": True},
            pending_starting_image_regen=None, error=None))

    def _generate_regenerated_clip(self, job_id: str) -> None:
        job = self._read(job_id)
        pending = job.get("pending_clip_regeneration")
        if not pending or not pending.get("image_ready"):
            self._set_failure(job_id, "The revised starting image is missing; resume image preparation before video generation.")
            return
        full_plan = self._load_plan(job)
        index = int(pending["clip_index"])
        original = full_plan.clips[index - 1]
        instructions = pending["instructions"]
        clip = original.model_copy(update={
            "starting_image_prompt": original.starting_image_prompt + f"\nClip-specific revision: {instructions}",
            "video_prompt": original.video_prompt + f"\nClip-specific revision: {instructions}",
        })
        single_plan = full_plan.model_copy(update={"clips": [clip]})
        asset_id = job["clips"][index - 1]["current_starting_image_asset"]
        image_path = Path(job["assets"][asset_id]["path"])
        output_dir = Path(pending["output_dir"])
        client = None
        try:
            client = self.client_factory()
            regeneration_estimate = estimate_multiclip_generation(single_plan, analysis_usd=Decimal("0"))
            outputs = run_multiclip_generation(single_plan, [Path(path) for path in full_plan.product_references],
                [], output_dir, client, pricing_verified=True, estimate_reliable=True,
                approved_max_usd=Decimal(pending["approved_max_usd"]) if pending.get("approved_max_usd") else regeneration_estimate.total_usd,
                resume=True, starting_images={index: image_path})
        except Exception as exc:
            label = f"clip_{index}_v{pending['version']}"
            self._collect_charges(job_id, output_dir, label)
            self._set_failure(job_id, str(exc))
            return
        finally:
            if client is not None:
                client.close()
        label = f"clip_{index}_v{pending['version']}"
        self._collect_charges(job_id, output_dir, label)
        self._save_clip_version(job_id, clip, image_path, outputs[0], self._clip_cost(output_dir / "generation-usage.json", index), instructions)
        self._update(job_id, lambda data: data.update(status="completed", stage="completed", provider_state="success",
            message=f"Clip {index} regenerated. Previous versions are preserved.", error=None,
            active_clip=None, active_stage=None, pending_clip_regeneration=None))

    def _regenerate_clip(self, job_id: str, clip_index: int, instructions: str) -> None:
        job = self._read(job_id)
        full_plan = self._load_plan(job)
        original = full_plan.clips[clip_index - 1]
        clip = original.model_copy(update={
            "starting_image_prompt": original.starting_image_prompt + f"\nClip-specific revision: {instructions}",
            "video_prompt": original.video_prompt + f"\nClip-specific revision: {instructions}",
        })
        old_version = job["clips"][clip_index - 1]["versions"][-1]
        continuity_path = Path(job["assets"][old_version["starting_image_asset"]]["path"])
        single_plan = full_plan.model_copy(update={"clips": [clip]})
        version_number = len(job["clips"][clip_index - 1]["versions"]) + 1
        output_dir = self._job_path(job_id).parent / "outputs" / f"clip-{clip_index:02d}" / f"version-{version_number}"
        client = None
        try:
            client = self.client_factory()
            outputs = run_multiclip_generation(
                single_plan,
                [Path(path) for path in full_plan.product_references],
                [continuity_path],
                output_dir,
                client,
                pricing_verified=True,
                estimate_reliable=True,
                on_progress=lambda event: self._update(job_id, lambda data: data.update(
                    stage="regenerating", status="regenerating",
                    provider_state=event.get("provider_state"),
                    message=f"Regenerating clip {clip_index}: {event['stage']}.",
                )),
            )
        except Exception as exc:
            self._collect_charges(job_id, output_dir, f"clip_{clip_index}_v{version_number}")
            self._set_failure(job_id, str(exc))
            return
        finally:
            if client is not None:
                client.close()
        self._collect_charges(job_id, output_dir, f"clip_{clip_index}_v{version_number}")
        image = next(output_dir.glob(f"clip-{clip_index:02d}-*-starting.png"))
        actual = self._clip_cost(output_dir / "generation-usage.json", clip_index)
        self._save_clip_version(job_id, clip, image, outputs[0], actual, instructions)
        self._update(job_id, lambda data: data.update(
            status="completed", stage="completed", provider_state="success",
            message=f"Clip {clip_index} regenerated. Previous versions are preserved.",
            active_clip=None, error=None,
        ))

    def _public_job(self, job: dict) -> dict:
        assets = {
            asset_id: {
                "url": f"/api/jobs/{job['id']}/assets/{asset_id}",
                "filename": asset["filename"],
                "media_type": asset["media_type"],
                "role": asset["role"],
            }
            for asset_id, asset in job["assets"].items()
        }
        clips = []
        for clip in job["clips"]:
            try:
                planned = self._load_plan(job).clips[int(clip["index"]) - 1]
                audio_fields = {
                    "audio_mode": planned.audio_mode,
                    "audio_requested": planned.audio_requested,
                    "audio_enabled": planned.audio_enabled,
                    "audio_direction": planned.audio_direction,
                }
            except (KeyError, IndexError, ValueError):
                audio_fields = {
                    "audio_mode": clip.get("audio_mode", "auto"),
                    "audio_requested": clip.get("audio_requested"),
                    "audio_enabled": clip.get("audio_enabled", bool(clip.get("preserve_audio", False))),
                    "audio_direction": clip.get("audio_direction", "No native audio selected for this legacy clip."),
                }
            versions = []
            for version in clip["versions"]:
                versions.append({
                    "version": version["version"],
                    "starting_image": assets.get(version["starting_image_asset"]),
                    "video": assets.get(version["video_asset"]),
                    "duration_seconds": version["duration_seconds"],
                    "revision": version["revision"],
                    "actual_kie_usd_equivalent": version["actual_kie_usd_equivalent"],
                    "audio_mode": version.get("audio_mode", "auto"),
                    "audio_enabled": version.get("audio_enabled", bool(version.get("audio_present", False))),
                    "audio_present": version.get("audio_present"),
                })
            current_asset_id = clip.get("current_starting_image_asset")
            if not current_asset_id and clip.get("versions"):
                current_asset_id = clip["versions"][-1].get("starting_image_asset")
            current_image = assets.get(current_asset_id)
            history = [{"version": item.get("version"), "created_at": item.get("created_at"),
                        "asset": assets.get(item.get("asset_id"))}
                       for item in clip.get("starting_image_history", [])]
            if not history:
                history = [{"version": version.get("version"), "created_at": None,
                            "asset": assets.get(version.get("starting_image_asset"))}
                           for version in clip.get("versions", [])]
            clips.append({key: value for key, value in clip.items() if key not in {"versions", "current_starting_image_asset", "starting_image_history"}} |
                         audio_fields | {"versions": versions, "current_starting_image": current_image, "starting_image_history": history})
        asset_records = job.get("assets", {})
        reference_id = job.get("reference_asset_id")
        if not reference_id and job.get("reference_video"):
            reference_id = next((asset_id for asset_id, item in asset_records.items()
                                 if item.get("role") == "reference_video" and Path(item.get("path", "")).resolve() == Path(job["reference_video"]).resolve()), None)
        product_ids = job.get("product_asset_ids")
        if product_ids is None:
            product_paths = {Path(path).resolve() for path in job.get("product_references", [])}
            product_ids = [asset_id for asset_id, item in asset_records.items()
                           if item.get("role") == "product_reference" and Path(item.get("path", "")).resolve() in product_paths]
        input_ids = ([reference_id] if reference_id else []) + list(product_ids)
        inputs = [{"asset_id": asset_id, **assets[asset_id]} for asset_id in input_ids if asset_id in assets]
        reference_filename = next(
            (asset["filename"] for asset in job["assets"].values() if asset["role"] == "reference_video"),
            Path(job["reference_video"]).name if job.get("reference_video") else None,
        )
        title = str(job.get("title") or "").strip() or _default_project_title(
            job.get("mode", "reference"), job.get("instructions", ""), reference_filename,
        )
        analysis = job["analysis"]
        video_info = dict(job["reference_video_info"])
        video_info.pop("path", None)
        return {
            "id": job["id"], "title": title, "mode": job.get("mode", "reference"), "hook": job.get("hook", "recommend"),
            "requested_clip_count": job.get("requested_clip_count", 3),
            "clip_count_selection": job.get("clip_count_selection", "fixed"),
            "seconds_per_clip": job.get("seconds_per_clip", 5),
            "duration_selection": job.get("duration_selection", "fixed"),
            "status": job["status"], "stage": job["stage"],
            "message": job["message"], "provider_state": job.get("provider_state"),
            "created_at": job["created_at"], "updated_at": job["updated_at"],
            "reference_video_info": video_info,
            "inputs": inputs, "instructions": job["instructions"],
            "analysis": analysis,
            "estimate": job["estimate"], "estimate_reliable": job["estimate_reliable"],
            "spending": self._project_spending(job),
            "project_spending_limit_usd": str(job.get("project_spending_limit_usd", DEFAULT_PROJECT_SPENDING_LIMIT_USD)),
            "director_questions": job.get("director_questions", []),
            "interview_answers": job.get("interview_answers", []),
            "can_resume": job.get("status") in {"failed", "interrupted"} and bool(job.get("plan_json")) and bool(job.get("clips")),
            "resume_operation": self._resume_image_retry_info(job),
            "images_approved": bool(job.get("images_approved_at")),
            "review_images_before_video": bool(job.get("review_images_before_video", job.get("status") == "awaiting_image_review")),
            "clip_regeneration_pending": bool(job.get("pending_clip_regeneration")),
            "scene_base_images": self.scene_base_images(job["id"], job),
            "clips": clips, "charges": job["charges"], "error": job.get("error"),
        }

    def _resume_image_retry_info(self, job: dict) -> dict | None:
        operation = job.get("resume_operation")
        if (operation and operation.get("kind") == "starting_image_retry"
                and job.get("status") not in {"failed", "interrupted"}):
            return operation
        if (job.get("status") not in {"failed", "interrupted"}
                or not job.get("review_images_before_video")
                or job.get("active_stage") != "preparing_images"):
            return None
        output_dir = self._job_path(job["id"]).parent / "outputs" / "initial"
        try:
            manifest = json.loads((output_dir / "generation-usage.json").read_text())
        except (OSError, ValueError, json.JSONDecodeError):
            return None
        failed_records = [task for task in manifest.get("tasks", [])
                          if task.get("kind") == "starting_image" and task.get("status") in {"failed", "fail"}
                          and task.get("clip_index") is not None]
        failed = {int(task["clip_index"]) for task in failed_records}
        missing = [index for index in sorted(failed)
                   if not any(output_dir.glob(f"clip-{index:02d}-*-starting.png"))]
        if not missing or any(index not in {clip.get("index") for clip in job.get("clips", [])} for index in missing):
            return None
        estimate = IMAGE_1K_USD * len(missing)
        clips_text = ", ".join(f"Clip {index}" for index in missing)
        label = f"Retry {clips_text} image" if len(missing) == 1 else f"Retry {len(missing)} missing images"
        latest_failure = next((task for task in reversed(failed_records) if int(task["clip_index"]) in missing), {})
        task_id = latest_failure.get("task_id")
        raw_credits = latest_failure.get("credits_consumed")
        usage_detail = f"The failed task reported {raw_credits} Kie credits; invoice usage is not verified." if raw_credits is not None else "The failed task's charge is unreported."
        provider_failure = latest_failure.get("provider_failure") or {}
        provider_code = provider_failure.get("code")
        provider_message = provider_failure.get("message")
        provider_detail = "; ".join(
            part for part in (
                f"code {provider_code}" if provider_code not in (None, "") else "",
                str(provider_message) if provider_message else "",
            ) if part
        )
        state_detail = latest_failure.get("provider_state", "unknown")
        failure_detail = (
            f"Failed task {task_id} ({state_detail} state"
            f"{'; ' + provider_detail if provider_detail else ''}). "
            if task_id else "The previous provider task failed. "
        )
        return {
            "kind": "starting_image_retry", "clip_indices": missing,
            "estimated_usd": f"{estimate:.4f}", "label": label,
            "description": f"{failure_detail}{usage_detail} Retry only the missing starting image for {clips_text}; saved images for other clips will be reused. Estimated new task cost: ${estimate:.2f}.",
        }

    def get_job(self, job_id: str) -> dict:
        public = self._public_job(self._read(job_id))
        public["storage_bytes"] = self._project_size(self._job_path(job_id).parent)
        return public

    def rename_project(self, job_id: str, title: str) -> dict:
        if not isinstance(title, str):
            raise AppError("Enter a project name.")
        normalized = re.sub(r"\s+", " ", title).strip()
        if not normalized:
            raise AppError("Project name cannot be empty.")
        if creative_instruction_length(normalized) > MAX_PROJECT_TITLE_LENGTH:
            raise AppError(f"Project names must be {MAX_PROJECT_TITLE_LENGTH} characters or fewer.")
        updated = self._update(job_id, lambda job: job.update(title=normalized))
        return self._public_job(updated)

    def _verified_jobs_root(self) -> Path:
        data_root = self.data_dir.resolve(strict=True)
        if self.jobs_dir.is_symlink():
            raise ProjectStorageError("Project storage is not a safe local directory; nothing was deleted.")
        jobs_root = self.jobs_dir.resolve(strict=True)
        if jobs_root.parent != data_root or not jobs_root.is_dir():
            raise ProjectStorageError("Project storage is not a safe local directory; nothing was deleted.")
        return jobs_root

    @staticmethod
    def _project_files(root: Path, *, reject_symlinks: bool = False) -> list[Path]:
        files = []
        for current, directories, names in os.walk(root, followlinks=False):
            current_path = Path(current)
            for name in list(directories):
                path = current_path / name
                if path.is_symlink():
                    if reject_symlinks:
                        raise ProjectStorageError("Project contains a symbolic link; nothing was deleted.")
                    continue
            for name in names:
                path = current_path / name
                if path.is_symlink():
                    if reject_symlinks:
                        raise ProjectStorageError("Project contains a symbolic link; nothing was deleted.")
                    continue
                if path.is_file():
                    try:
                        resolved = path.resolve(strict=True)
                    except FileNotFoundError:
                        # Atomic JSON writes can replace a temporary file between
                        # the directory scan and this check. It is no longer a
                        # managed file to count or delete.
                        continue
                    if root not in resolved.parents:
                        raise ProjectStorageError("Project files are outside the project storage directory; nothing was deleted.")
                    files.append(path)
        return files

    def _referenced_paths(self, exclude_ids: set[str] | None = None) -> set[Path]:
        references: set[Path] = set()
        exclude_ids = exclude_ids or set()
        for record in self.jobs_dir.glob("*/job.json"):
            if (record.parent.name in exclude_ids or record.parent.is_symlink() or record.is_symlink()
                    or record.parent.resolve().parent != self.jobs_dir.resolve()):
                continue
            try:
                job = json.loads(record.read_text())
            except (OSError, ValueError, json.JSONDecodeError):
                raise ProjectStorageError("Another project's files could not be verified; nothing was deleted.")
            candidates = [asset.get("path") for asset in job.get("assets", {}).values() if isinstance(asset, dict)]
            candidates += [job.get("reference_video"), *job.get("product_references", [])]
            for value in candidates:
                if isinstance(value, str) and value:
                    try:
                        references.add(Path(value).resolve())
                    except OSError:
                        continue
        return references

    @staticmethod
    def _job_asset_paths(job: dict) -> set[Path]:
        values = [asset.get("path") for asset in job.get("assets", {}).values() if isinstance(asset, dict)]
        values += [job.get("reference_video"), *job.get("product_references", [])]
        paths = set()
        for value in values:
            if isinstance(value, str) and value:
                try:
                    paths.add(Path(value).resolve())
                except OSError:
                    continue
        return paths

    def _project_size(self, root: Path, protected: set[Path] | None = None) -> int:
        protected = protected or set()
        try:
            jobs_root = self._verified_jobs_root()
            if root.is_symlink() or root.resolve(strict=True).parent != jobs_root:
                return 0
        except (OSError, ProjectStorageError):
            return 0
        total = 0
        inodes = set()
        for path in self._project_files(root):
            try:
                resolved = path.resolve(strict=True)
                if resolved in protected:
                    continue
                info = path.stat(follow_symlinks=False)
            except OSError:
                # Atomic job saves can replace a temporary file while usage is measured.
                continue
            inode = (info.st_dev, info.st_ino)
            if inode not in inodes:
                total += info.st_size
                inodes.add(inode)
        return total

    def _delete_projects_locked(self, job_ids: list[str]) -> dict:
        ids = list(dict.fromkeys(job_ids))
        if not ids:
            raise AppError("Select at least one project to delete.")
        jobs_root = self._verified_jobs_root()
        projects = []
        selected_asset_paths: set[Path] = set()
        for job_id in ids:
            job_path = self._job_path(job_id)
            root = job_path.parent
            if root.is_symlink() or not root.is_dir():
                raise JobNotFound("Project not found")
            resolved_root = root.resolve(strict=True)
            if resolved_root.parent != jobs_root or job_path.is_symlink() or not job_path.is_file() or job_path.resolve(strict=True).parent != resolved_root:
                raise ProjectStorageError("Project data is outside its project directory; nothing was deleted.")
            try:
                job = json.loads(job_path.read_text())
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                raise ProjectStorageError("Project data could not be verified; nothing was deleted.") from exc
            if job.get("status") in DELETION_BLOCKED_STATUSES:
                raise JobConflict("Cannot delete a project while work is active. Wait for it to finish first.")
            selected_asset_paths.update(self._job_asset_paths(job))
            files = self._project_files(resolved_root, reject_symlinks=True)
            projects.append((job_id, resolved_root, files))

        protected = self._referenced_paths(set(ids))
        recovered = sum(self._project_size(root, protected) for _, root, _ in projects)
        project_roots = [root.resolve() for _, root, _ in projects]
        reclaimable_shared = []
        for path in selected_asset_paths - protected:
            try:
                if jobs_root not in path.parents or any(root == path or root in path.parents for root in project_roots):
                    continue
                if path.is_symlink() or not path.is_file():
                    continue
                reclaimable_shared.append(path)
            except OSError:
                continue
        recovered += sum(path.stat(follow_symlinks=False).st_size for path in reclaimable_shared)
        try:
            for _, root, files in projects:
                shared = [path for path in files if path.resolve() in protected]
                if not shared:
                    shutil.rmtree(root)
                    continue
                # Preserve shared files in place; remove only this project's own files.
                for path in files:
                    if path.resolve() not in protected:
                        path.unlink()
                for current, directories, _ in os.walk(root, topdown=False, followlinks=False):
                    for directory in directories:
                        try:
                            (Path(current) / directory).rmdir()
                        except OSError:
                            pass
                try:
                    root.rmdir()
                except OSError:
                    # A shared file remains at its original managed path for its consumers.
                    pass
            for path in reclaimable_shared:
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
        except OSError as exc:
            raise ProjectStorageError("Project files could not be deleted. Check local storage and try again.") from exc
        return {"deleted_count": len(ids), "deleted_ids": ids, "recovered_bytes": recovered}

    def delete_project(self, job_id: str) -> dict:
        """Delete one inactive project while preserving any assets another project references."""
        self._job_path(job_id)
        with self.lock:
            result = self._delete_projects_locked([job_id])
        return {"id": job_id, "deleted": True, "recovered_bytes": result["recovered_bytes"]}

    def delete_projects(self, job_ids: list[str]) -> dict:
        if not isinstance(job_ids, list) or any(not isinstance(job_id, str) for job_id in job_ids):
            raise AppError("Choose valid projects to delete.")
        with self.lock:
            return self._delete_projects_locked(job_ids)

    def estimate_deletion(self, job_ids: list[str]) -> dict:
        if not isinstance(job_ids, list) or not job_ids or any(not isinstance(job_id, str) for job_id in job_ids):
            raise AppError("Select at least one project.")
        with self.lock:
            jobs_root = self._verified_jobs_root()
            roots = []
            for job_id in dict.fromkeys(job_ids):
                job_path = self._job_path(job_id)
                root = job_path.parent
                if root.is_symlink() or not root.is_dir() or root.resolve(strict=True).parent != jobs_root:
                    raise JobNotFound("Project not found")
                roots.append(root.resolve(strict=True))
            protected = self._referenced_paths(set(job_ids))
            own_bytes = sum(self._project_size(root, protected) for root in roots)
            selected_paths = set()
            for job_id in dict.fromkeys(job_ids):
                try:
                    selected_paths.update(self._job_asset_paths(json.loads(self._job_path(job_id).read_text())))
                except (OSError, ValueError, json.JSONDecodeError):
                    raise ProjectStorageError("Project data could not be verified; nothing was deleted.")
            additional = 0
            for path in selected_paths - protected:
                try:
                    if (jobs_root not in path.parents or any(root == path or root in path.parents for root in roots)
                            or path.is_symlink() or not path.is_file()):
                        continue
                    additional += path.stat(follow_symlinks=False).st_size
                except OSError:
                    continue
            return {"project_count": len(roots), "recoverable_bytes": own_bytes + additional}

    def storage_summary(self) -> dict:
        jobs = self.list_jobs()
        total = 0
        seen = set()
        try:
            self._verified_jobs_root()
            for path in self.jobs_dir.rglob("*"):
                if path.is_symlink() or not path.is_file():
                    continue
                stat = path.stat(follow_symlinks=False)
                inode = (stat.st_dev, stat.st_ino)
                if inode not in seen:
                    total += stat.st_size
                    seen.add(inode)
        except OSError:
            pass
        return {"project_count": len(jobs), "total_bytes": total}

    def cleanup_temporary_files(self, now: float | None = None) -> dict:
        """Remove only old, unreferenced transient files under managed job storage."""
        threshold = (now if now is not None else __import__("time").time()) - 48 * 60 * 60
        removed = 0
        recovered = 0
        suffixes = {".tmp", ".part", ".partial", ".download"}
        try:
            self._verified_jobs_root()
            active = {path.parent.name for path in self.jobs_dir.glob("*/job.json")
                      if not path.is_symlink() and self._status_for_path(path) in DELETION_BLOCKED_STATUSES}
            references = self._referenced_paths()
            for root in self.jobs_dir.iterdir():
                if root.is_symlink() or not root.is_dir() or root.name in active:
                    continue
                for current, directories, names in os.walk(root, followlinks=False):
                    directories[:] = [name for name in directories if not (Path(current) / name).is_symlink()]
                    for name in names:
                        path = Path(current) / name
                        if path.suffix.lower() not in suffixes or path.is_symlink() or not path.is_file():
                            continue
                        try:
                            if path.stat(follow_symlinks=False).st_mtime > threshold or path.resolve() in references:
                                continue
                            size = path.stat(follow_symlinks=False).st_size
                            path.unlink()
                            removed += 1
                            recovered += size
                        except OSError:
                            continue
        except (OSError, ProjectStorageError):
            return {"removed_count": removed, "recovered_bytes": recovered}
        return {"removed_count": removed, "recovered_bytes": recovered}

    def _status_for_path(self, path: Path) -> str | None:
        try:
            return json.loads(path.read_text()).get("status")
        except (OSError, ValueError, json.JSONDecodeError):
            return None

    def list_jobs(self) -> list[dict]:
        jobs = []
        for path in self.jobs_dir.glob("*/job.json"):
            try:
                public = self._public_job(json.loads(path.read_text()))
                public["storage_bytes"] = self._project_size(path.parent)
                jobs.append(public)
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                continue
        return sorted(jobs, key=lambda item: item["updated_at"], reverse=True)

    def get_asset(self, job_id: str, asset_id: str) -> tuple[Path, str, str]:
        job = self._read(job_id)
        asset = job["assets"].get(asset_id)
        if not asset:
            raise JobNotFound("Asset not found")
        path = Path(asset["path"]).resolve()
        job_root = self._job_path(job_id).parent.resolve()
        if job_root not in path.parents or not path.is_file():
            raise JobNotFound("Asset not found")
        return path, asset["media_type"], asset["filename"]


def _parse_multipart(content_type: str, body: bytes) -> tuple[dict[str, str], dict[str, list[tuple[str, bytes]]]]:
    header = f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("ascii")
    message = BytesParser(policy=policy.default).parsebytes(header + body)
    if not message.is_multipart():
        raise AppError("Expected a multipart form upload.")
    fields: dict[str, str] = {}
    files: dict[str, list[tuple[str, bytes]]] = {}
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if not name:
            continue
        value = part.get_payload(decode=True) or b""
        filename = part.get_filename()
        if filename:
            files.setdefault(name, []).append((Path(filename).name, value))
        else:
            charset = part.get_content_charset() or "utf-8"
            fields[name] = value.decode(charset, errors="replace")
    return fields, files


def make_handler(service: LocalJobService) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "EcommerceVideoLocal/1.0"

        def log_message(self, format: str, *args) -> None:
            return

        def _json(self, status: int, payload: dict) -> None:
            encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(encoded)

        def _check_local_origin(self) -> bool:
            host = self.headers.get("Host", "").split(":", 1)[0].strip("[]").lower()
            if host not in {"127.0.0.1", "localhost"}:
                self._json(HTTPStatus.FORBIDDEN, {"error": "This app only accepts requests from localhost."})
                return False
            origin = self.headers.get("Origin")
            if origin:
                origin_url = urlsplit(origin)
                origin_host = origin_url.hostname
                origin_port = origin_url.port or (443 if origin_url.scheme == "https" else 80)
                if origin_host not in {"127.0.0.1", "localhost"} or origin_port != self.server.server_port:
                    self._json(HTTPStatus.FORBIDDEN, {"error": "Cross-origin requests are not allowed."})
                    return False
            return True

        def _body(self) -> bytes:
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                raise AppError("The request size is invalid.") from None
            if length < 0 or length > MAX_REQUEST_BYTES:
                raise AppError("The request exceeds the combined upload limit.")
            return self.rfile.read(length)

        def _error(self, exc: Exception) -> None:
            if isinstance(exc, ConfirmationRequired):
                self._json(exc.status, {"error": str(exc), "confirmation_required": True, "estimate_usd": f"{exc.estimate_usd:.4f}"})
            elif isinstance(exc, AppError):
                self._json(exc.status, {"error": str(exc)})
            elif isinstance(exc, (ValueError, TypeError, json.JSONDecodeError)):
                self._json(HTTPStatus.BAD_REQUEST, {"error": "Check the submitted values and try again."})
            else:
                self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "The local application could not complete that request."})

        def do_GET(self) -> None:
            if not self._check_local_origin():
                return
            parsed = urlsplit(self.path)
            path = unquote(parsed.path)
            if path == "/":
                self._static("index.html")
                return
            if path.startswith("/static/"):
                self._static(path.removeprefix("/static/"))
                return
            pieces = path.strip("/").split("/")
            try:
                if pieces == ["api", "health"]:
                    self._json(HTTPStatus.OK, {"status": "ok", "host": "localhost"})
                elif pieces == ["api", "jobs"]:
                    self._json(HTTPStatus.OK, {"jobs": service.list_jobs(), "storage": service.storage_summary()})
                elif len(pieces) == 3 and pieces[:2] == ["api", "jobs"]:
                    self._json(HTTPStatus.OK, service.get_job(pieces[2]))
                elif len(pieces) == 5 and pieces[:2] == ["api", "jobs"] and pieces[3] == "assets":
                    download = parse_qs(parsed.query).get("download") == ["1"]
                    self._asset(pieces[2], pieces[4], download=download)
                elif len(pieces) == 6 and pieces[:2] == ["api", "jobs"] and pieces[3] == "scene-references":
                    file_path, media_type, _ = service.get_scene_reference(pieces[2], int(pieces[4]), int(pieces[5]))
                    data = file_path.read_bytes()
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", media_type)
                    self.send_header("Content-Length", str(len(data)))
                    self.send_header("X-Content-Type-Options", "nosniff")
                    self.send_header("Cache-Control", "private, no-store")
                    self.end_headers()
                    self.wfile.write(data)
                else:
                    self._json(HTTPStatus.NOT_FOUND, {"error": "Not found."})
            except Exception as exc:
                self._error(exc)

        def do_POST(self) -> None:
            if not self._check_local_origin():
                return
            parsed = urlsplit(self.path)
            path = unquote(parsed.path).strip("/").split("/")
            try:
                body = self._body()
                if path == ["api", "jobs"]:
                    fields, files = _parse_multipart(self.headers.get("Content-Type", ""), body)
                    if fields.get("mode") == "original":
                        review_images = fields.get("review_images_before_video", "true").lower() in {"1", "true", "on"}
                        result = service.create_original_job(fields.get("instructions", ""), fields.get("hook", "recommend"), int(fields.get("clip_count", "3")), int(fields.get("seconds_per_clip", "5")), files.get("product_images", []), review_images)
                        self._json(HTTPStatus.ACCEPTED, result)
                        return
                    videos = files.get("reference_video", [])
                    if len(videos) != 1:
                        raise AppError("Choose one reference video.")
                    result = service.create_job(
                        videos[0][0], videos[0][1], files.get("product_images", []),
                        fields.get("instructions", ""),
                        int(fields.get("clip_count", "3")),
                        int(fields.get("seconds_per_clip", "5")),
                        fields.get("review_images_before_video", "true").lower() in {"1", "true", "on"},
                        fields.get("hook", "recommend"),
                    )
                    self._json(HTTPStatus.ACCEPTED, result)
                    return
                if path == ["api", "drafts"]:
                    fields, files = _parse_multipart(self.headers.get("Content-Type", ""), body)
                    self._json(HTTPStatus.OK, service.save_draft(fields, files))
                    return
                if len(path) == 4 and path[:2] == ["api", "jobs"] and path[3] == "draft":
                    fields, files = _parse_multipart(self.headers.get("Content-Type", ""), body)
                    self._json(HTTPStatus.OK, service.save_draft(fields, files, path[2]))
                    return
                if (len(path) == 7 and path[:2] == ["api", "jobs"] and path[3] == "clips"
                        and path[5:] == ["starting-image", "regenerate"]
                        and self.headers.get("Content-Type", "").lower().startswith("multipart/form-data")):
                    fields, files = _parse_multipart(self.headers.get("Content-Type", ""), body)
                    approved = Decimal(fields["approved_max_usd"]) if fields.get("approved_max_usd") else None
                    masks = files.get("mask_image", [])
                    if len(masks) > 1:
                        raise AppError("Upload one edit mask at a time.")
                    result = service.regenerate_starting_image(
                        path[2], int(path[4]), fields.get("instructions", ""), approved,
                        masked_edit=fields.get("edit_mode") == "masked",
                        base_source=fields.get("base_source"), mask_upload=masks[0] if masks else None,
                    )
                    self._json(HTTPStatus.ACCEPTED, result)
                    return
                payload = json.loads(body or b"{}")
                if len(path) == 4 and path[:2] == ["api", "jobs"] and path[3] == "rename":
                    self._json(HTTPStatus.OK, service.rename_project(path[2], payload.get("title")))
                    return
                if len(path) == 4 and path[:2] == ["api", "jobs"] and path[3] == "delete":
                    self._json(HTTPStatus.OK, service.delete_project(path[2]))
                    return
                if path == ["api", "jobs", "delete-bulk"]:
                    self._json(HTTPStatus.OK, service.delete_projects(payload.get("job_ids")))
                    return
                if path == ["api", "jobs", "delete-estimate"]:
                    self._json(HTTPStatus.OK, service.estimate_deletion(payload.get("job_ids")))
                    return
                if len(path) == 4 and path[:2] == ["api", "jobs"] and path[3] == "review-images":
                    enabled = payload.get("enabled")
                    self._json(HTTPStatus.OK, service.set_review_images_before_video(path[2], enabled))
                    return
                if len(path) == 4 and path[:2] == ["api", "jobs"] and path[3] == "answer":
                    self._json(HTTPStatus.ACCEPTED, service.answer_original_interview(path[2], payload.get("answer", "")))
                    return
                if len(path) == 4 and path[:2] == ["api", "jobs"] and path[3] == "refine":
                    self._json(HTTPStatus.OK, service.refine_plan(path[2], payload.get("instructions", "")))
                    return
                if len(path) == 4 and path[:2] == ["api", "jobs"] and path[3] == "generate":
                    approved = Decimal(str(payload["approved_max_usd"])) if payload.get("approved_max_usd") is not None else None
                    review_images = payload.get("review_images")
                    self._json(HTTPStatus.ACCEPTED, service.start_generation(path[2], approved, resume=bool(payload.get("resume")), review_images=bool(review_images) if review_images is not None else None))
                    return
                if len(path) == 4 and path[:2] == ["api", "jobs"] and path[3] == "analyze":
                    self._json(HTTPStatus.ACCEPTED, service.analyze_draft(path[2]))
                    return
                if len(path) == 4 and path[:2] == ["api", "jobs"] and path[3] == "approve-images":
                    approved = Decimal(str(payload["approved_max_usd"])) if payload.get("approved_max_usd") is not None else None
                    self._json(HTTPStatus.ACCEPTED, service.approve_starting_images(path[2], approved))
                    return
                if len(path) == 6 and path[:2] == ["api", "jobs"] and path[3] == "clips" and path[5] == "regenerate":
                    approved = Decimal(str(payload["approved_max_usd"])) if payload.get("approved_max_usd") is not None else None
                    self._json(HTTPStatus.ACCEPTED, service.start_regeneration(path[2], int(path[4]), payload.get("instructions", ""), approved))
                    return
                if len(path) == 6 and path[:2] == ["api", "jobs"] and path[3] == "clips" and path[5] == "audio":
                    self._json(HTTPStatus.OK, service.set_clip_audio_mode(path[2], int(path[4]), payload.get("mode")))
                    return
                if len(path) == 7 and path[:2] == ["api", "jobs"] and path[3] == "clips" and path[5:] == ["starting-image", "regenerate"]:
                    approved = Decimal(str(payload["approved_max_usd"])) if payload.get("approved_max_usd") is not None else None
                    self._json(HTTPStatus.ACCEPTED, service.regenerate_starting_image(path[2], int(path[4]), payload.get("instructions", ""), approved))
                    return
                self._json(HTTPStatus.NOT_FOUND, {"error": "Not found."})
            except Exception as exc:
                self._error(exc)

        def _static(self, name: str) -> None:
            if name not in {"index.html", "app.css", "app.js"}:
                self._json(HTTPStatus.NOT_FOUND, {"error": "Not found."})
                return
            path = WEB_ROOT / name
            if not path.is_file():
                self._json(HTTPStatus.NOT_FOUND, {"error": "The frontend asset is missing."})
                return
            data = path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", mimetypes.guess_type(name)[0] or "application/octet-stream")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' blob:; media-src 'self' blob:; style-src 'self'; script-src 'self';")
            self.end_headers()
            self.wfile.write(data)

        def _asset(self, job_id: str, asset_id: str, *, download: bool = False) -> None:
            file_path, media_type, filename = service.get_asset(job_id, asset_id)
            size = file_path.stat().st_size
            start, end, partial = 0, size - 1, False
            range_header = self.headers.get("Range")
            if range_header and range_header.startswith("bytes="):
                try:
                    first, last = range_header.removeprefix("bytes=").split("-", 1)
                    if not first:
                        suffix_length = int(last)
                        if suffix_length <= 0 or size == 0:
                            raise ValueError
                        start = max(0, size - suffix_length)
                        end = size - 1
                    else:
                        start = int(first)
                        end = int(last) if last else size - 1
                    if start > end or start >= size:
                        raise ValueError
                    end = min(end, size - 1)
                    partial = True
                except ValueError:
                    self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                    return
            length = max(0, end - start + 1)
            self.send_response(HTTPStatus.PARTIAL_CONTENT if partial else HTTPStatus.OK)
            self.send_header("Content-Type", media_type)
            self.send_header("Content-Length", str(length))
            self.send_header("Accept-Ranges", "bytes")
            safe_filename = Path(filename).name
            fallback_filename = re.sub(r"[^A-Za-z0-9._-]", "_", safe_filename) or "download"
            disposition = "attachment" if download else "inline"
            self.send_header(
                "Content-Disposition",
                f"{disposition}; filename=\"{fallback_filename}\"; filename*=UTF-8''{quote(safe_filename, safe='')}",
            )
            self.send_header("X-Content-Type-Options", "nosniff")
            if partial:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            with file_path.open("rb") as file:
                file.seek(start)
                self.wfile.write(file.read(length))

    return Handler


def create_server(port: int = 8765, service: LocalJobService | None = None) -> ThreadingHTTPServer:
    app = service or LocalJobService()
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(app))
    server.daemon_threads = True
    return server
