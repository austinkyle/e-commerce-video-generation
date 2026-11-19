# E-Commerce Video Generator

**V1 Complete — Development Paused / Ready for Production Use**

A local-first creative production tool for turning product ideas or reference videos into separate, silent product clips. V1 supports reference recreation and original concepts, persistent Creative Director interviews, editable plans, starting-image review, per-clip regeneration, project spending history, and local project management.

## V1 capabilities

- **Reference Recreation:** analyze selected moments from a reference video, then plan and generate distinct product shots.
- **Original Concept:** describe a new idea without a reference video or product photo. The Creative Director asks adaptive follow-up questions, records answers, and builds a structured multi-clip plan.
- **Creative planning:** select or request a hook, refine the saved plan in natural language, and review clip roles, actions, framing, continuity, and cost estimates.
- **Image and video generation:** create starting images with Kie Sunburst and silent vertical clips with the configured Kie video model. User product photos and selected continuity images can guide image generation.
- **Starting-image review:** optionally pause after image preparation to inspect all planned images, regenerate only a selected image, and approve the set before videos are submitted. When review is off, image generation proceeds directly into video generation.
- **Recovery and history:** resume interrupted jobs using saved assets and provider task IDs; regenerate individual clips while retaining earlier versions.
- **Project management:** recent projects persist locally and can be renamed or deleted. Deletion removes only the selected project and its local files.
- **Spending visibility:** show a separate future-generation estimate and cumulative project spending records. Provider usage equivalents are distinguished from estimates and unknown charges; they are not represented as verified invoices.

The interface uses the Abyssal Teal dark design. Video is silent by default; there is no automatic retry or unrequested batch generation.

## Architecture

- Python local HTTP application serves the browser interface and JSON API.
- `src/ecom_video/workflow.py` contains planning, prompt compilation, estimates, task submission, recovery, and output handling shared by the browser and CLI.
- `src/ecom_video/providers/openai.py` provides reference analysis and the adaptive Original Concept director; `src/ecom_video/providers/kie.py` provides image and video task integration.
- Pydantic models validate analysis and saved creative plans. FFmpeg and FFprobe inspect media and prepare silent output files.
- Jobs, project histories, source inputs, generated images and videos, and usage records are stored as local files. No application database or cloud hosting is required.

## Requirements and setup

- macOS, Python 3.12 or newer, [`uv`](https://docs.astral.sh/uv/), FFmpeg, and FFprobe.
- OpenAI and Kie API keys are only needed for provider-backed analysis, interviews, images, or videos. Starting the local app and opening saved projects does not require paid generation.

From the project directory:

```sh
uv sync
cp .env.example .env
```

Add `OPENAI_API_KEY` and `KIE_API_KEY` to the local `.env` only when needed. Keep real keys there; `.env` and `.env.*` are excluded from Git while `.env.example` contains blank values. `APP_DATA_DIR` optionally changes the data root; it defaults to `data/` relative to the project directory.

## Start the browser app

```sh
uv run ecom-video serve
```

The app listens on `http://127.0.0.1:8765`. Keep the terminal open while using it and press Ctrl-C to stop. On macOS, double-click `Run E-Commerce Video Generator.command` for the same startup. The app runs locally; it is not a hosted production service.

## CLI

Inspect a video locally and extract frames:

```sh
uv run ecom-video inspect path/to/reference.mp4 --frames 12
```

Estimate a job without submitting requests:

```sh
uv run ecom-video estimate --seconds 12 --images 3
```

The CLI can also analyze supplied frames, prepare a multi-clip plan, and generate clips from a saved plan. Use `--dry-run` to inspect a generation estimate. The same spending policy applies to the browser and CLI: an explicitly started job with reliable, verified pricing at or below **$1.00 USD** can proceed; higher or uncertain estimates require an approved maximum. Never use approval to authorize an automatic retry or an unrequested batch.

## Project data and generated assets

By default, each saved project is in `data/jobs/<job-id>/`. Its `job.json` stores the project name, creative plan, interview and operation history, asset references, and spending records. Project inputs and outputs live in that job's directory; generated starting images, video versions, and provider usage manifests are stored under its `outputs/` tree. Local inspection frames and standalone CLI outputs may also be saved elsewhere under `data/` as selected by their commands.

The complete `data/` directory is ignored by Git. Keep it intact to preserve projects, generated media, task records, and spending history. Back it up separately if needed; deleting a project in the app permanently removes that project's local files.

## Verification

Run the automated suite and Python syntax compilation:

```sh
uv run pytest -q
uv run python -m compileall -q src tests
```

The suite uses mocked provider responses and does not make paid generation calls.

## Known limitations

- Image and video models can introduce occasional motion, anatomy, object-contact, or scene-continuity artifacts.
- Exact logo text, spelling, placement, and other fine branding details can vary between generations, even when references and prompts are supplied.
- First-generation results vary; reference images improve continuity but do not guarantee identical products, people, or environments.
- A generated starting image used as a continuity anchor is a useful visual reference, not a deterministic identity lock. Review images when continuity matters and regenerate selectively if needed.
- Provider-reported credits and locally calculated equivalents may not match an invoice. Missing provider usage remains unknown rather than being assumed to be zero.
- Clip lengths and visual transformations remain constrained by provider model capabilities; review generated clips before using them as commercial product claims.
- The app runs on the local machine and does not provide a remote service, shared-user authentication, or automated cloud backup.

## Future development backlog

No V2 work has started. Possible future work, requiring separate authorization, includes stronger exact-brand rendering, better evaluation of motion and physical realism, continuity measurements across generated clips, and a user-controlled project export and backup flow.

See [V1_HANDOFF.md](V1_HANDOFF.md) for the concise closeout summary and readiness record.
