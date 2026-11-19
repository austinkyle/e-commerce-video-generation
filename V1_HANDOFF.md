# V1 Closeout Handoff

## Status

**V1 Complete — Development Paused / Ready for Production Use**

This is a local production tool. “Ready for Production Use” records the approved V1 closeout status; it does not mean the app is hosted or that model outputs are guaranteed to be commercially accurate. The final RovePup test validated Original Concept generation, product-reference continuity, physical demonstrations, and the four-clip workflow. No additional generation was performed for closeout.

## Capabilities and verification

- Reference Recreation and Original Concept workflows, including adaptive interviews saved with project history.
- Structured, editable multi-clip creative plans, hook selection, starting-image review, selective image and clip regeneration, and interrupted-job recovery.
- Kie image/video generation and OpenAI analysis/director integration, with the existing cost estimate, approval, task tracking, and spending history safeguards.
- Persistent Recent Work projects with rename/delete actions and locally saved assets.
- Full automated test suite: **102 passed**. Provider calls in tests are mocked. Python compilation and local application/API smoke checks are part of closeout verification.

## Known limitations

- AI motion and physical interactions can contain occasional artifacts.
- Exact branding, especially fine logo text and placement, may vary.
- First-generation output varies; image references improve continuity but do not guarantee exact identity.
- Reported provider usage and local cost equivalents are not invoice verification. Unknown usage is recorded as unknown.
- Local files are not backed up automatically. The app is not hosted and does not provide multi-user authentication.

## Start locally

```sh
uv sync
cp .env.example .env
# Add provider keys to .env only when paid provider operations are needed.
uv run ecom-video serve
```

Open `http://127.0.0.1:8765`. The macOS launcher `Run E-Commerce Video Generator.command` starts the same app. `APP_DATA_DIR` can override the default data root.

## Data locations

- Projects, job histories, provider task records, spending entries, inputs, generated starting images, and video versions: `data/jobs/<job-id>/` (with per-job outputs under `outputs/`).
- Local analysis frames and standalone CLI artifacts: `data/` paths chosen by the command.
- `.env`: local provider keys and optional local settings; ignored by Git.
- `data/`: ignored by Git to keep private project data and generated media out of source commits. Preserve it during routine source maintenance and back it up separately.

## Future backlog

No V2 work is authorized or underway. If separately approved later, candidate work is exact-brand rendering, more systematic realism/continuity evaluation, and user-controlled project export/backup. Do not treat this list as active implementation scope.
