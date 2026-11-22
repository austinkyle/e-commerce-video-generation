# Generation Knowledge and Storage Map

This is an index to existing sources of truth, not a second playbook or a runtime retrieval system. New projects use the application code and configuration below; generation does not search old project folders.

## Reusable application knowledge

- `src/ecom_video/creative_playbook.py` — the only reusable creative playbook: frame-zero versus later action, continuity, physical realism, In-Store Reaction direction, and concise failure/correction patterns. Put approved, generalizable creative improvements here; keep product and scene facts in their project.
- `src/ecom_video/models.py` — creative-format directions for In-Store Reaction, 3-2-1 Reveal, Gifting, Versus, and custom concepts.
- `src/ecom_video/providers/openai.py` — reference analysis and Creative Director instructions; applies the playbook and preserves explicit creator direction, target-cast identity, shot timing, and audio decisions.
- `src/ecom_video/workflow.py` — shared prompt construction, preflight checks, product/character/environment reference roles, image and video submission, masked edits, task receipts, recovery, and output handling. Browser and CLI generation share this path.
- `src/ecom_video/providers/kie.py`, `src/ecom_video/config.py`, and `src/ecom_video/costs.py` — provider requests, credentials/configuration, estimates, and approval thresholds. Preserve the existing per-operation approval and per-project spending policies when changing generation behavior.
- `tests/` — regression coverage for formats, reference roles, identity continuity, audio selection and synchronization validation, masked editing, recovery, spending, and project cleanup.

## Editing knowledge

- `ECOM-VIDEO-EDITING-STANDARDS.md` — reusable story, trim, typography, audio, and export-review standards.
- `video-editing/reusable-assets/` — reusable Resolve title assets and their specifications.
- `video-editing/<project>/` — that edit's project file, source media, notes, renders, and reports. Do not put product-specific edit decisions in the shared standards unless they generalize.

DaVinci Resolve is part of the local editing workflow and reusable assets; the generator does not currently provide an integrated Resolve automation feature.

## Project records and deletion boundary

- `data/jobs/<job-id>/` — each project's source references, extracted frames, plan, generated assets, versions, approvals, provider task records, and costs. Usage manifests retain submitted prompts and reference roles where recorded by that workflow version. Preserve this data for reproduction and variations.
- `data/` is ignored by Git and is not a source of reusable system knowledge. The app's project-deletion and temporary-cleanup paths are scoped to managed job storage. Deletion checks references retained by other projects; cleanup removes only old, unreferenced temporary files. Tests exercise isolated temporary directories, symlinks, active-job refusal, and shared-asset retention.
- Keep credentials in the local `.env`; `.env.example` contains placeholders.

Do not copy full prompts or generation histories into the playbook. Record only generalizable, evidence-supported improvements. Mark uncertain techniques as experimental and include the conditions where they apply.

## Evidence and limitations

The app has regression-tested mock-provider paths for new product concepts and references, including a camping-lantern In-Store Reaction example, character continuity across settings, product-reference roles, native audio, and optional masked editing. These validate request construction without provider calls; they do not guarantee generated image or video quality. The estimated `$0.50–$0.60` target for a complete finished video is a long-term cost objective, not an enforced application threshold or a measured guarantee.

Local project files and the available development notes do not contain a complete archive of every historical experiment or conversation. Historical provider receipts may omit exact submitted prompts or image ordering; do not describe a reconstructed prompt as an exact request transcript.
