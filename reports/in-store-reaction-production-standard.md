# In-Store Reaction production standard: Christmas Village evidence

## Scope and saved-job state

Evidence was read from job `61cae587-1cc8-42b5-b282-ed5da539871a`, its local reference video and product photograph, saved plan, all current starting images, Clip 1 history, and each available `generation-usage.json`. No media was copied. No job, plan, asset, charge, audio setting, or generation task was changed.

The current selected starting images are the five assets below. The job itself is **`awaiting_image_review`**, `images_approved_at` is null, and all five `versions` arrays are empty. Therefore these are the current selections, but the saved record does not show formal image approval or video readiness. Clip 1 native audio remains enabled (`audio_requested=true`, `audio_mode=enabled`). No video task or video version exists.

| Clip | Selected image | Asset ID | SHA-256 |
|---|---|---|---|
| 1 | `outputs/clip-01/starting-image/version-8/clip-01-they-react-to-an-off-screen-surprise-the-wife-lo-starting.png` | `c964681c-87ce-448f-bbcc-ab0afef2de45` | `1b7d9c1f7f04f2bbc98e74dade8bfa836246ceb77e2b5147d65191574ed0ad28` |
| 2 | `outputs/clip-02/starting-image/version-4/clip-02-the-husband-takes-the-box-with-both-hands-and-ti-starting.png` | `cca764d4-2ffb-4453-a169-91d00f69781d` | `9fa864dbe227929dfc8d209c754c13a66e2f1167c7f8e264a73ed5c1bcbe3476` |
| 3 | `outputs/clip-03/starting-image/version-3/clip-03-the-husband-carefully-sets-the-table-down-and-sq-starting.png` | `4ed282a3-a72c-405d-8e28-77a0879847b7` | `c301850d6f5002c9315cb4f60a55c24db1ac5e62612a1b21be96bff5bd2600d8` |
| 4 | `outputs/initial/clip-04-he-presses-the-remote-once-holding-it-aimed-at-t-starting.png` | `17720c3b-ea7c-448f-b185-b11f5720acb9` | `c99ad645ef3f152fcc8f136f3194c08cea81f3c779aad320ff27925d9a027b76` |
| 5 | `outputs/initial/clip-05-camera-glides-and-settles-into-a-beauty-pass-ove-starting.png` | `0a869628-a4d7-4fa0-9c4f-035c434ddf1b` | `e64723f678a9974faa0c80051ae831b1ed06ff86f9974ef17970cfdca50d044c` |

## What the iterations show

- Clip 1 v1 shows the coffee already falling/spilling and the couple looking down, so the opening frame implies the accident already happened. The current v8 shows the same couple in a stocked warehouse aisle, both reacting toward the off-frame reveal, with the drink still upright. It has the strongest supported frame-zero composition of the saved Clip 1 versions.
- Clip 1 v5 and v7 put the couple on a white studio-like background. V6 returns to the warehouse but keeps their attention downward. This sequence distinguishes scene continuity and reaction geography problems from a general image-model failure.
- The current Clip 2 image is a useful same-scene couple/warehouse anchor. User-provided production history says v8 used Clip 2 for people/location and an earlier Clip 1 image only for the coffee pose, ignoring that image's bad background. The current v8 visually matches that stated strategy.
- The uploaded product photograph has a white background. It is evidence for product appearance, not the retail or home environment. The source video, its sampled frames, and the scene prompt carry scene and pacing information.
- The current Clip 3 image shows the table assembled and on the floor while the husband holds its frame, beyond the saved starting-state wording that says it is not yet centered/placed. The current Clip 4 image already shows the miniature village illuminated although its saved starting state says dark or minimally lit. Sampled source frames at 5.406s and 7.208s also show the table lit. These are project-specific plan/image alignment issues; the images and plan were left untouched.

## Provenance limits and spending evidence

The job manifest records task IDs, outcomes, timestamps, and credits. It does **not** retain the prompt sent for each historical image, the exact ordered image-reference selection, or the user's adjustment instructions. The user-described v8 reference roles can be retained as reconstructed context, but cannot be independently confirmed from those manifests. The correction causes above are based on visual comparison; where exact historical request payloads are absent, the report does not attribute a result to an unverified prompt change.

The saved `plan_json` remains in `data/jobs/61cae587-1cc8-42b5-b282-ed5da539871a/job.json` and preserves the current shot direction. It is not represented as the exact provider prompt that generated v8, because that historical request was not logged.

Saved charges show 102 Kie credits ($0.5100 equivalent) plus $0.011212 analysis, total $0.521212. The records account for 17 charged image requests at six credits each and one additional provider 500 failure with zero credits. The saved five-clip estimate is $0.7237. No provider was contacted during this standardization task.

## Reusable production rules now represented in code

The maintained format rules live in `src/ecom_video/creative_playbook.py`, which remains the only reusable creative playbook. It now records frame-zero versus later-action separation; the reference-led hook/reaction/reveal/discovery/demo/payoff progression without fixed clip counts or directions; camera/gaze geography; explicit environment continuity; same-scene identity and wardrobe; product identity versus product-photo background; and the observed failure/correction patterns. Project facts such as this couple, coffee, Christmas table, warehouse, rightward pan, and five scenes remain in the job plan rather than the permanent format defaults.

Existing planner paths use the selected hook direction. Image request construction keeps the actual prompt concise, labels product versus continuity roles, prioritizes matching-scene peers, puts an In-Store Reaction opening image before the product-visible anchor, and omits a generated cross-scene product anchor when frame zero hides the product. Selective starting-image regeneration uses the nearest saved same-group image for faces/location and the prior target image for pose/held-object position only. Image-only work does not compile video prompts. Usage manifests now retain future starting-image prompt text, reference filenames/roles, and a detectable revision note; no separate history or media-copy system was added.

The existing starting-image review panel now has an optional checklist for reference/setting, frame zero, continuity/spatial coherence, and physical realism. It is a user aid only: it performs no paid image analysis and does not gate approval.

## Limits

These are prompt and reference-selection improvements, not model retraining or guarantees. Image generation can still vary faces, hand anatomy, gaze, product geometry, package text, physical state, or background. Human review remains necessary for subtle fidelity and motion readiness. Historical prompt/reference lineage cannot be recovered retroactively from the existing receipts.
