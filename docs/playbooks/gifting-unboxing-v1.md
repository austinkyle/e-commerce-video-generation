# Gifting / Unboxing V1

## Purpose

Gifting / Unboxing recreates a supplied product advertisement as a production blueprint. Preserve its actual source shots, order, timing, actions, compositions, camera behavior, environments, captions, audio, and emotional progression. Apply only changes explicitly requested by the user, including only the continuity changes those substitutions require. Do not import In-Store Reaction actions or 3-2-1 Reveal choreography.

The functional story may move through gift curiosity, package or product discovery, reveal, first use, benefit, and payoff. This is not a fixed six-shot storyboard. Wrapping paper, a gift exchange, a recipient, an emotional reaction, or even unboxing may not appear. Source footage decides what happens. Never fabricate delight, surprise, dialogue, social behavior, commercial claims, safety, certification, licensing, or retailer affiliation.

## Reference extraction

For every distinct source shot, preserve its source interval and purpose, opening composition, camera and environment, people/animals/hands/props, product and package state, ordered actions and approximate timing, physical contact and cause/effect, camera movement and edits, required ending state, overlays, sound events, and continuity. Keep the observed source description in `source_observation`; record only explicitly authorized changes to that scene in `applied_user_changes`. Do not change source observations to match target substitutions. Do not merge or drop source scenes to meet a requested clip count.

## Packaging and product state

Treat packaging as a specific physical object: type, size, material, color, markings, orientation and visible inserts; exact sealed/open/partial/empty state; opening mechanism; hand and tool placement; contact; and the path by which the product emerges. Starting images show the correct frame-zero state. Video action performs the opening and removal continuously. Nothing teleports, dissolves, or emerges from an implausible location.

Use supplied product photographs for product identity and supplied scene references for composition separately. Keep the product's shape, scale, materials, controls, parts, and function consistent through only the stages visible in the reference: packaged, unboxed, displayed, fitted/installed, used, and result. Demonstrations show real visible controls, support, contact, cause, and result. Keep baby and animal positioning plausible; infer no safety or unverified capability claims. Preserve intentional variants as distinct products and recurring people/animals as the same identities.

## Prompt responsibilities

Starting images establish the scene, identities, wardrobe, frame-zero product/package state, camera, and visible contact points. They do not perform later actions or repair wrong product geometry. Video prompts specify the ordered physical action, timing, cause/effect, camera behavior, audio where supported, and required ending state. Keep product-photo backgrounds out of the scene unless they are actually part of it.

Use the shared prompt compaction path and 3,800-character provider video-prompt limit. Retain packaging actions, product demonstration, identity, environment, source timing, audio, and ending state when compacting. Motion-only revisions reuse approved starting images. Correct images only for actual frame-zero or geometry errors. Keep prior approved assets, clips, versions, and cost history; never automatically retry paid generation.

Use the existing DaVinci Resolve editing and typography workflow. Do not add a format-specific editing system.

## Canonical examples

Six uploaded videos represent five distinct examples because the first pair is identical. These observations are examples, not runtime choreography.

### A — Gaming-themed robe (two identical uploads, about 15.2 s)

- 0–5.1 s: gaming POV; package is presented and opened.
- 5.1–8.2 s: robe is revealed and worn, with front and back views.
- 8.2–13 s: outdoor, mirror, and dog-walking lifestyle shots.
- 13–15.2 s: robe is worn in a kitchen.
- Mechanism: novelty gift, unboxing, wearable reveal, humorous repeated use.

### B — Themed car-window decoration (about 16.9 s)

- 0–3.9 s: themed mailing package is handled and cut.
- 3.9–8.4 s: gloved hands expose the folded product.
- 8.4–11 s: setup at a car and phone interaction.
- 11–14.3 s: illuminated decoration appears on vehicles at night.
- 14.3–16.9 s: nighttime display and CTA.
- Mechanism: anticipation, tactile unboxing, setup, activated transformation.

### C — Car-shaped baby product (about 9.6 s)

- 0–2.3 s: large box opens and green car-shaped item is removed.
- 2.3–4.4 s: baby is positioned within the product.
- 4.4–5.6 s: wider nursery view.
- 5.6–9.6 s: additional colored variants.
- Mechanism: unexpected gift, physical reveal, practical context, product variants. Preserve the observed crying; do not invent happiness or infer sleep safety or certification.

### D — Dog winter clothing (about 11.7 s)

- 0–2.1 s: product is unpacked and displayed beside the dog.
- 2.1–4.2 s: outfit is positioned on the dog.
- 4.2–7.9 s: garment is fitted, adjusted, and zipped.
- 7.9–10.7 s: dog walks in a snowy city.
- 10.7–11.7 s: affectionate cuddle.
- Mechanism: seasonal need, thoughtful purchase, fit demonstration, real-world use, emotional payoff.

### E — Digital nail printer (about 11.5 s)

- 0–2.3 s: box opens and compact device is removed.
- 2.3–5.7 s: screen and design selection.
- 5.7–8.1 s: finger is positioned and nail printing is demonstrated.
- 8.1–9.3 s: result is shown.
- 9.3–11.5 s: manicure close-up and CTA.
- Mechanism: curiosity, unboxing, product interaction, visible transformation, result.

## Implementation boundary

Generic reusable prompt guidance lives in `creative_playbook.py`; the app's universal reference preservation remains shared with other reference formats. This playbook documents format behavior only. The uploaded example files are not runtime dependencies.
