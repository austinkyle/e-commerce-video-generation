# In-Store Reaction V1

This is the durable format playbook for the local E-Commerce Video Generator. The reusable reference-recreation rules live in `src/ecom_video/creative_playbook.py`; this document records the format-specific interpretation and one canonical example. The example is documentation only and is never loaded as job data or required at runtime.

## Format identity

In-Store Reaction can move through attention, curiosity, product reveal, demonstration, real-world benefit, and emotional or practical payoff. These are functions a source may combine or omit, not required beats or a fixed shot count. The supplied reference determines its scene boundaries, ordering, timing, actions, camera behavior, environment, and resolution.

For a reference recreation, the source is the production blueprint. Preservation is automatic; modification requires an explicit user request. Scope a requested change to its named subject, object, scene, or recurring identity. The observed source controls all untargeted details. Product photos and confirmed product facts control product appearance and capabilities. If a requested change makes an observed action impossible, explain that conflict instead of making up a replacement action.

## Analyze each reference independently

Analyze the full source from its first meaningful frame through its ending. For each distinct scene, retain absolute source timestamps, the frame-zero composition and physical state, visible subjects and props, ordered actions and within-shot timing, contact and cause-and-effect, camera movement, environment and lighting, ending state, audio and captions when observable, and continuity links. Distinguish observations from uncertainty and user-requested changes. Do not infer a typical format action or silently merge, reorder, compress, or replace source shots.

The reference-analysis schema stores an immutable-to-revision summary of each observed source shot in `source_observation`, alongside its absolute `start_seconds` and `end_seconds`. Target fields (`visual_description`, `starting_state`, `action`, `ending_state`, `camera`, and `product_interaction`) describe the requested execution. `applied_user_changes` lists only the explicit creator changes scoped to that shot. Creator instructions and confirmed product facts remain in job and plan context. Targeted revisions preserve source observations, shot indices, order, source intervals, durations, and untouched fields while appending the new explicit change to affected shots.

## Starting images and motion

Starting images reproduce one source shot's initial visible state. They specify who and what is visible, physical positioning and orientation, wardrobe and product condition, hand/object support, camera and framing, environment, and the state immediately before the first action. Later results belong in the video prompt.

Video prompts carry the source's ordered actions, approximate timing, initiating subject, physical contact and cause-and-effect, camera movement, ending state, and relevant audio. For difficult interactions, prioritize the essential visible action and its result. Preserve continuity for recurring people, products, animals, environments, wardrobe states, and props while allowing source-motivated scene changes. Product references establish product appearance only; they do not replace a source environment.

Review reference fidelity, essential action, product fidelity, continuity, and organic photographic realism separately. A polished clip that omits the selling action is not a successful recreation. Use existing previews and revision controls; do not add an automated critic or regeneration loop.

## Canonical example: babywearing hoodie

The supplied example is approximately 14.12 seconds, vertical 9:16, with four principal continuous shots. These are observations for this example, not reusable actions, settings, or required durations.

### 1. In-store reaction and reveal (0:00–0:03.5)

The mother is already surprised in a clothing store, holding her baby securely in one arm and iced coffee in the other. At about 1.2 seconds the cup leaves her hand and falls under gravity. She keeps looking toward the display as the handheld camera pans right, moving her out of frame and revealing a rack of pink babywearing hoodies. The continuous shot ends on the merchandise.

Preserve the sequence: established reaction, coffee release and fall, camera pan, identifiable product reveal. Keep the baby supported and ordinary store lighting. Do not add pointing, dialogue, an extra reaction shot, or a dramatic splash absent from the source.

### 2. Physical demonstration (0:03.5–0:06.5)

Cut to an outdoor walkway. The mother wears the pink integrated babywearing hoodie. The baby begins elevated above and partly aligned with the open pouch. She supports the baby with both hands, lowers the baby into the pouch, and adjusts the fabric while continuing support. The scene ends with the baby positioned inside.

The initial image must show the pre-placement state. Motion must show the visible downward placement and continuous support. Do not jump to the completed carry state, hold the baby against the hoodie instead, or invent straps, buckles, or mechanisms.

### 3. Everyday use (0:06.5–0:10.5)

Cut to an outdoor pumpkin market with wooden produce stalls, pumpkins, and overhead string lights. The mother wears the same hoodie with the baby in its pouch, moves naturally while holding a shopping bag, then lifts and adjusts the hood/fabric around the baby. Preserve the market, bag, hand interaction, and subtle everyday movement; do not reduce this to a static pose.

### 4. Emotional payoff (0:10.5–0:14.12)

Cut to an indoor nursery. The mother wears the same hoodie and gently touches, cuddles, and nuzzles the baby held close in the garment. Preserve the intimate composition and small, natural movement. End naturally without added dialogue or dramatic acting.

### Caption

The reference has a short top-positioned caption about the hoodie as a gift for a newborn mom, with emotional emojis. Preserve its words, position, timing, and general appearance when observable and unless the user changes it. Keep text direction in the existing editing workflow when available; do not add a caption-generation system.

## Example of scoped changes

If asked to make the mother Black and the hoodie pink, apply the identity change to the mother consistently and the color change wherever that hoodie appears. Preserve the baby, other people, all source settings, actions, scene order, timing, and wardrobe distinctions. In particular, do not put the hoodie on her during the store scene if she does not wear it there, and do not replace the pumpkin market with another seasonal setting. If the hoodie is already pink, no additional color change is needed.

## Implementation boundaries

Activate the reusable rules for `in_store_reaction` and its `wet_works` alias. Keep 3-2-1 Reveal, Gifting, Versus, Custom, and original-concept behavior unchanged. Reuse existing reference analysis, plan, prompt, continuity, regeneration, and preview paths. Do not introduce a new service, model, storage layer, interface, or paid generation step.
