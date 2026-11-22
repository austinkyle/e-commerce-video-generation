"""Version-controlled creative rules shared by planning and prompt generation."""

IN_STORE_REACTION_PLAYBOOK = (
    "In-Store Reaction: follow the reference's meaningful sequence: hook, immediate believable reaction, reveal, discovery, "
    "product-specific demonstration, and payoff. Keep reaction and reveal in one continuous hook when the reference supports it; "
    "place that unit in the first generated hook clip, without requiring it to start in the first second. When useful, keep it one continuous hook shot. "
    "Use natural handheld smartphone movement; derive pacing and shot lengths from its rhythm and never impose arbitrary fixed scene lengths or a fixed clip count. "
    "A gasp or hand over mouth, pointing, or brief freeze are optional reaction choices; choose only what fits the person and reference. A drink drop is optional. "
    "Unboxing and setup are optional; later scenes should use useful product-specific actions that advance the story. Show the distinctive product mechanism when relevant, "
    "and never imply stock at a named retailer unless actual distribution is established. "
    "Align eye line, expression, and camera movement with the stated reaction source; keep attention on it through incidental actions. "
    "For the opening, establish where the cause is relative to the people and camera; align gaze, head direction, composition, "
    "and planned camera movement with that location. Use the direction required by this scene; never assume left or right. "
    "Preserve the actual retail location, its recognizable layout, lighting, and depth in frame zero. Keep reactions natural and varied. "
    "Use the creator-requested target cast; when people are replaced, do not carry source-video identities into the generated scenes. "
    "Preserve the product's real mechanism and cause and effect. Build each short clip around one primary physical beat; "
    "include only the reaction and camera movement needed to clarify its cause and payoff. If the sequence cannot play at natural speed, "
    "simplify it or use a separate meaningful shot. Keep expressions proportionate, gaze motivated, and allow natural blinking and settling. "
    "Match the reference's handheld movement, pace, exposure, and depth; do not add cinematic motion or effects absent from the reference. "
    "For physical contact, establish a credible grip/support, weight, gravity, contact, and result. Preserve actual product geometry. "
    "When native sound is selected, name plausible visible sources and sync them to the action; otherwise keep output silent."
)

IN_STORE_REACTION_VIDEO_RULES = (
    "Align eye line, expression, and camera movement with the stated reaction source; keep attention on it through incidental actions. "
    "Keep reactions proportionate to the stimulus, with natural blinking and settling. "
    "Show one primary physical beat at natural speed; preserve grip, support, weight, gravity, contact, and the visible result. "
    "Match reference handheld pace; avoid unreferenced cinematic pushes, shallow focus, or artificial stabilization. "
    "Preserve product geometry, texture, exposure, and practical lights; add no glow beyond real product lights. "
    "Across motion, keep face, hair, age, body scale, wardrobe, skin texture, and scene lighting stable; let expression and pose change only with the action. "
    "Maintain natural anatomy, blinking, hand grips, and object contact; avoid facial shimmer, beauty smoothing, and identity drift."
)

IN_STORE_REACTION_IMAGE_RULES = (
    "For visible people, use subtle unretouched skin texture and tone variation, natural eyes and hair, ordinary proportions and slight asymmetry; avoid beauty smoothing or plastic shine. "
    "Match practical light, shadows, exposure, and natural color across people, product, and room; keep framing candid, not studio-polished. "
    "Use a relaxed, animation-ready pre-action pose with clear gaze, stable posture, and supported hand/object contact."
)

STARTING_STATE_RULES = (
    "Starting image is frame zero: depict only what is physically true before the first video action. Put all later movement, "
    "drops, impacts, activation, opening, and reveals in the video action. Follow the reference's actual initial state; do not "
    "force a product to be closed, hidden, or unlit when it starts open, visible, or illuminated. Preserve the specified location, "
    "lighting, background, and visible surfaces in frame zero."
)

CONTINUITY_RULES = (
    "Preserve each recurring person's face, skin tone, hair, age, and proportions across every scene assigned to that character, "
    "even when the environment, camera angle, pose, or wardrobe changes; use approved images from the matching character group "
    "as identity-only references and environment references for setting continuity. Preserve product geometry and distinctive details. "
    "Allow motivated wardrobe or location changes between scenes. Product photos define product "
    "appearance, not their unrelated background. Do not invent product features."
)

# User-approved corrections belong here as concise reusable patterns. Keep product- or
# project-specific facts in that project's plan and references, not in this format list.
IN_STORE_REACTION_FAILURE_PATTERNS = (
    "Coffee already falling at frame zero -> later video action leaked into the image request -> state the intact held cup in the image prompt and put the drop, impact, and splash only in video direction.",
    "People look down instead of toward the reveal -> cause, gaze, and camera geography were not resolved -> specify the cause's position and align gaze and camera move without assuming a direction.",
    "An off-screen product changes during its reveal -> the video model received no product image anchor -> pass the starting frame first and product-only references after it; do not treat reference backgrounds as the scene.",
    "A white studio replaces the retail aisle -> a product photo's background competed with scene guidance -> assign product photos to product details only and use a matching-scene image for the environment.",
    "An unrelated home scene influences a store hook -> references were selected by project membership instead of scene relevance -> use only matching continuity images and suppress cross-scene anchors when the product is outside frame.",
)

GENERAL_FAILURE_PATTERNS = (
    "The product appears before its reveal -> the frame-zero visibility state was vague -> state exactly whether the product is visible at the opening.",
    "A recurring character changes identity across scenes -> cast identity was tied to a location group or passed only as text -> reuse the approved image from that character group and let scene-specific wardrobe and setting instructions control the new shot.",
    "An activation shot starts in the wrong product state -> a generic closed/unlit assumption overrode the reference -> copy the actual state at the sampled start frame.",
    "A negated instruction triggers a contradiction warning -> validation treated a prohibition as a positive state -> validate affirmative frame-zero claims only.",
    "Starting-image prompts become bloated -> the detailed playbook or repeated directions were copied into the provider request -> keep durable guidance here and send only the concise, shot-relevant instructions within the provider budget.",
    "Image-only regeneration fails on a long video prompt -> unrelated video preflight ran on an image request -> skip video compilation when no video will be submitted.",
    "Repeated generations cannot be compared by cause -> prompts and selected references were not retained with task receipts -> save concise request and reference metadata in the existing usage manifest.",
)


def format_lesson_reminders(hook: str = "recommend") -> str:
    patterns = list(GENERAL_FAILURE_PATTERNS)
    if hook in {"in_store_reaction", "wet_works"}:
        patterns.extend(IN_STORE_REACTION_FAILURE_PATTERNS)
    return "Known failure patterns to avoid: " + " ".join(patterns)
