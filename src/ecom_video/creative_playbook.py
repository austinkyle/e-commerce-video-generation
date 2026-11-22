"""Version-controlled creative rules shared by planning and prompt generation."""

# Reusable for future reference-led formats. Only activate it for reference
# recreation; original concepts have no source production blueprint.
REFERENCE_RECREATION_RULES = (
    "Reference recreation: treat the supplied video as the production blueprint. Preserve its scene count, order, approximate timing, "
    "opening and ending states, ordered physical actions, camera, composition, location, lighting, wardrobe, cast behavior, "
    "product interaction, edit rhythm, captions, audio, and payoff wherever visible. Change only details explicitly requested by the creator; "
    "scope each change to the named people, objects, scenes, or recurring identities. The observed reference controls every untargeted detail. "
    "Product references and confirmed facts control product identity and capability. Never silently add, remove, merge, reorder, or reinterpret a shot or action. "
    "If a requested change conflicts with an observed action, state the specific conflict instead of inventing a substitute."
)
REFERENCE_RECREATION_PROMPT_RULES = (
    "Blueprint: preserve untargeted scene order/timing, states/actions, camera, setting, wardrobe, behavior, product use, edit, captions, audio, and payoff. "
    "Change only explicit targets; add, omit, merge, or reorder nothing."
)

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

COUNTDOWN_REVEAL_PLAYBOOK = (
    "3-2-1 Reveal is a curiosity-and-payoff format, not a literal countdown or fixed shot template. "
    "Build the source's actual progression from concealment or uncertainty through anticipation to its real reveal, "
    "then preserve the product proof and payoff that follow. The reveal may be a physical uncovering, a hard edit, "
    "or an activated result; use only the mechanism observed in the reference or requested by the creator. "
    "Never add countdown text, extra anticipation beats, a cover, a reveal action, fixed shot counts, or fixed durations "
    "that are not present in the source. Preserve intentional product variants, performers, locations, and product states. "
    "Keep packaged, unpacked, installed, inactive, and activated states distinct when the source shows them."
)
COUNTDOWN_REVEAL_IMAGE_RULES = (
    "Frame zero is the actual opening state of this source shot. For a covered reveal, match the cover's visible shape, "
    "size, material, color, folds, grip points, support surface, contact shadows, and occlusion; keep the revealed subject "
    "hidden until the source action. Use product references for product appearance only and character references for identity only. "
    "For later shots, show the correct source-specific product state, including packaging, installation, or activation only when visible."
)
COUNTDOWN_REVEAL_VIDEO_RULES = (
    "Preserve the observed reveal mechanism and source timing. If the source shows a physical cover, it moves only from its actual grip point, "
    "folds and exits naturally as the subject is progressively uncovered; do not teleport the cover, expose the subject early, "
    "or leave the reveal unfinished. For reveal-by-cut, preserve the hard edit and show the resulting use; do not animate an unobserved uncovering. "
    "For product proof, preserve observable contact, mechanism, and cause and effect. For activation, begin in the observed inactive state "
    "and show the physical trigger before the activated result. End in the specified visible state. "
    "The format name is not a countdown cue; add spoken or visible countdowns only when observed or explicitly requested."
)

GIFTING_UNBOXING_PLAYBOOK = (
    "Gifting / Unboxing is a product-desire and product-proof format. Use the source's actual progression toward discovery, "
    "reveal, first use, benefit, and payoff; this is a flexible story framework, never a required six-shot storyboard. "
    "A gift exchange, wrapping paper, recipient reaction, retail box, or even an unboxing may be absent. Preserve the actual "
    "recipient behavior, emotional level, product variants, environments, edits, captions, audio, and CTA. Do not invent surprise, "
    "delight, speech, social interaction, product claims, safety, certification, licensing, or retailer affiliation. "
    "Make the product desirable through the source's real presentation and prove its value only through observed or verified function."
)
GIFTING_UNBOXING_REFERENCE_RULES = (
    "For each package-bearing source shot, record package type, approximate dimensions, material, color, markings, inserts, "
    "contents orientation, opening mechanism, hand/tool contact, and how the product moves from inside to its visible ending position. "
    "Keep the unmodified source account in source_observation and list only applicable explicit changes in applied_user_changes. "
    "Distinguish scene-composition references from product-identity references and preserve intentional product variants."
)
GIFTING_UNBOXING_IMAGE_RULES = (
    "Frame zero must match the specified physical opening state; in reference recreation, match the package's exact observed condition or absence. "
    "Preserve package type, scale, material, color, relevant artwork, visible inserts, product orientation, "
    "hand/tool placement, and scene background. Product photos define product identity only, never the scene background. "
    "Do not show a product already removed when the source starts unopened; do not reveal hidden contents early."
)
GIFTING_UNBOXING_VIDEO_RULES = (
    "Show only the source's ordered physical actions: hands grip the actual flap/seal/tear/cut/zip mechanism, packaging "
    "opens or separates, inserts move as observed, and the product is removed from a plausible position with continuous contact. "
    "Never teleport, dissolve, or prematurely remove packaging or product. Preserve product geometry and scale through "
    "packaged, unboxed, displayed, fitted/installed, used, and result states only where present. Demonstrations must show "
    "the visible control, support, contact, real capability, cause, and ending state. Preserve real baby/animal support and fitting; "
    "infer no safety or capability claims. Keep motion and sounds aligned with the source; invent no dialogue, reaction, or music."
)

IN_STORE_REACTION_VIDEO_RULES = (
    "Keep reactions proportionate to the stimulus, with natural blinking. "
    "Show one primary physical beat at natural speed; preserve anatomy, grip, support, weight, gravity, contact, and the visible result. "
    "Match reference handheld pace; avoid unmotivated cinematic moves, shallow focus, or artificial stabilization. "
    "Preserve product geometry, texture, exposure, and practical lights; add no glow beyond real product lights. "
    "Across motion, keep face, hair, age, body scale, wardrobe, skin texture, and scene lighting stable; let expression and pose change only with the action. "
    "avoid facial shimmer, beauty smoothing, and identity drift."
)

IN_STORE_REACTION_IMAGE_RULES = (
    "Use natural unretouched skin, eyes, hair, and proportions. Match practical light and shadows; keep framing candid, not studio-polished. "
    "Show a relaxed pre-action pose, clear gaze, stable posture, and supported object contact."
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
    if hook == "countdown_reveal":
        patterns.extend((
            "A reveal is forced into a literal countdown -> the format name was mistaken for source content -> add countdown graphics or speech only when observed or requested.",
            "A cut-based reveal becomes a cloth pull -> the source edit mechanism was ignored -> preserve the hard cut and begin with the revealed use scene.",
            "A covered product is visible at frame zero -> occlusion geometry was underspecified -> describe the cover, grip, support, folds, and hidden target in the starting image.",
        ))
    return "Known failure patterns to avoid: " + " ".join(patterns)
