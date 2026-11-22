from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from types import SimpleNamespace

import pytest

from ecom_video.models import AnalysisResult
from ecom_video import cli
from ecom_video.workflow import (
    build_multiclip_plan,
    compile_video_prompt,
    estimate_multiclip_generation,
    find_saved_mask_for_scene,
    run_masked_image_edit,
    run_multiclip_generation,
    validate_starting_image_state,
    normalize_reference_segments,
    fit_reference_analysis_to_clip_count,
)
from ecom_video.costs import GenerationEstimate


def analysis_fixture() -> AnalysisResult:
    return AnalysisResult.model_validate({
        "summary": "A short product demonstration with a reveal.",
        "questions": [],
        "shots": [
            {
                "index": index,
                "start_seconds": (index - 1) * 2,
                "end_seconds": index * 2,
                "visual_description": description,
                "action": action,
                "camera": camera,
                "product_interaction": "The hands keep the product steady.",
                "hook_or_payoff": "A clear product reveal.",
                "uncertainties": [],
            }
            for index, description, action, camera in [
                (1, "Woman settles a kitten in the pet pouch", "Adjust the kitten in the pouch", "Handheld medium shot"),
                (2, "Close garment detail", "Adjust the opening", "Handheld close-up"),
                (3, "Centered reveal", "Let the kitten settle", "Handheld portrait"),
            ]
        ],
    })


def test_reveal_blanket_prompt_keeps_hidden_frame_lift_and_completed_reveal():
    analysis = AnalysisResult.model_validate({
        "summary": "A child costume reveal in a Christmas living room.", "questions": [], "shots": [{
            "index": 1, "start_seconds": 0, "end_seconds": 4.2, "creative_role": "Covered costume reveal",
            "visual_description": "A child sits beneath a white blanket in a Christmas living room.",
            "starting_state": "The white blanket fully conceals the seated child; the adult's hands hold its upper edge.",
            "action": "The adult lifts the blanket from its upper edge as it folds upward and clears the child's head and torso.",
            "ending_state": "The child is fully visible in a Christmas-tree costume; the blanket is out of frame.",
            "camera": "Vertical handheld medium shot, fixed on the seated child.", "product_interaction": "The adult lifts the blanket.",
            "hook_or_payoff": "The costume is revealed.", "source_observation": "Original: one continuous physical blanket lift, completed by 4.2s.",
            "applied_user_changes": ["Replace only the costume with the supplied product."],
            "continuity_group": "living-room", "character_groups": ["child", "adult"], "continuity_notes": [], "uncertainties": [],
        }],
    })
    plan = build_multiclip_plan(analysis, "Replace only the costume with the supplied product.", "Costume appearance from product reference.", "Christmas living room", 1, 5, hook="countdown_reveal")
    clip = plan.clips[0]
    video = compile_video_prompt(plan, clip)

    assert "fully conceals the seated child" in clip.starting_image_prompt
    assert "keep the revealed subject hidden" in clip.starting_image_prompt
    assert "lifts the blanket from its upper edge" in video
    assert "folds and exits naturally" in video
    assert "fully visible in a Christmas-tree costume" in video
    assert "not a countdown cue" in video.lower()
    assert "Original: one continuous physical blanket lift" in video
    assert "Replace only the costume" in video


def test_reveal_hard_cut_is_preserved_without_unobserved_uncovering():
    analysis = AnalysisResult.model_validate({
        "summary": "A projected interactive game demonstration.", "questions": [], "shots": [{
            "index": 1, "start_seconds": 0, "end_seconds": 5, "creative_role": "Reveal by hard cut",
            "visual_description": "A person holds a white cover in front of the camera, then the edit cuts to children using a projected game.",
            "starting_state": "The white cover fills the frame.",
            "action": "Hard cut from the cover to children aiming toy controllers at projected targets.",
            "ending_state": "Children play the projected game, with the game unit visible beside the projection.",
            "camera": "Handheld POV, then source cut to a wide room view.", "product_interaction": "Children aim the controllers at projected targets.",
            "hook_or_payoff": "The edit reveals the interactive game in use.",
            "source_observation": "Original transition is an edit cut; no one removes the white cover.",
            "applied_user_changes": [], "continuity_group": "playroom", "character_groups": ["children"],
            "continuity_notes": [], "uncertainties": [],
        }],
    })
    plan = build_multiclip_plan(analysis, "Recreate the reference.", "Interactive projection game; preserve observed capability.", "playroom", 1, 5, hook="countdown_reveal")
    prompt = compile_video_prompt(plan, plan.clips[0])

    assert "Hard cut from the cover" in prompt
    assert "preserve the hard edit" in prompt
    assert "no one removes the white cover" in prompt
    assert "do not animate an unobserved uncovering" in prompt.lower()
    assert "Preserve each source-required edit cut exactly" in prompt
    assert "No cuts, morphing" not in prompt


def test_reveal_retains_package_installation_and_activation_states_in_order():
    analysis = AnalysisResult.model_validate({
        "summary": "A wall-lighting product is unveiled, installed, and activated.", "questions": [], "shots": [
            {"index": 1, "start_seconds": 0, "end_seconds": 5, "creative_role": "Package reveal", "visual_description": "Covered retail pallet in a store.", "starting_state": "Pallet and packages are concealed by a cover.", "action": "A person lifts the cover.", "ending_state": "Stacked product packages are visible.", "camera": "Wide handheld view.", "product_interaction": "Cover is lifted from the pallet.", "hook_or_payoff": "Product packaging is unveiled.", "source_observation": "Covered pallet is unveiled.", "applied_user_changes": [], "continuity_group": "store", "character_groups": [], "continuity_notes": [], "uncertainties": []},
            {"index": 2, "start_seconds": 5, "end_seconds": 10, "creative_role": "Unpacked installation", "visual_description": "Unpacked light components are installed on the wall.", "starting_state": "Components and remote are unpacked; display is inactive.", "action": "Person mounts the light and adds ornaments.", "ending_state": "Installed display remains unlit.", "camera": "Medium close-up.", "product_interaction": "Mounts the fixture and ornaments.", "hook_or_payoff": "Installation is complete.", "source_observation": "Unpacked components installed before activation.", "applied_user_changes": [], "continuity_group": "home", "character_groups": ["installer"], "continuity_notes": [], "uncertainties": []},
            {"index": 3, "start_seconds": 10, "end_seconds": 15, "creative_role": "Activation payoff", "visual_description": "Remote activates the installed wall lights.", "starting_state": "Installed display is inactive; remote is in hand.", "action": "Person presses the remote and the wall lights illuminate.", "ending_state": "The completed wall display glows with the activated effect.", "camera": "Close-up of remote, then wide display.", "product_interaction": "Remote button activates the light.", "hook_or_payoff": "Illuminated final atmosphere.", "source_observation": "Activation follows installation.", "applied_user_changes": [], "continuity_group": "home", "character_groups": ["installer"], "continuity_notes": [], "uncertainties": []},
        ],
    })
    plan = build_multiclip_plan(analysis, "Preserve the reference sequence.", "Wall-lighting kit with remote.", "source locations", 3, 5, hook="countdown_reveal")
    prompts = [compile_video_prompt(plan, clip) for clip in plan.clips]

    assert [clip.source_start_seconds for clip in plan.clips] == [0, 5, 10]
    assert "Stacked product packages" in prompts[0]
    assert "Components and remote are unpacked" in prompts[1]
    assert "remains unlit" in prompts[1]
    assert "presses the remote" in prompts[2]
    assert "illuminate" in prompts[2]


def test_reveal_preserves_recurring_identity_across_intentional_variants_and_locations():
    analysis = AnalysisResult.model_validate({
        "summary": "A child models two intentional costume variants in different locations.", "questions": [], "shots": [
            {"index": 1, "start_seconds": 0, "end_seconds": 4, "creative_role": "Tree costume reveal", "visual_description": "Child wears a Christmas-tree costume in a living room.", "starting_state": "Child is concealed beneath a blanket.", "action": "Adult lifts the blanket.", "ending_state": "Child wears the tree costume.", "camera": "Medium shot.", "product_interaction": "Adult lifts blanket.", "hook_or_payoff": "Tree costume revealed.", "source_observation": "Tree costume reveal.", "applied_user_changes": [], "continuity_group": "living-room", "character_groups": ["child"], "continuity_notes": ["Same child's face and proportions."], "uncertainties": []},
            {"index": 2, "start_seconds": 4, "end_seconds": 8, "creative_role": "Gingerbread costume variation", "visual_description": "The same child wears an intentional gingerbread costume in the next shot.", "starting_state": "Child is already in the gingerbread costume outdoors.", "action": "Child turns toward camera.", "ending_state": "Gingerbread details are visible.", "camera": "Handheld full shot.", "product_interaction": "Costume is worn naturally.", "hook_or_payoff": "A second costume variant is shown.", "source_observation": "Intentional second costume variant in a new shot.", "applied_user_changes": [], "continuity_group": "winter-street", "character_groups": ["child"], "continuity_notes": ["Same child's face and proportions; allow the source's costume and location change."], "uncertainties": []},
        ],
    })
    plan = build_multiclip_plan(analysis, "Keep the child identity and source variants.", "Use exact costume details from product references.", "match each source shot", 2, 5, hook="countdown_reveal")

    assert plan.clips[0].character_groups == plan.clips[1].character_groups == ["child"]
    assert plan.clips[0].continuity_group != plan.clips[1].continuity_group
    assert "Christmas-tree costume" in plan.clips[0].visual_description
    assert "gingerbread costume" in plan.clips[1].visual_description
    assert "allow the source's costume and location change" in " ".join(plan.clips[1].continuity_notes)


def test_versus_reference_does_not_receive_reveal_choreography():
    analysis = analysis_fixture()
    plan = build_multiclip_plan(analysis, "Fairly compare both products.", "Two products.", "snowy driveway", 3, 5, hook="versus")
    prompt = compile_video_prompt(plan, plan.clips[0])

    assert "preserve the hard edit" not in prompt
    assert "do not add a literal countdown" not in prompt.lower()
    assert "concealed or uncertain" not in prompt


def test_reveal_motion_only_revision_keeps_format_rules_under_prompt_budget():
    analysis = AnalysisResult.model_validate({
        "summary": "A covered product is revealed.", "questions": [], "shots": [{
            "index": 1, "start_seconds": 0, "end_seconds": 4, "creative_role": "Physical reveal",
            "visual_description": "A covered device on a table.", "starting_state": "A blue cloth fully hides the device.",
            "action": "A hand lifts the cloth from the left edge until the device is fully visible.",
            "ending_state": "The device sits uncovered on the table.", "camera": "Handheld medium close-up.",
            "product_interaction": "Hand pulls the cloth from its edge.", "hook_or_payoff": "Device reveal.",
            "source_observation": "One uninterrupted cloth lift; the approved image has the covered starting geometry.",
            "applied_user_changes": [], "continuity_group": "table", "character_groups": [],
            "continuity_notes": [], "uncertainties": [],
        }],
    })
    plan = build_multiclip_plan(analysis, "Recreate the cover reveal.", "Use product image for device appearance.", "tabletop", 1, 5, hook="countdown_reveal")
    clip = plan.clips[0].model_copy(update={
        "video_prompt": plan.clips[0].video_prompt + "\nClip-specific revision: Motion-only correction: add a one-beat pause while the adult grips the upper hem before lifting.",
    })
    prompt = compile_video_prompt(plan, clip)

    assert "one-beat pause" in prompt
    assert "fully hides the device" in prompt
    assert "folds and exits naturally" in prompt
    assert len(prompt) <= 3800


def test_reveal_starting_images_keep_cover_hidden_and_use_product_reference_after_reveal(tmp_path):
    analysis = AnalysisResult.model_validate({
        "summary": "A covered device is revealed and demonstrated.", "questions": [], "shots": [
            {"index": 1, "start_seconds": 0, "end_seconds": 4, "creative_role": "Covered reveal", "visual_description": "A device is hidden beneath a blue cloth on a table.", "starting_state": "The device is fully hidden under the cloth.", "action": "A hand lifts the cloth from its edge until the device is fully revealed.", "ending_state": "The device is uncovered on the table.", "camera": "Vertical handheld medium shot.", "product_interaction": "Hand lifts the cloth.", "hook_or_payoff": "Device reveal.", "source_observation": "One continuous covered reveal.", "applied_user_changes": [], "continuity_group": "table", "character_groups": [], "continuity_notes": [], "uncertainties": []},
            {"index": 2, "start_seconds": 4, "end_seconds": 8, "creative_role": "Product proof", "visual_description": "The revealed device is used on the table.", "starting_state": "The device is visible and inactive on the table.", "action": "A hand presses the visible activation control; the device turns on.", "ending_state": "The device is visibly active.", "camera": "Close handheld view.", "product_interaction": "Press the observed activation control.", "hook_or_payoff": "Show the real activated result.", "source_observation": "Product use follows the reveal.", "applied_user_changes": [], "continuity_group": "table", "character_groups": [], "continuity_notes": [], "uncertainties": []},
        ],
    })
    product = tmp_path / "device.jpg"
    product.write_bytes(b"product-reference")
    plan = build_multiclip_plan(analysis, "Recreate the reveal and use.", "Device appearance from supplied product photo.", "tabletop", 2, 5, product_references=[product], hook="countdown_reveal")
    kie = FakeKie()

    run_multiclip_generation(plan, [product], [], tmp_path / "reveal-images", kie, images_only=True)
    image_calls = [call for call in kie.calls if call[0] in {"image", "text_image"}]
    manifest = json.loads((tmp_path / "reveal-images" / "generation-usage.json").read_text())

    assert len(image_calls) == 2
    assert image_calls[0][0] == "text_image"
    assert image_calls[1][0] == "image"
    assert "https://files.test/device.jpg" in image_calls[1][2]
    first_roles = manifest["tasks"][0]["request"]["references"]
    second_roles = manifest["tasks"][1]["request"]["references"]
    assert not any(ref["filename"] == "device.jpg" for ref in first_roles)
    assert {ref["filename"]: ref["role"] for ref in second_roles}["device.jpg"] == "product identity only"
    assert "@image2=product only" in compile_video_prompt(plan, plan.clips[1], product_reference_count=1)


def test_plan_uses_distinct_reference_shots_and_same_environment_for_image_and_video():
    plan = build_multiclip_plan(
        analysis_fixture(),
        instructions="Show a calm kitten already partly supported in the pouch.",
        product_facts="Red Santa hoodie with circular opening and belt graphic.",
        environment="naturally lit living room",
        clip_count=3,
        seconds_per_clip=5,
    )

    assert len(plan.clips) == 3
    assert len({clip.source_shot_index for clip in plan.clips}) == 3
    for clip in plan.clips:
        assert "naturally lit living room" in clip.starting_image_prompt
        assert "naturally lit living room" in clip.video_prompt
        assert len(clip.starting_image_prompt) < 1_000
        assert "baby" not in clip.starting_image_prompt.lower()
        assert "baby" not in clip.video_prompt.lower()
        assert clip.duration_seconds == 5


def test_auto_audio_silence_preserves_direction_without_claiming_native_audio():
    analysis = analysis_fixture()
    analysis.shots[0].audio_direction = "Intended store ambience and a gasp as the shopper reacts. Silent output."

    plan = build_multiclip_plan(
        analysis, "An in-store product discovery", "An unchanged teal product",
        "a generic retail aisle", clip_count=3, seconds_per_clip=5,
    )

    prompt = compile_video_prompt(plan, plan.clips[0])
    assert "Audio is disabled for this clip" in prompt
    assert "no audio stream" in prompt.lower()


@pytest.mark.parametrize("hook", ["in_store_reaction", "countdown_reveal", "gifting", "versus", "custom"])
def test_auto_audio_uses_creative_director_choice_across_all_formats(hook):
    analysis = analysis_fixture()
    analysis.shots[2].audio_requested = True
    analysis.shots[2].audio_direction = "Natural laugh and the product's soft mechanical click, synchronized to the action."

    plan = build_multiclip_plan(
        analysis, "Show the product clearly", "Preserve its real mechanism", "a casual home", 3, 5, hook=hook,
    )

    clip = plan.clips[2]
    assert clip.audio_mode == "auto"
    assert clip.audio_requested is True
    assert clip.audio_enabled is True
    prompt = compile_video_prompt(plan, clip)
    assert "smartphone video" in prompt
    assert "Native audio direction:" in prompt
    assert "Natural laugh" in prompt


def test_silent_audio_mode_overrides_auto_selection_without_losing_it():
    analysis = analysis_fixture()
    analysis.shots[0].audio_requested = True
    analysis.shots[0].audio_direction = "A synchronized cup impact and liquid splash."
    plan = build_multiclip_plan(analysis, "Show the drop", "Cup and drink", "a store", 3, 5, hook="in_store_reaction")
    clip = plan.clips[0]

    clip.audio_mode = "silent"
    assert clip.audio_enabled is False
    assert "silent smartphone video" in compile_video_prompt(plan, clip)

    clip.audio_mode = "auto"
    assert clip.audio_enabled is True


def test_legacy_silent_and_audio_plans_load_with_automatic_mode():
    base = build_multiclip_plan(analysis_fixture(), "a brief", "a product", "a room", 3, 5)
    payload = base.model_dump(mode="json")
    for clip in payload["clips"]:
        clip.pop("audio_mode", None)
        clip.pop("audio_requested", None)
    payload["clips"][1]["preserve_audio"] = True

    restored = type(base).model_validate(payload)

    assert restored.clips[0].audio_mode == "auto"
    assert restored.clips[0].audio_enabled is False
    assert restored.clips[1].audio_mode == "auto"
    assert restored.clips[1].audio_enabled is True


def test_in_store_reference_preserves_every_source_scene_and_its_timing():
    analysis = AnalysisResult.model_validate({
        "summary": "A shopper reacts to a laundry basketball hamper, then the product is installed, loaded, and shown in its final state.",
        "questions": [],
        "shots": [
            {"index": 1, "start_seconds": 0, "end_seconds": 2, "creative_role": "hook", "visual_description": "An older shopper reacts to the display.", "starting_state": "She is already noticing the shelves.", "action": "She pauses with a surprised expression.", "ending_state": "She turns toward the display.", "camera": "Handheld medium shot.", "product_interaction": "No touch.", "hook_or_payoff": "Immediate curiosity.", "continuity_group": "store", "continuity_notes": ["Same shopper and display."], "uncertainties": []},
            {"index": 2, "start_seconds": 2, "end_seconds": 4.2, "creative_role": "reveal", "visual_description": "She points her cane toward the Laundry Dunk display as the camera pans to the product.", "starting_state": "She faces the shelf.", "action": "She points with the cane while the camera follows to show the laundry hamper product.", "ending_state": "The product is identifiable on the display.", "camera": "Handheld pan to the shelf.", "product_interaction": "Cane points without touching the product.", "hook_or_payoff": "Clear product reveal.", "continuity_group": "store", "continuity_notes": ["Same shopper and display."], "uncertainties": []},
            {"index": 3, "start_seconds": 4.2, "end_seconds": 6.4, "creative_role": "installation", "visual_description": "The black mesh hamper is hung over the yellow door.", "starting_state": "The hamper is positioned at the top of the door.", "action": "A person steadies the rim and settles the mount over the door.", "ending_state": "The hamper hangs securely.", "camera": "Handheld medium shot.", "product_interaction": "Support the top rim while mounting.", "hook_or_payoff": "Show simple setup.", "continuity_group": "room", "continuity_notes": ["Same door and hamper."], "uncertainties": []},
            {"index": 4, "start_seconds": 6.4, "end_seconds": 8.4, "creative_role": "demonstration", "visual_description": "Laundry drops through the hamper opening.", "starting_state": "A few garments rest at the bottom.", "action": "Release a handful of laundry into the opening.", "ending_state": "The clothing settles in the mesh bag.", "camera": "Close handheld shot.", "product_interaction": "Laundry falls through the rim.", "hook_or_payoff": "Show the hamper in use.", "continuity_group": "room", "continuity_notes": ["Same door and hamper."], "uncertainties": []},
            {"index": 5, "start_seconds": 8.4, "end_seconds": 10.4, "creative_role": "rising payoff", "visual_description": "More garments are tossed into the same hamper.", "starting_state": "The hamper is partly full.", "action": "Drop a few more garments in sequence and let them settle.", "ending_state": "The hamper is visibly fuller.", "camera": "Angled close-up.", "product_interaction": "Clothes enter through the opening.", "hook_or_payoff": "Escalate the satisfying load test.", "continuity_group": "room", "continuity_notes": ["Keep the same fill progression."], "uncertainties": []},
            {"index": 6, "start_seconds": 10.4, "end_seconds": 14.5, "creative_role": "final payoff", "visual_description": "A full white basket sits below the mostly emptied hamper.", "starting_state": "The hamper remains mounted above the floor basket.", "action": "Reveal the filled floor basket and nearly empty hamper.", "ending_state": "The tidy final arrangement is clearly visible.", "camera": "Handheld wide vertical view.", "product_interaction": "Hamper remains mounted; basket collects laundry.", "hook_or_payoff": "Clear end-state payoff.", "continuity_group": "room", "continuity_notes": ["Same door, hamper, and basket."], "uncertainties": []},
        ],
    })

    plan = build_multiclip_plan(analysis, "Recreate the In-Store Reaction structure.", "Keep the black mesh hamper unchanged.", "the referenced scene", 6, 5, hook="in_store_reaction")

    assert len(plan.clips) == len(analysis.shots) == 6
    assert [(clip.source_start_seconds, clip.source_end_seconds) for clip in plan.clips] == [
        (0, 2), (2, 4.2), (4.2, 6.4), (6.4, 8.4), (8.4, 10.4), (10.4, 14.5),
    ]
    assert [clip.action for clip in plan.clips] == [shot.action for shot in analysis.shots]
    assert plan.clips[0].action == "She pauses with a surprised expression."
    assert "She points with the cane" in plan.clips[1].action
    assert "product is identifiable" in plan.clips[1].ending_state
    assert all(clip.duration_seconds == 5 for clip in plan.clips)
    assert plan.clips[-1].source_end_seconds == 14.5
    compact = compile_video_prompt(plan, plan.clips[0])
    assert "Hook direction: reproduce only the observed reaction" in compact
    assert "Reference segment 0.0-2.0s is the production blueprint" in compact
    assert "change only explicit targets; add, omit, merge, or reorder nothing" in compact.lower()


def test_in_store_reaction_uses_scene_specific_cast_durations_and_product_mechanics():
    analysis = AnalysisResult.model_validate({
        "summary": "In-Store Reaction ad with an elderly shopper at the mall hook, followed by distinct home scenes with a younger adult using the same colorful retro telephone.",
        "questions": [],
        "shots": [
            {"index": 1, "creative_role": "In-Store Reaction and reveal", "visual_description": "A shopper sees a display of colorful retro telephones.", "starting_state": "Shopper is already reacting beside the mall display.", "action": "She gasps, then points as the camera follows her attention and reveals the phones.", "ending_state": "Colorful retro telephones fill frame.", "camera": "Vertical handheld close reaction, then a fast pan to the display.", "product_interaction": "The gesture motivates the camera reveal.", "hook_or_payoff": "Resolve the reaction with a clear phone reveal.", "continuity_group": "mall_hook", "continuity_notes": ["Older shopper, mall display."], "target_duration_seconds": 4, "edit_duration_seconds": 3.5, "uncertainties": []},
            {"index": 2, "creative_role": "Unboxing", "visual_description": "At home, a younger adult opens the colorful telephone package.", "starting_state": "Closed product box on a side table.", "action": "Open the box and lift out the handset and base.", "ending_state": "Phone is displayed beside its packaging.", "camera": "Handheld close-up that tilts from the open box to the phone.", "product_interaction": "Lift the handset and base without changing their design.", "hook_or_payoff": "Reveal what is included.", "continuity_group": "home_setup", "continuity_notes": ["Same younger adult and same colorful retro telephone."], "target_duration_seconds": 3, "edit_duration_seconds": 2, "uncertainties": []},
            {"index": 3, "creative_role": "Setup demonstration", "visual_description": "The younger adult connects the telephone at home.", "starting_state": "Base is on a stable tabletop with the actual connector visible.", "action": "Connect the supplied cord to the matching visible port; do not invent ports or accessories.", "ending_state": "The connected phone base rests on the table.", "camera": "Handheld over-shoulder close-up that follows the connector into the matching port.", "product_interaction": "Align and insert the actual supplied connector into its matching port.", "hook_or_payoff": "Show how setup works.", "continuity_group": "home_setup", "continuity_notes": ["Same younger adult, home, and unchanged phone."], "target_duration_seconds": 3, "edit_duration_seconds": 2, "uncertainties": []},
            {"index": 4, "creative_role": "Use demonstration", "visual_description": "A first person lifts the handset and uses the retro telephone.", "starting_state": "Handset is seated on the base.", "action": "Lift the handset to the ear and speak naturally; keep the base and cord visible.", "ending_state": "The person is using the handset while the base stays on the table.", "camera": "Handheld medium close-up with a short push toward the handset and base.", "product_interaction": "Handset lifts from and remains physically connected to the base as designed.", "hook_or_payoff": "Show recognizable everyday use.", "continuity_group": "first_user", "continuity_notes": ["First user; keep the product identical."], "target_duration_seconds": 3, "edit_duration_seconds": 2.5, "uncertainties": []},
            {"index": 5, "creative_role": "Payoff", "visual_description": "A different person uses the same colorful retro telephone, then returns the handset to its cradle.", "starting_state": "Second person holds the handset beside the same phone base.", "action": "Speak briefly, then lower the handset into the cradle and let it settle.", "ending_state": "Colorful phone rests neatly on the table with handset seated.", "camera": "Handheld detail angle following the handset down into the cradle, ending on the complete phone.", "product_interaction": "The handset seats in its designed cradle; no unsupported ringing or other feature is shown.", "hook_or_payoff": "Pay off the product story with a clear final phone reveal.", "continuity_group": "second_user", "continuity_notes": ["Different person from the first-use clip; same unchanged telephone."], "target_duration_seconds": 3, "edit_duration_seconds": 2.5, "uncertainties": []},
        ],
    })

    plan = build_multiclip_plan(
        analysis, "Use the In-Store Reaction structure for a colorful retro telephone.",
        "A colorful retro telephone; match supplied product references and do not invent functions or accessories.",
        "mall display and simple home interiors as specified per scene", 5, 5, hook="in_store_reaction",
    )

    assert [clip.duration_seconds for clip in plan.clips] == [4, 3, 3, 3, 3]
    assert [clip.edit_duration_seconds for clip in plan.clips] == [3.5, 2, 2, 2.5, 2.5]
    assert plan.hook == "in_store_reaction"
    for clip in plan.clips:
        assert len(compile_video_prompt(plan, clip, product_reference_count=6)) <= 3800
    hook_prompt = compile_video_prompt(plan, plan.clips[0])
    assert "Hook direction:" in hook_prompt and "4-second clip" in hook_prompt
    assert "Opening hook:" not in hook_prompt
    assert "Intended final edit use: 3.5s" in hook_prompt
    assert "fast pan" in hook_prompt
    setup_prompt = compile_video_prompt(plan, plan.clips[2])
    assert "supplied connector" in setup_prompt and "matching port" in setup_prompt
    assert "Product-specific payoff:" not in setup_prompt
    payoff_prompt = compile_video_prompt(plan, plan.clips[4])
    assert "seats in its designed cradle" in payoff_prompt
    assert "elderly shopper" not in payoff_prompt.lower()
    assert "Different person from the first-use clip" in payoff_prompt
    assert "preserve the person already visible in the supplied starting image" in payoff_prompt.lower()


def test_canonical_in_store_reference_actions_states_and_scene_changes_are_preserved():
    analysis = AnalysisResult.model_validate({
        "summary": "Reference recreation. Make the mother Black; keep the hoodie pink only where worn in the source.",
        "questions": [],
        "shots": [
            {"index": 1, "start_seconds": 0, "end_seconds": 3.5, "creative_role": "In-store reaction and reveal",
             "visual_description": "A mother in a clothing store holds her baby and iced coffee beside a rack of pink babywearing hoodies.",
             "starting_state": "She is already surprised, supporting the baby in one arm and holding the intact iced coffee in her other hand; she wears a dark sleeveless top.",
             "action": "At about 1.2 seconds she releases the cup and it falls under gravity; her attention stays on the display as the handheld camera pans right, moving her out of frame and revealing the hoodie rack.",
             "ending_state": "The camera lands on the actual rack of pink babywearing hoodies.",
             "camera": "Vertical handheld medium reaction shot followed by a rightward pan to the rack.",
             "product_interaction": "The coffee leaves her hand; the baby remains supported.", "hook_or_payoff": "The product display resolves her reaction.",
             "continuity_group": "store", "character_groups": ["mother", "baby"], "continuity_notes": ["Black mother; dark sleeveless top in store; same baby."], "target_duration_seconds": 4, "uncertainties": [],
             "source_observation": "Original source: mother wears a dark sleeveless top, holds a supported baby and iced coffee, releases coffee at about 1.2 seconds, then the camera pans right to the hoodie rack.",
             "applied_user_changes": ["Make the mother Black; change no other character or wardrobe."]},
            {"index": 2, "start_seconds": 3.5, "end_seconds": 6.5, "creative_role": "Physical product demonstration",
             "visual_description": "On an outdoor walkway, the mother wears the pink integrated babywearing hoodie.",
             "starting_state": "The baby begins elevated above and partly aligned with the open pouch; the mother supports the baby with both hands.",
             "action": "She continuously supports the baby's body while lowering the baby visibly into the pouch, then adjusts the fabric without releasing support.",
             "ending_state": "The baby is positioned in the pouch and remains supported.", "camera": "Handheld medium view of the pouch and both supporting hands.",
             "product_interaction": "Visible downward placement into the integrated pouch.", "hook_or_payoff": "Show the garment's actual use.",
             "continuity_group": "walkway", "character_groups": ["mother", "baby"], "continuity_notes": ["Same Black mother and baby; pink hoodie."], "target_duration_seconds": 3, "uncertainties": [],
             "source_observation": "Original source: on the outdoor walkway, the baby starts above and partly aligned with the open pouch; mother supports with both hands, lowers the baby, adjusts fabric, and the baby ends inside.",
             "applied_user_changes": ["Make the mother Black; change no other character or action."]},
            {"index": 3, "start_seconds": 6.5, "end_seconds": 10.5, "creative_role": "Everyday lifestyle use",
             "visual_description": "At an outdoor pumpkin market with wooden stalls and string lights, the mother wears the same hoodie and carries the baby in its pouch.",
             "starting_state": "The mother holds a shopping bag while walking naturally through the market.",
             "action": "She moves naturally and lifts and adjusts the hood fabric around the baby.", "ending_state": "She continues through the market with the baby settled in the pouch.",
             "camera": "Casual handheld medium shot with subtle movement.", "product_interaction": "One hand adjusts the hood while the other holds the bag.",
             "hook_or_payoff": "Show everyday use.", "continuity_group": "market", "character_groups": ["mother", "baby"], "continuity_notes": ["Same Black mother, baby, and pink hoodie."], "target_duration_seconds": 4, "uncertainties": []},
            {"index": 4, "start_seconds": 10.5, "end_seconds": 14.12, "creative_role": "Emotional payoff",
             "visual_description": "In an indoor nursery the mother wears the hoodie and holds her baby close within the garment.",
             "starting_state": "Mother and baby are already close together.", "action": "She gently touches, cuddles, and nuzzles the baby with small natural movements.",
             "ending_state": "They remain close as the moment ends naturally.", "camera": "Intimate handheld close composition.",
             "product_interaction": "The baby stays close and supported within the garment.", "hook_or_payoff": "Quiet emotional resolution.",
             "continuity_group": "nursery", "character_groups": ["mother", "baby"], "continuity_notes": ["Same Black mother, baby, and pink hoodie."], "target_duration_seconds": 4, "uncertainties": []},
        ],
    })
    plan = build_multiclip_plan(analysis, "Make the mother Black; hoodie pink where worn in reference.", "Use the supplied product references.", "match each source scene", 4, 5, hook="wet_works")

    assert [(clip.source_start_seconds, clip.source_end_seconds) for clip in plan.clips] == [(0, 3.5), (3.5, 6.5), (6.5, 10.5), (10.5, 14.12)]
    assert "dark sleeveless top in store" in plan.clips[0].starting_image_prompt
    assert "pink integrated babywearing hoodie" in plan.clips[1].visual_description
    assert "Original source: mother wears a dark sleeveless top" in plan.clips[0].source_observation
    assert plan.clips[0].applied_user_changes == ["Make the mother Black; change no other character or wardrobe."]
    assert "She is already surprised" in plan.clips[0].starting_image_prompt
    assert "intact iced coffee" in plan.clips[0].starting_image_prompt
    assert "releases the cup" in compile_video_prompt(plan, plan.clips[0])
    assert "pans right" in compile_video_prompt(plan, plan.clips[0])
    assert "Observed original source scene: Original source: mother wears a dark sleeveless top" in compile_video_prompt(plan, plan.clips[0])
    assert "Explicit changes authorized for this scene: Make the mother Black" in compile_video_prompt(plan, plan.clips[0])
    assert "baby begins elevated above and partly aligned" in plan.clips[1].starting_image_prompt
    placement = compile_video_prompt(plan, plan.clips[1])
    assert "lowering the baby visibly into the pouch" in placement
    assert "continuously supports" in placement
    assert "pumpkin market" in plan.clips[2].starting_image_prompt
    assert "shopping bag" in compile_video_prompt(plan, plan.clips[2])
    assert "adjusts the hood fabric" in compile_video_prompt(plan, plan.clips[2])
    assert "nursery" in plan.clips[3].starting_image_prompt
    assert "cuddles, and nuzzles" in compile_video_prompt(plan, plan.clips[3])
    prompts = [compile_video_prompt(plan, clip) for clip in plan.clips]
    assert all(len(prompt) <= 3800 for prompt in prompts)
    assert "already surprised" in prompts[0]
    assert "intact iced coffee" in prompts[0]
    assert "falls under gravity" in prompts[0]
    assert "baby remains supported" in prompts[0]
    assert "pans right" in prompts[0]
    assert "actual rack of pink babywearing hoodies" in prompts[0]
    assert "handheld" in prompts[0].lower()
    audio_prompt = compile_video_prompt(plan, plan.clips[0].model_copy(update={
        "audio_mode": "enabled",
        "audio_direction": "Synchronize a realistic coffee-cup impact and spill sound to the cup striking the floor; no music.",
    }))
    assert "synchronize a realistic coffee-cup impact and spill sound" in audio_prompt.lower()
    assert "Synchronize each selected native sound to the visible event" in audio_prompt
    compact_drop = plan.clips[0].model_copy(update={
        "action": "She releases the iced coffee and the cup drops; she keeps the baby supported.",
    })
    assert "released object falling under gravity" in compile_video_prompt(plan, compact_drop).lower()
    for clip in plan.clips:
        prompt = prompts[clip.index - 1]
        assert "production blueprint" in prompt
        assert "change only explicit targets; add, omit, merge, or reorder nothing" in prompt.lower()


def test_different_in_store_reference_does_not_inherit_canonical_example():
    analysis = AnalysisResult.model_validate({"summary": "A shopper notices a compact umbrella display in an ordinary hardware store.", "questions": [], "shots": [{
        "index": 1, "start_seconds": 0, "end_seconds": 3, "creative_role": "Discovery",
        "visual_description": "A shopper pauses beside an umbrella display in a hardware store.",
        "starting_state": "The shopper holds a folded umbrella and looks at the shelf.",
        "action": "The shopper opens the umbrella once and looks at the canopy.",
        "ending_state": "The umbrella canopy is open beside the shelf.", "camera": "Static handheld medium framing.",
        "product_interaction": "The shopper opens the umbrella by hand.", "hook_or_payoff": "Reveal the compact canopy.",
        "continuity_group": "hardware-store", "character_groups": ["shopper"], "continuity_notes": ["Same shopper and umbrella."], "uncertainties": [],
    }]})
    plan = build_multiclip_plan(analysis, "Recreate the reference; replace only the umbrella color with blue.", "Compact umbrella with a manually opened canopy.", "match source", 1, 5, hook="in_store_reaction")
    prompt = plan.clips[0].starting_image_prompt + compile_video_prompt(plan, plan.clips[0])
    assert "blue" in prompt
    for inherited in ("coffee", "baby", "hoodie", "pumpkin", "nursery"):
        assert inherited not in prompt.lower()
    from ecom_video.workflow import starting_image_revision_prompt
    revised_image = starting_image_revision_prompt(plan, plan.clips[0], "Keep the umbrella blue.")
    assert "coffee" not in revised_image.lower()
    assert "folded umbrella and looks at the shelf" in revised_image
    assert "Requested revision: Keep the umbrella blue." in revised_image


def test_reference_plan_revision_prompt_preserves_source_scene_structure(monkeypatch):
    from ecom_video.providers import openai as provider

    data = {"summary": "A short product demonstration.", "questions": [], "shots": [
        {"index": 1, "start_seconds": 0, "end_seconds": 3, "creative_role": "Hook", "visual_description": "A shopper finds the product.",
         "starting_state": "The product remains on the shelf.", "action": "The shopper turns toward it.", "ending_state": "The shopper faces the product.",
         "target_duration_seconds": 3, "edit_duration_seconds": 2.5, "camera": "Handheld pan.", "product_interaction": "No contact.",
         "hook_or_payoff": "Discovery.", "audio_requested": False, "audio_direction": "Silent.", "continuity_group": "store",
         "character_groups": ["shopper"], "continuity_notes": ["Same aisle."], "uncertainties": []},
    ]}
    captured = {}
    response_payload = {
        "output": [{"content": [{
            "type": "output_text",
            "text": json.dumps({"summary": data["summary"], "changed_shot_indices": [1], "shots": data["shots"]}),
        }]}],
        "usage": {},
    }
    monkeypatch.setattr(provider, "require_key", lambda name: "mock-key")
    monkeypatch.setattr(provider.httpx, "post", lambda *args, **kwargs: captured.update(kwargs) or SimpleNamespace(
        raise_for_status=lambda: None, json=lambda: response_payload,
    ))

    provider.revise_creative_plan("Make only the shopper's jacket green.", data, {"mode": "reference", "hook": "in_store_reaction"})
    prompt = captured["json"]["input"][0]["content"][0]["text"]

    assert "production blueprint" in prompt
    assert "source scene boundary, order, source timestamp, target duration" in prompt
    assert "Keep original observations distinct from the user's requested changes" in prompt
    assert "Make only the shopper's jacket green." in prompt


def test_reference_analysis_prompt_scopes_changes_and_preserves_observed_footage(tmp_path, monkeypatch):
    from ecom_video.providers import openai as provider

    frame = tmp_path / "frame-0.000s.jpg"
    frame.write_bytes(b"mock-frame")
    result = AnalysisResult.model_validate({"summary": "A shopper discovers a blue umbrella.", "questions": [], "shots": [{
        "index": 1, "start_seconds": 0, "end_seconds": 3, "creative_role": "Discovery",
        "visual_description": "A shopper stands by an umbrella display.", "starting_state": "The umbrella is folded on a shelf.",
        "action": "The shopper lifts the umbrella from the shelf.", "ending_state": "The shopper holds the umbrella.",
        "target_duration_seconds": 3, "edit_duration_seconds": 3, "camera": "Handheld medium shot.",
        "product_interaction": "The shopper holds the handle.", "hook_or_payoff": "Show the umbrella.", "audio_requested": False,
        "audio_direction": "Silent.", "continuity_group": "store", "character_groups": ["shopper"],
        "continuity_notes": ["Same shopper."], "uncertainties": [],
    }]})
    captured = {}
    response_payload = {"output": [{"content": [{"type": "output_text", "text": result.model_dump_json()}]}], "usage": {}}
    monkeypatch.setattr(provider, "require_key", lambda name: "mock-key")
    monkeypatch.setattr(provider.httpx, "post", lambda *args, **kwargs: captured.update(kwargs) or SimpleNamespace(
        raise_for_status=lambda: None, json=lambda: response_payload,
    ))

    provider.analyze_reference([frame], "Change only the umbrella color to blue.", "Confirmed compact umbrella.", hook="wet_works")
    prompt = captured["json"]["input"][0]["content"][0]["text"]

    assert "faithful reference recreation" in prompt
    assert "preserve its scene count, order, approximate timing" in prompt.lower()
    assert "never omit or merge observed source scenes" in prompt.lower()
    assert "Change only the umbrella color to blue." in prompt
    assert "Source-video duration" not in prompt


def test_gifting_reference_analysis_captures_physical_unboxing_without_imposing_story_beats(tmp_path, monkeypatch):
    from ecom_video.providers import openai as provider

    frame = tmp_path / "frame-0.000s.jpg"
    frame.write_bytes(b"mock-frame")
    result = analysis_fixture()
    captured = {}
    payload = {"output": [{"content": [{"type": "output_text", "text": result.model_dump_json()}]}], "usage": {}}
    monkeypatch.setattr(provider, "require_key", lambda name: "mock-key")
    monkeypatch.setattr(provider.httpx, "post", lambda *args, **kwargs: captured.update(kwargs) or SimpleNamespace(
        raise_for_status=lambda: None, json=lambda: payload,
    ))

    provider.analyze_reference(
        [frame], "Replace only the robe with the supplied jacket.", "Use supplied product identity photos.",
        hook="gifting", clip_count=3, seconds_per_clip=5,
    )
    prompt = captured["json"]["input"][0]["content"][0]["text"].lower()

    assert "cannot remove or merge a source scene" in prompt
    assert "sealed/open/partial/empty state" in prompt
    assert "hand/tool placement" in prompt
    assert "source_observation" in prompt and "applied_user_changes" in prompt
    assert "do not infer delight" in prompt
    assert "six-stage framework" in prompt


def test_gifting_keeps_each_source_scene_and_compiles_opening_and_demo_physics():
    analysis = AnalysisResult.model_validate({
        "summary": "A quiet present becomes a useful product demo.", "questions": [], "shots": [
            {"index": 1, "start_seconds": 0, "end_seconds": 2.4, "creative_role": "Unboxing", "visual_description": "Hands open a kraft mailing box on a kitchen table.", "starting_state": "The kraft mailing box is fully sealed; the blue fabric pouch is inside beneath folded paper.", "action": "One hand steadies the box while the other peels its paper seal, folds back both top flaps, lifts the paper insert, and removes the blue pouch by its top seam.", "ending_state": "The open box and removed blue pouch are visible on the same table.", "camera": "Top-down handheld phone view of both hands and the box.", "product_interaction": "Fingers grip and pull the paper seal; the box flaps hinge outward.", "hook_or_payoff": "The pouch is physically revealed.", "source_observation": "Original: unopened kraft box, seal peeled, flaps folded outward, paper insert lifted, blue pouch removed by its top seam.", "applied_user_changes": ["Replace only the pouch with the supplied green version."], "continuity_group": "kitchen", "character_groups": ["hands-a"], "continuity_notes": ["Same hands and table."], "uncertainties": []},
            {"index": 2, "start_seconds": 2.4, "end_seconds": 5.7, "creative_role": "Product demonstration", "visual_description": "The green pouch is clipped to a backpack outdoors.", "starting_state": "The same green pouch is unzipped on a backpack strap; a hand holds the zipper tab.", "action": "The hand pulls the zipper fully closed, then attaches the existing clip to the backpack strap.", "ending_state": "The pouch hangs securely from the strap with zipper closed.", "camera": "Waist-height handheld close shot against the same outdoor path.", "product_interaction": "Pull the zipper tab along its track and fasten the visible clip to the strap.", "hook_or_payoff": "The useful carry function is shown.", "source_observation": "Original: zipper closes, pouch clip fastens to a backpack strap.", "applied_user_changes": ["Replace only the pouch with the supplied green version."], "continuity_group": "outdoor", "character_groups": ["hands-a"], "continuity_notes": ["Same hand identity, green pouch geometry unchanged."], "uncertainties": []},
        ]
    })
    plan = build_multiclip_plan(analysis, "Replace only the pouch color.", "Pouch has a zipper and one metal clip.", "match each source scene", len(analysis.shots), 5, hook="gifting")

    assert len(plan.clips) == 2
    assert [(clip.source_start_seconds, clip.source_end_seconds) for clip in plan.clips] == [(0, 2.4), (2.4, 5.7)]
    opening, demo = plan.clips
    assert "fully sealed" in opening.starting_image_prompt
    assert "Do not show a product already removed" in opening.starting_image_prompt
    assert "peels its paper seal" in compile_video_prompt(plan, opening)
    assert "removed by its top seam" in compile_video_prompt(plan, opening)
    assert "Original: unopened kraft box" in compile_video_prompt(plan, opening)
    assert "Replace only the pouch" in compile_video_prompt(plan, opening)
    assert "outdoors" in demo.starting_image_prompt
    assert "zipper fully closed" in compile_video_prompt(plan, demo)
    assert "hangs securely from the strap" in compile_video_prompt(plan, demo)
    assert "kitchen table" not in demo.starting_image_prompt
    assert len(compile_video_prompt(plan, opening)) <= 3800


def test_gifting_keeps_scene_reference_separate_from_product_identity(tmp_path):
    plan = build_multiclip_plan(
        analysis_fixture(), "Recreate the gift demo.", "Preserve the supplied pouch's exact design.",
        "the analyzed scenes", 3, 5, hook="gifting",
    )
    product = tmp_path / "product.jpg"
    scene = tmp_path / "scene.png"
    product.write_bytes(b"product")
    scene.write_bytes(b"scene")
    kie = FakeKie()

    run_multiclip_generation(
        plan, [product], [scene], tmp_path / "gifting-references", kie,
        images_only=True, continuity_reference_roles=["same-scene composition only"],
    )

    image = next(call for call in kie.calls if call[0] == "image")
    assert image[2][:2] == ["https://files.test/scene.png", "https://files.test/product.jpg"]
    assert "same-scene composition only" in image[1]
    assert "product identity only" in image[1]


def test_gifting_motion_revision_compacts_without_dropping_unboxing_mechanics():
    analysis = AnalysisResult.model_validate({
        "summary": "A gift is opened and used.", "questions": [], "shots": [{
            "index": 1, "start_seconds": 0, "end_seconds": 3.5, "creative_role": "Unboxing",
            "visual_description": "Hands open a small cardboard mailer on a desk.",
            "starting_state": "The cardboard mailer is sealed with the green pouch inside.",
            "action": "One hand steadies the mailer while the other tears the paper tab, folds the top flap back, and lifts the pouch by its side seam.",
            "ending_state": "The open mailer and pouch are visible on the desk.", "camera": "Overhead handheld phone close-up.",
            "product_interaction": "Tear the paper tab and lift the pouch by its side seam.",
            "hook_or_payoff": "A useful pouch is revealed.", "source_observation": "The source shows a physical tab tear, flap opening, then pouch removal.",
            "applied_user_changes": [], "continuity_group": "desk", "character_groups": ["hands"],
            "continuity_notes": ["Same hands, mailer, pouch and desk."], "uncertainties": [],
        }],
    })
    plan = build_multiclip_plan(analysis, "Recreate the source.", "Green fabric pouch with one zipper.", "desk", 1, 5, hook="gifting")
    clip = plan.clips[0].model_copy(update={
        "video_prompt": plan.clips[0].video_prompt + "\nClip-specific revision: Motion-only correction: pause briefly after the flap opens. " +
        " Preserve the supplied packaging and the same opening action." * 120,
    })
    prompt = compile_video_prompt(plan, clip)

    assert len(prompt) <= 3800
    assert "pause briefly after the flap opens" in prompt
    assert "tears the paper tab" in prompt
    assert "lifts the pouch by its side seam" in prompt
    assert "source shows a physical tab tear" in prompt


def test_non_in_store_compiled_prompts_keep_existing_global_summary_behavior():
    analysis = analysis_fixture()
    plan = build_multiclip_plan(analysis, "A product demo", "A product", "home", 3, 5, hook="recommend")
    prompt = compile_video_prompt(plan, plan.clips[0])
    image_prompt = plan.clips[0].starting_image_prompt
    assert "Ad concept: A short product demonstration with a reveal." in prompt
    assert "Follow only this clip's cast" not in prompt
    assert "Keep reactions proportionate to the stimulus" not in prompt
    assert "unretouched skin texture" not in image_prompt
    assert "animation-ready pre-action pose" not in image_prompt


def test_in_store_video_prompt_adds_compact_realism_and_action_density_rules():
    analysis = analysis_fixture()
    plan = build_multiclip_plan(
        analysis, "An In-Store Reaction product demo", "An unchanged teal product",
        "a generic retail aisle", 3, 5, hook="in_store_reaction",
    )

    prompt = compile_video_prompt(plan, plan.clips[0])
    image_prompt = plan.clips[0].starting_image_prompt

    assert "Keep reactions proportionate to the stimulus" in prompt
    assert "Show one primary physical beat" in prompt
    assert "Match reference handheld pace" in prompt
    assert "Preserve product geometry, texture, exposure, and practical lights" in prompt
    assert "natural unretouched skin" in image_prompt
    assert "eyes, hair, and proportions" in image_prompt
    assert "relaxed pre-action pose" in image_prompt
    assert "Match practical light and shadows" in image_prompt
    assert "keep face, hair, age, body scale, wardrobe, skin texture, and scene lighting stable" in prompt
    assert "avoid facial shimmer, beauty smoothing, and identity drift" in prompt
    assert len(image_prompt) < 1_000
    assert len(prompt) < 4_000

    audio_clip = plan.clips[0].model_copy(update={
        "audio_mode": "enabled", "audio_direction": "Natural cup impact and splash.",
    })
    audio_prompt = compile_video_prompt(plan, audio_clip)
    assert "Synchronize each selected native sound to the visible event" in audio_prompt


def test_in_store_opening_prompt_does_not_repeat_hook_payoff_or_audio_instructions():
    analysis = analysis_fixture()
    analysis.shots[0].audio_requested = True
    analysis.shots[0].audio_direction = "Natural store ambience and a brief gasp."
    plan = build_multiclip_plan(
        analysis, "An In-Store Reaction discovery", "An unchanged teal product",
        "a generic retail aisle", 3, 5, hook="in_store_reaction",
    )
    clip = plan.clips[0].model_copy(update={
        "creative_role": "hook / surprise reveal",
        "ending_state": "The shopper and product are both visible.",
        "scene_payoff": None,
    })

    prompt = compile_video_prompt(plan, clip)

    assert prompt.count("Hook direction:") == 1
    assert "Opening hook:" not in prompt
    assert "Product-specific payoff:" not in prompt
    assert prompt.count("Synchronize each selected native sound") == 1
    assert len(prompt) < 3_600


def test_in_store_reaction_source_controls_gaze_expression_and_camera():
    from ecom_video.creative_playbook import IN_STORE_REACTION_PLAYBOOK

    analysis = analysis_fixture()
    plan = build_multiclip_plan(
        analysis, "An In-Store Reaction discovery", "An unchanged teal product",
        "a generic retail aisle", 3, 5, hook="in_store_reaction",
    )
    clip = plan.clips[0].model_copy(update={"creative_role": "hook / surprise reveal"})

    prompt = compile_video_prompt(plan, clip)

    principle = "Align eye line, expression, and camera movement with the stated reaction source; keep attention on it through incidental actions."
    assert principle in IN_STORE_REACTION_PLAYBOOK
    assert "Hook direction: reproduce only the observed reaction" in prompt
    assert "Adjust the kitten in the pouch" in prompt


def test_in_store_starting_image_stays_natural_and_reaction_needs_no_coffee():
    data = analysis_fixture().model_dump()
    data["shots"][0].update({
        "creative_role": "product discovery hook",
        "visual_description": "A shopper walks through a camping store and notices a compact lantern display to the left.",
        "starting_state": "At frame zero, the shopper is walking naturally with relaxed shoulders; the lantern display is just outside the left edge.",
        "action": "The shopper notices the lantern, looks directly toward it, and gives a small genuine smile. The handheld camera follows the shopper's gaze left to reveal the lantern display.",
        "ending_state": "The lantern display is clear in frame as the shopper continues looking at it.",
        "camera": "Casual chest-height phone shot; pan left with the shopper's gaze to reveal the display.",
        "product_interaction": "No touch; the shopper points once toward the lantern.",
        "hook_or_payoff": "A clear discovery of the lantern display.",
    })
    plan = build_multiclip_plan(
        AnalysisResult.model_validate(data), "An authentic In-Store Reaction discovery",
        "A compact camping lantern; match its supplied reference and do not invent features.",
        "A practical camping store aisle", 1, 5, hook="in_store_reaction",
    )

    image_prompt = plan.clips[0].starting_image_prompt
    video_prompt = compile_video_prompt(plan, plan.clips[0])

    assert "relaxed shoulders" in image_prompt
    assert "natural unretouched skin" in image_prompt
    assert "Frame zero" in image_prompt
    assert "lantern" in video_prompt and "left" in video_prompt
    assert "Hook direction: reproduce only the observed reaction" in video_prompt
    assert "follows the shopper's gaze left" in video_prompt
    assert "camera follows the shopper's gaze left" in video_prompt
    assert "coffee" not in image_prompt.lower() + video_prompt.lower()
    assert "cup" not in image_prompt.lower() + video_prompt.lower()


def test_in_store_video_uses_start_frame_then_product_identity_reference(tmp_path, monkeypatch):
    data = analysis_fixture().model_dump()
    data["shots"][0].update({
        "creative_role": "product reaction and reveal hook",
        "visual_description": "A shopper walks past shelves and notices a lantern display just off-screen.",
        "starting_state": "At frame zero, the shopper walks naturally; the lantern display is outside the frame.",
        "action": "The shopper reacts to the lantern display and the camera pans to reveal it.",
        "ending_state": "The exact lantern from the product reference is clearly visible on the display.",
        "camera": "Handheld medium shot, following the shopper's gaze toward the lantern display.",
        "product_interaction": "The shopper does not touch the lantern.",
        "hook_or_payoff": "Reveal the lantern display.",
    })
    product = tmp_path / "lantern-reference.png"
    product.write_bytes(b"product")
    plan = build_multiclip_plan(
        AnalysisResult.model_validate(data), "An In-Store Reaction product discovery",
        "A compact camping lantern; preserve its supplied geometry and color.",
        "A practical camping store", 1, 5, product_references=[product], hook="in_store_reaction",
    )
    clip = plan.clips[0]
    clip = clip.model_copy(update={"audio_mode": "enabled", "audio_direction": "Natural store ambience and a brief gasp."})
    plan = plan.model_copy(update={"clips": [clip]})
    source = tmp_path / "starting-frame.png"
    source.write_bytes(b"starting frame")
    client = FakeKie()

    def fake_audio_export(src: Path, dst: Path, *, preserve_audio: bool = True) -> Path:
        dst.write_bytes(src.read_bytes())
        return dst
    monkeypatch.setattr("ecom_video.workflow.make_mp4", fake_audio_export)

    run_multiclip_generation(
        plan, [product], [], tmp_path / "video-reference-dry-run", client,
        starting_images={1: source},
    )

    video_call = next(call for call in client.calls if call[0] == "video")
    assert video_call[4]["reference_image_urls"] == ["https://files.test/lantern-reference.png"]
    assert "@image1" in video_call[1] and "@image2" in video_call[1]
    assert "@image2=product only" in video_call[1].lower()
    assert "use the reference video frames" not in video_call[1].lower()
    assert "Synchronize each selected native sound" in video_call[1]
    assert video_call[4]["enable_audio"] is None
    manifest = json.loads((tmp_path / "video-reference-dry-run" / "generation-usage.json").read_text())
    request = manifest["tasks"][0]["request"]
    assert [reference["role"] for reference in request["references"]] == [
        "exact starting frame; sets frame-zero scene and cast",
        "product identity only; ignore background",
    ]


def test_in_store_payoff_preserves_bottom_release_mechanism_without_manual_transfer():
    analysis = AnalysisResult.model_validate({
        "summary": "A shopper discovers a basketball laundry hamper, then sees its bottom-release feature empty it into a basket.",
        "questions": [],
        "shots": [{
            "index": 1, "start_seconds": 8, "end_seconds": 11, "creative_role": "Functional payoff",
            "visual_description": "The loaded hamper hangs above a laundry basket.",
            "starting_state": "A release panel closes the hamper bottom; a basket is centered below it.",
            "action": "Pull the product's release tab once so the bottom opens and gravity drops the laundry into the basket.",
            "ending_state": "Laundry lands in the basket and the opened hamper hangs mostly empty.",
            "camera": "Handheld close-up follows the release tab down to the basket, then tilts back to show the open bottom.",
            "product_interaction": "The built-in bottom release opens under the pull; gravity drops clothing directly into the basket. Do not manually transfer the laundry.",
            "hook_or_payoff": "A satisfying hands-free emptying payoff.", "continuity_group": "hamper_demo",
            "continuity_notes": ["Keep the same hamper, release mechanism, and basket."],
            "target_duration_seconds": 4, "edit_duration_seconds": 2.4, "uncertainties": [],
        }],
    })
    plan = build_multiclip_plan(analysis, "In-Store Reaction product demo", "Basketball laundry hamper with bottom-release feature.", "laundry room", 1, 5, hook="in_store_reaction")
    prompt = compile_video_prompt(plan, plan.clips[0])
    assert "Pull the product's release tab once" in prompt
    assert "gravity drops clothing directly into the basket" in prompt
    assert "Do not manually transfer the laundry" in prompt
    assert "satisfying hands-free emptying payoff" in prompt
    assert "4s" in prompt and "2.4s" in prompt


def test_plan_rejects_more_clips_than_distinct_analyzed_shots():
    with pytest.raises(ValueError, match="distinct analyzed shots"):
        build_multiclip_plan(
            analysis_fixture(), "creative direction", "product details",
            "living room", clip_count=4, seconds_per_clip=5,
        )


def test_forty_thousand_character_direction_stays_out_of_compacted_video_prompt():
    instructions = "x" * 40_000
    plan = build_multiclip_plan(
        analysis_fixture(), instructions, "A fixed product", "the analyzed scene", 1, 5, hook="custom",
    )

    assert plan.instructions == instructions
    prompt = compile_video_prompt(plan, plan.clips[0])
    assert len(prompt) <= 3_800
    assert instructions not in prompt


def test_original_null_source_segments_get_descriptive_titles():
    analysis = analysis_fixture()
    for shot in analysis.shots:
        shot.start_seconds = shot.end_seconds = None
    plan = build_multiclip_plan(analysis, "original concept", "product", "specific scenes", 3, 5)
    assert all(clip.source_start_seconds is None and clip.source_end_seconds is None for clip in plan.clips)
    assert all("reference segment" not in clip.title for clip in plan.clips)


def test_source_video_intervals_remain_distinct_from_generated_clip_duration():
    plan = build_multiclip_plan(
        analysis_fixture(), "Creative direction", "Product details", "Living room", 3, 5,
    )

    assert [(clip.source_start_seconds, clip.source_end_seconds) for clip in plan.clips] == [
        (0.0, 2.0), (2.0, 4.0), (4.0, 6.0),
    ]
    assert [clip.duration_seconds for clip in plan.clips] == [5, 5, 5]
    assert "0.0-2.0s" in plan.clips[0].title
    assert "4.0-6.0s" in plan.clips[2].video_prompt


def test_creative_direction_only_shot_has_no_invented_source_timestamps():
    analysis = analysis_fixture()
    analysis.shots[1].start_seconds = None
    analysis.shots[1].end_seconds = None
    plan = build_multiclip_plan(
        analysis, "Add a new indoor reaction shot", "Product details", "Living room", 3, 5,
    )

    clip = plan.clips[1]
    assert clip.source_start_seconds is None
    assert clip.source_end_seconds is None
    assert "creative direction" in clip.title.lower()
    assert "no exact reference segment" in clip.video_prompt.lower()
    assert "target clip duration: 5 seconds" in clip.video_prompt.lower()


def test_repeated_or_out_of_bounds_source_ranges_are_unmapped_instead_of_reused():
    analysis = AnalysisResult.model_validate({
        "summary": "Three scenes.", "questions": [], "shots": [
            {**shot.model_dump(), "start_seconds": 0, "end_seconds": 5}
            for shot in analysis_fixture().shots
        ],
    })

    normalized = normalize_reference_segments(analysis, reference_duration_seconds=11.84)
    ranges = [(shot.start_seconds, shot.end_seconds) for shot in normalized.shots]
    assert ranges[0] == (0.0, 5.0)
    assert ranges[1:] == [(None, None), (None, None)]
    assert len({pair for pair in ranges if pair[0] is not None}) == 1


def test_multiclip_estimate_counts_one_image_and_video_per_clip():
    plan = build_multiclip_plan(
        analysis_fixture(), "creative direction", "product details",
        "living room", clip_count=3, seconds_per_clip=5,
    )

    estimate = estimate_multiclip_generation(plan)
    assert estimate.image_count == 3
    assert estimate.generated_seconds == 15
    assert estimate.total_usd == Decimal("0.4275")


def test_long_brief_is_retained_in_plan_but_generation_prompts_are_scene_specific():
    shared = "Shared requirements: preserve the same product design, woman, kitten, and living-room light."
    instructions = shared + "\n\n" + "DETAILED_CREATIVE_NOTE_DO_NOT_DROP. " * 500
    analysis = AnalysisResult.model_validate({
        "summary": shared,
        "questions": [],
        "shots": [{
            "index": index, "start_seconds": index - 1, "end_seconds": index,
            "visual_description": f"Target composition for shot {index}.",
            "action": f"Target-only action {index}: adjust the garment naturally.",
            "camera": f"Handheld framing {index}.",
            "product_interaction": "Maintain believable physical contact and fabric folds.",
            "hook_or_payoff": f"Reveal garment detail {index}.", "uncertainties": [],
        } for index in range(1, 4)],
    })

    plan = build_multiclip_plan(
        analysis, instructions, "Preserve the product shown in reference images.",
        "a naturally lit living room", 3, 5,
    )

    assert plan.instructions == instructions
    for index, clip in enumerate(plan.clips, start=1):
        assert shared not in clip.starting_image_prompt and shared in clip.video_prompt
        assert f"Target-only action {index}" not in clip.starting_image_prompt
        assert f"Target-only action {index}" in clip.video_prompt
        assert "Frame zero:" in clip.starting_image_prompt
        assert "DETAILED_CREATIVE_NOTE_DO_NOT_DROP" not in clip.starting_image_prompt
        assert "DETAILED_CREATIVE_NOTE_DO_NOT_DROP" not in clip.video_prompt
        assert len(clip.starting_image_prompt) < 4_000
        assert len(clip.video_prompt) < 4_000


def test_scene_prompts_encode_role_start_action_payoff_camera_and_product_constraints():
    data = analysis_fixture().model_dump()
    data["shots"][0].update({
        "creative_role": "Hook: discover the gecko toy on the store shelf.",
        "starting_state": "The woman notices the boxed gecko toy at eye level and reaches toward it.",
        "ending_state": "She holds the box toward camera with a surprised smile.",
        "continuity_group": "store-hook",
        "continuity_notes": ["Keep the same boxed toy and woman appearance during this shot."],
    })
    plan = build_multiclip_plan(
        AnalysisResult.model_validate(data), "Show a playful product story", "Preserve the toy design",
        "toy store", 1, 5,
    )
    clip = plan.clips[0]
    assert "Frame zero:" in clip.starting_image_prompt
    assert "Concrete action:" not in clip.starting_image_prompt
    assert "Ending state or payoff:" not in clip.starting_image_prompt
    assert "Creative role:" in clip.video_prompt
    assert "Starting state:" in clip.video_prompt
    assert "Concrete action:" in clip.video_prompt
    assert "Ending state or payoff:" in clip.video_prompt
    assert "Product: Preserve the toy design" in clip.starting_image_prompt
    assert "Keep the product design accurate" in clip.video_prompt
    assert "Hook: discover the gecko toy" in clip.video_prompt
    assert "She holds the box toward camera" in clip.video_prompt
    assert "Avoid a mostly static product pose" in clip.video_prompt


def test_in_store_reaction_playbook_is_applied_to_hook_video_and_preflight_catches_state_conflicts():
    data = analysis_fixture().model_dump()
    data["shots"][0].update({
        "creative_role": "reaction / product reveal",
        "starting_state": "The shopper reacts toward an off-camera product; the product is not visible yet.",
        "action": "The camera quickly reframes to reveal the product.",
    })
    plan = build_multiclip_plan(
        AnalysisResult.model_validate(data), "Show a store reaction", "Preserve product appearance",
        "warehouse retail aisle", 3, 5, hook="in_store_reaction",
    )
    hook_video = compile_video_prompt(plan, plan.clips[0])
    assert "reproduce only the observed reaction" in hook_video
    assert "match reference handheld pace" in hook_video.lower()
    assert "preserve their order" in hook_video
    assert len(hook_video) <= 3_800
    assert "camera quickly reframes" in hook_video
    assert "camera quickly reframes" not in plan.clips[0].starting_image_prompt
    assert "warehouse retail aisle" in plan.clips[0].starting_image_prompt
    assert len(plan.clips[0].starting_image_prompt) < 1_000
    validate_starting_image_state(plan)

    coffee_drop = plan.clips[0].model_copy(update={
        "starting_state": "The wife holds her intact coffee cup upright.",
        "action": "The wife loses grip; the cup falls and coffee splashes onto the floor.",
        "starting_image_prompt": "Frame zero: the wife holds the cup securely upright.",
    })
    validate_starting_image_state(plan.model_copy(update={"clips": [coffee_drop]}))

    dry_then_splash = coffee_drop.model_copy(update={
        "starting_state": "The floor is dry; the cup is still in the wife's hand.",
        "action": "The cup hits the floor and coffee splashes.",
    })
    validate_starting_image_state(plan.model_copy(update={"clips": [dry_then_splash]}))

    dark_then_lit = coffee_drop.model_copy(update={
        "starting_state": "The product is unlit; its village lights are off.",
        "action": "He presses the remote and the village lights illuminate.",
    })
    validate_starting_image_state(plan.model_copy(update={"clips": [dark_then_lit]}))

    closed_then_open = coffee_drop.model_copy(update={
        "starting_state": "The retail box is closed and unopened.",
        "action": "The husband opens the box and removes the product.",
    })
    validate_starting_image_state(plan.model_copy(update={"clips": [closed_then_open]}))

    contradictory = coffee_drop.model_copy(update={
        "starting_state": "The wife holds her coffee upright while the cup is falling and coffee is splashing.",
    })
    with pytest.raises(ValueError, match="intact held drink conflicts"):
        validate_starting_image_state(plan.model_copy(update={"clips": [contradictory]}))

    negated = coffee_drop.model_copy(update={
        "starting_state": "The wife holds her coffee upright; no falling cup, no spill, and no splash.",
        "starting_image_prompt": "Frame zero: coffee held upright. Do not show a falling cup or spilled coffee.",
    })
    validate_starting_image_state(plan.model_copy(update={"clips": [negated]}))

    bad_environment = plan.clips[0].model_copy(update={
        "starting_state": "The shopper reacts in a warehouse retail aisle.",
        "starting_image_prompt": "Frame zero: the shopper stands against a white studio backdrop.",
    })
    bad_environment_plan = plan.model_copy(update={"clips": [bad_environment, *plan.clips[1:]]})
    with pytest.raises(ValueError, match="retail setting conflicts"):
        validate_starting_image_state(bad_environment_plan)


def test_successive_same_setting_clips_receive_shared_continuity_context_and_image(tmp_path):
    data = analysis_fixture().model_dump()
    data["shots"][0].update({"continuity_group": "store"})
    data["shots"][1].update({
        "continuity_group": "home-gecko-cat",
        "continuity_notes": ["Same living room, same gray tabby cat, same green-and-black gecko toy."],
    })
    data["shots"][2].update({
        "continuity_group": "home-gecko-cat",
        "continuity_notes": ["Continue with that same gray tabby cat, living room, and gecko toy."],
    })
    plan = build_multiclip_plan(
        AnalysisResult.model_validate(data), "Show the product story", "Keep the toy unchanged",
        "the scene described per shot", 3, 5,
    )
    assert "Same living room, same gray tabby cat" in plan.clips[1].video_prompt
    assert "same gray tabby cat" in plan.clips[2].video_prompt
    assert "home-gecko-cat" in plan.clips[2].video_prompt
    assert "Same living room, same gray tabby cat" in plan.clips[1].starting_image_prompt
    assert "Continue with that same gray tabby cat" in plan.clips[2].starting_image_prompt
    assert "Related shot 2:" not in plan.clips[1].starting_image_prompt

    product = tmp_path / "product-reference.jpg"
    continuity = tmp_path / "approved-continuity.jpg"
    product.write_bytes(b"product")
    continuity.write_bytes(b"continuity")
    kie = FakeKie()
    def fake_silent(source: Path, destination: Path) -> Path:
        destination.write_bytes(source.read_bytes())
        return destination
    from unittest.mock import patch
    with patch("ecom_video.workflow.make_silent_mp4", fake_silent):
        run_multiclip_generation(plan, [product], [continuity], tmp_path / "outputs", kie)
    image_calls = [call for call in kie.calls if call[0] == "image"]
    assert len(image_calls) == 3
    assert len(image_calls[0][2]) == 2
    assert len(image_calls[1][2]) == 3
    assert any("clip-02-" in url for url in image_calls[2][2])
    assert "continuity references define faces, wardrobe, location, and pose" in image_calls[2][1]


class FakeKie:
    def __init__(self):
        self.calls: list[tuple] = []

    def upload(self, path: Path) -> str:
        self.calls.append(("upload", path.name))
        return f"https://files.test/{path.name}"

    def create_image(self, prompt: str, input_urls: list[str], aspect_ratio: str = "9:16") -> str:
        self.calls.append(("image", prompt, input_urls, aspect_ratio))
        return f"image-task-{sum(call[0] == 'image' for call in self.calls)}"

    def create_text_image(self, prompt: str, aspect_ratio: str = "9:16") -> str:
        self.calls.append(("text_image", prompt, aspect_ratio))
        return f"image-task-{sum(call[0] in {'image', 'text_image'} for call in self.calls)}"

    def create_video(self, prompt: str, image_url: str, duration_seconds: int, **kwargs) -> str:
        self.calls.append(("video", prompt, image_url, duration_seconds, kwargs))
        return f"video-task-{sum(call[0] == 'video' for call in self.calls)}"

    def wait_for_task(self, task_id: str, on_state=None) -> dict:
        self.calls.append(("poll", task_id))
        if on_state:
            on_state("generating")
        credits = 6 if task_id.startswith("image") else 22.5
        return {
            "state": "success",
            "creditsConsumed": credits,
            "resultJson": json.dumps({"resultUrls": [f"https://files.test/{task_id}.mp4"]}),
        }

    def download_result(self, task_data: dict, destination: Path) -> Path:
        self.calls.append(("download", destination.name))
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"fixture")
        return destination


def test_non_in_store_images_keep_product_first_order_and_all_supplied_references(tmp_path):
    plan = build_multiclip_plan(
        analysis_fixture(), "A product story", "Preserve supplied product appearance",
        "warehouse retail aisle", 1, 5, hook="recommend",
    )
    product = tmp_path / "product-reference.jpg"
    clip2 = tmp_path / "clip-02-approved-warehouse.png"
    clip1 = tmp_path / "clip-01-coffee-pose.png"
    for path in (product, clip2, clip1):
        path.write_bytes(path.name.encode())
    prompt = (
        "Warehouse retail aisle with tall stocked shelves, overhead lights, and a visible concrete floor. "
        "Reference roles: use the approved Clip 2 image as the primary character and warehouse environment reference. "
        "Use the previous Clip 1 image only for the intact coffee-holding pose; ignore its white background."
    )
    clip = plan.clips[0].model_copy(update={
        "starting_image_prompt": prompt,
        "starting_state": "The couple reacts in the warehouse aisle; the wife holds coffee upright.",
    })
    plan = plan.model_copy(update={"clips": [clip]})
    kie = FakeKie()

    run_multiclip_generation(
        plan, [product], [clip2, clip1], tmp_path / "provider-request", kie,
        approved_max_usd=Decimal("0.50"), images_only=True,
    )

    request = next(call for call in kie.calls if call[0] == "image")
    assert request[2] == [
        "https://files.test/product-reference.jpg",
        "https://files.test/clip-02-approved-warehouse.png",
        "https://files.test/clip-01-coffee-pose.png",
    ]
    assert "Reference roles: use the approved Clip 2 image" in request[1]
    assert "continuity references define faces, wardrobe, location, and pose" in request[1]


def test_hidden_product_anchor_check_ignores_negative_prompt_constraints():
    from ecom_video.workflow import _product_hidden_at_start

    assert _product_hidden_at_start("No product visible at frame zero.")
    assert not _product_hidden_at_start("Do not put the product outside the opening frame.")


def test_in_store_dry_run_uses_scene_reference_without_off_frame_product_reference(tmp_path):
    data = analysis_fixture().model_dump()
    data["summary"] = "A shopper discovers an unexpected product in a retail store."
    data["shots"][0].update({
        "creative_role": "Immediate reaction hook",
        "visual_description": "A shopper in a stocked retail aisle notices a display off-camera.",
        "starting_state": "At frame zero, the shopper looks toward the display; the product is mostly outside the opening frame.",
        "action": "The shopper reacts and the handheld camera pans toward the display.",
        "continuity_group": "store-shoppers",
        "continuity_notes": ["Same shopper and stocked retail aisle."],
    })
    data["shots"][1].update({
        "creative_role": "Product discovery",
        "visual_description": "The same shopper examines the product display in the same retail aisle.",
        "starting_state": "At frame zero, the product is visible on the display.",
        "continuity_group": "store-shoppers",
        "continuity_notes": ["Same shopper and stocked retail aisle."],
    })
    product = tmp_path / "product-reference.jpg"
    continuity = tmp_path / "approved-scene.png"
    product.write_bytes(b"product")
    continuity.write_bytes(b"scene")
    plan = build_multiclip_plan(
        AnalysisResult.model_validate(data), "Show an authentic retail discovery", "A compact blue product",
        "a stocked retail aisle with overhead lights and visible depth", 2, 5,
        product_references=[product], continuity_references=[continuity], hook="in_store_reaction",
    )
    plan = plan.model_copy(update={"clips": [
        plan.clips[0].model_copy(update={"starting_image_prompt": plan.clips[0].starting_image_prompt + "\nRequested revision: Keep the opening cup upright."}),
        *plan.clips[1:],
    ]})
    assert "stocked retail aisle" in plan.clips[0].starting_image_prompt
    assert "frame zero" in plan.clips[0].starting_image_prompt.lower()
    assert "off-camera" in plan.clips[0].starting_image_prompt
    assert "pans toward" not in plan.clips[0].starting_image_prompt.lower()
    assert "pans toward" in plan.clips[0].action

    kie = FakeKie()
    run_multiclip_generation(
        plan, [product], [continuity], tmp_path / "dry-run", kie, images_only=True,
        continuity_reference_roles=["same-scene faces/clothes/location; use background"],
    )
    calls = [call for call in kie.calls if call[0] == "image"]
    assert len(calls) == 2
    assert calls[0][2] == ["https://files.test/approved-scene.png"]
    assert "product-reference.jpg" not in calls[0][1]
    assert "approved-scene.png = same-scene faces/clothes/location; use background" in calls[0][1]
    assert all("clip-01-" not in url for url in calls[0][2])
    assert any("clip-01-" in url for url in calls[1][2])
    manifest = json.loads((tmp_path / "dry-run" / "generation-usage.json").read_text())
    first_request = next(task["request"] for task in manifest["tasks"] if task["clip_index"] == 1)
    assert first_request["prompt"] == calls[0][1]
    assert first_request["references"] == [{"filename": "approved-scene.png", "role": "same-scene faces/clothes/location; use background"}]
    assert first_request["revision_note"] == "Keep the opening cup upright."


def test_in_store_visible_product_keeps_product_reference_after_scene_reference(tmp_path):
    analysis = analysis_fixture()
    plan = build_multiclip_plan(
        analysis, "Show a retail product discovery", "Preserve the product geometry",
        "a stocked warehouse aisle", 1, 5, hook="in_store_reaction",
    )
    clip = plan.clips[0].model_copy(update={
        "starting_state": "At frame zero, the product is visible on the shelf in the warehouse aisle.",
        "starting_image_prompt": "A warehouse aisle. The product is visible on the shelf.",
    })
    plan = plan.model_copy(update={"clips": [clip]})
    product = tmp_path / "product-reference.jpg"
    scene = tmp_path / "approved-warehouse.png"
    product.write_bytes(b"product")
    scene.write_bytes(b"scene")
    kie = FakeKie()

    run_multiclip_generation(
        plan, [product], [scene], tmp_path / "visible-product", kie,
        images_only=True,
    )

    request = next(call for call in kie.calls if call[0] == "image")
    assert request[2] == [
        "https://files.test/approved-warehouse.png",
        "https://files.test/product-reference.jpg",
    ]


@pytest.mark.parametrize("hook", ["countdown_reveal", "gifting", "versus"])
def test_other_hooks_keep_their_own_direction(hook):
    plan = build_multiclip_plan(analysis_fixture(), "Show the product story", "Verified product facts", "a home", 3, 5, hook=hook)
    prompt = compile_video_prompt(plan, plan.clips[0])
    assert "Opening hook: keep immediate reaction" not in prompt
    assert "In-Store Reaction ad:" not in prompt


@pytest.mark.parametrize("hook", ["versus"])
def test_other_formats_remain_runnable_without_reaction_specific_image_order(tmp_path, hook):
    data = analysis_fixture().model_dump()
    data["shots"][0]["creative_role"] = "Opening setup"
    data["shots"][1]["creative_role"] = "Product reveal"
    data["shots"][1]["starting_state"] = "The product is visible in frame."
    data["shots"][2]["creative_role"] = "Product demonstration"
    plan = build_multiclip_plan(
        AnalysisResult.model_validate(data), "Show distinct product beats", "Verified product details", "a home", 3, 5, hook=hook,
    )
    kie = FakeKie()
    output = tmp_path / hook
    run_multiclip_generation(plan, [], [], output, kie, images_only=True)
    manifest = json.loads((output / "generation-usage.json").read_text())
    assert [task["clip_index"] for task in manifest["tasks"]] == [2, 1, 3]
    assert not any("reaction-to-reveal" in call[1].lower() for call in kie.calls if call[0] in {"image", "text_image"})


def test_generation_submits_each_clip_separately_and_reuses_references(tmp_path, monkeypatch):
    plan = build_multiclip_plan(
        analysis_fixture(), "creative direction", "product details",
        "living room", clip_count=3, seconds_per_clip=5,
    )
    product = tmp_path / "product.jpg"
    continuity = tmp_path / "approved-frame.jpg"
    product.write_bytes(b"product")
    continuity.write_bytes(b"continuity")
    kie = FakeKie()

    def fake_silent(source: Path, destination: Path) -> Path:
        destination.write_bytes(source.read_bytes())
        return destination

    monkeypatch.setattr("ecom_video.workflow.make_silent_mp4", fake_silent)
    outputs = run_multiclip_generation(
        plan,
        [product],
        [continuity],
        tmp_path / "clips",
        kie,
    )

    assert len(outputs) == 3
    assert all(path.is_file() for path in outputs)
    assert sum(call[0] == "image" for call in kie.calls) == 3
    assert sum(call[0] == "video" for call in kie.calls) == 3
    image_calls = [call for call in kie.calls if call[0] == "image"]
    assert all(len(call[2]) in {2, 3} for call in image_calls)
    video_calls = [call for call in kie.calls if call[0] == "video"]
    assert [call[3] for call in video_calls] == [5, 5, 5]
    assert all(call[4]["resolution"] == "720p" for call in video_calls)
    assert all(call[4]["aspect_ratio"] == "9:16" for call in video_calls)
    usage = json.loads((tmp_path / "clips" / "generation-usage.json").read_text())
    assert usage["all_task_charges_reported"] is True
    assert usage["total_credits_consumed"] == "85.5"
    assert usage["total_usd_equivalent"] == "0.4275"


@pytest.mark.parametrize("audio_clip_index", [1, 2, 3, 4, 5])
def test_native_audio_can_be_enabled_on_any_clip_number(audio_clip_index, tmp_path, monkeypatch):
    base = build_multiclip_plan(
        analysis_fixture(), "direction", "product facts", "room", clip_count=3, seconds_per_clip=5,
    )
    clips = [
        base.clips[index % len(base.clips)].model_copy(update={
            "index": index + 1,
            "audio_mode": "enabled" if index == audio_clip_index - 1 else "silent",
            "audio_direction": "Synchronized physical sounds only; no music." if index == audio_clip_index - 1 else "Silent.",
        })
        for index in range(5)
    ]
    plan = base.model_copy(update={"clips": clips})
    product = tmp_path / "product.jpg"
    product.write_bytes(b"product")
    kie = FakeKie()
    preserved, silent = [], []

    def fake_export(source: Path, destination: Path, *, preserve_audio: bool = False) -> Path:
        (preserved if preserve_audio else silent).append(destination.name)
        destination.write_bytes(source.read_bytes())
        return destination

    monkeypatch.setattr("ecom_video.workflow.make_mp4", fake_export)
    monkeypatch.setattr("ecom_video.workflow.make_silent_mp4", lambda src, dst: fake_export(src, dst))
    outputs = run_multiclip_generation(
        plan, [product], [], tmp_path / "five-clips", kie, approved_max_usd=Decimal("1.00"),
    )

    video_calls = [call for call in kie.calls if call[0] == "video"]
    assert [call[4]["enable_audio"] for call in video_calls] == [None if index == audio_clip_index - 1 else False for index in range(5)]
    assert len(outputs) == 5
    assert len(preserved) == 1
    assert len(silent) == 4


def test_image_review_phase_prepares_every_image_then_reuses_them_for_videos(tmp_path, monkeypatch):
    plan = build_multiclip_plan(analysis_fixture(), "direction", "product facts", "room", clip_count=3, seconds_per_clip=5)
    product = tmp_path / "product.jpg"; product.write_bytes(b"product")
    output_dir = tmp_path / "review"; kie = FakeKie()
    monkeypatch.setattr("ecom_video.workflow.make_silent_mp4", lambda src, dst: (dst.write_bytes(src.read_bytes()), dst)[1])

    starting_images = run_multiclip_generation(plan, [product], [], output_dir, kie, images_only=True)

    assert len(starting_images) == 3
    assert all(path.is_file() for path in starting_images)
    assert not [call for call in kie.calls if call[0] == "video"]
    saved_bytes = [path.read_bytes() for path in starting_images]
    outputs = run_multiclip_generation(
        plan, [product], [], output_dir, kie, resume=True,
        starting_images={clip.index: path for clip, path in zip(plan.clips, starting_images)},
    )

    assert len(outputs) == 3
    assert len([call for call in kie.calls if call[0] == "image"]) == 3
    assert len([call for call in kie.calls if call[0] == "video"]) == 3
    assert [path.read_bytes() for path in starting_images] == saved_bytes


def test_reviewed_starting_images_require_approval_for_video_only_estimate(tmp_path):
    from ecom_video.workflow import run_multiclip_generation

    base = build_multiclip_plan(analysis_fixture(), "direction", "product facts", "room", clip_count=1, seconds_per_clip=5)
    clips = [base.clips[0].model_copy(update={"index": index, "duration_seconds": 5}) for index in range(1, 6)]
    plan = base.model_copy(update={"clips": clips})
    image = tmp_path / "approved-start.png"
    image.write_bytes(b"image")
    client = FakeKie()

    with pytest.raises(PermissionError, match=r"estimate \$0\.5625 exceeds \$0\.50"):
        run_multiclip_generation(
            plan, [], [], tmp_path / "reviewed", client,
            starting_images={index: image for index in range(1, 6)}, resume=True, analysis_usd=Decimal("0.0112"),
        )

    assert not any(call[0] == "video" for call in client.calls)


def test_image_only_preflight_skips_oversized_video_prompt_and_keeps_request_compact(tmp_path):
    plan = build_multiclip_plan(
        analysis_fixture(), "direction", "product facts", "room", clip_count=1, seconds_per_clip=5,
    ).model_copy(update={"summary": "long video-only context " * 300})
    kie = FakeKie()

    images = run_multiclip_generation(plan, [], [], tmp_path / "compact-image", kie, images_only=True)

    image_call = next(call for call in kie.calls if call[0] == "text_image")
    assert len(image_call[1]) < 1_000
    assert len(images) == 1 and images[0].is_file()
    assert not any(call[0] == "video" for call in kie.calls)


def test_resume_retries_only_a_terminally_failed_starting_image_and_preserves_attempt_history(tmp_path):
    from ecom_video.workflow import _slug

    plan = build_multiclip_plan(analysis_fixture(), "direction", "product facts", "room", clip_count=3, seconds_per_clip=5)
    output_dir = tmp_path / "retry-images"
    output_dir.mkdir()
    for clip in plan.clips[:2]:
        (output_dir / f"clip-{clip.index:02d}-{_slug(clip.action)}-starting.png").write_bytes(f"saved-{clip.index}".encode())
    failed = {"clip_index": 3, "kind": "starting_image", "task_id": "failed-image-3", "credits_consumed": "0.0", "status": "failed", "provider_state": "fail"}
    (output_dir / "generation-usage.json").write_text(json.dumps({
        "expected_generation_tasks": 6,
        "tasks": [
            {"clip_index": 1, "kind": "starting_image", "task_id": "image-1", "credits_consumed": "6", "status": "success"},
            {"clip_index": 2, "kind": "starting_image", "task_id": "image-2", "credits_consumed": "6", "status": "success"},
            failed,
        ],
    }))
    kie = FakeKie()

    images = run_multiclip_generation(plan, [], [], output_dir, kie, resume=True, images_only=True)

    assert len(images) == 3 and all(image.is_file() for image in images)
    assert sum(call[0] in {"image", "text_image"} for call in kie.calls) == 1
    assert not any(call[0] == "video" for call in kie.calls)
    assert (output_dir / f"clip-01-{_slug(plan.clips[0].action)}-starting.png").read_bytes() == b"saved-1"
    manifest = json.loads((output_dir / "generation-usage.json").read_text())
    assert any(task["task_id"] == "failed-image-3" and task["status"] == "failed" for task in manifest["tasks"])
    retry = next(task for task in manifest["tasks"] if task["clip_index"] == 3 and task["status"] == "success")
    assert retry["task_id"] != "failed-image-3" and retry["credits_consumed"] == "6"


def test_provider_failure_details_are_saved_in_the_usage_manifest(tmp_path):
    from ecom_video.providers.kie import KieTaskFailed

    class FailingKie(FakeKie):
        def wait_for_task(self, task_id: str, on_state=None) -> dict:
            raise KieTaskFailed(task_id, {
                "state": "fail", "failCode": "INTERNAL", "failMsg": "Temporary provider error",
            })

    plan = build_multiclip_plan(analysis_fixture(), "direction", "product facts", "room", clip_count=1, seconds_per_clip=5)
    output_dir = tmp_path / "failed-image"
    with pytest.raises(KieTaskFailed, match="image-task-1.*INTERNAL"):
        run_multiclip_generation(plan, [], [], output_dir, FailingKie(), images_only=True)

    manifest = json.loads((output_dir / "generation-usage.json").read_text())
    task = manifest["tasks"][0]
    assert task["task_id"] == "image-task-1"
    assert task["provider_failure"] == {"code": "INTERNAL", "message": "Temporary provider error"}


def test_image_review_seeds_product_identity_across_different_scene_groups(tmp_path):
    plan = build_multiclip_plan(analysis_fixture(), "A dog bowl ad", "Forest-green collapsible bowl", "varied scenes", 3, 5)
    roles = ["Hook: dog waits before the bowl is revealed", "Product reveal", "Water-fill demonstration"]
    groups = ["trail-hook", "home-reveal", "forest-demo"]
    clips = []
    for clip, role, group in zip(plan.clips, roles, groups):
        clips.append(clip.model_copy(update={
            "creative_role": role,
            "continuity_group": group,
            "visual_description": "The bowl stays hidden until the reveal." if "Hook" in role else f"Show the forest-green bowl during {role.lower()}.",
            "starting_state": "No bowl visible yet." if "Hook" in role else "The same forest-green bowl is visible.",
            "starting_image_prompt": "A trail scene; no bowl visible yet." if "Hook" in role else f"Show the same forest-green bowl for the {role.lower()}.",
        }))
    plan = plan.model_copy(update={"clips": clips})
    kie = FakeKie()
    images = run_multiclip_generation(plan, [], [], tmp_path / "review", kie, images_only=True)

    assert len(images) == 3
    assert [path.name[:7] for path in images] == ["clip-01", "clip-02", "clip-03"]
    image_calls = [call for call in kie.calls if call[0] in {"image", "text_image"}]
    assert image_calls[0][0] == "text_image"  # the reveal image is prepared first as the anchor
    anchor_url = f"https://files.test/{images[1].name}"
    assert image_calls[1][0] == "text_image" and "reference-image direction" not in image_calls[1][1].lower()
    assert image_calls[2][0] == "image" and anchor_url in image_calls[2][2]
    assert "product reference" in image_calls[2][1].lower()


def test_character_identity_reference_crosses_environment_and_wardrobe_changes(tmp_path):
    plan = build_multiclip_plan(analysis_fixture(), "A family product story", "Product facts", "varied scenes", 3, 5)
    clips = [
        plan.clips[0].model_copy(update={
            "continuity_group": "store",
            "character_groups": ["mother", "baby"],
            "starting_image_prompt": "Same Black mother holding her baby in a grocery aisle, wearing a green jacket.",
        }),
        plan.clips[1].model_copy(update={
            "continuity_group": "home",
            "starting_image_prompt": "A product detail on the living room table, no people visible.",
        }),
        plan.clips[2].model_copy(update={
            "continuity_group": "home",
            "character_groups": ["mother", "baby"],
            "starting_image_prompt": "The same Black mother and baby at home; she now wears a red sweater.",
        }),
    ]
    plan = plan.model_copy(update={"clips": clips})
    kie = FakeKie()
    product = tmp_path / "product.jpg"
    product.write_bytes(b"product")
    output_dir = tmp_path / "character-continuity"
    images = run_multiclip_generation(plan, [product], [], output_dir, kie, images_only=True)

    image_calls = [call for call in kie.calls if call[0] == "image"]
    first_character_url = f"https://files.test/{images[0].name}"
    third_call = next(call for call in image_calls if "clip-03-" in call[1] or "same Black mother" in call[1])
    assert third_call[2].count(first_character_url) == 1
    assert "character identity only" in third_call[1]
    assert "allow the current scene's wardrobe and setting" in third_call[1]
    manifest = json.loads((output_dir / "generation-usage.json").read_text())
    third_request = next(task["request"] for task in manifest["tasks"] if task["clip_index"] == 3)
    assert {reference["role"] for reference in third_request["references"]} == {
        "character identity only", "environment continuity only", "product identity only",
    }
    assert next(ref for ref in third_request["references"] if ref["role"] == "character identity only")["filename"] == images[0].name
    assert next(ref for ref in third_request["references"] if ref["role"] == "environment continuity only")["filename"] == images[1].name
    assert next(ref for ref in third_request["references"] if ref["role"] == "product identity only")["filename"] == product.name


def test_per_clip_environment_reference_is_separate_from_character_and_product_inputs(tmp_path):
    plan = build_multiclip_plan(
        analysis_fixture(), "A Target babywearing hoodie reveal", "Pink fleece hoodie with integrated pouch",
        "Target aisle", 1, 4, hook="in_store_reaction",
    )
    environment = tmp_path / "target-environment.jpg"
    identity = tmp_path / "clip-01-identity.png"
    product = tmp_path / "hoodie-product.jpg"
    for path in (environment, identity, product):
        path.write_bytes(path.name.encode())
    clip = plan.clips[0].model_copy(update={
        "character_groups": ["mother", "baby"],
        "environment_references": [str(environment)],
        "starting_state": "The pink babywearing hoodie and integrated baby pouch are visible.",
        "starting_image_prompt": "Show the same mother and baby in the Target aisle wearing the pink babywearing hoodie.",
    })
    plan = plan.model_copy(update={"clips": [clip]})
    kie = FakeKie()
    output_dir = tmp_path / "environment-reference"
    run_multiclip_generation(
        plan, [product], [], output_dir, kie, images_only=True,
        character_references=[identity],
    )

    image_call = next(call for call in kie.calls if call[0] == "image")
    assert image_call[2] == [
        "https://files.test/target-environment.jpg",
        "https://files.test/clip-01-identity.png",
        "https://files.test/hoodie-product.jpg",
    ]
    request = json.loads((output_dir / "generation-usage.json").read_text())["tasks"][0]["request"]
    assert request["references"] == [
        {"filename": environment.name, "role": "scene environment only"},
        {"filename": identity.name, "role": "character identity only"},
        {"filename": product.name, "role": "product identity only"},
    ]
    assert "ignore people, faces, pose, wardrobe, and product" in request["prompt"]
    assert "preserve this character's face, skin tone, hair, age, and body" in request["prompt"]
    assert "product details only; ignore background" in request["prompt"]


def test_resume_reuses_saved_identity_anchor_without_recreating_its_image(tmp_path):
    plan = build_multiclip_plan(analysis_fixture(), "A dog bowl ad", "Forest-green collapsible bowl", "varied scenes", 3, 5)
    clips = [clip.model_copy(update={"creative_role": role}) for clip, role in zip(plan.clips, ["Hook", "Product reveal", "Demonstration"])]
    plan = plan.model_copy(update={"clips": clips})
    output_dir = tmp_path / "resume"
    output_dir.mkdir()
    anchor = output_dir / "clip-02-anchor-starting.png"
    anchor.write_bytes(b"saved-anchor")
    kie = FakeKie()

    images = run_multiclip_generation(plan, [], [], output_dir, kie, images_only=True, resume=True)

    assert images[1] == anchor
    assert len([call for call in kie.calls if call[0] in {"image", "text_image"}]) == 2
    assert any(call[0] == "upload" and call[1] == anchor.name for call in kie.calls)


def test_compiled_prompt_prioritizes_contact_orientation_grip_and_simple_physical_action():
    plan = furglide_plan_with_long_provider_prompts()
    prompt = compile_video_prompt(plan, plan.clips[2])
    assert "working surface toward the fabric" in prompt
    assert "natural grip" in prompt
    assert "one primary physical movement" in prompt
    assert "visible contact point" in prompt


def test_failed_video_task_persists_provider_confirmed_credits_and_operation_time(tmp_path):
    plan = build_multiclip_plan(analysis_fixture(), "direction", "product facts", "room", clip_count=1, seconds_per_clip=5)
    reference = tmp_path / "product.jpg"
    reference.write_bytes(b"product")

    class FailedChargedVideo(FakeKie):
        def wait_for_task(self, task_id: str, on_state=None) -> dict:
            if task_id.startswith("video"):
                error = RuntimeError("Kie task failed: model rejected output")
                error.task_data = {"state": "fail", "creditsConsumed": 9, "failCode": "model_error"}
                raise error
            return super().wait_for_task(task_id, on_state)

    output_dir = tmp_path / "charged-failure"
    with pytest.raises(RuntimeError, match="Kie task failed"):
        run_multiclip_generation(plan, [reference], [], output_dir, FailedChargedVideo())

    manifest = json.loads((output_dir / "generation-usage.json").read_text())
    failed = next(task for task in manifest["tasks"] if task["kind"] == "video")
    assert failed["status"] == "failed"
    assert failed["credits_consumed"] == "9"
    assert failed["completed_at"]
    assert Decimal(manifest["total_credits_consumed"]) == Decimal("15")
    assert manifest["all_task_charges_reported"] is True


def test_uncertain_task_submission_is_recorded_and_not_automatically_resubmitted(tmp_path):
    plan = build_multiclip_plan(analysis_fixture(), "direction", "product facts", "room", clip_count=1, seconds_per_clip=5)
    reference = tmp_path / "product.jpg"
    reference.write_bytes(b"product")

    class UncertainVideoSubmission(FakeKie):
        def __init__(self):
            super().__init__()
            self.submissions = 0

        def create_video(self, prompt: str, image_url: str, duration_seconds: int, **kwargs) -> str:
            self.submissions += 1
            error = RuntimeError("provider response was lost")
            error.outcome_uncertain = True
            raise error

    provider = UncertainVideoSubmission()
    output_dir = tmp_path / "uncertain-submission"
    with pytest.raises(RuntimeError, match="provider response was lost"):
        run_multiclip_generation(plan, [reference], [], output_dir, provider)
    with pytest.raises(RuntimeError, match="Reconcile its status before retrying"):
        run_multiclip_generation(plan, [reference], [], output_dir, provider, resume=True)

    manifest = json.loads((output_dir / "generation-usage.json").read_text())
    uncertain = next(task for task in manifest["tasks"] if task["kind"] == "video")
    assert uncertain["status"] == "submission_unknown" and uncertain["task_id"] is None
    assert provider.submissions == 1


def test_resume_reuses_completed_clip_and_image_and_reconciles_submitted_image_task(tmp_path, monkeypatch):
    plan = build_multiclip_plan(
        analysis_fixture(), "creative direction", "product details",
        "living room", clip_count=3, seconds_per_clip=5,
    )
    output_dir = tmp_path / "clips"
    output_dir.mkdir()
    for filename, data in [
        ("clip-01-adjust-the-kitten-in-the-pouch-starting.png", b"image-1"),
        ("clip-01-adjust-the-kitten-in-the-pouch.mp4", b"video-1"),
        ("clip-02-adjust-the-opening-starting.png", b"image-2"),
    ]:
        (output_dir / filename).write_bytes(data)
    (output_dir / "generation-usage.json").write_text(json.dumps({
        "expected_generation_tasks": 6,
        "completed_generation_tasks": 3,
        "tasks": [
            {"clip_index": 1, "kind": "starting_image", "task_id": "image-done-1", "credits_consumed": "6", "status": "success"},
            {"clip_index": 1, "kind": "video", "task_id": "video-done-1", "credits_consumed": "22.5", "status": "success"},
            {"clip_index": 2, "kind": "starting_image", "task_id": "image-done-2", "credits_consumed": "6", "status": "success"},
            {"clip_index": 3, "kind": "starting_image", "task_id": "image-active-3", "credits_consumed": None, "status": "submitted"},
        ],
    }))
    kie = FakeKie()
    monkeypatch.setattr("ecom_video.workflow.make_silent_mp4", lambda src, dst: (dst.write_bytes(src.read_bytes()), dst)[1])
    monkeypatch.setattr("ecom_video.workflow.probe_video", lambda path: {"has_audio": False})

    outputs = run_multiclip_generation(
        plan, [], [], output_dir, kie, resume=True, approved_max_usd=Decimal("1.00"),
    )

    assert len(outputs) == 3
    assert outputs[0].read_bytes() == b"video-1"
    assert not [call for call in kie.calls if call[0] == "image"]
    assert [call[1] for call in kie.calls if call[0] == "poll"] == [
        "video-task-1", "image-active-3", "video-task-2",
    ]
    assert sum(call[0] == "video" for call in kie.calls) == 2
    assert (output_dir / "clip-02-adjust-the-opening-starting.png").read_bytes() == b"image-2"
    manifest = json.loads((output_dir / "generation-usage.json").read_text())
    image3 = next(task for task in manifest["tasks"] if task["task_id"] == "image-active-3")
    assert image3["status"] == "success"
    assert manifest["completed_generation_tasks"] == 6
    assert manifest["total_credits_consumed"] == "85.5"


def furglide_plan_with_long_provider_prompts():
    source = analysis_fixture()
    fourth = source.shots[2].model_copy(update={"index": 4})
    shots = [
        source.shots[0].model_copy(update={
            "index": 1, "start_seconds": None, "end_seconds": None,
            "creative_role": "Wet Works discovery hook",
            "visual_description": "A shopper discovers FurGlide on a shelf in a fictional pet store.",
            "starting_state": "The FurGlide box sits on a generic store shelf.",
            "action": "The shopper notices the box, pauses, then lifts it for a closer look.",
            "ending_state": "The shopper holds the box just off the shelf.",
            "camera": "Vertical handheld phone, quick push-in, eye level.",
            "continuity_group": "store_discovery",
            "continuity_notes": ["Fictional pet store; dark-teal cream FurGlide box with paw mark."],
            "product_interaction": "Lift and inspect the box.",
            "hook_or_payoff": "Curious in-store discovery.",
        }),
        source.shots[1].model_copy(update={
            "index": 2, "start_seconds": None, "end_seconds": None,
            "creative_role": "Product reveal",
            "visual_description": "At home beside the couch, the shopper opens FurGlide packaging.",
            "starting_state": "Closed box rests beside the same fur-covered couch.",
            "action": "Open the top flaps, lift out the teal tool, and turn it once to show its ridges.",
            "ending_state": "Hold the tool above the open box with the couch behind it.",
            "camera": "Vertical handheld close-up with a small tilt as the box opens.",
            "continuity_group": "home_couch",
            "continuity_notes": ["Same shopper, home, daylight, couch, product and package in home scenes."],
            "product_interaction": "Open the box and lift out the tool.",
            "hook_or_payoff": "Reveal the reusable ridged cleaning surface.",
        }),
        source.shots[2].model_copy(update={
            "index": 3, "start_seconds": None, "end_seconds": None,
            "creative_role": "Demonstration",
            "visual_description": "In the same home, FurGlide is used on loose pet hair on the couch.",
            "starting_state": "The cushion is covered with loose hair; the tool is poised above it.",
            "action": "Press down and make short overlapping strokes; fur gathers along the silicone edge.",
            "ending_state": "Small fur clumps sit beside a visibly cleaner path.",
            "camera": "Vertical handheld medium close-up angled slightly down.",
            "continuity_group": "home_couch",
            "continuity_notes": ["Same couch, shopper, daylight and unchanged FurGlide design."],
            "product_interaction": "Apply firm strokes across upholstery.",
            "hook_or_payoff": "Show a plausible partial clean.",
        }),
        fourth.model_copy(update={
            "index": 4, "start_seconds": None, "end_seconds": None,
            "creative_role": "Payoff",
            "visual_description": "Close view of the same couch after the cleaning strokes.",
            "starting_state": "A cleaner fabric path and gathered fur clump remain beside the tool.",
            "action": "Lift the tool, remove the fur clump, and hold on the cleaned upholstery.",
            "ending_state": "Clean fabric fills frame; the removed fur is off to one side.",
            "camera": "Vertical handheld close-up with a slight pull-back.",
            "continuity_group": "home_couch",
            "continuity_notes": ["Same couch, shopper, daylight and unchanged FurGlide design."],
            "product_interaction": "Lift off the collected fur and reveal the textile.",
            "hook_or_payoff": "Satisfying realistic clean-fabric payoff.",
        }),
    ]
    analysis = AnalysisResult.model_validate({
        "summary": "FurGlide ad: in-store discovery, home unboxing, couch demonstration and realistic payoff.",
        "questions": [], "shots": [shot.model_dump() for shot in shots],
    })
    plan = build_multiclip_plan(
        analysis, "Keep the footage authentic and silent.",
        "FurGlide is a 4.5 x 3 x 1.5 inch reusable pet-hair remover with matte teal housing and dark-gray washable silicone ridges. Short firm strokes gather loose fur. No motor or adhesive. Conceptual dark-teal, cream and white box with FurGlide name and paw mark.",
        "Fictional pet store for the hook; the same home living room and couch for the unboxing, demonstration and payoff.",
        4, 5,
    )
    for clip in plan.clips:
        clip.video_prompt += "\n" + ("LOW-PRIORITY REDUNDANT CONTEXT. " * 160)
    return plan


def test_resume_compacts_long_creative_plan_video_prompts_for_clips_two_to_four(tmp_path, monkeypatch):
    from ecom_video.workflow import _slug

    plan = furglide_plan_with_long_provider_prompts()
    output_dir = tmp_path / "furglide"
    output_dir.mkdir()
    image1 = output_dir / f"clip-01-{_slug(plan.clips[0].action)}-starting.png"
    video1 = output_dir / f"clip-01-{_slug(plan.clips[0].action)}.mp4"
    image2 = output_dir / f"clip-02-{_slug(plan.clips[1].action)}-starting.png"
    image1.write_bytes(b"saved image 1")
    video1.write_bytes(b"saved video 1")
    image2.write_bytes(b"saved image 2")
    (output_dir / "generation-usage.json").write_text(json.dumps({"tasks": [
        {"clip_index": 1, "kind": "starting_image", "task_id": "image-done-1", "credits_consumed": "6", "status": "success"},
        {"clip_index": 1, "kind": "video", "task_id": "video-done-1", "credits_consumed": "22.5", "status": "success"},
        {"clip_index": 2, "kind": "starting_image", "task_id": "image-done-2", "credits_consumed": "6", "status": "success"},
    ]}))
    original_prompts = [clip.video_prompt for clip in plan.clips]
    kie = FakeKie()
    monkeypatch.setattr("ecom_video.workflow.make_silent_mp4", lambda src, dst: (dst.write_bytes(src.read_bytes()), dst)[1])
    monkeypatch.setattr("ecom_video.workflow.probe_video", lambda path: {"has_audio": False})

    outputs = run_multiclip_generation(
        plan, [], [], output_dir, kie, resume=True, approved_max_usd=Decimal("1.00"),
    )

    video_requests = [call[1] for call in kie.calls if call[0] == "video"]
    assert len(outputs) == 4
    assert len(video_requests) == 3
    assert all(len(prompt) <= 3800 for prompt in video_requests)
    for clip in plan.clips[1:]:
        provider_prompt = compile_video_prompt(plan, clip)
        for required in (plan.product_facts, clip.starting_state, clip.action, clip.ending_state, clip.camera, *clip.continuity_notes):
            assert required in provider_prompt
    assert "LOW-PRIORITY REDUNDANT CONTEXT" in plan.clips[1].video_prompt
    assert [clip.video_prompt for clip in plan.clips] == original_prompts
    assert image1.read_bytes() == b"saved image 1"
    assert video1.read_bytes() == b"saved video 1"
    assert image2.read_bytes() == b"saved image 2"
    assert len([call for call in kie.calls if call[0] == "image"]) == 2


def _unique_oversized_product_facts() -> str:
    return " ".join(
        f"Confirmed specification detail {index:03d}: preserve unique geometry and material property code {index:03d}."
        for index in range(120)
    )


def test_compiler_deduplicates_exact_product_facts_without_losing_the_specification():
    plan = furglide_plan_with_long_provider_prompts()
    repeated_fact = "Confirmed product specification: the integrated pouch has a reinforced lower seam."
    plan.product_facts = (repeated_fact + " ") * 120

    prompt = compile_video_prompt(plan, plan.clips[1])

    assert repeated_fact in prompt
    # The first copy follows the provider-facing field label; one bare copy remains.
    assert prompt.count(repeated_fact) == 2
    assert len(prompt) <= 3800


def test_compiler_fails_actionably_when_essential_details_cannot_fit_provider_budget():
    plan = furglide_plan_with_long_provider_prompts()
    plan.product_facts = _unique_oversized_product_facts()

    with pytest.raises(ValueError, match="cannot be safely compacted.*Shorten the product specifications"):
        compile_video_prompt(plan, plan.clips[1])


def test_motion_only_revision_compacts_duplicate_scene_context_and_preserves_drop_audio():
    analysis = analysis_fixture()
    plan = build_multiclip_plan(
        analysis, "Recreate the store shot faithfully.",
        "The exact brown babywearing hoodie from supplied product photos.",
        "Target clothing store", 1, 4, hook="in_store_reaction",
    )
    plan.product_facts = "Product reference facts: " + (
        "The brown babywearing hoodie has a soft fabric body, integrated front pouch, hood, drawstrings, and matching seams. " * 12
    )
    clip = plan.clips[0].model_copy(update={
        "creative_role": "In-store reaction and reveal",
        "visual_description": "Inside a Target clothing aisle, the shocked mother looks toward brown hoodie merchandise while holding her baby and iced coffee.",
        "starting_state": "The mother is already shocked, holds an intact iced coffee cup, and securely supports the baby.",
        "action": "At about one second, she releases the cup and it falls under gravity; camera pans right to reveal the brown hoodies.",
        "ending_state": "The merchandise rack fills the frame; the mother and baby remain safe.",
        "camera": "Realistic handheld smartphone framing and a smooth pan right toward the hoodie rack.",
        "audio_mode": "enabled",
        "audio_requested": True,
        "audio_direction": "Retail ambience; realistic plastic cup impact and liquid spill synchronized with floor contact; no dialogue.",
        "continuity_notes": ["Same mother and baby; preserve the Target aisle, brown hoodies, and handheld smartphone realism."],
    })
    plan = plan.model_copy(update={"clips": [clip]})
    revision = (
        "Fix ONLY the coffee-drop action in Clip 1. Preserve the approved starting image, same mother and baby, Target environment, realistic handheld movement, and rightward camera pan revealing the brown hoodies.\n\n"
        "At approximately 1 second, the mother's fingers open and she accidentally releases the ENTIRE iced coffee cup. The cup, lid, straw, and liquid visibly fall together downward under gravity, completely separating from her hand.\n\n"
        "The previous version incorrectly showed coffee pouring out while the cup remained in her hand. Do NOT repeat that mistake.\n\n"
        "Include a realistic plastic-cup impact and liquid-spill sound synchronized with the cup hitting the floor.\n\n"
        "The mother remains shocked, looks toward the merchandise, and securely supports the baby throughout. The camera then pans right and finishes with the brown hoodies clearly visible.\n\n"
        "No extra gestures, cuts, dialogue, or changes to other clips."
    )
    revised = clip.model_copy(update={"video_prompt": clip.video_prompt + "\nClip-specific revision: " + revision})

    prompt = compile_video_prompt(plan, revised)

    assert len(prompt) <= 3800
    assert "entire iced coffee cup" in prompt.lower()
    assert "lid, straw, and liquid" in prompt.lower()
    assert "fall together downward under gravity" in prompt.lower()
    assert "completely separating from her hand" in prompt.lower()
    assert "do not pour liquid while the cup is held" in prompt.lower()
    assert "synchronized with floor contact" in prompt.lower() or "sync" in prompt.lower()
    assert "already shocked" in prompt.lower()
    assert "securely supports the baby" in prompt.lower()
    assert "Target clothing aisle" in prompt or "Target clothing aisle".lower() in prompt.lower()
    assert "pans right" in prompt.lower()
    assert "brown hoodies" in prompt.lower()


def test_final_prompt_compaction_keeps_clip2_insertion_requirements_across_revisions():
    plan = build_multiclip_plan(
        analysis_fixture(),
        "Keep the outdoor babywearing demonstration faithful to the approved direction.",
        "Brown integrated babywearing hoodie; preserve the supplied pouch construction and placement.",
        "Outdoor daylight setting with a neutral background", 3, 5, hook="in_store_reaction",
    )
    clip = plan.clips[1].model_copy(update={
        "source_start_seconds": None,
        "source_end_seconds": None,
        "creative_role": "Baby insertion demonstration",
        "visual_description": "Outdoors, the same Asian mother holds the baby above the open integrated hoodie pouch.",
        "starting_state": "The mother supports the baby above the open front carrier pouch; the baby is not yet seated.",
        "action": "She visibly lowers the baby's hips and lower torso into the pouch while continuously supporting the baby's head, back, and body with both hands.",
        "ending_state": "The baby is visibly seated inside the integrated hoodie pouch, not against its exterior.",
        "camera": "Natural handheld smartphone medium shot in steady outdoor daylight.",
        "audio_mode": "enabled",
        "audio_requested": True,
        "audio_direction": "Soft outdoor ambience and fabric rustle synchronized to the lowering and adjustment; no dialogue.",
        "continuity_notes": ["Same Asian mother and baby; keep the approved outdoor environment and natural phone-camera look."],
        "applied_user_changes": [
            "Use the Asian mother and Asian baby.",
            "Reconcile the current Creative Plan with the complete updated Creative Direction in confirmed context. Compare it with original_instructions, change only affected clips, and preserve all other approved plan details. Do not regenerate media.",
        ],
    })
    plan = plan.model_copy(update={"product_facts": "Brown integrated babywearing hoodie with an open front pouch.", "clips": [plan.clips[0], clip, *plan.clips[2:]]})
    revision = (
        "Correct ONLY the baby's insertion into the integrated babywearing hoodie. Preserve the existing outdoor setting, same Asian mother and baby, brown hoodie, approved starting image, and natural smartphone filming style. "
        "The mother begins holding the baby above and partially aligned with the open front carrier pouch. "
        "She gently lowers the baby's hips and lower torso INTO the integrated pouch while continuously supporting the baby's body with both hands. "
        "The pouch fabric naturally moves around the baby as the baby settles into the carrier. The mother then adjusts the fabric around the baby's body. "
        "By the end of the clip, the baby must be visibly SEATED INSIDE the integrated hoodie pouch, matching the supplied product reference—not hanging against the exterior or being held in front of it. "
        "Maintain believable anatomy, realistic fabric movement, appropriate support, and an unobstructed face. "
        "MANDATORY: Show the complete downward insertion and final seated position. Do not skip the insertion, invent carrier mechanisms, change the outdoor environment, or introduce additional actions."
    )
    revised_clip = clip.model_copy(update={
        "video_prompt": clip.video_prompt + "\nClip-specific revision: " + revision
        + "\nClip-specific revision: Preserve the outdoor environment, continuous support, and seated-inside ending."
    })
    revised_plan = plan.model_copy(update={"clips": [plan.clips[0], revised_clip, *plan.clips[2:]]})

    prompt = compile_video_prompt(revised_plan, revised_clip)

    assert len(prompt) <= 3800
    assert "visibly lowers" in prompt.lower() and "continuously supporting" in prompt.lower()
    assert "seated inside the integrated hoodie pouch" in prompt.lower()
    assert "not against its exterior" in prompt.lower()
    assert "do not skip the insertion" in prompt.lower()
    assert "outdoor daylight" in prompt.lower()
    assert "synchronized to the lowering" in prompt.lower() or "synchronize each selected native sound" in prompt.lower()
    assert "reconcile the current creative plan" not in prompt.lower()
    assert "do not regenerate media" not in prompt.lower()


def test_uncompactable_plan_is_rejected_before_any_provider_operation(tmp_path):
    plan = furglide_plan_with_long_provider_prompts()
    plan.product_facts = _unique_oversized_product_facts()
    kie = FakeKie()

    with pytest.raises(ValueError, match="cannot be safely compacted"):
        run_multiclip_generation(
            plan, [], [], tmp_path / "clips", kie, approved_max_usd=Decimal("1.00"),
        )

    assert kie.calls == []
    assert not (tmp_path / "clips").exists()


def test_compiler_preserves_selective_regeneration_revision():
    plan = furglide_plan_with_long_provider_prompts()
    revised = plan.clips[1].model_copy(update={
        "video_prompt": plan.clips[1].video_prompt + "\nClip-specific revision: Keep the branded box visible throughout the unboxing."
    })

    provider_prompt = compile_video_prompt(plan, revised)

    assert "Clip-specific revision: Keep the branded box visible throughout the unboxing." in provider_prompt
    assert len(provider_prompt) <= 3800


def test_runner_compacts_selective_regeneration_revision_before_video_submission(tmp_path, monkeypatch):
    plan = furglide_plan_with_long_provider_prompts()
    clip = plan.clips[1].model_copy(update={
        "video_prompt": plan.clips[1].video_prompt + "\nClip-specific revision: Keep the branded box visible throughout the unboxing."
    })
    single_plan = plan.model_copy(update={"clips": [clip]})
    kie = FakeKie()
    product = tmp_path / "product.jpg"
    product.write_bytes(b"product")
    monkeypatch.setattr("ecom_video.workflow.make_silent_mp4", lambda src, dst: (dst.write_bytes(src.read_bytes()), dst)[1])

    run_multiclip_generation(single_plan, [product], [], tmp_path / "regeneration", kie)

    submitted = next(call[1] for call in kie.calls if call[0] == "video")
    assert "Clip-specific revision: Keep the branded box visible throughout the unboxing." in submitted
    assert len(submitted) <= 3800


def test_generation_under_fifty_cents_runs_without_an_additional_approval(tmp_path, monkeypatch):
    plan = build_multiclip_plan(
        analysis_fixture(), "creative direction", "product details",
        "living room", clip_count=3, seconds_per_clip=5,
    )
    product = tmp_path / "product.jpg"
    product.write_bytes(b"product")
    kie = FakeKie()

    monkeypatch.setattr("ecom_video.workflow.make_silent_mp4", lambda src, dst: (dst.write_bytes(src.read_bytes()), dst)[1])
    outputs = run_multiclip_generation(plan, [product], [], tmp_path / "clips", kie)

    assert len(outputs) == 3
    assert sum(call[0] == "video" for call in kie.calls) == 3


def test_fifty_cent_threshold_requires_approval_above_limit():
    from ecom_video.workflow import generation_approval_required
    def estimate(total):
        return GenerationEstimate(1, 1, 1, Decimal(total), Decimal("0"), Decimal("0"))
    assert generation_approval_required(estimate("0.50")) is False
    assert generation_approval_required(estimate("0.5001")) is True
    assert generation_approval_required(estimate("0.10"), pricing_verified=False) is True
    assert generation_approval_required(estimate("0.10"), estimate_reliable=False) is True


def test_four_nine_second_clips_require_approval_over_fifty_cents(tmp_path, monkeypatch):
    plan = build_multiclip_plan(analysis_fixture(), "creative direction", "product details", "living room", 3, 9)
    product = tmp_path / "product.jpg"; product.write_bytes(b"product")
    kie = FakeKie()
    monkeypatch.setattr("ecom_video.workflow.make_silent_mp4", lambda src, dst: (dst.write_bytes(src.read_bytes()), dst)[1])
    with pytest.raises(PermissionError, match="approval"):
        run_multiclip_generation(plan, [product], [], tmp_path / "clips", kie)
    assert len(run_multiclip_generation(plan, [product], [], tmp_path / "clips-approved", kie,
                                        approved_max_usd=Decimal("0.70"))) == 3


def test_generation_over_fifty_cents_requires_approval_before_provider_calls(tmp_path):
    plan = build_multiclip_plan(
        analysis_fixture(), "creative direction", "product details",
        "living room", clip_count=3, seconds_per_clip=15,
    )
    product = tmp_path / "product.jpg"
    product.write_bytes(b"product")
    kie = FakeKie()

    with pytest.raises(PermissionError, match="approval"):
        run_multiclip_generation(plan, [product], [], tmp_path / "clips", kie)

    assert kie.calls == []


def test_generation_rejects_approval_below_estimate_without_provider_calls(tmp_path):
    plan = build_multiclip_plan(
        analysis_fixture(), "creative direction", "product details",
        "living room", clip_count=3, seconds_per_clip=5,
    )
    product = tmp_path / "product.jpg"
    product.write_bytes(b"product")
    kie = FakeKie()

    with pytest.raises(PermissionError, match="exceeds approved maximum"):
        run_multiclip_generation(
            plan, [product], [], tmp_path / "clips", kie,
            approved_max_usd=Decimal("0.42"),
        )

    assert kie.calls == []


def test_unverified_pricing_requires_approval_even_for_low_cost_job(tmp_path):
    plan = build_multiclip_plan(
        analysis_fixture(), "creative direction", "product details",
        "living room", clip_count=3, seconds_per_clip=5,
    )
    product = tmp_path / "product.jpg"
    product.write_bytes(b"product")
    kie = FakeKie()

    with pytest.raises(PermissionError, match="approval"):
        run_multiclip_generation(
            plan, [product], [], tmp_path / "clips", kie,
            pricing_verified=False,
        )

    assert kie.calls == []


def test_unreliable_estimate_requires_approval_even_for_low_cost_job(tmp_path):
    plan = build_multiclip_plan(
        analysis_fixture(), "creative direction", "product details",
        "living room", clip_count=3, seconds_per_clip=5,
    )
    product = tmp_path / "product.jpg"
    product.write_bytes(b"product")
    kie = FakeKie()

    with pytest.raises(PermissionError, match="estimate is unreliable"):
        run_multiclip_generation(
            plan, [product], [], tmp_path / "clips", kie,
            estimate_reliable=False,
        )

    assert kie.calls == []


def test_cli_starts_verified_job_below_threshold_without_approval_flag(tmp_path, monkeypatch):
    plan = build_multiclip_plan(
        analysis_fixture(), "creative direction", "product details",
        "living room", clip_count=3, seconds_per_clip=5,
    )
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(plan.model_dump_json())
    args = SimpleNamespace(
        plan=str(plan_path), output_dir=str(tmp_path / "clips"),
        approve_max_usd=None, require_approval=False, dry_run=False,
    )
    calls = {}
    fake_client = SimpleNamespace(close=lambda: None)
    monkeypatch.setattr("ecom_video.providers.kie.KieClient", lambda: fake_client)

    def fake_run(*args, **kwargs):
        calls["kwargs"] = kwargs
        return [tmp_path / "clips" / "clip-01.mp4"]

    monkeypatch.setattr(cli, "run_multiclip_generation", fake_run)
    cli._generate_multiclip(args)

    assert calls["kwargs"]["approved_max_usd"] is None
    assert calls["kwargs"]["pricing_verified"] is True
    assert calls["kwargs"]["estimate_reliable"] is True


def test_cli_dry_run_never_initializes_kie_or_calls_generation(tmp_path, monkeypatch):
    plan = build_multiclip_plan(
        analysis_fixture(), "creative direction", "product details",
        "living room", clip_count=3, seconds_per_clip=5,
    )
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(plan.model_dump_json())
    args = SimpleNamespace(
        plan=str(plan_path), output_dir=str(tmp_path / "clips"),
        approve_max_usd=None, require_approval=False, dry_run=True,
    )
    monkeypatch.setattr(
        "ecom_video.providers.kie.KieClient",
        lambda: pytest.fail("Kie must not be initialized for a dry run"),
    )
    monkeypatch.setattr(
        cli, "run_multiclip_generation",
        lambda *a, **kw: pytest.fail("generation runner must not be called for a dry run"),
    )

    cli._generate_multiclip(args)


@pytest.mark.parametrize("require_approval", [False, True])
def test_cli_holds_over_threshold_or_uncertain_job_without_approval(tmp_path, monkeypatch, require_approval):
    plan = build_multiclip_plan(
        analysis_fixture(), "creative direction", "product details",
        "living room", clip_count=3, seconds_per_clip=5 if require_approval else 15,
    )
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(plan.model_dump_json())
    args = SimpleNamespace(
        plan=str(plan_path), output_dir=str(tmp_path / "clips"),
        approve_max_usd=None, require_approval=require_approval, dry_run=False,
    )
    monkeypatch.setattr(
        "ecom_video.providers.kie.KieClient",
        lambda: pytest.fail("Kie must not be initialized while approval is required"),
    )
    monkeypatch.setattr(
        cli, "run_multiclip_generation",
        lambda *a, **kw: pytest.fail("generation runner must not be called before approval"),
    )

    cli._generate_multiclip(args)


@pytest.mark.parametrize(("clip_index", "scene", "state"), [
    (2, "outdoor park path", "Mother holds baby naturally in her arms during the hoodie reveal."),
    (3, "Christmas market", "Mother walks casually with baby supported in the integrated pouch."),
    (4, "nursery beside the crib", "Mother cuddles the baby in the integrated pouch."),
])
def test_masked_edit_compiles_environment_identity_product_and_scene_roles(
    tmp_path, clip_index, scene, state,
):
    class MaskedEditClient:
        def __init__(self):
            self.calls = []

        def upload(self, path):
            self.calls.append(("upload", Path(path).name))
            return f"https://files.test/{Path(path).name}"

        def create_masked_image_edit(self, prompt, input_urls, mask_url, aspect_ratio="9:16"):
            self.calls.append(("masked", prompt, list(input_urls), mask_url, aspect_ratio))
            return "masked-task"

        def wait_for_task(self, task_id, on_state=None):
            self.calls.append(("poll", task_id))
            return {"state": "success", "creditsConsumed": 8.1,
                    "resultJson": json.dumps({"resultUrls": ["https://files.test/result.png"]})}

        def download_result(self, task_data, destination):
            self.calls.append(("download", destination.name))
            destination.write_bytes(b"edited-image")
            return destination

    scene_base = tmp_path / f"clip-{clip_index:02d}-{scene.replace(' ', '-')}.png"
    identity = tmp_path / "clip-01-identity.png"
    product = tmp_path / "hoodie-product.png"
    mask = tmp_path / f"clip-{clip_index:02d}-mask.png"
    for path, contents in ((scene_base, b"scene"), (identity, b"identity"), (product, b"product"), (mask, b"mask")):
        path.write_bytes(contents)

    plan = build_multiclip_plan(
        analysis_fixture(), "same cast", "pink babywearing hoodie", scene,
        clip_count=3, seconds_per_clip=4,
    )
    target = plan.clips[min(clip_index - 1, len(plan.clips) - 1)].model_copy(update={
        "index": clip_index,
        "visual_description": f"The same mother and baby appear in the {scene}.",
        "starting_state": state,
        "starting_image_prompt": f"Frame zero in the {scene}: {state}",
        "character_groups": ["mother", "baby"],
    })
    plan = plan.model_copy(update={
        "clips": [target], "product_references": [str(product)],
    })
    client = MaskedEditClient()

    result = run_masked_image_edit(
        plan, target, scene_base, mask, [product], [identity],
        tmp_path / "masked-output", client,
        instructions="Replace only her clothing with a forest-green cardigan.",
    )

    request = next(call for call in client.calls if call[0] == "masked")
    assert [url.rsplit("/", 1)[-1] for url in request[2]] == [scene_base.name, identity.name, product.name]
    assert request[3] == f"https://files.test/{mask.name}"
    assert request[4] == "9:16"
    assert "first input is the selected scene base and environment only" in request[1].lower()
    assert "identity only" in request[1].lower()
    assert "ignore that reference's pose, expression, wardrobe, and environment" in request[1].lower()
    assert "user-requested masked change" in request[1].lower()
    assert "replace only her clothing with a forest-green cardigan." in request[1].lower()
    assert "only within the transparent editable mask" in request[1].lower()
    assert "product appearance only" in request[1].lower()
    assert state.lower() in request[1].lower()
    assert result.is_file()
    manifest = json.loads((result.parent / "generation-usage.json").read_text())
    assert manifest["tasks"][0]["request"]["compiled_input_url_order"] == [scene_base.name, identity.name, product.name]
    assert [item["role"] for item in manifest["tasks"][0]["request"]["references"]] == [
        "scene environment only", "character identity only", "product identity only",
    ]


def test_saved_mask_lookup_matches_source_hash_for_environment_first_scene_bases(tmp_path):
    import hashlib

    from ecom_video.workflow import find_saved_mask_for_scene

    outputs = tmp_path / "outputs"
    sources = [tmp_path / name for name in ("clip-02-source.jpg", "clip-03-v1.png", "clip-04-source.jpg")]
    for index, source in enumerate(sources, start=2):
        source.write_bytes(f"scene-{index}".encode())
        version = outputs / f"clip-{index:02d}" / "starting-image" / f"version-{index}"
        version.mkdir(parents=True)
        mask = version / f"clip-{index:02d}-mask.png"
        mask.write_bytes(f"mask-{index}".encode())
        manifest = {
            "tasks": [{"request": {
                "references": [{"sha256": hashlib.sha256(source.read_bytes()).hexdigest()}],
                "mask": {"filename": mask.name, "sha256": hashlib.sha256(mask.read_bytes()).hexdigest(), "dimensions": [8, 12]},
            }}]
        }
        (version / "generation-usage.json").write_text(json.dumps(manifest))

    assert [find_saved_mask_for_scene(outputs, source).name for source in sources] == [
        "clip-02-mask.png", "clip-03-mask.png", "clip-04-mask.png",
    ]
    assert find_saved_mask_for_scene(outputs, tmp_path / "unrelated.png") is None
    assert find_saved_mask_for_scene(outputs, sources[0], target="different edit target") is None
