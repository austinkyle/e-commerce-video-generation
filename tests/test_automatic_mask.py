from pathlib import Path

import pytest

from ecom_video.automatic_mask import AutomaticMaskError, extract_edit_target


@pytest.mark.parametrize(("instruction", "target"), [
    ("Replace the pink hoodie with a plain black T-shirt. Preserve everything else.", "pink hoodie"),
    ("Change a person's hairstyle to a short bob.", "person's hairstyle"),
    ("Recolor the gecko toy blue.", "gecko toy"),
    ("Remove the coffee cup.", "coffee cup"),
    ("Correct the facial expression.", "facial expression"),
])
def test_extract_edit_target(instruction, target):
    assert extract_edit_target(instruction) == target


def test_target_extraction_rejects_ambiguous_instruction():
    with pytest.raises(AutomaticMaskError, match="couldn't identify the existing object"):
        extract_edit_target("Make it look better.")
