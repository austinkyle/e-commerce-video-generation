"""Local natural-language object grounding and source-aligned alpha masks."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path


DETECTOR_MODEL = "IDEA-Research/grounding-dino-tiny"
SEGMENTER_MODEL = "facebook/sam2.1-hiera-tiny"
MIN_DETECTION_CONFIDENCE = 0.15
MIN_MASK_COVERAGE = 0.002
MAX_MASK_COVERAGE = 0.55


class AutomaticMaskError(ValueError):
    """Raised when the requested edit cannot be isolated safely."""


def extract_edit_target(instructions: str) -> str:
    """Extract the existing object/region from common edit instructions."""
    text = " ".join(instructions.strip().rstrip(". ").split())
    patterns = (
        r"re-?colou?r\s+(?:only\s+)?(?:the\s+)?(.+?)\s+(?:black|white|red|blue|green|yellow|pink|purple|orange|brown|gray|grey)",
        r"(?:replace|remove|erase|delete|edit|change|recolor|re-colou?r|adjust)\s+(?:only\s+)?(?:the\s+)?(.+?)\s+(?:with|to|into|from)\s+.+",
        r"(?:change|adjust|fix|correct)\s+(?:the\s+)?(?:color|colour|appearance|style|shape|expression)\s+of\s+(?:the\s+)?(.+)",
        r"(?:change|adjust|fix|correct|remove|erase|delete|replace|recolor|re-colou?r|make)\s+(?:only\s+)?(?:the\s+)?(.+)",
    )
    for pattern in patterns:
        match = re.fullmatch(pattern, text, flags=re.IGNORECASE)
        if match:
            target = match.group(1).strip(" .,!?")
            target = re.sub(r"^(?:a|an|the)\s+", "", target, flags=re.IGNORECASE)
            target = re.sub(r"\s+(?:while|and)\s+preserv(?:e|ing)\b.*$", "", target, flags=re.IGNORECASE)
            if target.lower() in {"it", "this", "that", "it look better", "everything"}:
                continue
            if target:
                return target
    raise AutomaticMaskError(
        "I couldn't identify the existing object in that instruction. Start with “replace [existing item] with …”, “change [item] to …”, or “remove [item]”."
    )


@lru_cache(maxsize=1)
def _load_models():
    try:
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, Sam2Model, Sam2Processor
    except ImportError as exc:
        raise AutomaticMaskError(
            "Automatic masking needs the optional local vision models. Install them with `uv sync --extra auto-mask`; no image is sent to a segmentation service."
        ) from exc

    # Keep models lazy and resident only while an edit is being prepared.
    detector_processor = AutoProcessor.from_pretrained(DETECTOR_MODEL, use_fast=False)
    detector = AutoModelForZeroShotObjectDetection.from_pretrained(DETECTOR_MODEL)
    segmenter_processor = Sam2Processor.from_pretrained(SEGMENTER_MODEL, use_fast=False)
    segmenter = Sam2Model.from_pretrained(SEGMENTER_MODEL)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    return torch, detector_processor, detector.to(device).eval(), segmenter_processor, segmenter.to(device).eval(), device


def _detect(image, phrase, torch, processor, model, device, threshold=MIN_DETECTION_CONFIDENCE):
    inputs = processor(images=image, text=phrase, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model(**inputs)
    found = processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids, threshold=threshold, text_threshold=threshold,
        target_sizes=[(image.height, image.width)],
    )[0]
    return [
        (str(label).lower().strip(" ."), float(score), [float(v) for v in box.tolist()])
        for label, score, box in zip(found["text_labels"], found["scores"], found["boxes"])
    ]


def _target_phrases(target: str) -> list[str]:
    phrases = [target]
    substitutions = {
        "hoodie": ("sweatshirt", "fleece hoodie"),
        "sweatshirt": ("hoodie", "fleece sweatshirt"),
        "cup": ("mug",),
        "mug": ("cup",),
        "gecko": ("lizard",),
        "sneakers": ("shoes",),
        "cellphone": ("mobile phone",),
    }
    for word, alternatives in substitutions.items():
        if re.search(rf"\b{word}\b", target, re.IGNORECASE):
            phrases.extend(re.sub(rf"\b{word}\b", alternative, target, flags=re.IGNORECASE) for alternative in alternatives)
    if re.search(r"\b(hair|hairstyle|hair style)\b", target, re.IGNORECASE):
        phrases.append("hair")
    if re.search(r"\b(face|facial expression|eyes|mouth)\b", target, re.IGNORECASE):
        phrases.append("face")
    return list(dict.fromkeys(phrase.strip() for phrase in phrases))


def _box_area_fraction(box, width, height):
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1) / (width * height)


def _hsv_color_mask(image, color: str):
    """Simple color gate used only when the requested existing item names a color."""
    import numpy as np
    hsv = np.asarray(image.convert("HSV"))
    hue, saturation, value = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    ranges = {
        "pink": ((220, 255), 15, 35), "purple": ((190, 235), 45, 45),
        "blue": ((135, 195), 45, 45), "green": ((55, 135), 45, 45),
        "yellow": ((28, 48), 55, 85), "orange": ((8, 29), 55, 75),
        "red": ((0, 10), 55, 60), "brown": ((8, 32), 45, 35),
    }
    if color == "black":
        return value < 65
    if color in {"white", "gray", "grey"}:
        return (value > 175) & (saturation < 55) if color == "white" else saturation < 35
    limits = ranges.get(color)
    if not limits:
        return None
    (lo, hi), minimum_saturation, minimum_value = limits
    hue_match = (hue >= lo) & (hue <= hi)
    if color == "red":
        hue_match |= hue >= 245
    return hue_match & (saturation >= minimum_saturation) & (value >= minimum_value)


def _box_mask(image, boxes, torch, processor, model, device):
    inputs = processor(images=image, input_boxes=[[box for box in boxes]], return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model(**inputs)
    masks = processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0]
    scores = outputs.iou_scores[0]
    import numpy as np
    result = np.zeros((image.height, image.width), dtype=bool)
    for index, candidates in enumerate(masks):
        best = int(scores[index].argmax().item())
        result |= candidates[best].numpy().astype(bool)
    return result


def generate_automatic_mask(source_path: Path, instructions: str, output_path: Path) -> dict:
    """Ground the instruction locally, segment the target, and save provider alpha semantics."""
    try:
        from PIL import Image, ImageFilter
        import numpy as np
    except ImportError as exc:
        raise AutomaticMaskError("Automatic masking requires the optional Pillow package (`uv sync --extra auto-mask`).") from exc

    target = extract_edit_target(instructions)
    source_path = Path(source_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    try:
        image = Image.open(source_path).convert("RGB")
    except OSError as exc:
        raise AutomaticMaskError("The selected scene image could not be decoded for automatic masking.") from exc
    torch, detector_processor, detector, sam_processor, sam, device = _load_models()

    phrases = _target_phrases(target)
    detections = []
    for phrase in phrases:
        expected = re.sub(r"[^a-z0-9 ]", "", phrase.lower()).strip()
        for label, score, box in _detect(image, phrase, torch, detector_processor, detector, device, threshold=0.15):
            # Restrict results to the requested concept; model subword fragments and
            # unrelated high-scoring objects must never become edit targets.
            normalized = re.sub(r"[^a-z0-9 ]", "", label.replace("##", "")).strip()
            if normalized == expected and score >= MIN_DETECTION_CONFIDENCE:
                detections.append((label, score, box))
    if not detections:
        raise AutomaticMaskError(f"I couldn't confidently locate “{target}” in this image. No mask was saved and no image request was sent.")
    # Ambiguous multiple matches are unsafe unless one detection is clearly strongest.
    detections.sort(key=lambda item: item[1], reverse=True)
    # Merge repeated detections of the same object across query synonyms.
    unique = []
    for item in detections:
        x1, y1, x2, y2 = item[2]
        duplicate = False
        for prior in unique:
            a1, b1, a2, b2 = prior[2]
            iw, ih = max(0, min(x2, a2) - max(x1, a1)), max(0, min(y2, b2) - max(y1, b1))
            intersection = iw * ih
            union = (x2 - x1) * (y2 - y1) + (a2 - a1) * (b2 - b1) - intersection
            if union > 0 and intersection / union > 0.6:
                duplicate = True
                break
        if not duplicate:
            unique.append(item)
    detections = unique
    if len(detections) > 1 and detections[1][1] >= detections[0][1] * 0.82:
        raise AutomaticMaskError(f"I found more than one likely “{target}”. Choose a clearer saved image; no mask was saved and no image request was sent.")
    chosen = detections[0]
    target_mask = _box_mask(image, [chosen[2]], torch, sam_processor, sam, device)
    color_match = re.search(r"\b(black|white|red|blue|green|yellow|pink|purple|orange|brown|gray|grey)\b", target, re.IGNORECASE)
    if color_match:
        color_gate = _hsv_color_mask(image, color_match.group(1).lower())
        if color_gate is not None:
            target_mask &= color_gate
            # Fleece and knit textures can fragment a color-gated region. Close
            # only small internal gaps; protected regions are subtracted below.
            binary = Image.fromarray(target_mask.astype(np.uint8) * 255)
            target_mask = np.asarray(
                binary.filter(ImageFilter.MaxFilter(31)).filter(ImageFilter.MinFilter(31))
            ) > 0
    coverage = float(target_mask.mean())
    if not MIN_MASK_COVERAGE <= coverage <= MAX_MASK_COVERAGE:
        raise AutomaticMaskError(f"The “{target}” selection covered an unsafe part of the image ({coverage:.0%}). No paid request was sent.")

    # Check and subtract protected neighboring people/object regions before accepting the mask.
    protected_labels = ["face", "hair", "hand", "baby", "coffee cup", "cup"]
    target_lower = target.lower()
    editable_labels = {
        "hair": bool(re.search(r"\b(hair|hairstyle|hair style)\b", target_lower)),
        "face": bool(re.search(r"\b(face|facial expression|eyes|mouth)\b", target_lower)),
        "hand": bool(re.search(r"\bhand(s)?\b", target_lower)),
        "baby": bool(re.search(r"\b(baby|infant)\b", target_lower)),
        "cup": bool(re.search(r"\b(cup|mug)\b", target_lower)),
    }
    protected_labels = [
        label for label in protected_labels
        if not editable_labels.get(label, False)
        and not (editable_labels["cup"] and label in {"coffee cup", "cup"})
    ]
    protected = _detect(
        image,
        ". ".join(protected_labels) + ".",
        torch, detector_processor, detector, device, threshold=0.3,
    )
    protected = [entry for entry in protected if entry[0] in {"face", "hair", "hand", "baby", "coffee cup", "cup"}
                 and _box_area_fraction(entry[2], image.width, image.height) < 0.25]
    protected_mask = _box_mask(image, [entry[2] for entry in protected], torch, sam_processor, sam, device) if protected else np.zeros_like(target_mask)
    overlap = float((target_mask & protected_mask).sum() / max(1, target_mask.sum()))
    editable = target_mask & ~protected_mask
    coverage = float(editable.mean())
    if not MIN_MASK_COVERAGE <= coverage <= MAX_MASK_COVERAGE or coverage < 0.35 * float(target_mask.mean()):
        raise AutomaticMaskError(f"The safe “{target}” mask is too small or too broad ({coverage:.0%}; protected overlap {overlap:.0%}). No paid request was sent.")

    # Provider contract: alpha 0 means editable; alpha 255 means preserve.
    alpha = np.where(editable, 0, 255).astype(np.uint8)
    rgba = np.full((image.height, image.width, 4), 255, dtype=np.uint8)
    rgba[:, :, 3] = alpha
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, mode="RGBA").save(output_path, format="PNG")
    return {
        "target": target,
        "detector": DETECTOR_MODEL,
        "segmenter": SEGMENTER_MODEL,
        "confidence": chosen[1],
        "coverage": coverage,
        "protected_overlap": overlap,
        "dimensions": [image.width, image.height],
        "device": device,
    }
