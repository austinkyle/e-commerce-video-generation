from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING, ROUND_HALF_UP


IMAGE_1K_USD = Decimal("0.03")
MASKED_EDIT_1K_USD = Decimal("0.05")
VIDEO_720P_PER_SECOND_USD = Decimal("0.0225")
ANALYSIS_ALLOWANCE_USD = Decimal("0.02")
KIE_USD_PER_CREDIT = Decimal("0.005")
DEFAULT_PROJECT_SPENDING_LIMIT_USD = Decimal("6.00")
ESTIMATE_DISPLAY_QUANTUM_USD = Decimal("0.0001")


def estimate_within_approved_amount(estimate: Decimal, approved: Decimal) -> bool:
    """Compare at the four-decimal precision shown in generation confirmations."""
    shown_estimate = estimate.quantize(ESTIMATE_DISPLAY_QUANTUM_USD, rounding=ROUND_HALF_UP)
    shown_approval = approved.quantize(ESTIMATE_DISPLAY_QUANTUM_USD, rounding=ROUND_HALF_UP)
    return shown_estimate <= shown_approval


def reserve_estimate_usd(estimate: Decimal) -> Decimal:
    """Round estimated spend upward to display precision before checking a project cap."""
    return estimate.quantize(ESTIMATE_DISPLAY_QUANTUM_USD, rounding=ROUND_CEILING)


@dataclass(frozen=True)
class GenerationEstimate:
    image_count: int
    requested_seconds: int
    generated_seconds: int
    image_usd: Decimal
    video_usd: Decimal
    analysis_usd: Decimal

    @property
    def total_usd(self) -> Decimal:
        return self.image_usd + self.video_usd + self.analysis_usd


def estimate_generation(seconds: int, images: int, analysis_usd: Decimal = ANALYSIS_ALLOWANCE_USD) -> GenerationEstimate:
    if seconds < 1:
        raise ValueError("seconds must be at least 1")
    if images < 0:
        raise ValueError("images cannot be negative")
    rounded_seconds = int(Decimal(seconds).to_integral_value(rounding=ROUND_CEILING))
    return GenerationEstimate(
        image_count=images,
        requested_seconds=seconds,
        generated_seconds=rounded_seconds,
        image_usd=IMAGE_1K_USD * images,
        video_usd=VIDEO_720P_PER_SECOND_USD * rounded_seconds,
        analysis_usd=analysis_usd,
    )


def price_openai_analysis_usage(usage: dict) -> Decimal | None:
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    if input_tokens is None or output_tokens is None:
        return None
    cached = usage.get("input_tokens_details", {}).get("cached_tokens", 0)
    return (
        (Decimal(input_tokens - cached) * Decimal("0.75")
         + Decimal(cached) * Decimal("0.075")
         + Decimal(output_tokens) * Decimal("4.50"))
        / Decimal(1_000_000)
    )
