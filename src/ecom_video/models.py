from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class ShotPlan(BaseModel):
    index: int = Field(ge=1)
    start_seconds: float | None = Field(default=None, ge=0)
    end_seconds: float | None = Field(default=None, gt=0)
    creative_role: str = ""
    visual_description: str
    starting_state: str = ""
    action: str
    ending_state: str = ""
    camera: str
    product_interaction: str
    hook_or_payoff: str
    continuity_group: str = ""
    continuity_notes: list[str] = Field(default_factory=list)
    uncertainties: list[str]

    @model_validator(mode="after")
    def validate_reference_interval(self) -> "ShotPlan":
        if (self.start_seconds is None) != (self.end_seconds is None):
            raise ValueError("source start and end times must both be provided or both be null")
        if self.start_seconds is not None and self.end_seconds <= self.start_seconds:
            raise ValueError("source end time must be after source start time")
        return self


class AnalysisResult(BaseModel):
    summary: str
    questions: list[str]
    shots: list[ShotPlan]


HOOK_TEMPLATES = {
    "wet_works": "Wet Works · Open on a believable in-store discovery and immediate human reaction; reveal product details only when the shopper engages, and never imply real retailer distribution.",
    "countdown_reveal": "3-2-1 Reveal · Use three distinct visual beats of anticipation, increasing proximity or evidence, then a clear product reveal; do not reduce the hook to overlay text.",
    "gifting": "Gifting · Show who gives or discovers the product, the physical handoff or reveal, and a specific natural emotional response that pays off why it is a good gift.",
    "custom": "Custom · Follow the creator's hook and arrange distinct setup, demonstration, and payoff shots only where they serve the concept.",
}


def hook_direction(hook: str) -> str:
    if hook == "recommend":
        return "Recommend the most product-appropriate hook. Build a coherent ad with a distinct attention hook, useful demonstration, and satisfying payoff; omit any redundant beat."
    return HOOK_TEMPLATES.get(hook, HOOK_TEMPLATES["custom"])
