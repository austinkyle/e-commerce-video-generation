from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from ecom_video.creative_playbook import IN_STORE_REACTION_PLAYBOOK

AudioMode = Literal["auto", "enabled", "silent"]


class ShotPlan(BaseModel):
    index: int = Field(ge=1)
    start_seconds: float | None = Field(default=None, ge=0)
    end_seconds: float | None = Field(default=None, gt=0)
    creative_role: str = ""
    visual_description: str
    starting_state: str = ""
    action: str
    ending_state: str = ""
    target_duration_seconds: int | None = Field(default=None, ge=1, le=15)
    edit_duration_seconds: float | None = Field(default=None, ge=0.5, le=15)
    camera: str
    product_interaction: str
    hook_or_payoff: str
    audio_requested: bool = False
    audio_direction: str = "Silent output; no generated audio is available."
    continuity_group: str = ""
    character_groups: list[str] = Field(default_factory=list)
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
    "in_store_reaction": IN_STORE_REACTION_PLAYBOOK + " Never imply stock at a named retailer unless established.",
    "countdown_reveal": "3-2-1 Reveal · Use three distinct visual beats of anticipation, increasing proximity or evidence, then a clear product reveal; do not reduce the hook to overlay text.",
    "gifting": "Gifting · Show who gives or discovers the product, the physical handoff or reveal, and a specific natural emotional response that pays off why it is a good gift.",
    "versus": "Versus · Make a fair, visually clear side-by-side comparison of the specified products or approaches, then resolve the difference with an observable product-specific result. Do not invent comparative claims.",
    "custom": "Custom · Follow the creator's hook and arrange distinct setup, demonstration, and payoff shots only where they serve the concept.",
}


def hook_direction(hook: str) -> str:
    if hook == "recommend":
        return "Recommend the most product-appropriate hook. Build a coherent ad with a distinct attention hook, useful demonstration, and satisfying payoff; omit any redundant beat."
    if hook == "wet_works":
        hook = "in_store_reaction"
    return HOOK_TEMPLATES.get(hook, HOOK_TEMPLATES["custom"])
