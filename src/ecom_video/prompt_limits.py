"""Text limits for provider request fields."""

KIE_VIDEO_PROMPT_MAX_CHARACTERS = 4096
KIE_VIDEO_PROMPT_SAFE_CHARACTERS = 3800


def kie_video_prompt_length(text: str) -> int:
    """Count Unicode scalar values and UTF-16 units conservatively for Kie's counter."""
    return max(
        len(text),
        len(text.encode("utf-16-le")) // 2,
    )
