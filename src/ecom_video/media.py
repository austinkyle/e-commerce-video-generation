from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


def _binary(name: str) -> str:
    found = shutil.which(name)
    if not found:
        raise RuntimeError(f"{name} is required. Install FFmpeg and ensure {name} is on PATH.")
    return found


def probe_video(path: Path) -> dict:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    result = subprocess.run(
        [_binary("ffprobe"), "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    )
    data = json.loads(result.stdout)
    video = next((stream for stream in data.get("streams", []) if stream.get("codec_type") == "video"), None)
    if video is None:
        raise ValueError("The selected file has no video stream")
    streams = data.get("streams", [])

    def seconds(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    video_start = seconds(video.get("start_time"))
    audio_streams = [stream for stream in streams if stream.get("codec_type") == "audio"]
    audio_offsets = [
        audio_start - video_start
        for stream in audio_streams
        if (audio_start := seconds(stream.get("start_time"))) is not None and video_start is not None
    ]
    return {"path": str(path), "duration": float(data.get("format", {}).get("duration", 0)),
            "width": video.get("width"), "height": video.get("height"),
            "rotation": next((x.get("rotation") for x in video.get("side_data_list", []) if "rotation" in x), 0),
            "has_audio": bool(audio_streams), "audio_stream_count": len(audio_streams),
            "video_start_time": video_start,
            "audio_video_start_offsets": audio_offsets,
            "audio_video_start_offset": audio_offsets[0] if audio_offsets else None}


def sample_frames(path: Path, output_dir: Path, count: int = 12, long_edge: int = 768) -> list[Path]:
    """Extract evenly spaced representative frames; no media leaves the Mac."""
    if count < 1 or count > 24:
        raise ValueError("frame count must be between 1 and 24")
    if long_edge < 128 or long_edge > 2048:
        raise ValueError("long_edge must be between 128 and 2048")
    info = probe_video(path)
    duration = info["duration"]
    if duration <= 0:
        raise ValueError("Video duration is unavailable")
    output_dir.mkdir(parents=True, exist_ok=True)
    source = Path(info["path"])
    outputs: list[Path] = []
    for i in range(count):
        # Avoid seeking exactly to EOF, which can yield an empty final frame.
        # Span the complete source, including a frame near its end so later scenes are visible.
        endpoint = max(0.0, duration - min(0.25, duration * 0.1))
        timestamp = duration / 2 if count == 1 else endpoint * i / (count - 1)
        target = output_dir / f"frame-{i + 1:02d}-{timestamp:.3f}s.jpg"
        subprocess.run([_binary("ffmpeg"), "-hide_banner", "-loglevel", "error", "-ss", f"{timestamp:.3f}",
                        "-i", str(source), "-frames:v", "1", "-vf",
                        f"scale={long_edge}:{long_edge}:force_original_aspect_ratio=decrease", "-q:v", "3", "-y", str(target)],
                       check=True, capture_output=True, text=True)
        outputs.append(target)
    return outputs


def make_mp4(source: Path, destination: Path, *, preserve_audio: bool = False) -> Path:
    """Remux provider output, retaining its first audio stream only when requested."""
    source = source.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    source_info = probe_video(source)
    if preserve_audio and not source_info["has_audio"]:
        raise RuntimeError("The provider returned no audio stream; refusing to mark this clip audio-complete.")
    destination = destination.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [_binary("ffmpeg"), "-hide_banner", "-loglevel", "error", "-i", str(source), "-map", "0:v:0"]
    if preserve_audio:
        command += ["-map", "0:a?", "-c:a", "copy"]
    else:
        command += ["-an"]
    command += ["-c:v", "copy", "-movflags", "+faststart", "-y", str(destination)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    info = probe_video(destination)
    if info["has_audio"] != preserve_audio:
        destination.unlink(missing_ok=True)
        expected = "an audio" if preserve_audio else "no audio"
        raise RuntimeError(f"MP4 audio validation failed: output does not contain {expected} stream as requested")
    if preserve_audio:
        if info["audio_stream_count"] != source_info["audio_stream_count"]:
            destination.unlink(missing_ok=True)
            raise RuntimeError("MP4 audio validation failed: not all source audio streams were preserved")
        source_offsets = source_info["audio_video_start_offsets"]
        output_offsets = info["audio_video_start_offsets"]
        if len(source_offsets) != source_info["audio_stream_count"] or len(output_offsets) != info["audio_stream_count"]:
            destination.unlink(missing_ok=True)
            raise RuntimeError("MP4 audio timestamp metadata is missing; synchronization could not be verified")
        if any(abs(before - after) > 0.02 for before, after in zip(source_offsets, output_offsets)):
            destination.unlink(missing_ok=True)
            raise RuntimeError("MP4 audio validation failed: audio/video timestamp offsets changed during remux")
    return destination


def make_silent_mp4(source: Path, destination: Path) -> Path:
    """Copy video without an audio stream, then verify the resulting container."""
    return make_mp4(source, destination, preserve_audio=False)
