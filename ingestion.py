from __future__ import annotations
"""
ingestion.py — Video, audio, and transcript ingestion pipeline.

Pipeline:
  1. ingest_video()        — OpenCV keyframe extraction + scene cut detection
  2. transcribe_audio()    — Whisper audio → text transcript
  3. transcript_to_bundle() — Reformat transcript into script-analyzer bundle format
  4. analyze_acoustics()   — librosa energy/tempo/pitch per scene
"""

import io, os, re, math, json, hashlib, tempfile, traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import pandas as pd
import numpy as np

# ── Optional heavy imports (guarded) ────────────────────────────────────────
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import whisper
    HAS_WHISPER = True
except ImportError:
    HAS_WHISPER = False

try:
    from scenedetect import open_video, SceneManager
    from scenedetect.detectors import ContentDetector
    HAS_SCENEDETECT = True
except ImportError:
    HAS_SCENEDETECT = False

try:
    import librosa
    import soundfile as sf
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False


# ════════════════════════════════════════════════════════════════════════════
# 1. VIDEO INGESTION
# ════════════════════════════════════════════════════════════════════════════

def ingest_video(video_path: str, max_keyframes: int = 50) -> Dict[str, Any]:
    """
    Extract scene cuts and keyframes from a video file.

    Returns:
        {
          "scene_cuts": List[float],        # timestamps in seconds
          "keyframes": List[np.ndarray],    # BGR images
          "keyframe_times": List[float],    # timestamps for each keyframe
          "fps": float,
          "duration_seconds": float,
          "total_frames": int,
          "error": str | None
        }
    """
    result = {
        "scene_cuts": [],
        "keyframes": [],
        "keyframe_times": [],
        "fps": 0.0,
        "duration_seconds": 0.0,
        "total_frames": 0,
        "error": None,
    }

    if not HAS_CV2:
        result["error"] = "opencv-python not installed. Run: pip install opencv-python"
        return result

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            result["error"] = f"Could not open video file: {video_path}"
            return result

        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        result["fps"] = fps
        result["total_frames"] = total_frames
        result["duration_seconds"] = duration

        # Scene detection
        if HAS_SCENEDETECT:
            try:
                video = open_video(video_path)
                scene_manager = SceneManager()
                scene_manager.add_detector(ContentDetector(threshold=27.0))
                scene_manager.detect_scenes(video)
                scene_list = scene_manager.get_scene_list()
                result["scene_cuts"] = [s[0].get_seconds() for s in scene_list]
            except Exception as e:
                # Fall back to simple interval sampling
                result["scene_cuts"] = list(np.arange(0, duration, 30.0))
        else:
            # No scene detection — sample every 30 seconds
            result["scene_cuts"] = list(np.arange(0, duration, 30.0))

        # Keyframe extraction — evenly spaced up to max_keyframes
        if result["scene_cuts"]:
            sample_times = result["scene_cuts"]
        else:
            sample_times = list(np.linspace(0, duration, min(max_keyframes, 20)))

        # Limit keyframes
        if len(sample_times) > max_keyframes:
            indices = np.linspace(0, len(sample_times) - 1, max_keyframes, dtype=int)
            sample_times = [sample_times[i] for i in indices]

        keyframes = []
        keyframe_times = []
        for t in sample_times:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ret, frame = cap.read()
            if ret:
                keyframes.append(frame)
                keyframe_times.append(t)

        cap.release()
        result["keyframes"] = keyframes
        result["keyframe_times"] = keyframe_times

    except Exception as e:
        result["error"] = f"Video ingestion error: {traceback.format_exc()}"

    return result


def keyframe_to_png_bytes(frame: np.ndarray, max_width: int = 640) -> bytes:
    """Convert OpenCV BGR frame to PNG bytes for display in Streamlit."""
    if not HAS_CV2:
        return b""
    h, w = frame.shape[:2]
    if w > max_width:
        scale = max_width / w
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
    _, buf = cv2.imencode(".png", frame)
    return buf.tobytes()


# ════════════════════════════════════════════════════════════════════════════
# 2. AUDIO TRANSCRIPTION
# ════════════════════════════════════════════════════════════════════════════

_whisper_model_cache: Dict[str, Any] = {}

def load_whisper_model(model_size: str = "base") -> Optional[Any]:
    """Load and cache a Whisper model."""
    if not HAS_WHISPER:
        return None
    if model_size not in _whisper_model_cache:
        _whisper_model_cache[model_size] = whisper.load_model(model_size)
    return _whisper_model_cache[model_size]


def transcribe_audio(
    video_path: str,
    model_size: str = "base",
    language: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Transcribe audio from a video file using Whisper.

    Returns:
        {
          "text": str,                      # full transcript
          "segments": List[Dict],           # [{start, end, text}, ...]
          "language": str,
          "error": str | None
        }
    """
    result = {"text": "", "segments": [], "language": "en", "error": None}

    if not HAS_WHISPER:
        result["error"] = "openai-whisper not installed. Run: pip install openai-whisper"
        return result

    try:
        model = load_whisper_model(model_size)
        if model is None:
            result["error"] = "Failed to load Whisper model."
            return result

        options = {}
        if language:
            options["language"] = language

        transcription = model.transcribe(video_path, **options)
        result["text"] = transcription.get("text", "").strip()
        result["segments"] = [
            {
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"].strip(),
            }
            for seg in transcription.get("segments", [])
        ]
        result["language"] = transcription.get("language", "en")

    except Exception as e:
        result["error"] = f"Transcription error: {traceback.format_exc()}"

    return result


# ════════════════════════════════════════════════════════════════════════════
# 3. TRANSCRIPT → SCRIPT-ANALYZER BUNDLE FORMAT
# ════════════════════════════════════════════════════════════════════════════

def transcript_to_script_text(segments: List[Dict], scene_cuts: List[float]) -> str:
    """
    Convert Whisper segments + scene cut timestamps into a script-like
    text format that analyze_script() can parse.

    Format produced:
        INT. SCENE 1 - (0:00 - 0:45)

        NARRATOR
        Dialogue from segment...

        INT. SCENE 2 - (0:45 - 1:30)
        ...
    """
    if not segments:
        return ""

    # Build scene boundaries from scene cuts
    if scene_cuts:
        boundaries = sorted(scene_cuts) + [float("inf")]
    else:
        # No scene cuts — group into 60-second blocks
        max_time = max(s["end"] for s in segments) if segments else 60
        boundaries = list(range(0, int(max_time) + 60, 60)) + [float("inf")]

    lines = []
    current_scene = 0

    for i, boundary_end in enumerate(boundaries[1:], start=1):
        boundary_start = boundaries[i - 1]

        # Collect segments in this scene
        scene_segs = [
            s for s in segments
            if s["start"] >= boundary_start and s["start"] < boundary_end
        ]
        if not scene_segs:
            continue

        start_ts = _fmt_timestamp(boundary_start)
        end_ts = _fmt_timestamp(min(boundary_end, scene_segs[-1]["end"]))
        current_scene += 1

        lines.append(f"\nINT. SCENE {current_scene} - ({start_ts} - {end_ts})\n")

        for seg in scene_segs:
            text = seg["text"].strip()
            if text:
                lines.append("NARRATOR")
                lines.append(text)
                lines.append("")

    return "\n".join(lines)


def _fmt_timestamp(seconds: float) -> str:
    if seconds == float("inf"):
        return "--:--"
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}:{s:02d}"


def segments_to_scene_df(segments: List[Dict], scene_cuts: List[float]) -> pd.DataFrame:
    """
    Build a scene_df directly from Whisper segments without going through
    the script text parser. Useful when analyze_script() isn't available.
    """
    if not segments:
        return pd.DataFrame()

    if scene_cuts:
        boundaries = sorted(scene_cuts) + [float("inf")]
    else:
        max_time = max(s["end"] for s in segments) if segments else 60
        boundaries = list(range(0, int(max_time) + 60, 60)) + [float("inf")]

    rows = []
    for i, boundary_end in enumerate(boundaries[1:], start=1):
        boundary_start = boundaries[i - 1]
        scene_segs = [
            s for s in segments
            if s["start"] >= boundary_start and s["start"] < boundary_end
        ]
        if not scene_segs:
            continue

        text = " ".join(s["text"].strip() for s in scene_segs)
        words = len(text.split())
        duration = scene_segs[-1]["end"] - scene_segs[0]["start"]

        rows.append({
            "scene_id": i,
            "start_time": scene_segs[0]["start"],
            "end_time": scene_segs[-1]["end"],
            "duration_seconds": round(duration, 1),
            "dialogue_words": words,
            "scene_text": text,
            "characters": ["NARRATOR"],
        })

    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════════
# 4. ACOUSTIC ANALYSIS
# ════════════════════════════════════════════════════════════════════════════

def analyze_acoustics(video_path: str, scene_cuts: List[float]) -> pd.DataFrame:
    """
    Extract acoustic features (energy, tempo, pitch) per scene using librosa.

    Returns DataFrame with columns:
        scene_id, start_time, end_time, rms_energy, tempo_bpm, pitch_mean
    """
    if not HAS_LIBROSA:
        return pd.DataFrame()

    try:
        y, sr = librosa.load(video_path, sr=22050, mono=True)
        duration = librosa.get_duration(y=y, sr=sr)

        if scene_cuts:
            boundaries = sorted(scene_cuts) + [duration]
        else:
            boundaries = list(np.arange(0, duration, 30.0)) + [duration]

        rows = []
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]
            start_sample = int(start * sr)
            end_sample = int(end * sr)
            segment = y[start_sample:end_sample]

            if len(segment) < sr * 0.5:  # Skip very short segments
                continue

            # RMS energy
            rms = float(np.sqrt(np.mean(segment ** 2)))

            # Tempo
            try:
                tempo, _ = librosa.beat.beat_track(y=segment, sr=sr)
                tempo = float(tempo)
            except Exception:
                tempo = 0.0

            # Pitch (mean fundamental frequency)
            try:
                pitches, magnitudes = librosa.piptrack(y=segment, sr=sr)
                pitch_vals = pitches[magnitudes > np.median(magnitudes)]
                pitch_mean = float(np.mean(pitch_vals)) if len(pitch_vals) > 0 else 0.0
            except Exception:
                pitch_mean = 0.0

            rows.append({
                "scene_id": i + 1,
                "start_time": round(start, 1),
                "end_time": round(end, 1),
                "rms_energy": round(rms, 4),
                "tempo_bpm": round(tempo, 1),
                "pitch_mean": round(pitch_mean, 1),
            })

        return pd.DataFrame(rows)

    except Exception as e:
        return pd.DataFrame()


# ════════════════════════════════════════════════════════════════════════════
# 5. BUNDLE BUILDER
# ════════════════════════════════════════════════════════════════════════════

def build_visual_bundle(
    scene_df: pd.DataFrame,
    acoustic_df: pd.DataFrame,
    transcript_text: str,
    video_info: Dict[str, Any],
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Assemble the visual bundle — a superset of the script analyzer bundle
    with additional visual/audio layers.

    Compatible keys with script analyzer: summary, scene_df, char_df, emo_df, cop_edges, talk_edges
    Additional keys: acoustic_df, keyframe_times, scene_cuts, transcript_text
    """
    total_scenes = len(scene_df)
    total_words = int(scene_df["dialogue_words"].sum()) if not scene_df.empty else 0
    duration = video_info.get("duration_seconds", 0)

    summary = {
        "scenes": total_scenes,
        "characters": 1,  # Will be enriched if NLP pipeline available
        "dialogue_words": total_words,
        "plotline_counts": {"Main": total_scenes},
        "duration_seconds": round(duration, 1),
        "fps": video_info.get("fps", 0),
        "total_frames": video_info.get("total_frames", 0),
    }

    # Minimal char_df for compatibility with script analyzer charts
    if not scene_df.empty:
        char_df = pd.DataFrame([{
            "character": "NARRATOR",
            "scenes": total_scenes,
            "dialogue_words": total_words,
            "words": total_words,
            "lines_est": total_scenes,
            "dialogue_share": 1.0,
        }])
    else:
        char_df = pd.DataFrame()

    return {
        "summary": summary,
        "scene_df": scene_df,
        "char_df": char_df,
        "emo_df": pd.DataFrame(),       # populated later if NLP available
        "cop_edges": pd.DataFrame(),    # populated later if multi-character
        "talk_edges": pd.DataFrame(),   # populated later if multi-character
        "acoustic_df": acoustic_df,
        "keyframe_times": video_info.get("keyframe_times", []),
        "scene_cuts": video_info.get("scene_cuts", []),
        "transcript_text": transcript_text,
        "metadata": metadata,
    }


def video_hash(video_bytes: bytes) -> str:
    return hashlib.md5(video_bytes[:65536]).hexdigest()[:8]
