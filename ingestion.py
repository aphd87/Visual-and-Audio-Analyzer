from __future__ import annotations
"""
ingestion.py — Video, audio, and URL ingestion pipeline.

Sources supported:
  - Video file upload (mp4, mov, mkv, avi, m4v)
  - Audio file upload (mp3, wav, m4a, ogg)
  - YouTube URLs
  - Podcast URLs (RSS, direct mp3, SoundCloud, Spotify public episodes)
  - Any yt-dlp supported source (~1000+ sites)

Pipeline:
  1. download_url()          — yt-dlp URL → local audio/video file
  2. ingest_video()          — OpenCV keyframe + scene cut extraction
  3. transcribe_audio()      — Whisper audio → timestamped segments
  4. segments_to_scene_df()  — segments + cuts → scene_df
  5. analyze_acoustics()     — librosa energy/tempo/pitch per scene
  6. build_visual_bundle()   — assemble final bundle dict
"""

import io, os, re, math, json, hashlib, tempfile, traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable

import pandas as pd
import numpy as np

# ── Optional heavy imports ───────────────────────────────────────────────────
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

try:
    import yt_dlp
    HAS_YTDLP = True
except ImportError:
    HAS_YTDLP = False


# ════════════════════════════════════════════════════════════════════════════
# 1. URL DOWNLOAD (yt-dlp)
# ════════════════════════════════════════════════════════════════════════════

_DRM_DOMAINS = ["open.spotify.com/track", "open.spotify.com/album", "music.apple.com"]

def is_drm_url(url: str) -> bool:
    return any(d in url for d in _DRM_DOMAINS)

def is_audio_only_url(url: str) -> bool:
    audio_indicators = [
        "podcast", "episode", "soundcloud.com", "anchor.fm", "buzzsprout",
        "podbean", "libsyn", "transistor.fm", ".mp3", ".m4a", ".wav",
        "spotify.com/episode", "spotify.com/show",
    ]
    return any(ind in url.lower() for ind in audio_indicators)


def download_url(
    url: str,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """
    Download audio/video from a URL using yt-dlp.

    Returns dict with: file_path, title, duration, uploader,
                       is_audio_only, thumbnail_url, description, error
    """
    result = {
        "file_path": None, "title": "", "duration": 0.0, "uploader": "",
        "is_audio_only": False, "thumbnail_url": "", "description": "", "error": None,
    }

    if not HAS_YTDLP:
        result["error"] = "yt-dlp not installed. Run: pip install yt-dlp"
        return result

    if is_drm_url(url):
        result["error"] = (
            "This URL is DRM-protected (Spotify music / Apple Music tracks cannot be downloaded). "
            "For Spotify podcasts, try the episode URL. Most podcasts are also on YouTube or SoundCloud."
        )
        return result

    tmp_dir = tempfile.mkdtemp()
    output_template = os.path.join(tmp_dir, "%(title)s.%(ext)s")
    is_audio = is_audio_only_url(url)

    def _hook(d):
        if progress_callback is None:
            return
        if d.get("status") == "downloading":
            pct = d.get("_percent_str", "").strip()
            speed = d.get("_speed_str", "").strip()
            eta = d.get("_eta_str", "").strip()
            progress_callback(f"⬇️ Downloading... {pct}  |  {speed}  |  ETA {eta}")
        elif d.get("status") == "finished":
            progress_callback("✅ Download complete — processing file...")

    ydl_opts = {
        "outtmpl": output_template,
        "progress_hooks": [_hook],
        "quiet": True,
        "no_warnings": True,
    }

    if is_audio:
        ydl_opts.update({
            "format": "bestaudio/best",
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }],
        })
        result["is_audio_only"] = True
    else:
        ydl_opts["format"] = "bestvideo[height<=720]+bestaudio/best[height<=720]/best"

    try:
        if progress_callback:
            progress_callback("🔍 Fetching content info...")

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)

        downloaded = list(Path(tmp_dir).glob("*"))
        if not downloaded:
            result["error"] = "Download completed but no file found. Try a different URL."
            return result

        result["file_path"] = str(downloaded[0])
        result["title"] = info.get("title", "Unknown")
        result["duration"] = float(info.get("duration") or 0)
        result["uploader"] = info.get("uploader", info.get("channel", ""))
        result["thumbnail_url"] = info.get("thumbnail", "")
        result["description"] = (info.get("description") or "")[:500]

        if progress_callback:
            progress_callback(f"✅ Ready: {result['title']} ({result['duration']/60:.1f} min)")

    except Exception as e:
        err = str(e)
        if "Sign in" in err or "login" in err.lower():
            result["error"] = "Content requires authentication (private/members-only). Try a public URL."
        elif "Unsupported URL" in err:
            result["error"] = (
                "URL not recognized. Supported: YouTube, SoundCloud, Vimeo, podcast feeds, "
                "and 1000+ other sites. Direct Spotify music is not supported."
            )
        elif "not available" in err.lower():
            result["error"] = "Content unavailable — may be region-locked or removed."
        else:
            result["error"] = f"Download failed: {err[:300]}"

    return result


def get_url_preview(url: str) -> Dict[str, Any]:
    """Fetch title/duration metadata without downloading — for preview."""
    if not HAS_YTDLP:
        return {"error": "yt-dlp not installed"}
    if is_drm_url(url):
        return {"error": "DRM-protected URL — podcasts and video content only"}
    try:
        with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True}) as ydl:
            info = ydl.extract_info(url, download=False)
        return {
            "title": info.get("title", "Unknown"),
            "duration": float(info.get("duration") or 0),
            "uploader": info.get("uploader", info.get("channel", "")),
            "thumbnail_url": info.get("thumbnail", ""),
            "description": (info.get("description") or "")[:300],
            "error": None,
        }
    except Exception as e:
        return {"error": str(e)[:200]}


# ════════════════════════════════════════════════════════════════════════════
# 2. VIDEO INGESTION
# ════════════════════════════════════════════════════════════════════════════

def ingest_video(video_path: str, max_keyframes: int = 50) -> Dict[str, Any]:
    result = {
        "scene_cuts": [], "keyframes": [], "keyframe_times": [],
        "fps": 0.0, "duration_seconds": 0.0, "total_frames": 0, "error": None,
    }
    if not HAS_CV2:
        result["error"] = "opencv-python not installed."
        return result
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            result["error"] = f"Could not open file: {video_path}"
            return result
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        result.update({"fps": fps, "total_frames": total_frames, "duration_seconds": duration})

        if HAS_SCENEDETECT:
            try:
                video = open_video(video_path)
                sm = SceneManager()
                sm.add_detector(ContentDetector(threshold=27.0))
                sm.detect_scenes(video)
                result["scene_cuts"] = [s[0].get_seconds() for s in sm.get_scene_list()]
            except Exception:
                result["scene_cuts"] = list(np.arange(0, duration, 30.0))
        else:
            result["scene_cuts"] = list(np.arange(0, duration, 30.0))

        sample_times = result["scene_cuts"] or list(np.linspace(0, duration, min(max_keyframes, 20)))
        if len(sample_times) > max_keyframes:
            idxs = np.linspace(0, len(sample_times) - 1, max_keyframes, dtype=int)
            sample_times = [sample_times[i] for i in idxs]

        keyframes, keyframe_times = [], []
        for t in sample_times:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ret, frame = cap.read()
            if ret:
                keyframes.append(frame)
                keyframe_times.append(t)
        cap.release()
        result["keyframes"] = keyframes
        result["keyframe_times"] = keyframe_times
    except Exception:
        result["error"] = traceback.format_exc()
    return result


def keyframe_to_png_bytes(frame, max_width: int = 640) -> bytes:
    if not HAS_CV2:
        return b""
    h, w = frame.shape[:2]
    if w > max_width:
        frame = cv2.resize(frame, (max_width, int(h * max_width / w)))
    _, buf = cv2.imencode(".png", frame)
    return buf.tobytes()


# ════════════════════════════════════════════════════════════════════════════
# 3. AUDIO TRANSCRIPTION
# ════════════════════════════════════════════════════════════════════════════

_whisper_cache: Dict[str, Any] = {}

def load_whisper_model(size: str = "base"):
    if not HAS_WHISPER:
        return None
    if size not in _whisper_cache:
        _whisper_cache[size] = whisper.load_model(size)
    return _whisper_cache[size]


def transcribe_audio(
    file_path: str,
    model_size: str = "base",
    language: Optional[str] = None,
) -> Dict[str, Any]:
    result = {"text": "", "segments": [], "language": "en", "error": None}
    if not HAS_WHISPER:
        result["error"] = "openai-whisper not installed."
        return result
    try:
        model = load_whisper_model(model_size)
        opts = {}
        if language:
            opts["language"] = language
        tx = model.transcribe(file_path, **opts)
        result["text"] = tx.get("text", "").strip()
        result["segments"] = [
            {"start": s["start"], "end": s["end"], "text": s["text"].strip()}
            for s in tx.get("segments", [])
        ]
        result["language"] = tx.get("language", "en")
    except Exception:
        result["error"] = traceback.format_exc()
    return result


# ════════════════════════════════════════════════════════════════════════════
# 4. SEGMENTS → SCENE DF
# ════════════════════════════════════════════════════════════════════════════

def segments_to_scene_df(segments: List[Dict], scene_cuts: List[float]) -> pd.DataFrame:
    if not segments:
        return pd.DataFrame()
    max_time = max(s["end"] for s in segments)
    boundaries = (sorted(scene_cuts) + [float("inf")]) if scene_cuts else (list(range(0, int(max_time) + 60, 60)) + [float("inf")])
    rows = []
    for i in range(len(boundaries) - 1):
        segs = [s for s in segments if boundaries[i] <= s["start"] < boundaries[i + 1]]
        if not segs:
            continue
        text = " ".join(s["text"].strip() for s in segs)
        rows.append({
            "scene_id": len(rows) + 1,
            "start_time": round(segs[0]["start"], 1),
            "end_time": round(segs[-1]["end"], 1),
            "duration_seconds": round(segs[-1]["end"] - segs[0]["start"], 1),
            "dialogue_words": len(text.split()),
            "scene_text": text,
        })
    return pd.DataFrame(rows)


def transcript_to_script_text(segments: List[Dict], scene_cuts: List[float]) -> str:
    if not segments:
        return ""
    max_time = max(s["end"] for s in segments)
    boundaries = (sorted(scene_cuts) + [float("inf")]) if scene_cuts else (list(range(0, int(max_time) + 60, 60)) + [float("inf")])
    lines = []
    scene_num = 0
    for i in range(len(boundaries) - 1):
        segs = [s for s in segments if boundaries[i] <= s["start"] < boundaries[i + 1]]
        if not segs:
            continue
        scene_num += 1
        t_s, t_e = int(segs[0]["start"]), int(min(boundaries[i + 1], segs[-1]["end"]))
        lines.append(f"\nINT. SCENE {scene_num} - ({t_s//60}:{t_s%60:02d} - {t_e//60}:{t_e%60:02d})\n")
        for seg in segs:
            if seg["text"].strip():
                lines.append("SPEAKER\n" + seg["text"].strip() + "\n")
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════
# 5. ACOUSTIC ANALYSIS
# ════════════════════════════════════════════════════════════════════════════

def analyze_acoustics(file_path: str, scene_cuts: List[float]) -> pd.DataFrame:
    if not HAS_LIBROSA:
        return pd.DataFrame()
    try:
        y, sr = librosa.load(file_path, sr=22050, mono=True)
        duration = librosa.get_duration(y=y, sr=sr)
        boundaries = (sorted(scene_cuts) + [duration]) if scene_cuts else list(np.arange(0, duration + 30, 30.0))
        rows = []
        for i in range(len(boundaries) - 1):
            seg = y[int(boundaries[i] * sr):int(boundaries[i + 1] * sr)]
            if len(seg) < sr * 0.5:
                continue
            rms = float(np.sqrt(np.mean(seg ** 2)))
            try:
                tempo, _ = librosa.beat.beat_track(y=seg, sr=sr)
                tempo = float(tempo)
            except Exception:
                tempo = 0.0
            try:
                pitches, magnitudes = librosa.piptrack(y=seg, sr=sr)
                pv = pitches[magnitudes > np.median(magnitudes)]
                pitch_mean = float(np.mean(pv)) if len(pv) > 0 else 0.0
            except Exception:
                pitch_mean = 0.0
            rows.append({
                "scene_id": i + 1,
                "start_time": round(boundaries[i], 1),
                "end_time": round(boundaries[i + 1], 1),
                "rms_energy": round(rms, 4),
                "tempo_bpm": round(tempo, 1),
                "pitch_mean": round(pitch_mean, 1),
            })
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()


# ════════════════════════════════════════════════════════════════════════════
# 6. BUNDLE BUILDER
# ════════════════════════════════════════════════════════════════════════════

def build_visual_bundle(
    scene_df: pd.DataFrame,
    acoustic_df: pd.DataFrame,
    transcript_text: str,
    video_info: Dict[str, Any],
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    n = len(scene_df)
    words = int(scene_df["dialogue_words"].sum()) if not scene_df.empty else 0
    duration = video_info.get("duration_seconds", 0)

    char_df = pd.DataFrame([{
        "character": "SPEAKER", "scenes": n, "dialogue_words": words,
        "words": words, "lines_est": n, "dialogue_share": 1.0,
    }]) if n > 0 else pd.DataFrame()

    return {
        "summary": {
            "scenes": n, "characters": 1, "dialogue_words": words,
            "plotline_counts": {"Main": n}, "duration_seconds": round(duration, 1),
            "fps": video_info.get("fps", 0), "total_frames": video_info.get("total_frames", 0),
        },
        "scene_df": scene_df,
        "char_df": char_df,
        "emo_df": pd.DataFrame(),
        "cop_edges": pd.DataFrame(),
        "talk_edges": pd.DataFrame(),
        "acoustic_df": acoustic_df,
        "keyframe_times": video_info.get("keyframe_times", []),
        "scene_cuts": video_info.get("scene_cuts", []),
        "transcript_text": transcript_text,
        "metadata": metadata,
    }


def content_hash(data: bytes) -> str:
    return hashlib.md5(data[:65536]).hexdigest()[:8]
