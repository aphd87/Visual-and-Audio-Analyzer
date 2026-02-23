from __future__ import annotations
import importlib.util
import streamlit as st

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Video and Audio Analyzer",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state ────────────────────────────────────────────────────────────
for key in ["current_visual_bundle", "current_visual_metadata", "current_transcript",
            "current_keyframes", "current_keyframe_times",
            "ca_bundle", "cb_bundle"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ── Color palette ────────────────────────────────────────────────────────────
COLOR_PALETTE = {
    "primary":   "#1e90ff",
    "secondary": "rgba(30, 144, 255, 0.15)",
    "accent":    "#ff6b6b",
    "neutral":   "#d3d3d3",
    "success":   "#4ecdc4",
    "warning":   "#ffe66d",
}

# ── Dependency flags ─────────────────────────────────────────────────────────
HAS_WHISPER     = importlib.util.find_spec("whisper")      is not None
HAS_CV2         = importlib.util.find_spec("cv2")          is not None
HAS_SCENEDETECT = importlib.util.find_spec("scenedetect")  is not None
HAS_LIBROSA     = importlib.util.find_spec("librosa")      is not None
HAS_YTDLP       = importlib.util.find_spec("yt_dlp")       is not None

try:
    from anthropic import Anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False
