from __future__ import annotations
import traceback
import streamlit as st

# ── Page config (must be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="TV & Film Visual Analyzer",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state init ───────────────────────────────────────────────────────
if "visual_library" not in st.session_state:
    st.session_state["visual_library"] = []
if "current_visual_bundle" not in st.session_state:
    st.session_state["current_visual_bundle"] = None
if "current_visual_metadata" not in st.session_state:
    st.session_state["current_visual_metadata"] = {}
if "current_transcript" not in st.session_state:
    st.session_state["current_transcript"] = ""

# ── Color palette (matches Script Analyzer for visual consistency) ───────────
COLOR_PALETTE = {
    "primary":   "#1e90ff",
    "secondary": "rgba(30, 144, 255, 0.15)",
    "accent":    "#ff6b6b",
    "neutral":   "#d3d3d3",
    "success":   "#4ecdc4",
    "warning":   "#ffe66d",
}

# ── Availability flags ───────────────────────────────────────────────────────
import importlib.util

HAS_WHISPER    = importlib.util.find_spec("whisper") is not None
HAS_CV2        = importlib.util.find_spec("cv2") is not None
HAS_SCENEDETECT = importlib.util.find_spec("scenedetect") is not None
HAS_LIBROSA    = importlib.util.find_spec("librosa") is not None
HAS_YTDLP      = importlib.util.find_spec("yt_dlp") is not None

try:
    from anthropic import Anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False
