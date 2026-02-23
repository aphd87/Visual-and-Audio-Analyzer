from __future__ import annotations
"""
app.py — TV & Film Visual Analyzer
Entry point for Streamlit Cloud.

Repo: TV-and-Film-Visual-Analyzer
Companion to: TV-and-Film-Script-Analyzer

Pipeline:
  Video/Audio → Whisper transcription → Scene detection → Acoustic analysis
  → Same analytical bundle format as Script Analyzer → Charts & insights
"""

import traceback
import streamlit as st

# constants.py must be imported first (sets page config)
from constants import *

from pages_upload import page_upload_and_analyze, safe_render
from pages_analysis import page_analysis
from pages_library import page_library


# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.title("🎥 TV & Film Visual Analyzer")
st.caption(
    "Ingest produced video content — transcribe audio, detect scenes, extract acoustic features, "
    "and apply narrative analytics to episodes and films without needing a script."
)

# ── Dependency status sidebar ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ System Status")
    deps = {
        "🎙️ Whisper": HAS_WHISPER,
        "🎬 OpenCV": HAS_CV2,
        "✂️ SceneDetect": HAS_SCENEDETECT,
        "🎵 Librosa": HAS_LIBROSA,
        "🤖 Anthropic": HAS_ANTHROPIC,
    }
    for name, available in deps.items():
        status = "✅" if available else "❌"
        st.write(f"{status} {name}")

    st.divider()
    st.markdown("### 📖 About")
    st.caption(
        "Visual Analyzer works on **produced content** — no script required. "
        "Upload a video file, transcribe it, and run the full analytics pipeline."
    )
    st.caption("For **script-only** analysis, use the [Script Analyzer](https://github.com/aphd87/TV-and-Film-Script-Analyzer).")

# ── Navigation tabs ───────────────────────────────────────────────────────────
tab_ingest, tab_analysis, tab_library = st.tabs([
    "📤 Ingest",
    "📊 Analysis",
    "📚 Library",
])

with tab_ingest:
    safe_render(page_upload_and_analyze, "Ingest")

with tab_analysis:
    safe_render(page_analysis, "Analysis")

with tab_library:
    safe_render(page_library, "Library")
