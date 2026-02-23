from __future__ import annotations

import streamlit as st
from constants import HAS_WHISPER, HAS_CV2, HAS_SCENEDETECT, HAS_LIBROSA, HAS_YTDLP, HAS_ANTHROPIC
from pages_ingest import page_ingest, safe_render
from pages_analysis import page_analysis
from pages_compare import page_compare
from pages_academic import page_academic


# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ⚙️ System Status")
    for name, flag in [
        ("🎙️ Whisper — audio transcription",   HAS_WHISPER),
        ("🎬 OpenCV — video frame extraction", HAS_CV2),
        ("✂️ SceneDetect — scene cuts",        HAS_SCENEDETECT),
        ("🎵 Librosa — acoustic analysis",     HAS_LIBROSA),
        ("⬇️ yt-dlp — URL downloading",        HAS_YTDLP),
        ("🤖 Anthropic — AI summaries",        HAS_ANTHROPIC),
    ]:
        st.markdown(f"{'✅' if flag else '❌'} {name}")

    st.divider()
    st.markdown("## 📖 Library Reference")

    with st.expander("🎙️ Whisper"):
        st.caption(
            "**OpenAI Whisper** is a general-purpose speech recognition model trained on 680,000 hours "
            "of multilingual audio. It transcribes speech to text with high accuracy across 99 languages. "
            "Models range from `tiny` (fastest, ~39M params) to `large` (most accurate, ~1.5B params). "
            "In this app, `base` is the default balance of speed and accuracy for most content."
        )

    with st.expander("🎬 OpenCV"):
        st.caption(
            "**OpenCV** (Open Source Computer Vision Library) is the world's most widely used computer "
            "vision framework. Here it handles video file reading, frame extraction at precise timestamps, "
            "and keyframe capture at scene cut points. Supports MP4, MOV, MKV, AVI, and most containers."
        )

    with st.expander("✂️ SceneDetect"):
        st.caption(
            "**PySceneDetect** detects scene boundaries using content-aware analysis — it measures "
            "frame-to-frame differences in color histograms and luminance to identify cut points. "
            "This is how continuous video is segmented into discrete 'scenes' analogous to a script."
        )

    with st.expander("🎵 Librosa"):
        st.caption(
            "**Librosa** is a Python library for audio and music analysis. It extracts three acoustic "
            "features per scene: **RMS energy** (loudness/intensity), **tempo** (estimated BPM via "
            "beat tracking), and **pitch** (fundamental frequency). These enable acoustic intelligence "
            "analysis — identifying high-intensity moments and energy arcs across content."
        )

    with st.expander("⬇️ yt-dlp"):
        st.caption(
            "**yt-dlp** supports 1,000+ websites including YouTube, SoundCloud, Vimeo, Dailymotion, "
            "and podcast RSS feeds. It handles format selection, metadata retrieval, and audio extraction. "
            "Note: DRM-protected content (Spotify music, Apple Music) cannot be downloaded."
        )

    with st.expander("🤖 Anthropic / Claude"):
        st.caption(
            "**Claude** is used for AI-powered narrative intelligence summaries in the Analysis tab. "
            "Given a transcript, it generates a structured report covering narrative summary, emotional "
            "register, key themes, and analytical observations. Requires ANTHROPIC_API_KEY in secrets."
        )

    st.divider()
    st.caption("Companion: [Script Analyzer](https://github.com/aphd87/TV-and-Film-Script-Analyzer)")
    st.caption("Large-scale batch analysis: use `batch_runner.py` from the repo.")


# ════════════════════════════════════════════════════════════════════════════
# HEADER & INTRO
# ════════════════════════════════════════════════════════════════════════════

st.title("🎥 Video and Audio Analyzer")
st.markdown("### *Computational content intelligence for produced media*")

st.markdown("""
This platform applies rigorous computational analysis to **produced video and audio content** —
no script required. Upload a file, paste a URL, or point it at an entire YouTube channel or podcast feed.
The pipeline transcribes dialogue via Whisper, detects scene structure via OpenCV and SceneDetect,
extracts acoustic features via Librosa, and produces a full analytical suite covering narrative pacing,
NLP word analysis, topic modeling, and acoustic intelligence.

Designed for media researchers, entertainment industry analysts, and computational social scientists
working at the intersection of content analysis, econometrics, and machine learning.
""")

c1, c2, c3, c4 = st.columns(4)
c1.info("**📤 Ingest**\nYouTube, podcasts, file upload, or transcript paste")
c2.info("**📊 Analysis**\nScene structure, pacing, acoustics, NLP, AI summary")
c3.info("**⚖️ Compare**\nSide-by-side of up to 10 items")
c4.info("**🎓 Academic**\nBatch analysis, ML/econometrics, Excel export")

st.markdown("---")


# ════════════════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════════════════

tab_ingest, tab_analysis, tab_compare, tab_academic = st.tabs([
    "📤 Ingest",
    "📊 Analysis",
    "⚖️ Compare",
    "🎓 Academic",
])

with tab_ingest:
    safe_render(page_ingest, "Ingest")

with tab_analysis:
    safe_render(page_analysis, "Analysis")

with tab_compare:
    safe_render(page_compare, "Compare")

with tab_academic:
    safe_render(page_academic, "Academic")
