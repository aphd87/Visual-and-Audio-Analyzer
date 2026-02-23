from __future__ import annotations
"""
pages_ingest.py — Ingest tab: video upload, URL input (YouTube/podcast/SoundCloud),
and transcript paste. Handles download progress, transcription, and bundle assembly.
"""

import os, tempfile, traceback
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd
import streamlit as st

from constants import COLOR_PALETTE, HAS_WHISPER, HAS_CV2, HAS_SCENEDETECT, HAS_LIBROSA, HAS_YTDLP
from ingestion import (
    download_url, get_url_preview, ingest_video, transcribe_audio,
    segments_to_scene_df, transcript_to_script_text, analyze_acoustics,
    build_visual_bundle, keyframe_to_png_bytes, content_hash,
)


def safe_render(page_fn, name: str = "Page"):
    try:
        page_fn()
    except Exception:
        st.error(f"❌ Crash while rendering: {name}")
        st.code(traceback.format_exc())


def _dep_banner():
    missing = []
    if not HAS_WHISPER:    missing.append("Whisper")
    if not HAS_CV2:        missing.append("OpenCV")
    if not HAS_LIBROSA:    missing.append("Librosa")
    if not HAS_YTDLP:      missing.append("yt-dlp")
    if missing:
        st.warning(f"**Limited functionality** — missing packages: {', '.join(missing)}. Check requirements.txt.")


def _metadata_row() -> Dict[str, Any]:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        title = st.text_input("Title", value="", key="va_title")
    with col2:
        content_type = st.selectbox("Type", ["TV Episode", "Movie", "Podcast", "Short Film", "Commercial", "Pilot"], key="va_type")
    with col3:
        genre = st.text_input("Genre", value="Drama", key="va_genre")
    with col4:
        year = st.number_input("Year", min_value=1950, max_value=2030, value=2024, key="va_year")
    return {"title": title.strip(), "type": content_type, "genre": genre.strip(), "year": int(year)}


def _whisper_options() -> tuple:
    col1, col2 = st.columns(2)
    with col1:
        model = st.selectbox(
            "Whisper Model",
            ["tiny", "base", "small", "medium"],
            index=1,
            help="tiny=fastest, medium=most accurate. 'base' works for most content.",
            key="va_whisper_model",
        )
    with col2:
        lang = st.text_input(
            "Language (optional)",
            value="",
            placeholder="e.g. en, es, fr — blank = auto-detect",
            key="va_language",
        )
    return model, lang.strip() if lang.strip() else None


def _run_pipeline(file_path: str, metadata: Dict, whisper_model: str, language: Optional[str], is_audio: bool):
    """Run the full ingestion pipeline on a local file and save to session state."""

    # Step 1: Video/scene ingestion (skip for audio-only)
    video_info = {"scene_cuts": [], "keyframes": [], "keyframe_times": [], "fps": 0, "duration_seconds": 0, "total_frames": 0}
    if not is_audio and HAS_CV2:
        with st.spinner("🎬 Detecting scenes and extracting keyframes..."):
            video_info = ingest_video(file_path)
            if video_info.get("error"):
                st.warning(f"Scene detection issue: {video_info['error']} — continuing with interval sampling.")
                video_info["scene_cuts"] = []
        n_cuts = len(video_info["scene_cuts"])
        duration = video_info["duration_seconds"]
        st.success(f"✅ Video: {duration:.0f}s — {n_cuts} scene cuts — {len(video_info['keyframes'])} keyframes")

    # Step 2: Transcription
    with st.spinner(f"🎙️ Transcribing with Whisper ({whisper_model})..."):
        tx = transcribe_audio(file_path, model_size=whisper_model, language=language)
        if tx.get("error"):
            st.error(f"Transcription failed: {tx['error']}")
            return

    word_count = len(tx["text"].split())
    st.success(f"✅ Transcription: {word_count:,} words — {len(tx['segments'])} segments — language: {tx['language']}")

    # Step 3: Scene df
    scene_df = segments_to_scene_df(tx["segments"], video_info["scene_cuts"])

    # Step 4: Acoustics
    acoustic_df = pd.DataFrame()
    if HAS_LIBROSA:
        with st.spinner("🎵 Analyzing acoustics..."):
            acoustic_df = analyze_acoustics(file_path, video_info["scene_cuts"])
        if not acoustic_df.empty:
            st.success(f"✅ Acoustics: {len(acoustic_df)} segments analyzed")

    # Step 5: Build bundle
    transcript_text = transcript_to_script_text(tx["segments"], video_info["scene_cuts"])
    bundle = build_visual_bundle(
        scene_df=scene_df,
        acoustic_df=acoustic_df,
        transcript_text=transcript_text,
        video_info=video_info,
        metadata=metadata,
    )

    # Persist
    st.session_state["current_visual_bundle"] = bundle
    st.session_state["current_visual_metadata"] = metadata
    st.session_state["current_transcript"] = tx["text"]
    st.session_state["current_keyframes"] = video_info.get("keyframes", [])
    st.session_state["current_keyframe_times"] = video_info.get("keyframe_times", [])

    st.success(f"🎉 Analysis complete — **{metadata.get('title') or 'Content'}** ready. Go to the **Analysis** tab.")


# ════════════════════════════════════════════════════════════════════════════
# MAIN PAGE
# ════════════════════════════════════════════════════════════════════════════

def page_ingest():
    st.subheader("📤 Ingest")
    st.caption("Upload a video/audio file, paste a URL (YouTube, podcast, SoundCloud), or paste a transcript directly.")

    _dep_banner()
    st.divider()

    st.markdown("### 📋 Content Details")
    meta_base = _metadata_row()
    st.divider()

    # ── Source selection ─────────────────────────────────────────────────────
    source = st.radio(
        "Source",
        ["🔗 URL (YouTube / Podcast / SoundCloud)", "📁 Upload File", "📝 Paste Transcript"],
        horizontal=True,
        key="va_source",
    )

    # ════════════════════════════════════════════════════════════════════════
    # BRANCH A — URL
    # ════════════════════════════════════════════════════════════════════════
    if source == "🔗 URL (YouTube / Podcast / SoundCloud)":

        st.markdown("### 🔗 Content URL")
        url = st.text_input(
            "Paste URL",
            placeholder="https://youtube.com/watch?v=... or https://open.spotify.com/episode/... or any podcast link",
            key="va_url_input",
        )

        # URL preview
        if url and url.startswith("http"):
            if st.button("👁️ Preview", key="va_preview_btn"):
                with st.spinner("Fetching info..."):
                    preview = get_url_preview(url)
                if preview.get("error"):
                    st.warning(f"Preview unavailable: {preview['error']}")
                else:
                    col_t, col_d = st.columns([3, 1])
                    col_t.markdown(f"**{preview['title']}**  \n{preview.get('uploader', '')}")
                    duration = preview.get("duration", 0)
                    col_d.metric("Duration", f"{int(duration//60)}m {int(duration%60)}s" if duration else "—")
                    if preview.get("description"):
                        st.caption(preview["description"])

        st.markdown("### ⚙️ Transcription Settings")
        whisper_model, language = _whisper_options()

        if st.button("⬇️ Download & Analyze", type="primary", key="va_url_btn", disabled=not HAS_YTDLP):
            if not url or not url.startswith("http"):
                st.warning("Please enter a valid URL.")
                return

            # Live progress display
            progress_placeholder = st.empty()

            def update_progress(msg: str):
                progress_placeholder.info(msg)

            # Download
            dl = download_url(url, progress_callback=update_progress)
            if dl.get("error"):
                st.error(dl["error"])
                return

            progress_placeholder.empty()

            # Fill metadata from URL info if user left title blank
            metadata = dict(meta_base)
            if not metadata["title"]:
                metadata["title"] = dl.get("title", "Untitled")
            metadata["source"] = "url"
            metadata["url"] = url
            metadata["uploader"] = dl.get("uploader", "")

            try:
                _run_pipeline(
                    file_path=dl["file_path"],
                    metadata=metadata,
                    whisper_model=whisper_model,
                    language=language,
                    is_audio=dl.get("is_audio_only", False),
                )
            finally:
                # Clean up downloaded file
                try:
                    os.unlink(dl["file_path"])
                    os.rmdir(os.path.dirname(dl["file_path"]))
                except Exception:
                    pass

        if not HAS_YTDLP:
            st.error("yt-dlp is not installed. Add `yt-dlp` to requirements.txt and redeploy.")

    # ════════════════════════════════════════════════════════════════════════
    # BRANCH B — FILE UPLOAD
    # ════════════════════════════════════════════════════════════════════════
    elif source == "📁 Upload File":
        st.markdown("### 📁 Upload Video or Audio")
        uploaded = st.file_uploader(
            "Choose file",
            type=["mp4", "mov", "mkv", "avi", "m4v", "mp3", "wav", "m4a", "ogg"],
            key="va_file_uploader",
        )

        if uploaded:
            ext = Path(uploaded.name).suffix.lower()
            is_audio = ext in [".mp3", ".wav", ".m4a", ".ogg"]
            st.info(f"📁 **{uploaded.name}** — {uploaded.size / 1_000_000:.1f} MB — {'Audio' if is_audio else 'Video'}")

            st.markdown("### ⚙️ Transcription Settings")
            whisper_model, language = _whisper_options()

            if st.button("🔍 Analyze", type="primary", key="va_file_btn"):
                metadata = dict(meta_base)
                if not metadata["title"]:
                    metadata["title"] = Path(uploaded.name).stem
                metadata["source"] = "upload"
                metadata["filename"] = uploaded.name

                with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                    tmp.write(uploaded.read())
                    tmp_path = tmp.name

                try:
                    _run_pipeline(
                        file_path=tmp_path,
                        metadata=metadata,
                        whisper_model=whisper_model,
                        language=language,
                        is_audio=is_audio,
                    )
                finally:
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass

    # ════════════════════════════════════════════════════════════════════════
    # BRANCH C — PASTE TRANSCRIPT
    # ════════════════════════════════════════════════════════════════════════
    else:
        st.markdown("### 📝 Paste Transcript")
        st.caption("Paste plain text, Whisper output, SRT subtitles, or any timestamped transcript.")
        transcript_input = st.text_area(
            "Transcript text",
            height=300,
            placeholder="Paste transcript here...",
            key="va_transcript_paste",
        )

        if st.button("🔍 Analyze Transcript", type="primary", key="va_transcript_btn"):
            if not transcript_input or len(transcript_input.strip()) < 50:
                st.warning("Please paste at least 50 characters.")
                return

            metadata = dict(meta_base)
            if not metadata["title"]:
                metadata["title"] = "Pasted Transcript"
            metadata["source"] = "paste"

            lines = [l.strip() for l in transcript_input.split("\n") if l.strip()]
            segments = [{"start": i * 5.0, "end": (i + 1) * 5.0, "text": line} for i, line in enumerate(lines)]
            scene_cuts = list(range(0, len(lines) * 5, 60))
            scene_df = segments_to_scene_df(segments, scene_cuts)

            bundle = build_visual_bundle(
                scene_df=scene_df,
                acoustic_df=pd.DataFrame(),
                transcript_text=transcript_input,
                video_info={"scene_cuts": scene_cuts, "keyframes": [], "keyframe_times": [],
                            "fps": 0, "duration_seconds": len(lines) * 5, "total_frames": 0},
                metadata=metadata,
            )

            st.session_state["current_visual_bundle"] = bundle
            st.session_state["current_visual_metadata"] = metadata
            st.session_state["current_transcript"] = transcript_input
            st.session_state["current_keyframes"] = []
            st.session_state["current_keyframe_times"] = []

            st.success(f"✅ Transcript analyzed — {len(scene_df)} scenes. Go to **Analysis** tab.")

    # ════════════════════════════════════════════════════════════════════════
    # QUICK SUMMARY (if content already loaded)
    # ════════════════════════════════════════════════════════════════════════
    bundle = st.session_state.get("current_visual_bundle")
    if bundle:
        st.divider()
        meta = st.session_state.get("current_visual_metadata", {})
        sm = bundle.get("summary", {})
        duration = sm.get("duration_seconds", 0)
        st.markdown(f"### ✅ Currently Loaded: **{meta.get('title', 'Untitled')}**")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Scenes", sm.get("scenes", 0))
        c2.metric("Words", f"{sm.get('dialogue_words', 0):,}")
        c3.metric("Duration", f"{int(duration//60)}m {int(duration%60)}s" if duration else "—")
        c4.metric("Source", meta.get("source", "—").capitalize())

        # Keyframe preview strip
        keyframes = st.session_state.get("current_keyframes", [])
        if keyframes and HAS_CV2:
            st.markdown("**Keyframe preview:**")
            preview_frames = keyframes[:8]
            cols = st.columns(len(preview_frames))
            kf_times = st.session_state.get("current_keyframe_times", [])
            for i, (col, frame) in enumerate(zip(cols, preview_frames)):
                with col:
                    t = kf_times[i] if i < len(kf_times) else 0
                    col.image(keyframe_to_png_bytes(frame), caption=f"{int(t//60)}:{int(t%60):02d}", use_column_width=True)
