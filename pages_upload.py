from __future__ import annotations
"""
pages_upload.py — Upload & Analyze page for TV & Film Visual Analyzer.

Handles:
  - Video file upload or YouTube URL
  - Whisper transcription with model size selection
  - Scene detection and keyframe grid
  - Acoustic analysis
  - Bundle assembly and session state persistence
"""

import io, os, re, tempfile, traceback
from pathlib import Path
from typing import Dict, List, Any, Optional

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from constants import (
    COLOR_PALETTE, HAS_WHISPER, HAS_CV2, HAS_SCENEDETECT,
    HAS_LIBROSA, HAS_YTDLP, HAS_ANTHROPIC
)
from ingestion import (
    ingest_video, transcribe_audio, segments_to_scene_df,
    transcript_to_script_text, analyze_acoustics,
    build_visual_bundle, keyframe_to_png_bytes, video_hash
)


def safe_render(page_fn, name: str = "Page"):
    try:
        page_fn()
    except Exception:
        st.error(f"❌ Crash while rendering: {name}")
        st.code(traceback.format_exc())


# ════════════════════════════════════════════════════════════════════════════
# DEPENDENCY STATUS BANNER
# ════════════════════════════════════════════════════════════════════════════

def _render_dep_status():
    deps = {
        "🎙️ Whisper (transcription)": HAS_WHISPER,
        "🎬 OpenCV (video)": HAS_CV2,
        "✂️ SceneDetect": HAS_SCENEDETECT,
        "🎵 Librosa (acoustics)": HAS_LIBROSA,
    }
    missing = [k for k, v in deps.items() if not v]
    if missing:
        st.warning(
            f"**Some features unavailable** — missing: {', '.join(missing)}. "
            "Check that all packages in requirements.txt are installed."
        )


# ════════════════════════════════════════════════════════════════════════════
# MAIN PAGE
# ════════════════════════════════════════════════════════════════════════════

def page_upload_and_analyze():
    st.subheader("🎥 Upload & Analyze")
    st.caption("Ingest produced video content — extract transcript, scenes, and acoustic features for analysis.")

    _render_dep_status()
    st.divider()

    # ── Input method ────────────────────────────────────────────────────────
    st.markdown("### 📤 Input")
    input_method = st.radio(
        "Source",
        ["Upload Video File", "Paste Transcript (text only)"],
        horizontal=True,
        key="va_input_method",
    )

    # ── Metadata ────────────────────────────────────────────────────────────
    st.markdown("### 📋 Content Details")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        title = st.text_input("Title", value="", key="va_title")
    with col2:
        content_type = st.selectbox("Type", ["TV Episode", "Movie", "Short Film", "Commercial", "Pilot"], key="va_type")
    with col3:
        genre = st.text_input("Genre", value="Drama", key="va_genre")
    with col4:
        year = st.number_input("Year", min_value=1950, max_value=2030, value=2024, key="va_year")

    st.divider()

    # ════════════════════════════════════════════════════════════════════════
    # BRANCH A: VIDEO FILE UPLOAD
    # ════════════════════════════════════════════════════════════════════════
    if input_method == "Upload Video File":
        uploaded_file = st.file_uploader(
            "Choose a video file",
            type=["mp4", "mov", "mkv", "avi", "m4v"],
            key="va_video_uploader",
        )

        if uploaded_file:
            st.info(f"📁 **{uploaded_file.name}** — {uploaded_file.size / 1_000_000:.1f} MB")

            # Whisper model selection
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                whisper_model = st.selectbox(
                    "Whisper Model",
                    ["tiny", "base", "small", "medium"],
                    index=1,
                    help="larger = more accurate but slower. 'base' is recommended for most use cases.",
                    key="va_whisper_model",
                )
            with col_m2:
                language = st.text_input(
                    "Language (optional)",
                    value="",
                    placeholder="e.g. en, es, fr — leave blank for auto-detect",
                    key="va_language",
                )

            if st.button("🔍 Analyze Video", type="primary", key="va_analyze_btn"):
                with tempfile.NamedTemporaryFile(suffix=Path(uploaded_file.name).suffix, delete=False) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                try:
                    metadata = {
                        "title": title.strip() or uploaded_file.name,
                        "type": content_type,
                        "genre": genre.strip(),
                        "year": int(year),
                        "source": "upload",
                        "filename": uploaded_file.name,
                    }

                    # Step 1: Video ingestion
                    with st.spinner("🎬 Extracting scenes and keyframes..."):
                        video_info = ingest_video(tmp_path)
                        if video_info.get("error"):
                            st.error(f"Video error: {video_info['error']}")
                            return

                    st.success(
                        f"✅ Video loaded — {video_info['duration_seconds']:.0f}s, "
                        f"{len(video_info['scene_cuts'])} scene cuts detected, "
                        f"{len(video_info['keyframes'])} keyframes extracted."
                    )

                    # Step 2: Transcription
                    with st.spinner(f"🎙️ Transcribing audio with Whisper ({whisper_model})..."):
                        lang = language.strip() if language.strip() else None
                        transcript_result = transcribe_audio(tmp_path, model_size=whisper_model, language=lang)
                        if transcript_result.get("error"):
                            st.error(f"Transcription error: {transcript_result['error']}")
                            return

                    word_count = len(transcript_result["text"].split())
                    st.success(
                        f"✅ Transcription complete — {word_count:,} words, "
                        f"{len(transcript_result['segments'])} segments, "
                        f"language: {transcript_result['language']}"
                    )

                    # Step 3: Scene df from segments
                    with st.spinner("📊 Building scene structure..."):
                        scene_df = segments_to_scene_df(
                            transcript_result["segments"],
                            video_info["scene_cuts"],
                        )

                    # Step 4: Acoustic analysis
                    acoustic_df = pd.DataFrame()
                    if HAS_LIBROSA:
                        with st.spinner("🎵 Analyzing acoustics..."):
                            acoustic_df = analyze_acoustics(tmp_path, video_info["scene_cuts"])

                    # Step 5: Assemble bundle
                    transcript_text = transcript_to_script_text(
                        transcript_result["segments"],
                        video_info["scene_cuts"],
                    )

                    bundle = build_visual_bundle(
                        scene_df=scene_df,
                        acoustic_df=acoustic_df,
                        transcript_text=transcript_text,
                        video_info=video_info,
                        metadata=metadata,
                    )

                    # Persist to session state
                    st.session_state["current_visual_bundle"] = bundle
                    st.session_state["current_visual_metadata"] = metadata
                    st.session_state["current_transcript"] = transcript_result["text"]
                    st.session_state["current_keyframes"] = video_info["keyframes"]
                    st.session_state["current_keyframe_times"] = video_info["keyframe_times"]

                    # Add to library
                    vid_id = video_hash(uploaded_file.getvalue() if hasattr(uploaded_file, 'getvalue') else b"")
                    _add_to_visual_library(vid_id, metadata, bundle)

                    st.success(f"✅ Analysis complete — saved to Visual Library: {metadata['title']}")

                finally:
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass

    # ════════════════════════════════════════════════════════════════════════
    # BRANCH B: PASTE TRANSCRIPT
    # ════════════════════════════════════════════════════════════════════════
    else:
        transcript_input = st.text_area(
            "Paste transcript text",
            height=300,
            placeholder="Paste a plain transcript here — can be Whisper output, a .srt file, or any time-stamped text...",
            key="va_transcript_paste",
        )

        if st.button("🔍 Analyze Transcript", type="primary", key="va_transcript_btn"):
            if not transcript_input or len(transcript_input.strip()) < 50:
                st.warning("Please paste at least 50 characters of transcript text.")
                return

            metadata = {
                "title": title.strip() or "Untitled",
                "type": content_type,
                "genre": genre.strip(),
                "year": int(year),
                "source": "transcript_paste",
            }

            with st.spinner("📊 Analyzing transcript..."):
                # Parse as simple segments (no timestamps)
                raw_lines = [l.strip() for l in transcript_input.split("\n") if l.strip()]
                segments = [{"start": i * 5.0, "end": (i + 1) * 5.0, "text": line}
                            for i, line in enumerate(raw_lines)]
                scene_cuts = list(range(0, len(raw_lines) * 5, 60))
                scene_df = segments_to_scene_df(segments, scene_cuts)
                transcript_text = transcript_input

                bundle = build_visual_bundle(
                    scene_df=scene_df,
                    acoustic_df=pd.DataFrame(),
                    transcript_text=transcript_text,
                    video_info={"scene_cuts": scene_cuts, "keyframes": [], "keyframe_times": [],
                                "fps": 0, "duration_seconds": len(raw_lines) * 5, "total_frames": 0},
                    metadata=metadata,
                )

            st.session_state["current_visual_bundle"] = bundle
            st.session_state["current_visual_metadata"] = metadata
            st.session_state["current_transcript"] = transcript_input
            st.session_state["current_keyframes"] = []
            st.session_state["current_keyframe_times"] = []

            _add_to_visual_library("paste_" + str(abs(hash(transcript_input)))[:8], metadata, bundle)
            st.success(f"✅ Transcript analyzed — {len(scene_df)} scenes built.")

    # ════════════════════════════════════════════════════════════════════════
    # RESULTS (renders from session state, survives widget interactions)
    # ════════════════════════════════════════════════════════════════════════
    bundle = st.session_state.get("current_visual_bundle")
    if bundle is None:
        return

    metadata = st.session_state.get("current_visual_metadata", {})
    scene_df = bundle.get("scene_df", pd.DataFrame())
    acoustic_df = bundle.get("acoustic_df", pd.DataFrame())
    keyframes = st.session_state.get("current_keyframes", [])
    keyframe_times = st.session_state.get("current_keyframe_times", [])

    st.divider()
    st.markdown(f"## 📊 Results — {metadata.get('title', 'Untitled')}")

    # ── Summary metrics ──────────────────────────────────────────────────────
    sm = bundle.get("summary", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Scenes", f"{sm.get('scenes', 0):,}")
    c2.metric("Dialogue Words", f"{sm.get('dialogue_words', 0):,}")
    duration = sm.get("duration_seconds", 0)
    c3.metric("Duration", f"{int(duration//60)}m {int(duration%60)}s" if duration else "—")
    c4.metric("Scene Cuts", f"{len(bundle.get('scene_cuts', [])):,}")

    # ── Scene pacing chart ───────────────────────────────────────────────────
    if not scene_df.empty and "dialogue_words" in scene_df.columns:
        st.markdown("### 📈 Scene Pacing — Dialogue Words per Scene")
        fig = px.bar(
            scene_df,
            x="scene_id",
            y="dialogue_words",
            title="Dialogue Density Across Scenes",
            color="dialogue_words",
            color_continuous_scale=["#d3d3d3", "#87ceeb", "#1e90ff"],
        )
        fig.update_layout(height=350, xaxis_title="Scene", yaxis_title="Words", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # ── Acoustic analysis ────────────────────────────────────────────────────
    if not acoustic_df.empty:
        st.markdown("### 🎵 Acoustic Profile")
        col_a, col_b = st.columns(2)

        with col_a:
            fig_energy = px.line(
                acoustic_df,
                x="scene_id",
                y="rms_energy",
                title="Audio Energy per Scene",
                markers=True,
                color_discrete_sequence=[COLOR_PALETTE["primary"]],
            )
            fig_energy.update_layout(height=300, xaxis_title="Scene", yaxis_title="RMS Energy")
            st.plotly_chart(fig_energy, use_container_width=True)

        with col_b:
            if "tempo_bpm" in acoustic_df.columns and acoustic_df["tempo_bpm"].sum() > 0:
                fig_tempo = px.bar(
                    acoustic_df,
                    x="scene_id",
                    y="tempo_bpm",
                    title="Estimated Tempo per Scene (BPM)",
                    color_discrete_sequence=[COLOR_PALETTE["accent"]],
                )
                fig_tempo.update_layout(height=300, xaxis_title="Scene", yaxis_title="BPM")
                st.plotly_chart(fig_tempo, use_container_width=True)

    # ── Keyframe grid ────────────────────────────────────────────────────────
    if keyframes and HAS_CV2:
        st.markdown("### 🖼️ Keyframe Grid")
        st.caption(f"{len(keyframes)} keyframes extracted at scene cut points.")

        cols_per_row = 4
        for row_start in range(0, min(len(keyframes), 20), cols_per_row):
            cols = st.columns(cols_per_row)
            for col_idx, frame_idx in enumerate(range(row_start, min(row_start + cols_per_row, len(keyframes)))):
                with cols[col_idx]:
                    png_bytes = keyframe_to_png_bytes(keyframes[frame_idx])
                    t = keyframe_times[frame_idx] if frame_idx < len(keyframe_times) else 0
                    m, s = int(t // 60), int(t % 60)
                    st.image(png_bytes, caption=f"{m}:{s:02d}", use_column_width=True)

    # ── Transcript viewer ────────────────────────────────────────────────────
    transcript = st.session_state.get("current_transcript", "")
    if transcript:
        with st.expander("📝 Full Transcript", expanded=False):
            st.text_area("Transcript", value=transcript, height=400, key="va_transcript_viewer", disabled=True)
            st.download_button(
                "⬇️ Download Transcript (.txt)",
                data=transcript.encode("utf-8"),
                file_name=f"{metadata.get('title', 'transcript')}.txt",
                mime="text/plain",
            )


# ════════════════════════════════════════════════════════════════════════════
# LIBRARY HELPER
# ════════════════════════════════════════════════════════════════════════════

def _add_to_visual_library(vid_id: str, metadata: Dict, bundle: Dict):
    """Add analyzed content to the visual library in session state."""
    lib = st.session_state.get("visual_library", [])
    existing_ids = [item.get("id") for item in lib]
    if vid_id not in existing_ids:
        lib.append({
            "id": vid_id,
            "title": metadata.get("title", "Untitled"),
            "metadata": metadata,
            "bundle": bundle,
        })
        st.session_state["visual_library"] = lib
