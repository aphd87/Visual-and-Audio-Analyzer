from __future__ import annotations
"""
pages_compare.py — Compare two pieces of content side by side.
Each item can come from a URL, file upload, or transcript paste.
"""

import os, tempfile, traceback
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from constants import COLOR_PALETTE, HAS_WHISPER, HAS_CV2, HAS_LIBROSA, HAS_YTDLP
from ingestion import (
    download_url, ingest_video, transcribe_audio,
    segments_to_scene_df, analyze_acoustics,
    build_visual_bundle, content_hash,
)


def _ingest_one(
    label: str,
    key_prefix: str,
) -> Optional[Dict[str, Any]]:
    """
    Render ingest controls for one item and return a bundle dict when ready,
    or None if not yet analyzed.
    """
    st.markdown(f"#### {label}")

    source = st.radio(
        "Source",
        ["🔗 URL", "📁 Upload", "📝 Paste Transcript"],
        horizontal=True,
        key=f"{key_prefix}_source",
    )

    col1, col2 = st.columns(2)
    with col1:
        title = st.text_input("Title", key=f"{key_prefix}_title")
    with col2:
        content_type = st.selectbox("Type", ["TV Episode", "Movie", "Podcast", "Short Film", "Other"], key=f"{key_prefix}_type")

    whisper_model = st.selectbox("Whisper Model", ["tiny", "base", "small"], index=1, key=f"{key_prefix}_model")
    lang_input = st.text_input("Language (blank = auto)", value="", key=f"{key_prefix}_lang")
    language = lang_input.strip() if lang_input.strip() else None

    bundle_key = f"{key_prefix}_bundle"

    # ── URL ──────────────────────────────────────────────────────────────────
    if source == "🔗 URL":
        url = st.text_input("URL", placeholder="YouTube, podcast, SoundCloud...", key=f"{key_prefix}_url")
        if st.button(f"⬇️ Download & Analyze", key=f"{key_prefix}_url_btn", disabled=not HAS_YTDLP):
            if not url.startswith("http"):
                st.warning("Enter a valid URL.")
                return None
            progress = st.empty()
            dl = download_url(url, progress_callback=lambda m: progress.info(m))
            if dl.get("error"):
                st.error(dl["error"])
                return None
            progress.empty()
            meta = {"title": title or dl.get("title", "Untitled"), "type": content_type, "source": "url", "url": url}
            bundle = _run_pipeline(dl["file_path"], meta, whisper_model, language, dl.get("is_audio_only", False))
            try:
                os.unlink(dl["file_path"])
                os.rmdir(os.path.dirname(dl["file_path"]))
            except Exception:
                pass
            if bundle:
                st.session_state[bundle_key] = bundle
                st.success(f"✅ {meta['title']} ready.")

    # ── File upload ───────────────────────────────────────────────────────────
    elif source == "📁 Upload":
        uploaded = st.file_uploader("File", type=["mp4","mov","mkv","avi","m4v","mp3","wav","m4a","ogg"], key=f"{key_prefix}_file")
        if uploaded and st.button("🔍 Analyze", key=f"{key_prefix}_file_btn"):
            ext = Path(uploaded.name).suffix.lower()
            is_audio = ext in [".mp3",".wav",".m4a",".ogg"]
            meta = {"title": title or Path(uploaded.name).stem, "type": content_type, "source": "upload"}
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name
            bundle = _run_pipeline(tmp_path, meta, whisper_model, language, is_audio)
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
            if bundle:
                st.session_state[bundle_key] = bundle
                st.success(f"✅ {meta['title']} ready.")

    # ── Paste ─────────────────────────────────────────────────────────────────
    else:
        text = st.text_area("Transcript", height=200, key=f"{key_prefix}_paste")
        if st.button("🔍 Analyze", key=f"{key_prefix}_paste_btn"):
            if len(text.strip()) < 50:
                st.warning("Need at least 50 characters.")
                return None
            meta = {"title": title or "Pasted Transcript", "type": content_type, "source": "paste"}
            lines = [l.strip() for l in text.split("\n") if l.strip()]
            segments = [{"start": i*5.0, "end": (i+1)*5.0, "text": line} for i, line in enumerate(lines)]
            scene_cuts = list(range(0, len(lines)*5, 60))
            scene_df = segments_to_scene_df(segments, scene_cuts)
            bundle = build_visual_bundle(
                scene_df=scene_df, acoustic_df=pd.DataFrame(), transcript_text=text,
                video_info={"scene_cuts": scene_cuts, "keyframes": [], "keyframe_times": [],
                            "fps": 0, "duration_seconds": len(lines)*5, "total_frames": 0},
                metadata=meta,
            )
            st.session_state[bundle_key] = bundle
            st.success(f"✅ {meta['title']} ready.")

    # Return whatever's in session state for this slot
    return st.session_state.get(bundle_key)


def _run_pipeline(file_path, meta, whisper_model, language, is_audio):
    video_info = {"scene_cuts":[], "keyframes":[], "keyframe_times":[], "fps":0, "duration_seconds":0, "total_frames":0}
    if not is_audio and HAS_CV2:
        with st.spinner("🎬 Detecting scenes..."):
            video_info = ingest_video(file_path)
            if video_info.get("error"):
                video_info["scene_cuts"] = []

    with st.spinner(f"🎙️ Transcribing ({whisper_model})..."):
        tx = transcribe_audio(file_path, model_size=whisper_model, language=language)
        if tx.get("error"):
            st.error(tx["error"])
            return None

    scene_df = segments_to_scene_df(tx["segments"], video_info["scene_cuts"])

    acoustic_df = pd.DataFrame()
    if HAS_LIBROSA:
        with st.spinner("🎵 Acoustics..."):
            acoustic_df = analyze_acoustics(file_path, video_info["scene_cuts"])

    return build_visual_bundle(
        scene_df=scene_df, acoustic_df=acoustic_df,
        transcript_text=tx["text"], video_info=video_info, metadata=meta,
    )


# ════════════════════════════════════════════════════════════════════════════
# MAIN PAGE
# ════════════════════════════════════════════════════════════════════════════

def page_compare():
    st.subheader("⚖️ Compare")
    st.caption("Analyze two pieces of content side by side — any combination of YouTube URLs, uploads, or transcripts.")

    col_a, col_b = st.columns(2)

    with col_a:
        bundle_a = _ingest_one("🅰️ Content A", "ca")

    with col_b:
        bundle_b = _ingest_one("🅱️ Content B", "cb")

    if bundle_a is None or bundle_b is None:
        st.info("Load both Content A and Content B above to see the comparison.")
        return

    st.divider()
    st.markdown("## 📊 Comparison Results")

    meta_a = bundle_a.get("metadata", {})
    meta_b = bundle_b.get("metadata", {})
    title_a = meta_a.get("title", "Content A")
    title_b = meta_b.get("title", "Content B")

    # ── Summary metrics ──────────────────────────────────────────────────────
    st.markdown("### Summary Metrics")
    sm_a = bundle_a.get("summary", {})
    sm_b = bundle_b.get("summary", {})

    def dur_str(d):
        return f"{int(d//60)}m {int(d%60)}s" if d else "—"

    summary_df = pd.DataFrame([
        {"Metric": "Scenes",         title_a: sm_a.get("scenes", 0),                           title_b: sm_b.get("scenes", 0)},
        {"Metric": "Total Words",    title_a: f"{sm_a.get('dialogue_words', 0):,}",            title_b: f"{sm_b.get('dialogue_words', 0):,}"},
        {"Metric": "Duration",       title_a: dur_str(sm_a.get("duration_seconds", 0)),        title_b: dur_str(sm_b.get("duration_seconds", 0))},
        {"Metric": "Words/Scene",    title_a: f"{sm_a.get('dialogue_words',0)/max(sm_a.get('scenes',1),1):.1f}", title_b: f"{sm_b.get('dialogue_words',0)/max(sm_b.get('scenes',1),1):.1f}"},
    ])
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # ── Pacing overlay ───────────────────────────────────────────────────────
    st.markdown("### Narrative Pacing Overlay")
    st.caption("Scene positions normalized to 0–100% so content of different lengths is comparable.")

    scene_a = bundle_a.get("scene_df", pd.DataFrame())
    scene_b = bundle_b.get("scene_df", pd.DataFrame())

    fig_pace = go.Figure()
    colors = [COLOR_PALETTE["primary"], COLOR_PALETTE["accent"]]

    for idx, (scene_df, label) in enumerate([(scene_a, title_a), (scene_b, title_b)]):
        if scene_df.empty or "dialogue_words" not in scene_df.columns:
            continue
        n = len(scene_df)
        x_pct = [i / n * 100 for i in range(n)]
        rolling = scene_df["dialogue_words"].rolling(window=5, min_periods=1).mean().tolist()
        fig_pace.add_trace(go.Scatter(
            x=x_pct, y=rolling, name=label,
            line=dict(color=colors[idx], width=2.5), mode="lines",
        ))

    for pct, lbl in [(25, "Act 2"), (75, "Act 3")]:
        fig_pace.add_vline(x=pct, line_dash="dash", line_color="#aaa",
                           annotation_text=lbl, annotation_position="top")

    fig_pace.update_layout(
        height=400, xaxis_title="Story Position (%)", yaxis_title="Words (5-scene rolling avg)",
        legend=dict(orientation="h", y=1.08), xaxis=dict(ticksuffix="%"),
    )
    st.plotly_chart(fig_pace, use_container_width=True)

    # ── Acoustic overlay ─────────────────────────────────────────────────────
    ac_a = bundle_a.get("acoustic_df", pd.DataFrame())
    ac_b = bundle_b.get("acoustic_df", pd.DataFrame())

    if not ac_a.empty or not ac_b.empty:
        st.markdown("### Acoustic Energy Overlay")
        fig_ac = go.Figure()
        for idx, (ac_df, label) in enumerate([(ac_a, title_a), (ac_b, title_b)]):
            if ac_df.empty or "rms_energy" not in ac_df.columns:
                continue
            n = len(ac_df)
            x_pct = [i / n * 100 for i in range(n)]
            fig_ac.add_trace(go.Scatter(
                x=x_pct, y=ac_df["rms_energy"].tolist(), name=label,
                line=dict(color=colors[idx], width=2), mode="lines",
            ))
        fig_ac.update_layout(
            height=320, xaxis_title="Story Position (%)", yaxis_title="RMS Energy",
            legend=dict(orientation="h", y=1.08), xaxis=dict(ticksuffix="%"),
        )
        st.plotly_chart(fig_ac, use_container_width=True)
        st.caption("Higher energy = louder/more intense audio. Useful for comparing dramatic tension across episodes.")

    # ── Word frequency comparison ─────────────────────────────────────────────
    st.markdown("### Word Frequency Comparison")
    from collections import Counter

    stop_words = {
        "the","a","an","and","or","but","in","on","at","to","for","of","with",
        "is","it","this","that","was","are","be","have","i","you","he","she",
        "we","they","do","not","so","if","as","my","your","um","uh","yeah","just","like",
    }

    def get_top_words(bundle, n=15):
        tx = bundle.get("transcript_text", "")
        if not tx:
            return pd.DataFrame()
        words = [w.lower().strip(".,!?\"'") for w in tx.split() if w.lower().strip(".,!?\"'") not in stop_words and len(w.strip(".,!?\"'")) > 2]
        freq = Counter(words)
        return pd.DataFrame(freq.most_common(n), columns=["Word", "Count"])

    wf_a = get_top_words(bundle_a)
    wf_b = get_top_words(bundle_b)

    col1, col2 = st.columns(2)
    with col1:
        if not wf_a.empty:
            fig_wa = px.bar(wf_a, x="Count", y="Word", orientation="h",
                            title=f"Top Words — {title_a}",
                            color_discrete_sequence=[COLOR_PALETTE["primary"]])
            fig_wa.update_layout(height=400, yaxis={"categoryorder":"total ascending"})
            st.plotly_chart(fig_wa, use_container_width=True)
    with col2:
        if not wf_b.empty:
            fig_wb = px.bar(wf_b, x="Count", y="Word", orientation="h",
                            title=f"Top Words — {title_b}",
                            color_discrete_sequence=[COLOR_PALETTE["accent"]])
            fig_wb.update_layout(height=400, yaxis={"categoryorder":"total ascending"})
            st.plotly_chart(fig_wb, use_container_width=True)

    # Shared vs exclusive vocabulary
    if not wf_a.empty and not wf_b.empty:
        words_a = set(wf_a["Word"])
        words_b = set(wf_b["Word"])
        shared = words_a & words_b
        excl_a = words_a - words_b
        excl_b = words_b - words_a

        st.markdown("### Vocabulary Overlap")
        vc1, vc2, vc3 = st.columns(3)
        vc1.metric("Shared Top Words", len(shared))
        vc2.metric(f"Unique to {title_a[:20]}", len(excl_a))
        vc3.metric(f"Unique to {title_b[:20]}", len(excl_b))
        if shared:
            st.caption(f"**Shared:** {', '.join(sorted(shared))}")
