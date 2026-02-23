from __future__ import annotations
"""
pages_compare.py — Compare up to 10 pieces of content side by side.
Items can come from URLs, file uploads, or transcript pastes.
Each item is ingested independently and stored in session state.
"""

import os, tempfile, traceback
from pathlib import Path
from typing import Dict, Any, Optional, List
from collections import Counter

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from constants import COLOR_PALETTE, HAS_WHISPER, HAS_CV2, HAS_LIBROSA, HAS_YTDLP
from ingestion import (
    download_url, ingest_video, transcribe_audio,
    segments_to_scene_df, analyze_acoustics,
    build_visual_bundle,
)

MAX_ITEMS = 10
COLORS = [
    "#1e90ff","#ff6b6b","#4ecdc4","#ffe66d","#a29bfe",
    "#fd79a8","#00b894","#e17055","#74b9ff","#55efc4",
]

STOP_WORDS = {
    "the","a","an","and","or","but","in","on","at","to","for","of","with",
    "is","it","this","that","was","are","be","have","i","you","he","she",
    "we","they","do","not","so","if","as","my","your","um","uh","yeah","just","like",
}


# ════════════════════════════════════════════════════════════════════════════
# INGEST PANEL FOR ONE ITEM
# ════════════════════════════════════════════════════════════════════════════

def _ingest_panel(slot_idx: int) -> Optional[Dict]:
    """Render ingest controls for one slot. Returns bundle dict or None."""
    key = f"cmp_{slot_idx}"
    bundle_key = f"{key}_bundle"
    label = f"Item {slot_idx + 1}"

    with st.expander(f"{'✅' if st.session_state.get(bundle_key) else '⬜'} {label}", expanded=not bool(st.session_state.get(bundle_key))):
        source = st.radio("Source", ["🔗 URL", "📁 Upload", "📝 Paste"], horizontal=True, key=f"{key}_src")

        col1, col2 = st.columns(2)
        with col1:
            title = st.text_input("Title (optional)", key=f"{key}_title")
        with col2:
            content_type = st.selectbox("Type", ["TV Episode","Movie","Podcast","Other"], key=f"{key}_type")

        whisper_model = st.selectbox("Whisper Model", ["tiny","base","small"], index=0, key=f"{key}_model")
        lang_raw = st.text_input("Language (blank=auto)", value="", key=f"{key}_lang")
        language = lang_raw.strip() or None

        if source == "🔗 URL":
            url = st.text_input("URL", placeholder="YouTube, podcast, SoundCloud...", key=f"{key}_url")
            if st.button(f"⬇️ Add Item {slot_idx+1}", key=f"{key}_url_btn", disabled=not HAS_YTDLP):
                if not url.startswith("http"):
                    st.warning("Enter a valid URL.")
                    return None
                prog = st.empty()
                dl = download_url(url, progress_callback=lambda m: prog.info(m))
                if dl.get("error"):
                    st.error(dl["error"])
                    return None
                prog.empty()
                meta = {"title": title or dl.get("title","Untitled"), "type": content_type,
                        "source": "url", "url": url, "uploader": dl.get("uploader","")}
                bundle = _run_pipeline(dl["file_path"], meta, whisper_model, language, dl.get("is_audio_only",False))
                _cleanup(dl["file_path"])
                if bundle:
                    st.session_state[bundle_key] = bundle
                    st.success(f"✅ {meta['title']}")

        elif source == "📁 Upload":
            uploaded = st.file_uploader("File", type=["mp4","mov","mkv","avi","m4v","mp3","wav","m4a","ogg"], key=f"{key}_file")
            if uploaded and st.button(f"🔍 Add Item {slot_idx+1}", key=f"{key}_file_btn"):
                ext = Path(uploaded.name).suffix.lower()
                is_audio = ext in [".mp3",".wav",".m4a",".ogg"]
                meta = {"title": title or Path(uploaded.name).stem, "type": content_type, "source": "upload"}
                with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                    tmp.write(uploaded.read())
                    tmp_path = tmp.name
                bundle = _run_pipeline(tmp_path, meta, whisper_model, language, is_audio)
                _cleanup(tmp_path)
                if bundle:
                    st.session_state[bundle_key] = bundle
                    st.success(f"✅ {meta['title']}")

        else:  # Paste
            text = st.text_area("Transcript", height=150, key=f"{key}_paste")
            if st.button(f"🔍 Add Item {slot_idx+1}", key=f"{key}_paste_btn"):
                if len(text.strip()) < 50:
                    st.warning("Need at least 50 characters.")
                    return None
                meta = {"title": title or f"Item {slot_idx+1}", "type": content_type, "source": "paste"}
                lines = [l.strip() for l in text.split("\n") if l.strip()]
                segments = [{"start": i*5.0,"end":(i+1)*5.0,"text": line} for i, line in enumerate(lines)]
                scene_cuts = list(range(0, len(lines)*5, 60))
                scene_df = segments_to_scene_df(segments, scene_cuts)
                bundle = build_visual_bundle(
                    scene_df=scene_df, acoustic_df=pd.DataFrame(), transcript_text=text,
                    video_info={"scene_cuts":scene_cuts,"keyframes":[],"keyframe_times":[],
                                "fps":0,"duration_seconds":len(lines)*5,"total_frames":0},
                    metadata=meta,
                )
                st.session_state[bundle_key] = bundle
                st.success(f"✅ {meta['title']}")

        # Clear button
        if st.session_state.get(bundle_key):
            if st.button(f"🗑️ Clear slot {slot_idx+1}", key=f"{key}_clear"):
                st.session_state[bundle_key] = None
                st.rerun()

    return st.session_state.get(bundle_key)


def _run_pipeline(file_path, meta, whisper_model, language, is_audio):
    video_info = {"scene_cuts":[],"keyframes":[],"keyframe_times":[],"fps":0,"duration_seconds":0,"total_frames":0}
    if not is_audio and HAS_CV2:
        with st.spinner("🎬 Detecting scenes..."):
            vi = ingest_video(file_path)
            if not vi.get("error"):
                vi["keyframes"] = []
                video_info = vi
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
    return build_visual_bundle(scene_df=scene_df, acoustic_df=acoustic_df,
                                transcript_text=tx["text"], video_info=video_info, metadata=meta)


def _cleanup(fp):
    try:
        if fp and os.path.exists(fp):
            os.unlink(fp)
            parent = os.path.dirname(fp)
            if os.path.isdir(parent) and not os.listdir(parent):
                os.rmdir(parent)
    except Exception:
        pass


# ════════════════════════════════════════════════════════════════════════════
# MAIN PAGE
# ════════════════════════════════════════════════════════════════════════════

def page_compare():
    st.subheader("⚖️ Compare")
    st.caption(f"Load up to {MAX_ITEMS} items — any mix of YouTube URLs, file uploads, or transcript pastes.")

    n_slots = st.slider("Number of items to compare", min_value=2, max_value=MAX_ITEMS, value=2, key="cmp_n_slots")

    st.divider()
    st.markdown("### 📥 Load Items")

    # Render ingest panels
    bundles = []
    cols_per_row = min(n_slots, 2)
    for row_start in range(0, n_slots, cols_per_row):
        cols = st.columns(cols_per_row)
        for col_idx in range(cols_per_row):
            slot_idx = row_start + col_idx
            if slot_idx >= n_slots:
                break
            with cols[col_idx]:
                b = _ingest_panel(slot_idx)
                if b:
                    bundles.append((slot_idx, b))

    loaded_bundles = [b for _, b in sorted(bundles, key=lambda x: x[0])]

    if len(loaded_bundles) < 2:
        st.info(f"Load at least 2 items above to see the comparison. ({len(loaded_bundles)}/{n_slots} loaded)")
        return

    st.divider()
    st.markdown(f"## 📊 Comparison — {len(loaded_bundles)} items")

    # ── Summary table ─────────────────────────────────────────────────────────
    st.markdown("### Summary Metrics")
    summary_rows = []
    for b in loaded_bundles:
        meta = b.get("metadata", {})
        sm = b.get("summary", {})
        dur = sm.get("duration_seconds", 0)
        n = max(sm.get("scenes", 1), 1)
        summary_rows.append({
            "Title": meta.get("title","Untitled")[:40],
            "Type": meta.get("type",""),
            "Scenes": sm.get("scenes", 0),
            "Total Words": f"{sm.get('dialogue_words',0):,}",
            "Duration": f"{int(dur//60)}m {int(dur%60)}s" if dur else "—",
            "Words/Scene": f"{sm.get('dialogue_words',0)/n:.1f}",
        })
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    # ── Pacing overlay ────────────────────────────────────────────────────────
    st.markdown("### Narrative Pacing Overlay")
    st.caption("All content normalized to 0–100% story position for direct comparison regardless of length.")

    fig_pace = go.Figure()
    for idx, b in enumerate(loaded_bundles):
        title = b.get("metadata", {}).get("title","Untitled")[:30]
        scene_df = b.get("scene_df", pd.DataFrame())
        if scene_df.empty or "dialogue_words" not in scene_df.columns:
            continue
        n = len(scene_df)
        x_pct = [i/n*100 for i in range(n)]
        rolling = scene_df["dialogue_words"].rolling(window=5, min_periods=1).mean().tolist()
        fig_pace.add_trace(go.Scatter(x=x_pct, y=rolling, name=title,
                                       line=dict(color=COLORS[idx % len(COLORS)], width=2), mode="lines"))

    for pct, lbl in [(25,"Act 2"), (75,"Act 3")]:
        fig_pace.add_vline(x=pct, line_dash="dash", line_color="#aaa",
                           annotation_text=lbl, annotation_position="top")
    fig_pace.update_layout(height=420, xaxis_title="Story Position (%)", yaxis_title="Words (5-scene rolling avg)",
                            legend=dict(orientation="h", y=1.1), xaxis=dict(ticksuffix="%"))
    st.plotly_chart(fig_pace, use_container_width=True)

    # ── Bar comparisons ───────────────────────────────────────────────────────
    st.markdown("### Feature Comparison")
    titles_short = [b.get("metadata",{}).get("title","?")[:25] for b in loaded_bundles]

    col1, col2 = st.columns(2)
    with col1:
        words_per_scene = [
            b.get("summary",{}).get("dialogue_words",0) / max(b.get("summary",{}).get("scenes",1),1)
            for b in loaded_bundles
        ]
        fig_wps = px.bar(x=titles_short, y=words_per_scene, title="Avg Words per Scene",
                          color=titles_short, color_discrete_sequence=COLORS)
        fig_wps.update_layout(height=350, showlegend=False, xaxis_title="", yaxis_title="Words/Scene")
        st.plotly_chart(fig_wps, use_container_width=True)

    with col2:
        durations = [b.get("summary",{}).get("duration_seconds",0)/60 for b in loaded_bundles]
        fig_dur = px.bar(x=titles_short, y=durations, title="Duration (minutes)",
                          color=titles_short, color_discrete_sequence=COLORS)
        fig_dur.update_layout(height=350, showlegend=False, xaxis_title="", yaxis_title="Minutes")
        st.plotly_chart(fig_dur, use_container_width=True)

    # ── Acoustic overlay ──────────────────────────────────────────────────────
    acoustic_available = any(not b.get("acoustic_df", pd.DataFrame()).empty for b in loaded_bundles)
    if acoustic_available:
        st.markdown("### Acoustic Energy Overlay")
        fig_ac = go.Figure()
        for idx, b in enumerate(loaded_bundles):
            title = b.get("metadata",{}).get("title","Untitled")[:30]
            ac = b.get("acoustic_df", pd.DataFrame())
            if ac.empty or "rms_energy" not in ac.columns:
                continue
            n = len(ac)
            x_pct = [i/n*100 for i in range(n)]
            fig_ac.add_trace(go.Scatter(x=x_pct, y=ac["rms_energy"].tolist(), name=title,
                                         line=dict(color=COLORS[idx % len(COLORS)], width=2)))
        fig_ac.update_layout(height=340, xaxis_title="Story Position (%)", yaxis_title="RMS Energy",
                              legend=dict(orientation="h", y=1.1), xaxis=dict(ticksuffix="%"))
        st.plotly_chart(fig_ac, use_container_width=True)
        st.caption("Higher energy = louder/more intense audio. Useful for comparing dramatic tension across episodes.")

    # ── Word frequency comparison ─────────────────────────────────────────────
    st.markdown("### Top Words Comparison")

    def get_top_words(b, n=15):
        tx = b.get("transcript_text","")
        if not tx:
            return pd.DataFrame()
        words = [w.lower().strip(".,!?\"'") for w in tx.split()
                 if w.lower().strip(".,!?\"'") not in STOP_WORDS and len(w.strip(".,!?\"'")) > 2]
        freq = Counter(words)
        return pd.DataFrame(freq.most_common(n), columns=["Word","Count"])

    # Up to 5 per row
    cols_per_row = min(len(loaded_bundles), 3)
    for row_start in range(0, len(loaded_bundles), cols_per_row):
        cols = st.columns(cols_per_row)
        for col_idx in range(cols_per_row):
            idx = row_start + col_idx
            if idx >= len(loaded_bundles):
                break
            b = loaded_bundles[idx]
            title = b.get("metadata",{}).get("title","?")[:25]
            wf = get_top_words(b)
            if not wf.empty:
                with cols[col_idx]:
                    fig_wf = px.bar(wf, x="Count", y="Word", orientation="h",
                                     title=title, color_discrete_sequence=[COLORS[idx % len(COLORS)]])
                    fig_wf.update_layout(height=350, yaxis={"categoryorder":"total ascending"},
                                          showlegend=False, margin=dict(l=10,r=10,t=40,b=10))
                    st.plotly_chart(fig_wf, use_container_width=True)

    # ── Vocabulary overlap ────────────────────────────────────────────────────
    if len(loaded_bundles) >= 2:
        st.markdown("### Vocabulary Overlap")
        word_sets = []
        for b in loaded_bundles:
            wf = get_top_words(b, n=30)
            word_sets.append(set(wf["Word"].tolist()) if not wf.empty else set())

        shared_all = word_sets[0]
        for ws in word_sets[1:]:
            shared_all = shared_all & ws

        overlap_rows = []
        for i, b in enumerate(loaded_bundles):
            title = b.get("metadata",{}).get("title","?")[:30]
            unique_to_this = word_sets[i] - set().union(*[word_sets[j] for j in range(len(word_sets)) if j != i])
            overlap_rows.append({
                "Title": title,
                "Unique Vocab": len(unique_to_this),
                "Sample Unique Words": ", ".join(sorted(unique_to_this)[:5]),
            })

        st.metric("Words shared across ALL items (top 30 each)", len(shared_all))
        if shared_all:
            st.caption(f"**Shared:** {', '.join(sorted(shared_all))}")
        st.dataframe(pd.DataFrame(overlap_rows), use_container_width=True, hide_index=True)

    # ── Lexical diversity comparison ──────────────────────────────────────────
    st.markdown("### Lexical Diversity")
    lex_titles, lex_vals = [], []
    for b in loaded_bundles:
        tx = b.get("transcript_text","")
        if not tx:
            continue
        words = tx.split()
        if len(words) > 0:
            lex_titles.append(b.get("metadata",{}).get("title","?")[:25])
            lex_vals.append(round(len(set(w.lower() for w in words)) / len(words) * 100, 1))
    if lex_vals:
        fig_lex = px.bar(x=lex_titles, y=lex_vals, title="Lexical Diversity (% unique words)",
                          color=lex_titles, color_discrete_sequence=COLORS)
        fig_lex.update_layout(height=320, showlegend=False, xaxis_title="", yaxis_title="% Unique Words")
        st.plotly_chart(fig_lex, use_container_width=True)
        st.caption("Higher = more varied vocabulary. Lower = more repetitive language (common in genre-constrained content).")
