from __future__ import annotations
"""
pages_academic.py — Academic Research tab.

Features:
  - Channel/playlist batch ingestion (YouTube, podcast RSS)
  - Feature extraction to DataFrame
  - OLS, Fixed Effects, DiD, RF/XGBoost, LDA, Time Series modeling
  - Excel export
  - Links to CLI batch runner for large-scale analysis
"""

import io, os, tempfile, traceback, json
from typing import Dict, List, Any, Optional
from pathlib import Path

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from constants import COLOR_PALETTE, HAS_YTDLP, HAS_WHISPER, HAS_LIBROSA
from ingestion import (
    download_url, transcribe_audio, ingest_video,
    segments_to_scene_df, analyze_acoustics,
    build_visual_bundle,
)
from academic_modeling import (
    bundle_to_feature_row, bundles_to_dataframe,
    run_ols, run_fixed_effects, run_did,
    run_random_forest, run_xgboost, run_lda,
    run_time_series, export_to_excel, get_numeric_cols,
)


# ════════════════════════════════════════════════════════════════════════════
# CHANNEL / PLAYLIST URL FETCHER
# ════════════════════════════════════════════════════════════════════════════

def fetch_channel_urls(channel_url: str, max_videos: int = 20) -> Dict[str, Any]:
    """Use yt-dlp to list video URLs from a channel or playlist without downloading."""
    if not HAS_YTDLP:
        return {"urls": [], "error": "yt-dlp not installed"}
    try:
        import yt_dlp
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "extract_flat": True,
            "playlistend": max_videos,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(channel_url, download=False)

        entries = info.get("entries", []) or []
        urls = []
        for e in entries[:max_videos]:
            url = e.get("url") or e.get("webpage_url")
            if url and not url.startswith("http"):
                url = f"https://www.youtube.com/watch?v={url}"
            if url:
                urls.append({
                    "url": url,
                    "title": e.get("title", "Unknown"),
                    "duration": e.get("duration", 0),
                    "uploader": e.get("uploader", info.get("uploader", "")),
                })
        return {"urls": urls, "channel_title": info.get("title", channel_url), "error": None}
    except Exception as e:
        return {"urls": [], "error": str(e)[:300]}


def fetch_rss_urls(rss_url: str, max_episodes: int = 20) -> Dict[str, Any]:
    """Parse a podcast RSS feed and return episode URLs."""
    try:
        import urllib.request
        import xml.etree.ElementTree as ET

        with urllib.request.urlopen(rss_url, timeout=15) as resp:
            xml_content = resp.read()

        root = ET.fromstring(xml_content)
        ns = {"itunes": "http://www.itunes.com/dtds/podcast-1.0.dtd"}
        channel = root.find("channel")
        if channel is None:
            return {"urls": [], "error": "No channel element found in RSS feed"}

        channel_title = channel.findtext("title", "Unknown Podcast")
        urls = []
        for item in channel.findall("item")[:max_episodes]:
            title = item.findtext("title", "Unknown Episode")
            enclosure = item.find("enclosure")
            audio_url = enclosure.get("url") if enclosure is not None else None
            if audio_url:
                duration_str = item.findtext("itunes:duration", "", ns) or ""
                duration_secs = _parse_duration(duration_str)
                urls.append({
                    "url": audio_url,
                    "title": title,
                    "duration": duration_secs,
                    "uploader": channel_title,
                })

        return {"urls": urls, "channel_title": channel_title, "error": None}
    except Exception as e:
        return {"urls": [], "error": f"RSS parse error: {str(e)[:200]}"}


def _parse_duration(s: str) -> int:
    try:
        parts = s.split(":")
        if len(parts) == 3:
            return int(parts[0])*3600 + int(parts[1])*60 + int(parts[2])
        if len(parts) == 2:
            return int(parts[0])*60 + int(parts[1])
        return int(s)
    except Exception:
        return 0


# ════════════════════════════════════════════════════════════════════════════
# BATCH INGEST PIPELINE
# ════════════════════════════════════════════════════════════════════════════

def batch_ingest_items(
    url_items: List[Dict],
    whisper_model: str = "base",
    progress_table_placeholder=None,
) -> List[Dict]:
    """
    Process a list of URL items, returning a list of {metadata, bundle} dicts.
    Updates a Streamlit placeholder with a live progress table.
    """
    results = []
    status_rows = []

    for i, item in enumerate(url_items):
        url = item["url"]
        title = item.get("title", f"Item {i+1}")
        status_rows.append({
            "Title": title[:50], "Status": "⏳ Queued",
            "Words": "—", "Scenes": "—", "Duration": "—",
        })

        if progress_table_placeholder:
            progress_table_placeholder.dataframe(
                pd.DataFrame(status_rows), use_container_width=True, hide_index=True
            )

        try:
            # Download
            status_rows[-1]["Status"] = "⬇️ Downloading"
            if progress_table_placeholder:
                progress_table_placeholder.dataframe(pd.DataFrame(status_rows), use_container_width=True, hide_index=True)

            dl = download_url(url)
            if dl.get("error"):
                status_rows[-1]["Status"] = f"❌ {dl['error'][:40]}"
                if progress_table_placeholder:
                    progress_table_placeholder.dataframe(pd.DataFrame(status_rows), use_container_width=True, hide_index=True)
                continue

            fp = dl["file_path"]
            is_audio = dl.get("is_audio_only", False)
            meta = {
                "title": dl.get("title") or title,
                "type": item.get("type", "Unknown"),
                "genre": item.get("genre", ""),
                "year": item.get("year", ""),
                "source": "batch_url",
                "url": url,
                "uploader": dl.get("uploader", ""),
                "duration_raw": dl.get("duration", 0),
            }

            # Video ingestion
            video_info = {"scene_cuts":[], "keyframes":[], "keyframe_times":[], "fps":0, "duration_seconds": dl.get("duration",0), "total_frames":0}
            if not is_audio:
                status_rows[-1]["Status"] = "🎬 Scene detect"
                if progress_table_placeholder:
                    progress_table_placeholder.dataframe(pd.DataFrame(status_rows), use_container_width=True, hide_index=True)
                vi = ingest_video(fp)
                if not vi.get("error"):
                    video_info = vi

            # Transcribe
            status_rows[-1]["Status"] = "🎙️ Transcribing"
            if progress_table_placeholder:
                progress_table_placeholder.dataframe(pd.DataFrame(status_rows), use_container_width=True, hide_index=True)

            tx = transcribe_audio(fp, model_size=whisper_model)
            if tx.get("error"):
                status_rows[-1]["Status"] = f"❌ Transcription failed"
                continue

            scene_df = segments_to_scene_df(tx["segments"], video_info["scene_cuts"])

            # Acoustics
            acoustic_df = pd.DataFrame()
            if HAS_LIBROSA and os.path.exists(fp):
                status_rows[-1]["Status"] = "🎵 Acoustics"
                if progress_table_placeholder:
                    progress_table_placeholder.dataframe(pd.DataFrame(status_rows), use_container_width=True, hide_index=True)
                acoustic_df = analyze_acoustics(fp, video_info["scene_cuts"])

            bundle = build_visual_bundle(
                scene_df=scene_df, acoustic_df=acoustic_df,
                transcript_text=tx["text"], video_info=video_info, metadata=meta,
            )

            sm = bundle["summary"]
            dur = sm.get("duration_seconds", 0)
            status_rows[-1].update({
                "Status": "✅ Done",
                "Words": f"{sm.get('dialogue_words',0):,}",
                "Scenes": sm.get("scenes", 0),
                "Duration": f"{int(dur//60)}m {int(dur%60)}s",
            })
            results.append({"metadata": meta, "bundle": bundle})

        except Exception as e:
            status_rows[-1]["Status"] = f"❌ Error: {str(e)[:40]}"
        finally:
            # Clean up temp file
            try:
                if "fp" in locals() and fp and os.path.exists(fp):
                    os.unlink(fp)
                    parent = os.path.dirname(fp)
                    if os.path.isdir(parent) and not os.listdir(parent):
                        os.rmdir(parent)
            except Exception:
                pass
            if progress_table_placeholder:
                progress_table_placeholder.dataframe(pd.DataFrame(status_rows), use_container_width=True, hide_index=True)

    return results


# ════════════════════════════════════════════════════════════════════════════
# MAIN PAGE
# ════════════════════════════════════════════════════════════════════════════

def page_academic():
    st.subheader("🎓 Academic Research")
    st.caption(
        "Batch-ingest entire YouTube channels, podcast feeds, or URL lists. "
        "Extract a full feature dataset, run ML and econometric models, and export to Excel."
    )

    st.info(
        "**Scale guidance:** This tab handles batches of up to ~50 items comfortably in Streamlit. "
        "For large-scale research (hundreds to tens of thousands of videos), use the **CLI Batch Runner** "
        "(`batch_runner.py`) which runs independently of the browser and supports multi-day jobs. "
        "See the sidebar for instructions."
    )

    st.divider()

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 1: BATCH COLLECTION
    # ════════════════════════════════════════════════════════════════════════
    st.markdown("## 1️⃣ Build Your Dataset")

    collection_method = st.radio(
        "Collection method",
        ["📺 YouTube Channel / Playlist", "🎙️ Podcast RSS Feed", "📋 Manual URL List"],
        horizontal=True,
        key="acad_method",
    )

    url_items = []

    if collection_method == "📺 YouTube Channel / Playlist":
        channel_url = st.text_input(
            "Channel or Playlist URL",
            placeholder="https://www.youtube.com/@ChannelName or https://youtube.com/playlist?list=...",
            key="acad_channel_url",
        )
        col1, col2, col3 = st.columns(3)
        with col1:
            max_videos = st.number_input("Max videos", min_value=1, max_value=50, value=10, key="acad_max_vids")
        with col2:
            content_type = st.selectbox("Content Type", ["TV Episode","Movie","Documentary","Other"], key="acad_vtype")
        with col3:
            genre = st.text_input("Genre tag", value="", key="acad_vgenre")

        if st.button("🔍 Fetch Video List", key="acad_fetch_btn", disabled=not HAS_YTDLP):
            with st.spinner("Fetching channel info..."):
                result = fetch_channel_urls(channel_url, max_videos=int(max_videos))
            if result.get("error"):
                st.error(result["error"])
            else:
                items = [{**u, "type": content_type, "genre": genre} for u in result["urls"]]
                st.session_state["acad_url_items"] = items
                st.success(f"Found {len(items)} videos from **{result['channel_title']}**")
                preview_df = pd.DataFrame([{"Title": u["title"][:60], "Duration": f"{int(u['duration']//60)}m {int(u['duration']%60)}s"} for u in items])
                st.dataframe(preview_df, use_container_width=True, hide_index=True)

    elif collection_method == "🎙️ Podcast RSS Feed":
        rss_url = st.text_input(
            "Podcast RSS Feed URL",
            placeholder="https://feeds.simplecast.com/... or any RSS/Atom feed URL",
            key="acad_rss_url",
        )
        col1, col2 = st.columns(2)
        with col1:
            max_eps = st.number_input("Max episodes", min_value=1, max_value=50, value=10, key="acad_max_eps")
        with col2:
            genre = st.text_input("Genre tag", value="Podcast", key="acad_pgenre")

        if st.button("🔍 Parse Feed", key="acad_rss_btn"):
            with st.spinner("Parsing RSS feed..."):
                result = fetch_rss_urls(rss_url, max_episodes=int(max_eps))
            if result.get("error"):
                st.error(result["error"])
            else:
                items = [{**u, "type": "Podcast", "genre": genre} for u in result["urls"]]
                st.session_state["acad_url_items"] = items
                st.success(f"Found {len(items)} episodes from **{result['channel_title']}**")
                preview_df = pd.DataFrame([{"Title": u["title"][:60], "Duration": f"{int(u['duration']//60)}m {int(u['duration']%60)}s"} for u in items])
                st.dataframe(preview_df, use_container_width=True, hide_index=True)

    else:  # Manual URL list
        url_text = st.text_area(
            "Paste URLs (one per line)",
            height=150,
            placeholder="https://youtube.com/watch?v=...\nhttps://youtube.com/watch?v=...",
            key="acad_manual_urls",
        )
        col1, col2 = st.columns(2)
        with col1:
            content_type = st.selectbox("Content Type", ["TV Episode","Movie","Podcast","Other"], key="acad_mtype")
        with col2:
            genre = st.text_input("Genre tag", value="", key="acad_mgenre")

        if url_text.strip():
            urls = [u.strip() for u in url_text.strip().split("\n") if u.strip().startswith("http")]
            st.caption(f"{len(urls)} valid URLs detected.")
            items = [{"url": u, "title": f"Item {i+1}", "duration": 0, "type": content_type, "genre": genre, "uploader": ""} for i, u in enumerate(urls)]
            st.session_state["acad_url_items"] = items

    url_items = st.session_state.get("acad_url_items", [])

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 2: BATCH INGEST
    # ════════════════════════════════════════════════════════════════════════
    if url_items:
        st.divider()
        st.markdown("## 2️⃣ Batch Ingest & Feature Extraction")

        col1, col2 = st.columns(2)
        with col1:
            whisper_model = st.selectbox("Whisper Model", ["tiny","base","small"], index=0,
                                          help="'tiny' is fastest for batch runs. Use 'small' for higher accuracy.",
                                          key="acad_whisper")
        with col2:
            st.metric("Items queued", len(url_items))

        if st.button("🚀 Start Batch Analysis", type="primary", key="acad_run_btn"):
            st.markdown("### ⏳ Progress")
            progress_placeholder = st.empty()
            results = batch_ingest_items(url_items, whisper_model=whisper_model,
                                          progress_table_placeholder=progress_placeholder)

            if results:
                feature_df = bundles_to_dataframe(results)
                st.session_state["acad_feature_df"] = feature_df
                st.session_state["acad_bundles"] = results
                st.success(f"✅ Batch complete — {len(results)}/{len(url_items)} items processed. {len(feature_df.columns)} features extracted.")
            else:
                st.error("No items processed successfully.")

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 3: MODELING
    # ════════════════════════════════════════════════════════════════════════
    feature_df = st.session_state.get("acad_feature_df")

    if feature_df is not None and not feature_df.empty:
        st.divider()
        st.markdown("## 3️⃣ Modeling")

        # Dataset preview
        with st.expander("📋 Feature Dataset Preview", expanded=False):
            st.dataframe(feature_df, use_container_width=True, hide_index=True)
            st.caption(f"{len(feature_df)} rows × {len(feature_df.columns)} columns")

        numeric_cols = get_numeric_cols(feature_df)
        all_cols = feature_df.columns.tolist()

        model_results = {}
        lda_results = None

        model_type = st.selectbox(
            "Select Model",
            ["OLS Regression", "Fixed Effects Panel", "Difference-in-Differences",
             "Random Forest", "XGBoost", "LDA Topic Model", "Time Series Trend"],
            key="acad_model_select",
        )

        # ── OLS ──────────────────────────────────────────────────────────────
        if model_type == "OLS Regression":
            col1, col2 = st.columns(2)
            with col1:
                outcome = st.selectbox("Outcome variable", numeric_cols, key="ols_outcome")
            with col2:
                predictors = st.multiselect("Predictors", [c for c in numeric_cols if c != outcome], key="ols_pred")
            if predictors and st.button("Run OLS", key="ols_run"):
                with st.spinner("Running OLS..."):
                    result = run_ols(feature_df, outcome, predictors)
                model_results["OLS"] = result
                _render_regression_results(result)

        # ── Fixed Effects ─────────────────────────────────────────────────────
        elif model_type == "Fixed Effects Panel":
            col1, col2, col3 = st.columns(3)
            with col1:
                outcome = st.selectbox("Outcome", numeric_cols, key="fe_outcome")
            with col2:
                entity_col = st.selectbox("Entity (e.g. channel)", all_cols, key="fe_entity")
            with col3:
                time_col = st.selectbox("Time (e.g. year)", all_cols, key="fe_time")
            predictors = st.multiselect("Predictors", [c for c in numeric_cols if c not in [outcome]], key="fe_pred")
            if predictors and st.button("Run Fixed Effects", key="fe_run"):
                with st.spinner("Running Fixed Effects..."):
                    result = run_fixed_effects(feature_df, outcome, predictors, entity_col, time_col)
                model_results["Fixed Effects"] = result
                _render_regression_results(result)

        # ── DiD ───────────────────────────────────────────────────────────────
        elif model_type == "Difference-in-Differences":
            col1, col2, col3 = st.columns(3)
            with col1:
                outcome = st.selectbox("Outcome", numeric_cols, key="did_outcome")
            with col2:
                treatment_col = st.selectbox("Treatment indicator (0/1)", all_cols, key="did_treat")
            with col3:
                post_col = st.selectbox("Post-period indicator (0/1)", all_cols, key="did_post")
            controls = st.multiselect("Control variables (optional)", [c for c in numeric_cols if c not in [outcome]], key="did_ctrl")
            if st.button("Run DiD", key="did_run"):
                with st.spinner("Running DiD..."):
                    result = run_did(feature_df, outcome, treatment_col, post_col, controls or None)
                model_results["DiD"] = result
                if result.get("error"):
                    st.error(result["error"])
                else:
                    st.metric("DiD Estimate", f"{result['did_estimate']:.4f}")
                    st.metric("p-value", f"{result['did_pvalue']:.4f} {result['did_sig']}")
                    st.info(result.get("interpretation", ""))
                    _render_regression_results(result)

        # ── Random Forest ─────────────────────────────────────────────────────
        elif model_type == "Random Forest":
            col1, col2 = st.columns(2)
            with col1:
                outcome = st.selectbox("Outcome", numeric_cols, key="rf_outcome")
            with col2:
                task = st.selectbox("Task", ["regression", "classification"], key="rf_task")
            predictors = st.multiselect("Features", [c for c in numeric_cols if c != outcome], key="rf_pred")
            if predictors and st.button("Run Random Forest", key="rf_run"):
                with st.spinner("Training Random Forest..."):
                    result = run_random_forest(feature_df, outcome, predictors, task=task)
                model_results["Random Forest"] = result
                _render_importance_results(result)

        # ── XGBoost ───────────────────────────────────────────────────────────
        elif model_type == "XGBoost":
            col1, col2 = st.columns(2)
            with col1:
                outcome = st.selectbox("Outcome", numeric_cols, key="xgb_outcome")
            with col2:
                task = st.selectbox("Task", ["regression", "classification"], key="xgb_task")
            predictors = st.multiselect("Features", [c for c in numeric_cols if c != outcome], key="xgb_pred")
            if predictors and st.button("Run XGBoost", key="xgb_run"):
                with st.spinner("Training XGBoost..."):
                    result = run_xgboost(feature_df, outcome, predictors, task=task)
                model_results["XGBoost"] = result
                _render_importance_results(result)

        # ── LDA ───────────────────────────────────────────────────────────────
        elif model_type == "LDA Topic Model":
            bundles = st.session_state.get("acad_bundles", [])
            transcripts = [b.get("bundle", {}).get("transcript_text", "") for b in bundles]
            titles = [b.get("metadata", {}).get("title", f"Doc {i}") for i, b in enumerate(bundles)]
            valid = [(t, ti) for t, ti in zip(transcripts, titles) if len(t.strip()) > 100]

            if len(valid) < 2:
                st.warning("Need at least 2 documents with transcripts for LDA.")
            else:
                n_topics = st.slider("Number of topics", 2, 10, 5, key="lda_ntopics")
                if st.button("Run LDA", key="lda_run"):
                    with st.spinner("Running LDA topic modeling..."):
                        txts, tits = zip(*valid)
                        lda_results = run_lda(list(txts), list(tits), n_topics=n_topics)
                    if lda_results.get("error"):
                        st.error(lda_results["error"])
                    else:
                        st.metric("Perplexity", lda_results["perplexity"])
                        st.caption("Lower perplexity = better model fit.")
                        for topic, words in lda_results["topics"].items():
                            st.markdown(f"**{topic}:** {', '.join(words)}")
                        st.session_state["acad_lda"] = lda_results

        # ── Time Series ───────────────────────────────────────────────────────
        elif model_type == "Time Series Trend":
            col1, col2 = st.columns(2)
            with col1:
                time_col = st.selectbox("Time variable", all_cols, key="ts_time")
            with col2:
                value_cols = st.multiselect("Variables to analyze", numeric_cols, key="ts_vals")
            if value_cols and st.button("Run Trend Analysis", key="ts_run"):
                with st.spinner("Analyzing trends..."):
                    result = run_time_series(feature_df, time_col, value_cols)
                model_results["Time Series"] = result
                if result.get("error"):
                    st.error(result["error"])
                else:
                    for col_name, stats in result.get("results", {}).items():
                        st.markdown(f"**{col_name}**: {stats['trend_direction']} "
                                    f"(slope={stats['trend_slope']:.4f}, p={stats['trend_pvalue']:.4f}{stats['trend_sig']}, R²={stats['r2']:.3f})")

                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            y=feature_df[col_name].dropna().tolist(),
                            name=col_name, line=dict(color=COLOR_PALETTE["neutral"]),
                        ))
                        fig.add_trace(go.Scatter(
                            y=stats["rolling_avg"], name="Rolling Avg",
                            line=dict(color=COLOR_PALETTE["primary"], width=2),
                        ))
                        fig.update_layout(height=280, title=f"{col_name} over time", showlegend=True)
                        st.plotly_chart(fig, use_container_width=True)

        # ════════════════════════════════════════════════════════════════════
        # SECTION 4: EXPORT
        # ════════════════════════════════════════════════════════════════════
        st.divider()
        st.markdown("## 4️⃣ Export to Excel")
        lda_for_export = st.session_state.get("acad_lda")

        if st.button("📥 Generate Excel Report", type="primary", key="acad_export"):
            with st.spinner("Building Excel workbook..."):
                try:
                    out_path = "/tmp/academic_analysis.xlsx"
                    export_to_excel(
                        feature_df=feature_df,
                        model_results=model_results if model_results else None,
                        lda_results=lda_for_export,
                        output_path=out_path,
                    )
                    with open(out_path, "rb") as f:
                        excel_bytes = f.read()

                    st.download_button(
                        "⬇️ Download Excel Report",
                        data=excel_bytes,
                        file_name="academic_analysis.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )
                    st.success("✅ Excel report ready — click above to download.")
                    st.caption("Sheets: Feature Dataset | Descriptive Statistics | Model Results | LDA Topics")
                except Exception as e:
                    st.error(f"Export error: {e}")

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 5: CLI BATCH RUNNER INFO
    # ════════════════════════════════════════════════════════════════════════
    st.divider()
    with st.expander("⚡ Large-Scale Analysis: CLI Batch Runner", expanded=False):
        st.markdown("""
**For 100s to 50,000+ videos**, use `batch_runner.py` which runs independently of the browser:

```bash
# Basic usage
python batch_runner.py --urls urls.txt --output results/ --model tiny --workers 4

# Full options
python batch_runner.py \\
  --channel https://youtube.com/@ChannelName \\
  --max-videos 500 \\
  --output results/ \\
  --model small \\
  --workers 8 \\
  --resume \\
  --export-excel
```

**Key features:**
- Resume interrupted jobs (`--resume` flag skips already-processed items)
- Parallel workers (`--workers N` for multi-core processing)
- Checkpoint saving every N items so nothing is lost if it crashes
- Final Excel export with all features and model results
- Can run for days unattended on a cloud VM (AWS EC2, Google Cloud, etc.)

**Recommended VM for 30k+ videos:**
- AWS EC2 `c5.4xlarge` (16 vCPU, 32 GB RAM) ~$0.68/hr
- Estimated throughput: ~200-400 videos/hour depending on length and Whisper model
- 30k videos at 30 min avg = ~75-150 hours compute time
        """)


# ════════════════════════════════════════════════════════════════════════════
# RENDERING HELPERS
# ════════════════════════════════════════════════════════════════════════════

def _render_regression_results(result: Dict):
    if result.get("error"):
        st.error(result["error"])
        return
    col1, col2, col3 = st.columns(3)
    col1.metric("N", result.get("n", "—"))
    col2.metric("R²", result.get("r2") or result.get("r2_within", "—"))
    col3.metric("Adj. R²", result.get("r2_adj", "—"))

    coefs = result.get("coefficients", {})
    if coefs:
        rows = []
        for var, stats in coefs.items():
            if isinstance(stats, dict):
                rows.append({"Variable": var, "Coef": stats.get("coef",""), "SE": stats.get("se",""),
                             "t": stats.get("t",""), "p": stats.get("p",""), "Sig": stats.get("sig","")})
        if rows:
            coef_df = pd.DataFrame(rows)
            st.dataframe(coef_df, use_container_width=True, hide_index=True)
            st.caption("Significance: *** p<0.001  ** p<0.01  * p<0.05  . p<0.10")


def _render_importance_results(result: Dict):
    if result.get("error"):
        st.error(result["error"])
        return
    metric_key = "cv_r2" if "cv_r2" in result else "cv_accuracy"
    st.metric(f"{result.get('model','Model')} CV Score", f"{result.get(metric_key, '—'):.4f}")
    st.caption(f"±{result.get(metric_key+'_std', 0):.4f} std across folds")

    imp = result.get("feature_importance", {})
    if imp:
        imp_df = pd.DataFrame(list(imp.items()), columns=["Feature", "Importance"]).sort_values("Importance", ascending=False)
        fig = px.bar(imp_df, x="Importance", y="Feature", orientation="h",
                     color="Importance", color_continuous_scale=["#d3d3d3","#87ceeb","#1e90ff"],
                     title="Feature Importance")
        fig.update_layout(height=max(300, len(imp_df)*25), yaxis={"categoryorder":"total ascending"})
        st.plotly_chart(fig, use_container_width=True)
