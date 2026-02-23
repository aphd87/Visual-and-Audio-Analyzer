from __future__ import annotations
"""
pages_analysis.py — Deep analysis of transcribed visual content.

Tabs within this page:
  - Scene Structure
  - Narrative Pacing
  - Acoustic Intelligence
  - Transcript NLP
"""

import re
from typing import Dict, List, Any, Optional

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from constants import COLOR_PALETTE, HAS_ANTHROPIC


def page_analysis():
    st.subheader("📊 Content Analysis")
    st.caption("Deep analysis of transcribed and visual content. Upload a video first in the Ingest tab.")

    bundle = st.session_state.get("current_visual_bundle")
    if bundle is None:
        st.info("📤 No content loaded yet. Go to **Ingest** and upload a video or paste a transcript.")
        return

    metadata = st.session_state.get("current_visual_metadata", {})
    scene_df = bundle.get("scene_df", pd.DataFrame())
    acoustic_df = bundle.get("acoustic_df", pd.DataFrame())
    transcript = st.session_state.get("current_transcript", "")

    title = metadata.get("title", "Untitled")
    st.markdown(f"**Analyzing:** {title} | {metadata.get('type', '')} | {metadata.get('genre', '')} | {metadata.get('year', '')}")
    st.divider()

    tab_scene, tab_pacing, tab_acoustic, tab_nlp = st.tabs([
        "🎬 Scene Structure",
        "📈 Narrative Pacing",
        "🎵 Acoustic Intelligence",
        "📝 Transcript NLP",
    ])

    # ════════════════════════════════════════════════════════════════════════
    # TAB 1: SCENE STRUCTURE
    # ════════════════════════════════════════════════════════════════════════
    with tab_scene:
        st.markdown("### Scene Overview")

        if scene_df.empty:
            st.info("No scene data available.")
        else:
            sm = bundle.get("summary", {})
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Scenes", sm.get("scenes", 0))
            c2.metric("Total Words", f"{sm.get('dialogue_words', 0):,}")
            duration = sm.get("duration_seconds", 0)
            c3.metric("Duration", f"{int(duration//60)}m {int(duration%60)}s" if duration else "—")
            avg_words = scene_df["dialogue_words"].mean() if "dialogue_words" in scene_df.columns else 0
            c4.metric("Avg Words/Scene", f"{avg_words:.0f}")

            st.divider()

            # Scene length distribution
            st.markdown("#### Scene Length Distribution")
            if "dialogue_words" in scene_df.columns:
                fig_hist = px.histogram(
                    scene_df,
                    x="dialogue_words",
                    nbins=20,
                    title="Distribution of Scene Lengths (Words)",
                    color_discrete_sequence=[COLOR_PALETTE["primary"]],
                )
                fig_hist.update_layout(height=350, xaxis_title="Dialogue Words", yaxis_title="Scenes")
                st.plotly_chart(fig_hist, use_container_width=True)
                st.caption("Shorter scenes = faster pacing. Longer scenes = dialogue-heavy or exposition-heavy sequences.")

            # Scene table
            st.markdown("#### Scene Table")
            display_cols = [c for c in ["scene_id", "start_time", "end_time", "duration_seconds", "dialogue_words"]
                            if c in scene_df.columns]
            if display_cols:
                st.dataframe(
                    scene_df[display_cols].rename(columns={
                        "scene_id": "Scene",
                        "start_time": "Start (s)",
                        "end_time": "End (s)",
                        "duration_seconds": "Duration (s)",
                        "dialogue_words": "Words",
                    }),
                    use_container_width=True,
                    hide_index=True,
                )

            # Act structure estimation
            st.markdown("#### Estimated Act Structure")
            n_scenes = len(scene_df)
            act1_end = int(n_scenes * 0.25)
            act2_end = int(n_scenes * 0.75)

            act_df = pd.DataFrame([
                {"Act": "Act 1 (Setup)", "Scenes": f"1–{act1_end}", "Pct": "0–25%"},
                {"Act": "Act 2 (Confrontation)", "Scenes": f"{act1_end+1}–{act2_end}", "Pct": "25–75%"},
                {"Act": "Act 3 (Resolution)", "Scenes": f"{act2_end+1}–{n_scenes}", "Pct": "75–100%"},
            ])
            st.dataframe(act_df, use_container_width=True, hide_index=True)
            st.caption("Based on Syd Field's three-act model. Scene boundaries estimated from video cut detection.")

    # ════════════════════════════════════════════════════════════════════════
    # TAB 2: NARRATIVE PACING
    # ════════════════════════════════════════════════════════════════════════
    with tab_pacing:
        st.markdown("### Narrative Pacing")

        if scene_df.empty or "dialogue_words" not in scene_df.columns:
            st.info("No pacing data available.")
        else:
            # Rolling average pacing
            scene_df_copy = scene_df.copy()
            scene_df_copy["rolling_avg"] = scene_df_copy["dialogue_words"].rolling(window=5, min_periods=1).mean()

            fig_pacing = go.Figure()
            fig_pacing.add_trace(go.Bar(
                x=scene_df_copy["scene_id"],
                y=scene_df_copy["dialogue_words"],
                name="Words per Scene",
                marker_color=COLOR_PALETTE["neutral"],
                opacity=0.6,
            ))
            fig_pacing.add_trace(go.Scatter(
                x=scene_df_copy["scene_id"],
                y=scene_df_copy["rolling_avg"],
                name="5-Scene Rolling Average",
                line=dict(color=COLOR_PALETTE["primary"], width=2.5),
            ))

            # Mark act boundaries
            n = len(scene_df_copy)
            for pct, label in [(0.25, "Act 2"), (0.75, "Act 3")]:
                scene_mark = int(n * pct)
                fig_pacing.add_vline(
                    x=scene_mark,
                    line_dash="dash",
                    line_color=COLOR_PALETTE["accent"],
                    annotation_text=label,
                    annotation_position="top",
                )

            fig_pacing.update_layout(
                title="Narrative Pacing — Words per Scene with Rolling Average",
                height=400,
                xaxis_title="Scene Number",
                yaxis_title="Dialogue Words",
                legend=dict(orientation="h", y=1.08),
            )
            st.plotly_chart(fig_pacing, use_container_width=True)

            # Pacing insights
            st.markdown("#### Pacing Insights")
            total = len(scene_df_copy)
            act1 = scene_df_copy.iloc[:int(total*0.25)]
            act2 = scene_df_copy.iloc[int(total*0.25):int(total*0.75)]
            act3 = scene_df_copy.iloc[int(total*0.75):]

            a1_avg = act1["dialogue_words"].mean() if not act1.empty else 0
            a2_avg = act2["dialogue_words"].mean() if not act2.empty else 0
            a3_avg = act3["dialogue_words"].mean() if not act3.empty else 0

            col1, col2, col3 = st.columns(3)
            col1.metric("Act 1 Avg Words/Scene", f"{a1_avg:.0f}")
            col2.metric("Act 2 Avg Words/Scene", f"{a2_avg:.0f}")
            col3.metric("Act 3 Avg Words/Scene", f"{a3_avg:.0f}")

            if a3_avg < a2_avg * 0.8:
                st.success("✅ **Accelerating finale** — Act 3 has shorter, faster scenes. Strong structural pacing.")
            elif a3_avg > a2_avg * 1.2:
                st.warning("⚠️ **Slowing finale** — Act 3 is more dialogue-heavy than Act 2. Consider tightening the ending.")
            else:
                st.info("ℹ️ **Consistent pacing** across acts.")

            # Peak/trough moments
            st.markdown("#### Notable Pacing Moments")
            if len(scene_df_copy) > 5:
                peak_scene = scene_df_copy.loc[scene_df_copy["dialogue_words"].idxmax()]
                trough_scene = scene_df_copy.loc[scene_df_copy["dialogue_words"].idxmin()]
                col_p, col_t = st.columns(2)
                col_p.info(f"🔺 **Most dialogue-heavy**: Scene {int(peak_scene['scene_id'])} ({int(peak_scene['dialogue_words'])} words)")
                col_t.info(f"🔻 **Sparsest scene**: Scene {int(trough_scene['scene_id'])} ({int(trough_scene['dialogue_words'])} words)")

    # ════════════════════════════════════════════════════════════════════════
    # TAB 3: ACOUSTIC INTELLIGENCE
    # ════════════════════════════════════════════════════════════════════════
    with tab_acoustic:
        st.markdown("### Acoustic Intelligence")

        if acoustic_df.empty:
            st.info("Acoustic data not available. Re-analyze with a video file and Librosa installed.")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Avg Energy (RMS)", f"{acoustic_df['rms_energy'].mean():.4f}")
            if "tempo_bpm" in acoustic_df.columns:
                valid_tempo = acoustic_df[acoustic_df["tempo_bpm"] > 0]["tempo_bpm"]
                c2.metric("Avg Tempo", f"{valid_tempo.mean():.0f} BPM" if not valid_tempo.empty else "—")
            if "pitch_mean" in acoustic_df.columns:
                valid_pitch = acoustic_df[acoustic_df["pitch_mean"] > 0]["pitch_mean"]
                c3.metric("Avg Pitch", f"{valid_pitch.mean():.0f} Hz" if not valid_pitch.empty else "—")

            st.divider()

            # Energy over time
            fig_e = px.area(
                acoustic_df,
                x="scene_id",
                y="rms_energy",
                title="Audio Energy Profile (RMS per Scene)",
                color_discrete_sequence=[COLOR_PALETTE["primary"]],
            )
            fig_e.update_layout(height=320, xaxis_title="Scene", yaxis_title="RMS Energy")
            st.plotly_chart(fig_e, use_container_width=True)
            st.caption("High energy peaks often correspond to action, confrontation, or climactic moments.")

            # Combine acoustic + pacing if both available
            if not scene_df.empty and "dialogue_words" in scene_df.columns:
                st.markdown("#### Acoustic vs. Dialogue Correlation")
                merged = pd.merge(
                    scene_df[["scene_id", "dialogue_words"]],
                    acoustic_df[["scene_id", "rms_energy"]],
                    on="scene_id",
                    how="inner",
                )
                if not merged.empty:
                    fig_corr = px.scatter(
                        merged,
                        x="dialogue_words",
                        y="rms_energy",
                        title="Dialogue Volume vs Audio Energy",
                        trendline="ols",
                        color_discrete_sequence=[COLOR_PALETTE["primary"]],
                    )
                    fig_corr.update_layout(height=350, xaxis_title="Dialogue Words", yaxis_title="RMS Energy")
                    st.plotly_chart(fig_corr, use_container_width=True)
                    st.caption("Positive correlation: louder/more energetic scenes tend to have more dialogue. "
                               "Negative correlation: action/music-driven scenes have high energy but sparse dialogue.")

    # ════════════════════════════════════════════════════════════════════════
    # TAB 4: TRANSCRIPT NLP
    # ════════════════════════════════════════════════════════════════════════
    with tab_nlp:
        st.markdown("### Transcript NLP")

        if not transcript:
            st.info("No transcript available.")
        else:
            words = transcript.split()
            word_count = len(words)
            unique_words = len(set(w.lower().strip(".,!?\"'") for w in words))
            sentences = [s.strip() for s in re.split(r'[.!?]', transcript) if s.strip()]
            avg_sentence_len = word_count / len(sentences) if sentences else 0

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Words", f"{word_count:,}")
            c2.metric("Unique Words", f"{unique_words:,}")
            c3.metric("Lexical Diversity", f"{unique_words/word_count*100:.1f}%" if word_count > 0 else "—")
            c4.metric("Avg Sentence Length", f"{avg_sentence_len:.1f} words")

            st.divider()

            # Word frequency
            st.markdown("#### Most Frequent Words")
            from collections import Counter
            stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
                          "of", "with", "is", "it", "this", "that", "was", "are", "be", "have",
                          "i", "you", "he", "she", "we", "they", "do", "not", "so", "if", "as",
                          "my", "your", "his", "her", "our", "its", "me", "him", "us", "them"}
            word_freq = Counter(
                w.lower().strip(".,!?\"'") for w in words
                if w.lower().strip(".,!?\"'") not in stop_words
                and len(w.strip(".,!?\"'")) > 2
            )
            top_words = pd.DataFrame(word_freq.most_common(20), columns=["Word", "Count"])

            if not top_words.empty:
                fig_words = px.bar(
                    top_words,
                    x="Count",
                    y="Word",
                    orientation="h",
                    title="Top 20 Words (excluding stopwords)",
                    color="Count",
                    color_continuous_scale=["#d3d3d3", "#87ceeb", "#1e90ff"],
                )
                fig_words.update_layout(height=500, yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig_words, use_container_width=True)

            # Claude AI summary (if available)
            if HAS_ANTHROPIC:
                st.divider()
                st.markdown("#### 🤖 AI Narrative Summary")
                st.caption("Uses Claude to generate a narrative intelligence report from the transcript.")

                if st.button("Generate AI Summary", key="va_ai_summary_btn"):
                    with st.spinner("Generating narrative summary with Claude..."):
                        try:
                            from anthropic import Anthropic
                            client = Anthropic()
                            excerpt = transcript[:4000]
                            response = client.messages.create(
                                model="claude-sonnet-4-20250514",
                                max_tokens=800,
                                messages=[{
                                    "role": "user",
                                    "content": (
                                        f"You are a media analyst. Analyze this transcript excerpt from a "
                                        f"{metadata.get('type', 'production')} titled '{metadata.get('title', 'Unknown')}'.\n\n"
                                        f"Provide:\n"
                                        f"1. Narrative summary (2-3 sentences)\n"
                                        f"2. Tone and emotional register\n"
                                        f"3. Key themes\n"
                                        f"4. Notable narrative observations\n\n"
                                        f"Transcript excerpt:\n{excerpt}"
                                    )
                                }]
                            )
                            summary_text = response.content[0].text
                            st.markdown(summary_text)
                        except Exception as e:
                            st.error(f"AI summary error: {e}")
            else:
                st.info("💡 Add an Anthropic API key in Streamlit secrets to enable AI narrative summaries.")
