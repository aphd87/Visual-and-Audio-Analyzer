from __future__ import annotations
"""
pages_analysis.py — Deep single-content analysis.
Tabs: Scene Structure | Narrative Pacing | Acoustic Intelligence | Transcript NLP
"""

import re
from collections import Counter
from typing import Dict, Any

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from constants import COLOR_PALETTE, HAS_ANTHROPIC


def page_analysis():
    st.subheader("📊 Analysis")
    st.caption("Deep analysis of a single piece of content. Ingest something in the **Ingest** tab first.")

    bundle = st.session_state.get("current_visual_bundle")
    if bundle is None:
        st.info("📤 Nothing loaded yet — go to **Ingest** first.")
        return

    meta = st.session_state.get("current_visual_metadata", {})
    scene_df = bundle.get("scene_df", pd.DataFrame())
    acoustic_df = bundle.get("acoustic_df", pd.DataFrame())
    transcript = st.session_state.get("current_transcript", "")

    st.markdown(
        f"**{meta.get('title','Untitled')}** | {meta.get('type','')} | "
        f"{meta.get('genre','')} | {meta.get('year','')} | "
        f"Source: {meta.get('source','').replace('_',' ').title()}"
    )
    st.divider()

    tab_scene, tab_pacing, tab_acoustic, tab_nlp = st.tabs([
        "🎬 Scene Structure",
        "📈 Narrative Pacing",
        "🎵 Acoustic Intelligence",
        "📝 Transcript NLP",
    ])

    # ════════════════════════════════════════════════════════════════════════
    # TAB 1 — SCENE STRUCTURE
    # ════════════════════════════════════════════════════════════════════════
    with tab_scene:
        if scene_df.empty:
            st.info("No scene data available.")
        else:
            sm = bundle.get("summary", {})
            duration = sm.get("duration_seconds", 0)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Scenes", sm.get("scenes", 0))
            c2.metric("Total Words", f"{sm.get('dialogue_words', 0):,}")
            c3.metric("Duration", f"{int(duration//60)}m {int(duration%60)}s" if duration else "—")
            avg = scene_df["dialogue_words"].mean() if "dialogue_words" in scene_df.columns else 0
            c4.metric("Avg Words/Scene", f"{avg:.0f}")
            st.divider()

            if "dialogue_words" in scene_df.columns:
                fig = px.histogram(scene_df, x="dialogue_words", nbins=20,
                                   title="Scene Length Distribution",
                                   color_discrete_sequence=[COLOR_PALETTE["primary"]])
                fig.update_layout(height=320, xaxis_title="Words per Scene", yaxis_title="Count")
                st.plotly_chart(fig, use_container_width=True)

            display_cols = [c for c in ["scene_id","start_time","end_time","duration_seconds","dialogue_words"] if c in scene_df.columns]
            st.dataframe(scene_df[display_cols].rename(columns={
                "scene_id":"Scene","start_time":"Start (s)","end_time":"End (s)",
                "duration_seconds":"Duration (s)","dialogue_words":"Words",
            }), use_container_width=True, hide_index=True)

            n = len(scene_df)
            st.markdown("#### Estimated Act Structure")
            st.dataframe(pd.DataFrame([
                {"Act":"Act 1 — Setup",         "Scenes":f"1–{int(n*0.25)}",               "Position":"0–25%"},
                {"Act":"Act 2 — Confrontation", "Scenes":f"{int(n*0.25)+1}–{int(n*0.75)}", "Position":"25–75%"},
                {"Act":"Act 3 — Resolution",    "Scenes":f"{int(n*0.75)+1}–{n}",           "Position":"75–100%"},
            ]), use_container_width=True, hide_index=True)
            st.caption("Based on Syd Field's three-act model. Scene boundaries estimated from audio cut detection.")

    # ════════════════════════════════════════════════════════════════════════
    # TAB 2 — NARRATIVE PACING
    # ════════════════════════════════════════════════════════════════════════
    with tab_pacing:
        if scene_df.empty or "dialogue_words" not in scene_df.columns:
            st.info("No pacing data available.")
        else:
            df = scene_df.copy()
            df["rolling_avg"] = df["dialogue_words"].rolling(window=5, min_periods=1).mean()
            n = len(df)

            fig = go.Figure()
            fig.add_trace(go.Bar(x=df["scene_id"], y=df["dialogue_words"],
                                  name="Words/Scene", marker_color=COLOR_PALETTE["neutral"], opacity=0.6))
            fig.add_trace(go.Scatter(x=df["scene_id"], y=df["rolling_avg"],
                                      name="5-Scene Avg", line=dict(color=COLOR_PALETTE["primary"], width=2.5)))
            for pct, label in [(0.25, "Act 2"), (0.75, "Act 3")]:
                fig.add_vline(x=int(n*pct), line_dash="dash", line_color=COLOR_PALETTE["accent"],
                              annotation_text=label, annotation_position="top")
            fig.update_layout(title="Narrative Pacing", height=400,
                              xaxis_title="Scene", yaxis_title="Dialogue Words",
                              legend=dict(orientation="h", y=1.08))
            st.plotly_chart(fig, use_container_width=True)

            a1 = df.iloc[:int(n*0.25)]["dialogue_words"].mean()
            a2 = df.iloc[int(n*0.25):int(n*0.75)]["dialogue_words"].mean()
            a3 = df.iloc[int(n*0.75):]["dialogue_words"].mean()
            c1, c2, c3 = st.columns(3)
            c1.metric("Act 1 Avg Words", f"{a1:.0f}")
            c2.metric("Act 2 Avg Words", f"{a2:.0f}")
            c3.metric("Act 3 Avg Words", f"{a3:.0f}")

            if a3 < a2 * 0.8:
                st.success("✅ **Accelerating finale** — Act 3 pacing tightens. Strong structural rhythm.")
            elif a3 > a2 * 1.2:
                st.warning("⚠️ **Slowing finale** — Act 3 is heavier than Act 2. Consider tightening.")
            else:
                st.info("ℹ️ **Consistent pacing** across all three acts.")

            if len(df) > 5:
                peak = df.loc[df["dialogue_words"].idxmax()]
                trough = df.loc[df["dialogue_words"].idxmin()]
                cp, ct = st.columns(2)
                cp.info(f"🔺 **Peak**: Scene {int(peak['scene_id'])} — {int(peak['dialogue_words'])} words")
                ct.info(f"🔻 **Sparsest**: Scene {int(trough['scene_id'])} — {int(trough['dialogue_words'])} words")

    # ════════════════════════════════════════════════════════════════════════
    # TAB 3 — ACOUSTIC INTELLIGENCE
    # ════════════════════════════════════════════════════════════════════════
    with tab_acoustic:
        if acoustic_df.empty:
            st.info("Acoustic data not available. Re-ingest with a video or audio file (requires Librosa).")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Avg Energy (RMS)", f"{acoustic_df['rms_energy'].mean():.4f}")
            valid_tempo = acoustic_df[acoustic_df["tempo_bpm"] > 0]["tempo_bpm"] if "tempo_bpm" in acoustic_df.columns else pd.Series()
            c2.metric("Avg Tempo", f"{valid_tempo.mean():.0f} BPM" if not valid_tempo.empty else "—")
            valid_pitch = acoustic_df[acoustic_df["pitch_mean"] > 0]["pitch_mean"] if "pitch_mean" in acoustic_df.columns else pd.Series()
            c3.metric("Avg Pitch", f"{valid_pitch.mean():.0f} Hz" if not valid_pitch.empty else "—")
            st.divider()

            fig_e = px.area(acoustic_df, x="scene_id", y="rms_energy",
                             title="Audio Energy Profile",
                             color_discrete_sequence=[COLOR_PALETTE["primary"]])
            fig_e.update_layout(height=300, xaxis_title="Scene", yaxis_title="RMS Energy")
            st.plotly_chart(fig_e, use_container_width=True)
            st.caption("Energy peaks often correspond to high-intensity moments, confrontations, or music cues.")

            if "tempo_bpm" in acoustic_df.columns and acoustic_df["tempo_bpm"].sum() > 0:
                fig_t = px.bar(acoustic_df, x="scene_id", y="tempo_bpm",
                                title="Estimated Tempo per Scene (BPM)",
                                color_discrete_sequence=[COLOR_PALETTE["accent"]])
                fig_t.update_layout(height=280, xaxis_title="Scene", yaxis_title="BPM")
                st.plotly_chart(fig_t, use_container_width=True)

            if not scene_df.empty and "dialogue_words" in scene_df.columns:
                merged = pd.merge(scene_df[["scene_id","dialogue_words"]],
                                   acoustic_df[["scene_id","rms_energy"]], on="scene_id", how="inner")
                if not merged.empty:
                    corr = merged["dialogue_words"].corr(merged["rms_energy"])
                    fig_c = px.scatter(merged, x="dialogue_words", y="rms_energy",
                                       title=f"Dialogue vs Energy (r = {corr:.2f})",
                                       trendline="ols",
                                       color_discrete_sequence=[COLOR_PALETTE["primary"]])
                    fig_c.update_layout(height=320, xaxis_title="Dialogue Words", yaxis_title="RMS Energy")
                    st.plotly_chart(fig_c, use_container_width=True)
                    if corr > 0.4:
                        st.info("Positive correlation — louder scenes carry more dialogue.")
                    elif corr < -0.4:
                        st.info("Negative correlation — high-energy scenes are action/music-driven with sparse dialogue.")
                    else:
                        st.info("Weak correlation — audio energy and dialogue volume are relatively independent.")

    # ════════════════════════════════════════════════════════════════════════
    # TAB 4 — TRANSCRIPT NLP
    # ════════════════════════════════════════════════════════════════════════
    with tab_nlp:
        if not transcript:
            st.info("No transcript available.")
        else:
            stop_words = {
                "the","a","an","and","or","but","in","on","at","to","for","of","with",
                "is","it","this","that","was","are","be","have","i","you","he","she",
                "we","they","do","not","so","if","as","my","your","his","her","our",
                "its","me","him","us","them","just","like","know","think","get","going",
                "yeah","right","okay","well","um","uh","actually","really",
            }
            words = transcript.split()
            word_count = len(words)
            unique = len(set(w.lower().strip(".,!?\"'") for w in words))
            sentences = [s.strip() for s in re.split(r'[.!?]', transcript) if s.strip()]
            avg_sent = word_count / len(sentences) if sentences else 0

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Words", f"{word_count:,}")
            c2.metric("Unique Words", f"{unique:,}")
            c3.metric("Lexical Diversity", f"{unique/word_count*100:.1f}%" if word_count > 0 else "—")
            c4.metric("Avg Sentence", f"{avg_sent:.1f} words")
            st.divider()

            word_freq = Counter(
                w.lower().strip(".,!?\"'") for w in words
                if w.lower().strip(".,!?\"'") not in stop_words and len(w.strip(".,!?\"'")) > 2
            )
            top_words = pd.DataFrame(word_freq.most_common(20), columns=["Word","Count"])
            if not top_words.empty:
                fig_w = px.bar(top_words, x="Count", y="Word", orientation="h",
                                title="Top 20 Words (stopwords removed)",
                                color="Count",
                                color_continuous_scale=["#d3d3d3","#87ceeb","#1e90ff"])
                fig_w.update_layout(height=500, yaxis={"categoryorder":"total ascending"}, showlegend=False)
                st.plotly_chart(fig_w, use_container_width=True)

            st.divider()
            st.markdown("#### Topic Signals")
            topic_seeds = {
                "Conflict / Drama":      ["fight","argue","conflict","crisis","angry","tension","blame","threat"],
                "Emotion / Feeling":     ["feel","love","fear","hope","sad","happy","hurt","worried","care","trust"],
                "Money / Business":      ["money","deal","cost","pay","profit","market","budget","revenue","invest"],
                "Technology":            ["technology","data","system","digital","app","software","algorithm","ai","code"],
                "Family / Relationship": ["family","mother","father","sister","brother","marriage","child","home"],
                "Power / Politics":      ["power","control","government","leader","vote","decision","authority","law"],
            }
            lower_tx = transcript.lower()
            topic_scores = {t: sum(lower_tx.count(kw) for kw in kws) for t, kws in topic_seeds.items()}
            topic_scores = {k: v for k, v in topic_scores.items() if v > 0}
            if topic_scores:
                topic_df = pd.DataFrame(sorted(topic_scores.items(), key=lambda x: x[1], reverse=True),
                                         columns=["Topic","Signal Strength"])
                fig_t = px.bar(topic_df, x="Signal Strength", y="Topic", orientation="h",
                                title="Topic Signal Strength",
                                color="Signal Strength",
                                color_continuous_scale=["#d3d3d3","#87ceeb","#1e90ff"])
                fig_t.update_layout(height=320, yaxis={"categoryorder":"total ascending"}, showlegend=False)
                st.plotly_chart(fig_t, use_container_width=True)
                st.caption("Signal strength = keyword frequency. Indicates dominant themes — not a substitute for full semantic NLP (available in the Academic tab via LDA).")

            st.divider()
            with st.expander("📝 Full Transcript", expanded=False):
                st.text_area("", value=transcript, height=350, disabled=True, key="va_tx_view")
                st.download_button("⬇️ Download Transcript (.txt)", data=transcript.encode("utf-8"),
                                    file_name=f"{meta.get('title','transcript')}.txt", mime="text/plain")

            if HAS_ANTHROPIC:
                st.divider()
                st.markdown("#### 🤖 AI Narrative Summary")
                if st.button("Generate Summary", key="va_ai_btn"):
                    with st.spinner("Generating with Claude..."):
                        try:
                            from anthropic import Anthropic
                            client = Anthropic()
                            response = client.messages.create(
                                model="claude-sonnet-4-20250514", max_tokens=800,
                                messages=[{"role":"user","content":(
                                    f"Analyze this transcript from a {meta.get('type','production')} "
                                    f"titled '{meta.get('title','Unknown')}'.\n\n"
                                    f"Provide:\n1. Narrative summary (2-3 sentences)\n"
                                    f"2. Tone and emotional register\n3. Key themes\n"
                                    f"4. Notable observations\n\nTranscript:\n{transcript[:4000]}"
                                )}]
                            )
                            st.markdown(response.content[0].text)
                        except Exception as e:
                            st.error(f"AI error: {e}")
            else:
                st.info("💡 Add ANTHROPIC_API_KEY to Streamlit secrets to enable AI summaries.")
