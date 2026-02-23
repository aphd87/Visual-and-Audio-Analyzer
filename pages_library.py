from __future__ import annotations
"""
pages_library.py — Visual Library: manage and compare analyzed video content.
"""

from typing import Dict, List, Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from constants import COLOR_PALETTE


def page_library():
    st.subheader("📚 Visual Library")
    st.caption("Manage analyzed episodes and films. Build a library to enable cross-content comparison.")

    lib = st.session_state.get("visual_library", [])

    if not lib:
        st.info("📤 No content in library yet. Analyze a video in the **Ingest** tab to get started.")
        return

    st.success(f"**{len(lib)} item(s)** in your Visual Library.")
    st.divider()

    # ── Library table ────────────────────────────────────────────────────────
    st.markdown("### 📋 Library Contents")
    lib_rows = []
    for item in lib:
        meta = item.get("metadata", {})
        bundle = item.get("bundle", {})
        sm = bundle.get("summary", {})
        duration = sm.get("duration_seconds", 0)
        lib_rows.append({
            "Title": meta.get("title", "Untitled"),
            "Type": meta.get("type", "—"),
            "Genre": meta.get("genre", "—"),
            "Year": meta.get("year", "—"),
            "Scenes": sm.get("scenes", 0),
            "Words": f"{sm.get('dialogue_words', 0):,}",
            "Duration": f"{int(duration//60)}m {int(duration%60)}s" if duration else "—",
        })

    lib_df = pd.DataFrame(lib_rows)
    st.dataframe(lib_df, use_container_width=True, hide_index=True)

    st.divider()

    # ── Cross-library comparison ─────────────────────────────────────────────
    if len(lib) >= 2:
        st.markdown("### ⚖️ Library Comparison")

        titles = [item.get("title", item.get("metadata", {}).get("title", "Untitled")) for item in lib]
        selected_titles = st.multiselect(
            "Select items to compare (2–4)",
            options=titles,
            default=titles[:min(3, len(titles))],
            key="lib_compare_select",
        )

        selected_items = [item for item in lib if item.get("title", item.get("metadata", {}).get("title", "")) in selected_titles]

        if len(selected_items) >= 2:
            # Scene count comparison
            comp_rows = []
            for item in selected_items:
                meta = item.get("metadata", {})
                sm = item.get("bundle", {}).get("summary", {})
                duration = sm.get("duration_seconds", 0)
                comp_rows.append({
                    "Title": meta.get("title", "Untitled"),
                    "Scenes": sm.get("scenes", 0),
                    "Words": sm.get("dialogue_words", 0),
                    "Duration (s)": duration,
                    "Words/Scene": round(sm.get("dialogue_words", 0) / max(sm.get("scenes", 1), 1), 1),
                })

            comp_df = pd.DataFrame(comp_rows)

            col1, col2 = st.columns(2)

            with col1:
                fig_scenes = px.bar(
                    comp_df,
                    x="Title",
                    y="Scenes",
                    title="Scene Count Comparison",
                    color="Title",
                    color_discrete_sequence=["#1e90ff", "#ff6b6b", "#4ecdc4", "#ffe66d"],
                )
                fig_scenes.update_layout(height=350, showlegend=False)
                st.plotly_chart(fig_scenes, use_container_width=True)

            with col2:
                fig_words = px.bar(
                    comp_df,
                    x="Title",
                    y="Words/Scene",
                    title="Avg Words per Scene",
                    color="Title",
                    color_discrete_sequence=["#1e90ff", "#ff6b6b", "#4ecdc4", "#ffe66d"],
                )
                fig_words.update_layout(height=350, showlegend=False)
                st.plotly_chart(fig_words, use_container_width=True)

            # Pacing overlay
            st.markdown("#### Pacing Overlay — Dialogue Words per Scene")
            fig_overlay = go.Figure()
            colors = ["#1e90ff", "#ff6b6b", "#4ecdc4", "#ffe66d"]

            for idx, item in enumerate(selected_items):
                title = item.get("metadata", {}).get("title", "Untitled")
                scene_df = item.get("bundle", {}).get("scene_df", pd.DataFrame())
                if scene_df.empty or "dialogue_words" not in scene_df.columns:
                    continue
                # Normalize scene positions to 0-100%
                n = len(scene_df)
                x_pct = [i / n * 100 for i in range(n)]
                rolling = scene_df["dialogue_words"].rolling(window=5, min_periods=1).mean().tolist()

                fig_overlay.add_trace(go.Scatter(
                    x=x_pct,
                    y=rolling,
                    name=title,
                    line=dict(color=colors[idx % len(colors)], width=2),
                    mode="lines",
                ))

            fig_overlay.update_layout(
                height=400,
                xaxis_title="Story Position (%)",
                yaxis_title="Words (5-scene rolling avg)",
                legend=dict(orientation="h", y=1.08),
                xaxis=dict(ticksuffix="%"),
            )
            st.plotly_chart(fig_overlay, use_container_width=True)
            st.caption("Normalized to story position % so episodes of different lengths are comparable.")

            # Acoustic comparison if available
            acoustic_available = any(
                not item.get("bundle", {}).get("acoustic_df", pd.DataFrame()).empty
                for item in selected_items
            )
            if acoustic_available:
                st.markdown("#### Acoustic Energy Overlay")
                fig_acoustic = go.Figure()
                for idx, item in enumerate(selected_items):
                    title = item.get("metadata", {}).get("title", "Untitled")
                    acoustic_df = item.get("bundle", {}).get("acoustic_df", pd.DataFrame())
                    if acoustic_df.empty or "rms_energy" not in acoustic_df.columns:
                        continue
                    n = len(acoustic_df)
                    x_pct = [i / n * 100 for i in range(n)]
                    fig_acoustic.add_trace(go.Scatter(
                        x=x_pct,
                        y=acoustic_df["rms_energy"].tolist(),
                        name=title,
                        line=dict(color=colors[idx % len(colors)], width=2),
                    ))
                fig_acoustic.update_layout(
                    height=350,
                    xaxis_title="Story Position (%)",
                    yaxis_title="RMS Energy",
                    legend=dict(orientation="h", y=1.08),
                    xaxis=dict(ticksuffix="%"),
                )
                st.plotly_chart(fig_acoustic, use_container_width=True)

    # ── Remove items ─────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### 🗑️ Remove from Library")
    remove_title = st.selectbox("Select item to remove", options=["—"] + [item.get("metadata", {}).get("title", "Untitled") for item in lib], key="lib_remove_select")
    if remove_title != "—":
        if st.button(f"Remove '{remove_title}'", key="lib_remove_btn"):
            st.session_state["visual_library"] = [
                item for item in lib
                if item.get("metadata", {}).get("title", "Untitled") != remove_title
            ]
            st.success(f"Removed '{remove_title}' from library.")
            st.rerun()
