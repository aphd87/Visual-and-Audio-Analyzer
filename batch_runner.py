#!/usr/bin/env python3
"""
batch_runner.py — CLI batch processor for TV & Film Visual Analyzer.

For large-scale academic research: 100s to 50,000+ videos.
Runs independently of the browser/Streamlit UI.

Usage:
    # Process a list of URLs
    python batch_runner.py --urls urls.txt --output results/ --model tiny

    # Process a YouTube channel
    python batch_runner.py --channel https://youtube.com/@ChannelName --max-videos 500 --output results/

    # Process a podcast RSS feed
    python batch_runner.py --rss https://feeds.example.com/podcast.xml --max-episodes 200 --output results/

    # Resume an interrupted job
    python batch_runner.py --urls urls.txt --output results/ --resume

    # With parallel workers and Excel export
    python batch_runner.py --urls urls.txt --output results/ --workers 4 --export-excel

Author: TV & Film Visual Analyzer
"""

import argparse
import csv
import json
import os
import sys
import time
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

warnings.filterwarnings("ignore")

# ── Dependency check ─────────────────────────────────────────────────────────
REQUIRED = ["yt_dlp", "whisper", "pandas", "numpy"]
OPTIONAL = ["cv2", "librosa", "xgboost", "statsmodels"]

for pkg in REQUIRED:
    try:
        __import__(pkg)
    except ImportError:
        print(f"ERROR: Required package '{pkg}' not installed. Run: pip install {pkg}")
        sys.exit(1)

import pandas as pd
import numpy as np


# ════════════════════════════════════════════════════════════════════════════
# URL COLLECTION
# ════════════════════════════════════════════════════════════════════════════

def collect_from_channel(channel_url: str, max_videos: int) -> List[Dict]:
    import yt_dlp
    print(f"  Fetching channel: {channel_url}")
    ydl_opts = {"quiet": True, "no_warnings": True, "extract_flat": True, "playlistend": max_videos}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(channel_url, download=False)
    entries = info.get("entries", []) or []
    items = []
    for e in entries[:max_videos]:
        url = e.get("url") or e.get("webpage_url", "")
        if url and not url.startswith("http"):
            url = f"https://www.youtube.com/watch?v={url}"
        if url:
            items.append({
                "url": url, "title": e.get("title", "Unknown"),
                "duration": e.get("duration", 0),
                "uploader": e.get("uploader", info.get("uploader", "")),
                "type": "Video", "genre": "",
            })
    print(f"  Found {len(items)} items from {info.get('title', channel_url)}")
    return items


def collect_from_rss(rss_url: str, max_episodes: int) -> List[Dict]:
    import urllib.request
    import xml.etree.ElementTree as ET
    print(f"  Parsing RSS: {rss_url}")
    with urllib.request.urlopen(rss_url, timeout=15) as resp:
        root = ET.fromstring(resp.read())
    ns = {"itunes": "http://www.itunes.com/dtds/podcast-1.0.dtd"}
    channel = root.find("channel")
    channel_title = channel.findtext("title", "Unknown")
    items = []
    for item in (channel.findall("item") or [])[:max_episodes]:
        enc = item.find("enclosure")
        url = enc.get("url") if enc is not None else None
        if url:
            items.append({
                "url": url, "title": item.findtext("title", "Unknown Episode"),
                "duration": 0, "uploader": channel_title, "type": "Podcast", "genre": "Podcast",
            })
    print(f"  Found {len(items)} episodes from {channel_title}")
    return items


def collect_from_file(filepath: str) -> List[Dict]:
    urls = []
    with open(filepath) as f:
        for line in f:
            url = line.strip()
            if url and url.startswith("http"):
                urls.append({"url": url, "title": "", "duration": 0, "uploader": "", "type": "Unknown", "genre": ""})
    print(f"  Loaded {len(urls)} URLs from {filepath}")
    return urls


# ════════════════════════════════════════════════════════════════════════════
# SINGLE ITEM PROCESSOR
# ════════════════════════════════════════════════════════════════════════════

def process_item(
    item: Dict,
    whisper_model: str = "base",
    checkpoint_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Process a single URL item and return a feature row dict."""
    url = item["url"]
    title = item.get("title", url)

    # Check checkpoint
    if checkpoint_dir:
        cp_file = Path(checkpoint_dir) / f"{_url_hash(url)}.json"
        if cp_file.exists():
            print(f"  [SKIP] {title[:50]} (already processed)")
            with open(cp_file) as f:
                return json.load(f)

    print(f"  [START] {title[:60]}")
    t_start = time.time()
    row = {"url": url, "title": item.get("title",""), "uploader": item.get("uploader",""),
           "type": item.get("type",""), "genre": item.get("genre",""), "status": "error"}

    try:
        # Import here to avoid module-level issues with optional deps
        from ingestion import (
            download_url, ingest_video, transcribe_audio,
            segments_to_scene_df, analyze_acoustics, build_visual_bundle,
        )
        from academic_modeling import bundle_to_feature_row

        # Download
        dl = download_url(url)
        if dl.get("error"):
            row["error"] = dl["error"]
            return row

        fp = dl["file_path"]
        is_audio = dl.get("is_audio_only", False)
        meta = {
            "title": dl.get("title") or title, "type": item.get("type",""),
            "genre": item.get("genre",""), "year": item.get("year",""),
            "source": "batch_cli", "url": url, "uploader": dl.get("uploader",""),
        }

        # Video ingestion
        video_info = {"scene_cuts":[], "keyframes":[], "keyframe_times":[],
                      "fps":0, "duration_seconds": dl.get("duration",0), "total_frames":0}
        try:
            import cv2
            if not is_audio:
                vi = ingest_video(fp)
                if not vi.get("error"):
                    video_info = vi
                    video_info["keyframes"] = []  # Don't store frames in batch mode
        except ImportError:
            pass

        # Transcribe
        tx = transcribe_audio(fp, model_size=whisper_model)
        if tx.get("error"):
            row["error"] = "Transcription failed"
            return row

        scene_df = segments_to_scene_df(tx["segments"], video_info["scene_cuts"])

        # Acoustics
        acoustic_df = pd.DataFrame()
        try:
            import librosa
            acoustic_df = analyze_acoustics(fp, video_info["scene_cuts"])
        except ImportError:
            pass

        bundle = build_visual_bundle(
            scene_df=scene_df, acoustic_df=acoustic_df,
            transcript_text=tx["text"], video_info=video_info, metadata=meta,
        )

        # Extract features
        row = bundle_to_feature_row(bundle, meta)
        row["status"] = "success"
        row["process_time_seconds"] = round(time.time() - t_start, 1)

        # Checkpoint
        if checkpoint_dir:
            with open(cp_file, "w") as f:
                json.dump(row, f, default=str)

        print(f"  [DONE] {title[:50]} — {row.get('n_scenes',0)} scenes, "
              f"{row.get('total_words',0):,} words ({row['process_time_seconds']:.0f}s)")

    except Exception as e:
        row["error"] = str(e)
        row["traceback"] = traceback.format_exc()[-500:]
        print(f"  [ERROR] {title[:50]}: {str(e)[:80]}")

    finally:
        try:
            if "fp" in locals() and fp and os.path.exists(fp):
                os.unlink(fp)
                parent = os.path.dirname(fp)
                if os.path.isdir(parent) and not os.listdir(parent):
                    os.rmdir(parent)
        except Exception:
            pass

    return row


def _url_hash(url: str) -> str:
    import hashlib
    return hashlib.md5(url.encode()).hexdigest()[:12]


# ════════════════════════════════════════════════════════════════════════════
# MAIN RUNNER
# ════════════════════════════════════════════════════════════════════════════

def run_batch(
    items: List[Dict],
    output_dir: str,
    whisper_model: str = "base",
    n_workers: int = 1,
    checkpoint: bool = True,
    export_excel: bool = False,
    checkpoint_every: int = 10,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = str(output_path / "checkpoints") if checkpoint else None
    if checkpoint_dir:
        Path(checkpoint_dir).mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print(f"TV & Film Visual Analyzer — Batch Runner")
    print(f"{'='*60}")
    print(f"Items to process: {len(items)}")
    print(f"Whisper model:    {whisper_model}")
    print(f"Workers:          {n_workers}")
    print(f"Output dir:       {output_dir}")
    print(f"Checkpointing:    {'on' if checkpoint else 'off'}")
    print(f"{'='*60}\n")

    start = time.time()
    all_rows = []
    errors = []

    if n_workers == 1:
        for i, item in enumerate(items):
            print(f"[{i+1}/{len(items)}]", end=" ")
            row = process_item(item, whisper_model=whisper_model, checkpoint_dir=checkpoint_dir)
            all_rows.append(row)
            if row.get("status") != "success":
                errors.append(row)
            # Checkpoint CSV every N items
            if checkpoint and (i+1) % checkpoint_every == 0:
                _save_csv(all_rows, output_path / "results_partial.csv")
                print(f"  → Checkpoint saved ({i+1} items)")
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(process_item, item, whisper_model, checkpoint_dir): (i, item)
                for i, item in enumerate(items)
            }
            completed = 0
            for future in as_completed(futures):
                i, item = futures[future]
                try:
                    row = future.result()
                    all_rows.append(row)
                    if row.get("status") != "success":
                        errors.append(row)
                except Exception as e:
                    errors.append({"url": item["url"], "error": str(e)})
                completed += 1
                print(f"  Progress: {completed}/{len(items)} ({completed/len(items)*100:.1f}%)")
                if checkpoint and completed % checkpoint_every == 0:
                    _save_csv(all_rows, output_path / "results_partial.csv")

    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"Batch complete: {len(all_rows)} processed, {len(errors)} errors")
    print(f"Total time: {elapsed/3600:.1f}h ({elapsed/len(items):.0f}s/item avg)")
    print(f"{'='*60}\n")

    # Save results
    feature_df = pd.DataFrame(all_rows)
    csv_path = output_path / "results.csv"
    _save_csv(all_rows, csv_path)
    print(f"CSV saved: {csv_path}")

    if export_excel:
        try:
            from academic_modeling import export_to_excel
            xl_path = str(output_path / "results.xlsx")
            export_to_excel(feature_df=feature_df, output_path=xl_path)
            print(f"Excel saved: {xl_path}")
        except Exception as e:
            print(f"Excel export failed: {e}")

    # Error log
    if errors:
        error_path = output_path / "errors.json"
        with open(error_path, "w") as f:
            json.dump(errors, f, indent=2, default=str)
        print(f"Error log: {error_path} ({len(errors)} items)")

    return feature_df


def _save_csv(rows: List[Dict], path):
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


# ════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="TV & Film Visual Analyzer — CLI Batch Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # YouTube channel
  python batch_runner.py --channel https://youtube.com/@3Blue1Brown --max-videos 50 --output results/

  # Podcast RSS feed
  python batch_runner.py --rss https://feeds.simplecast.com/xyz.xml --max-episodes 100 --output results/

  # URL list file (one URL per line)
  python batch_runner.py --urls urls.txt --output results/ --model small --workers 4 --export-excel

  # Resume interrupted job
  python batch_runner.py --urls urls.txt --output results/ --resume
        """
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--channel", help="YouTube channel or playlist URL")
    source.add_argument("--rss", help="Podcast RSS feed URL")
    source.add_argument("--urls", help="Path to text file with one URL per line")

    parser.add_argument("--output", required=True, help="Output directory for results")
    parser.add_argument("--model", default="base",
                        choices=["tiny","base","small","medium"],
                        help="Whisper model size (default: base)")
    parser.add_argument("--max-videos", type=int, default=50, help="Max videos from channel (default: 50)")
    parser.add_argument("--max-episodes", type=int, default=50, help="Max podcast episodes (default: 50)")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers (default: 1)")
    parser.add_argument("--resume", action="store_true", help="Skip already-processed items")
    parser.add_argument("--export-excel", action="store_true", help="Generate Excel report")
    parser.add_argument("--no-checkpoint", action="store_true", help="Disable checkpointing")
    parser.add_argument("--checkpoint-every", type=int, default=10,
                        help="Save partial results every N items (default: 10)")

    args = parser.parse_args()

    # Collect items
    print("\nCollecting items...")
    if args.channel:
        items = collect_from_channel(args.channel, args.max_videos)
    elif args.rss:
        items = collect_from_rss(args.rss, args.max_episodes)
    else:
        items = collect_from_file(args.urls)

    if not items:
        print("No items found. Exiting.")
        sys.exit(1)

    run_batch(
        items=items,
        output_dir=args.output,
        whisper_model=args.model,
        n_workers=args.workers,
        checkpoint=not args.no_checkpoint,
        export_excel=args.export_excel,
        checkpoint_every=args.checkpoint_every,
    )


if __name__ == "__main__":
    main()
