from __future__ import annotations
"""
youtube_api.py — YouTube Data API v3 integration.

Uses the official API to fetch video metadata and captions/transcripts,
bypassing yt-dlp entirely for YouTube URLs. No IP bans, no 403s.

Requires: YOUTUBE_API_KEY in Streamlit secrets or environment variable.

Quota cost:
  - videos.list (metadata):  1 unit
  - captions.list (list):    50 units
  - captions.download:       200 units (requires OAuth — not used here)
  - timedText (public XML):  0 units   ← what we use

Free tier: 10,000 units/day (~100 full caption fetches/day).
"""

import re
import os
import json
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from typing import Dict, List, Any, Optional


# ════════════════════════════════════════════════════════════════════════════
# API KEY RESOLUTION
# ════════════════════════════════════════════════════════════════════════════

def get_api_key() -> Optional[str]:
    """Resolve YouTube API key from Streamlit secrets or environment."""
    # Try Streamlit secrets first
    try:
        import streamlit as st
        key = st.secrets.get("YOUTUBE_API_KEY") or st.secrets.get("youtube_api_key")
        if key:
            return str(key).strip()
    except Exception:
        pass
    # Fall back to environment variable
    return os.environ.get("YOUTUBE_API_KEY", "").strip() or None


def has_api_key() -> bool:
    return bool(get_api_key())


# ════════════════════════════════════════════════════════════════════════════
# VIDEO ID EXTRACTION
# ════════════════════════════════════════════════════════════════════════════

def extract_video_id(url: str) -> Optional[str]:
    """Extract YouTube video ID from any YouTube URL format."""
    patterns = [
        r"(?:v=|/v/|/embed/|youtu\.be/|/shorts/)([A-Za-z0-9_-]{11})",
        r"^([A-Za-z0-9_-]{11})$",  # bare ID
    ]
    for pattern in patterns:
        m = re.search(pattern, url)
        if m:
            return m.group(1)
    return None


def is_youtube_url(url: str) -> bool:
    return bool(extract_video_id(url)) and (
        "youtube.com" in url or "youtu.be" in url or len(url) == 11
    )


# ════════════════════════════════════════════════════════════════════════════
# VIDEO METADATA
# ════════════════════════════════════════════════════════════════════════════

def get_video_metadata(video_id: str, api_key: str) -> Dict[str, Any]:
    """Fetch video title, duration, channel, description via Data API v3."""
    url = (
        f"https://www.googleapis.com/youtube/v3/videos"
        f"?part=snippet,contentDetails"
        f"&id={video_id}"
        f"&key={api_key}"
    )
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read())
        items = data.get("items", [])
        if not items:
            return {"error": f"Video not found: {video_id}"}
        item = items[0]
        snippet = item.get("snippet", {})
        details = item.get("contentDetails", {})

        # Parse ISO 8601 duration (PT1H2M3S)
        duration_str = details.get("duration", "PT0S")
        duration_secs = _parse_iso_duration(duration_str)

        return {
            "title":       snippet.get("title", "Unknown"),
            "channel":     snippet.get("channelTitle", ""),
            "description": (snippet.get("description") or "")[:500],
            "duration":    duration_secs,
            "thumbnail":   (snippet.get("thumbnails", {}).get("high", {}) or {}).get("url", ""),
            "published":   snippet.get("publishedAt", ""),
            "tags":        snippet.get("tags", []),
            "error":       None,
        }
    except urllib.error.HTTPError as e:
        if e.code == 403:
            return {"error": "YouTube API quota exceeded or key invalid. Check YOUTUBE_API_KEY in secrets."}
        return {"error": f"API error {e.code}: {e.reason}"}
    except Exception as e:
        return {"error": str(e)[:200]}


def _parse_iso_duration(s: str) -> int:
    """Convert ISO 8601 duration string to seconds."""
    m = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", s)
    if not m:
        return 0
    h = int(m.group(1) or 0)
    mins = int(m.group(2) or 0)
    secs = int(m.group(3) or 0)
    return h * 3600 + mins * 60 + secs


# ════════════════════════════════════════════════════════════════════════════
# CAPTION FETCHING (no quota cost — public timedText endpoint)
# ════════════════════════════════════════════════════════════════════════════

def fetch_captions(video_id: str, lang: str = "en") -> Dict[str, Any]:
    """
    Fetch auto-generated or manual captions from YouTube's public timedText endpoint.
    Returns timestamped segments compatible with Whisper output format.

    No API key required. No quota consumed.
    Works for any video with captions enabled (most YouTube videos).
    """
    result = {"text": "", "segments": [], "language": lang, "source": "captions", "error": None}

    # Try auto-generated captions first, then manual
    caption_urls = [
        f"https://www.youtube.com/api/timedtext?v={video_id}&lang={lang}&fmt=json3",
        f"https://www.youtube.com/api/timedtext?v={video_id}&lang={lang}&kind=asr&fmt=json3",
        f"https://www.youtube.com/api/timedtext?v={video_id}&lang=en&fmt=json3",
        f"https://www.youtube.com/api/timedtext?v={video_id}&lang=en&kind=asr&fmt=json3",
    ]

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }

    data = None
    for cap_url in caption_urls:
        try:
            req = urllib.request.Request(cap_url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as resp:
                raw = resp.read()
                if len(raw) > 10:
                    data = json.loads(raw)
                    break
        except Exception:
            continue

    if data is None:
        # Fallback: try XML format
        return _fetch_captions_xml(video_id, lang, headers)

    # Parse json3 format
    segments = []
    full_text_parts = []

    for event in data.get("events", []):
        start_ms = event.get("tStartMs", 0)
        dur_ms = event.get("dDurationMs", 0)
        segs = event.get("segs", [])
        if not segs:
            continue
        text = "".join(s.get("utf8", "") for s in segs).strip()
        text = re.sub(r"\s+", " ", text).strip()
        if not text or text in ("[Music]", "[Applause]", "[Laughter]"):
            continue
        start_s = start_ms / 1000
        end_s = (start_ms + dur_ms) / 1000
        segments.append({"start": round(start_s, 2), "end": round(end_s, 2), "text": text})
        full_text_parts.append(text)

    if not segments:
        return {"text": "", "segments": [], "language": lang,
                "source": "captions", "error": "No captions found for this video. Try Upload File instead."}

    result["text"] = " ".join(full_text_parts)
    result["segments"] = segments
    return result


def _fetch_captions_xml(video_id: str, lang: str, headers: Dict) -> Dict[str, Any]:
    """XML caption fallback."""
    result = {"text": "", "segments": [], "language": lang, "source": "captions_xml", "error": None}
    urls = [
        f"https://www.youtube.com/api/timedtext?v={video_id}&lang={lang}",
        f"https://www.youtube.com/api/timedtext?v={video_id}&lang=en",
        f"https://www.youtube.com/api/timedtext?v={video_id}&lang={lang}&kind=asr",
    ]
    for url in urls:
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as resp:
                xml_data = resp.read()
            if len(xml_data) < 20:
                continue
            root = ET.fromstring(xml_data)
            segments = []
            texts = []
            for elem in root.findall("text"):
                start = float(elem.get("start", 0))
                dur = float(elem.get("dur", 2))
                text = (elem.text or "").strip()
                # Unescape HTML entities
                text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").replace("&#39;", "'").replace("&quot;", '"')
                text = re.sub(r"\s+", " ", text).strip()
                if text:
                    segments.append({"start": round(start, 2), "end": round(start + dur, 2), "text": text})
                    texts.append(text)
            if segments:
                result["text"] = " ".join(texts)
                result["segments"] = segments
                return result
        except Exception:
            continue

    result["error"] = "No captions found. Try Upload File instead."
    return result


# ════════════════════════════════════════════════════════════════════════════
# FULL YOUTUBE PIPELINE (metadata + captions, no download needed)
# ════════════════════════════════════════════════════════════════════════════

def fetch_youtube_content(
    url: str,
    lang: str = "en",
    progress_callback=None,
) -> Dict[str, Any]:
    """
    Main entry point for YouTube URLs.
    Returns a result dict compatible with download_url() output,
    but with transcript pre-populated from captions (no Whisper needed).

    result keys: title, duration, uploader, transcript_text, segments,
                 language, thumbnail_url, description, error, source
    """
    result = {
        "file_path": None,
        "title": "", "duration": 0.0, "uploader": "",
        "is_audio_only": True,
        "thumbnail_url": "", "description": "",
        "transcript_text": "", "segments": [],
        "language": lang,
        "source": "youtube_api",
        "error": None,
    }

    video_id = extract_video_id(url)
    if not video_id:
        result["error"] = f"Could not extract YouTube video ID from: {url}"
        return result

    api_key = get_api_key()

    # Step 1: Metadata (requires API key if available, skip gracefully if not)
    if api_key:
        if progress_callback:
            progress_callback("📋 Fetching video metadata via YouTube API...")
        meta = get_video_metadata(video_id, api_key)
        if meta.get("error"):
            if progress_callback:
                progress_callback(f"⚠️ Metadata fetch failed: {meta['error']} — continuing with captions...")
        else:
            result["title"]         = meta.get("title", "")
            result["duration"]      = meta.get("duration", 0)
            result["uploader"]      = meta.get("channel", "")
            result["thumbnail_url"] = meta.get("thumbnail", "")
            result["description"]   = meta.get("description", "")
    else:
        # No API key — use oembed for basic metadata (free, no key needed)
        if progress_callback:
            progress_callback("📋 Fetching basic metadata (no API key)...")
        try:
            oembed_url = f"https://www.youtube.com/oembed?url={urllib.parse.quote(url)}&format=json"
            with urllib.request.urlopen(oembed_url, timeout=8) as resp:
                oembed = json.loads(resp.read())
            result["title"]   = oembed.get("title", f"YouTube {video_id}")
            result["uploader"] = oembed.get("author_name", "")
        except Exception:
            result["title"] = f"YouTube Video {video_id}"

    # Step 2: Captions
    if progress_callback:
        progress_callback("📝 Fetching captions...")

    cap = fetch_captions(video_id, lang=lang)
    if cap.get("error"):
        result["error"] = (
            f"Captions unavailable: {cap['error']}\n\n"
            "**To analyze this video:** Download it locally and use **Upload File** instead."
        )
        return result

    result["transcript_text"] = cap["text"]
    result["segments"]        = cap["segments"]
    result["language"]        = cap.get("language", lang)

    word_count = len(cap["text"].split())
    if progress_callback:
        progress_callback(
            f"✅ Captions loaded — {word_count:,} words, "
            f"{len(cap['segments'])} segments"
        )

    return result
