# 🎥 TV & Film Visual Analyzer

Companion to the [TV & Film Script Analyzer](https://github.com/aphd87/TV-and-Film-Script-Analyzer).

Analyzes **produced video content** — no script required. Upload an episode or film, transcribe the audio, detect scene cuts, extract acoustic features, and run narrative analytics.

---

## What It Does

| Feature | Description |
|---|---|
| 🎙️ Whisper Transcription | Converts dialogue to text — any language, auto-detected |
| ✂️ Scene Detection | OpenCV + SceneDetect identify cut points and keyframes |
| 🎵 Acoustic Analysis | Librosa extracts energy, tempo, pitch per scene |
| 📊 Narrative Analytics | Pacing, act structure, dialogue density, NLP word analysis |
| 📚 Visual Library | Build a catalog of analyzed content for cross-show comparison |
| 🤖 AI Summary | Claude-powered narrative intelligence reports |

---

## Key Difference vs. Script Analyzer

The Script Analyzer requires a `.txt` or `.fdx` script file. The Visual Analyzer ingests **any video file** and works entirely from the produced content — useful for:

- Content you don't have scripts for
- Competitor analysis
- Catalog-level analytics for streaming platforms
- Post-production narrative review

---

## Setup

```bash
git clone https://github.com/aphd87/TV-and-Film-Visual-Analyzer
cd TV-and-Film-Visual-Analyzer
pip install -r requirements.txt
python -m spacy download en_core_web_sm
streamlit run app.py
```

### Anthropic API Key (for AI summaries)

Create `.streamlit/secrets.toml`:
```toml
ANTHROPIC_API_KEY = "your-key-here"
```

---

## File Structure

```
app.py                  # Entry point
constants.py            # Config, session state, dependency flags
ingestion.py            # Video → transcription → bundle pipeline
pages_upload.py         # Ingest tab
pages_analysis.py       # Analysis tab (scene, pacing, acoustic, NLP)
pages_library.py        # Library tab (catalog + comparison)
requirements.txt
```

---

## Bundle Format

Output is compatible with the Script Analyzer bundle structure:

```python
{
  "summary":        {...},          # scenes, words, duration
  "scene_df":       pd.DataFrame,  # per-scene data
  "char_df":        pd.DataFrame,  # character stats (from transcript)
  "emo_df":         pd.DataFrame,  # emotion data (NLP)
  "cop_edges":      pd.DataFrame,  # co-presence network
  "talk_edges":     pd.DataFrame,  # dialogue network
  "acoustic_df":    pd.DataFrame,  # energy, tempo, pitch per scene  ← visual-only
  "keyframe_times": List[float],   # timestamps of extracted frames  ← visual-only
  "transcript_text": str,          # full transcript                 ← visual-only
}
```

---

## Roadmap

- [ ] YouTube URL ingestion via yt-dlp
- [ ] Multi-speaker diarization (distinguish characters by voice)
- [ ] CLIP-based visual scene labeling (interior/exterior, day/night)
- [ ] Export to Script Analyzer format for cross-tool comparison
- [ ] Side-by-side script vs. produced episode diff

---

*Built by Anthony Palomba | Darden School of Business | GBK Collective*
