# Task 3 — Setlist API Reconciliation Agent
### Warner Chappell Music · AI Automation Analyst Assessment

An agentic Streamlit application that fetches live tour setlist data from an external API, reconciles each track against an internal controlled-songs catalog, and outputs a `matched_setlists.csv` report — using deterministic pre-processing first, with an LLM only for genuinely ambiguous cases.

---

## Quick Start

```bash
# 1. Enter the directory
cd task3_setlist_matching

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the app
streamlit run app.py
```

In the sidebar:
- Paste your **Groq** or **OpenAI** API key
- Leave "Local file" selected (uses bundled `tour_payload.json` + `catalog.csv`)
- Click **▶ Run Reconciliation**

To use a live URL instead, host `tour_payload.json` on GitHub Gist, select "URL" in the sidebar, and paste the raw URL.

---

## Environment

| Variable | Purpose |
|---|---|
| `GROQ_API_KEY` | Auto-detected — no sidebar entry needed |
| `OPENAI_API_KEY` | Fallback |

---

## Files

```
app.py               ← Streamlit UI (WC theme, tabs, stats, download)
pipeline.py          ← Core logic (API fetch, catalog load, matching, LLM)
catalog.csv          ← 13-song internal controlled catalog (sample data)
tour_payload.json    ← Mock tour API response (2 shows, 10 setlist tracks)
requirements.txt
README.md
```

**Output** (after running):
```
matched_setlists.csv ← Reconciliation report
```

---

## Architecture

```
Stage 1: API Ingestion
  fetch_tour_data(source)
    → supports URL (urllib, timeout=15, error handling) or local file path
    → validates JSON structure and "status": "success"

Stage 2: Catalog Load
  load_catalog(file)
    → accepts path or file-like object (Streamlit UploadedFile compatible)
    → returns list of dicts

Stage 3: Flatten Setlist
  flatten_setlist(payload)
    → explodes nested shows[].setlist[] into flat per-track rows
    → preserves show_date, venue_name, city, artist, tour

Stage 4: Deterministic Pre-Processing (no LLM cost)
  deterministic_match(track, catalog)
    → Exact string match
    → Normalized match (lowercase + punctuation stripped)
    → Qualifier-stripped match: removes "(Acoustic)", "(Extended Jam)", etc.
    → Medley detection: splits on " / " and matches each part independently

Stage 5: Agentic Fuzzy Matching (LLM — only for unresolved tracks)
  llm_fuzzy_match(track, catalog, client, model)
    → One LLM call per unresolved track
    → Handles abbreviations, misspellings, garbled text
    → Medley awareness (returns multiple matches from one input)
    → False-positive prevention via explicit system prompt rules
    → Returns matched catalog_id, confidence level, and reasoning
```

---

## Design Decisions

### API & Data Handling

`fetch_tour_data()` handles both URL and local file sources with a unified interface. For URLs:
- Uses `urllib.request` (no extra dependency) with a 15-second timeout
- Catches `HTTPError` (bad status codes) and `URLError` (network failures) separately
- Validates `status: "success"` field before processing

The nested JSON structure (`data.shows[].setlist[]`) is flattened in `flatten_setlist()` by iterating the two levels of nesting explicitly, producing one dict per setlist track. This is fully deterministic and requires no LLM.

### Cost & Speed Optimisation (Pre-Processing)

The deterministic stage attempts four match strategies in ascending cost order:

| Strategy | Example | LLM needed? |
|---|---|---|
| Exact string match | `"Neon Dreams"` = `"Neon Dreams"` | No |
| Normalized match | `"Neon Dreams"` = `"neon dreams"` | No |
| Qualifier-stripped | `"Velocity (Extended Jam)"` → `"velocity"` = `"velocity"` | No |
| Medley split | `"Desert Rain / Ocean Avenue"` → both matched individually | No |

**In the provided dataset, 8 of 10 tracks are resolved deterministically**, meaning only 2 require LLM calls — an 80% LLM cost saving. The UI shows this saving as a percentage bar in the stats strip.

The pre-processing step is intentionally conservative: it only matches when there is a clear, rule-based equivalence. Ambiguous or low-confidence cases are always passed to the LLM rather than risking a false positive.

### Prompt Engineering & Agent Design

The LLM system prompt addresses three key failure modes explicitly:

**Abbreviations/Variations:** The prompt names this scenario directly ("Tokyo" might mean "Midnight in Tokyo") and instructs the model to use judgment about likely intent given the artist context.

**Medley handling:** The prompt instructs the model to return an array of `matches` (not a single match), so one input can produce multiple catalog matches. The response schema enforces this — the code then aggregates multiple catalog IDs with `"; "` separator and uses the lowest-confidence rating of the group.

**False positive prevention (covers):** The prompt includes an explicit rule: *"Never match a cover song to a catalog song with a similar-sounding title."* It also includes a worked example ("Smsls Lk Tn Sprt" is likely "Smells Like Teen Spirit" — not in our catalog). The model is instructed to return `match_confidence: "None"` and not force a match, rather than return a vague similarity.

**Confidence levels are defined precisely:**
- `Exact` — reserved for the deterministic stage only
- `High` — LLM is confident the match is correct
- `Review` — possible match but human verification recommended (e.g., partial medley match, ambiguous abbreviation)
- `None` — not in catalog (cover, uncontrolled, or garbled with no clear match)

### Conflict & Exception Handling

| Scenario | Handling |
|---|---|
| Network timeout / HTTP error | `RuntimeError` caught in UI, shown as error box |
| Invalid JSON in API response | `JSONDecodeError` caught, user-facing error |
| `status != "success"` in API | Explicit check, error raised |
| LLM returns markdown-fenced JSON | Regex strips ` ```json ``` ` before parsing |
| LLM returns unexpected schema | Falls back to `confidence: None` with error note |
| LLM call fails entirely | Error captured in `match_notes`, pipeline continues |
| Catalog file not found | Error raised and shown in UI |
| Track appears in multiple shows | Processed independently per occurrence (correct behaviour) |

### Scalability & Reliability

**Volume**: The current approach makes one LLM call per unresolved track. For large tours (hundreds of shows, thousands of tracks):
- Batch the unresolved tracks into a single LLM call using a list input format to reduce round-trips
- Cache results keyed on `(track_name, catalog_hash)` — the same garbled track name will appear across shows and should not be re-processed
- Run LLM calls concurrently with `asyncio` + the async OpenAI client

**Format shifts**: The JSON flattening logic (`flatten_setlist`) is tied to the current schema. If the API adds nesting or renames fields, `fetch_tour_data` can be extended with a schema validation step (e.g., `pydantic`) that fails early with a clear error rather than silently producing bad data.

**Deterministic expansion**: New qualifier patterns (e.g., "(Studio)", "(Feat. X)") can be added to `_strip_qualifiers()` as a simple regex list extension with no impact on the LLM stage.

---

## Output Format

`matched_setlists.csv` columns:

| Column | Description | Example |
|---|---|---|
| `show_date` | ISO 8601 date | `2024-11-15` |
| `venue_name` | Venue from API | `The Echo Lounge` |
| `setlist_track_name` | Raw track from API | `Tokyo (Acoustic)` |
| `matched_catalog_id` | Our catalog ID, or `None` | `CAT-002` |
| `match_confidence` | `Exact`, `High`, `Review`, or `None` | `High` |
| `match_notes` | Reasoning for the match decision | `Qualifier-stripped match` |

---

## Expected Results (with bundled data)

| Track | Expected Match | Confidence | Method |
|---|---|---|---|
| Neon Dreams | CAT-001 | Exact | Deterministic |
| Tokyo (Acoustic) | CAT-002 | High | LLM → "Tokyo" abbreviation of "Tokyo Midnight" |
| Desert Rain / Ocean Avenue | CAT-003; CAT-012 | High | Medley split |
| Wonderwall | None | None | LLM → uncontrolled cover |
| Shattered Glass | CAT-004 | High/Review | LLM → fuzzy match to "Shatter" |
| Velocity (Extended Jam) | CAT-007 | High | Qualifier-strip → "Velocity" |
| Golden Gate | CAT-006 | Exact | Deterministic |
| Midnight In Tokyo | CAT-013 | Exact | Deterministic (normalized) |
| Smsls Lk Tn Sprt | None | None | LLM → garbled, not in catalog |

---

## Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | UI framework |
| `openai` | OpenAI-compatible client (also works with Groq via `base_url`) |
| `pandas` | DataFrame display |

No additional dependencies — JSON/CSV handling uses Python stdlib only.
