"""
pipeline.py  —  Setlist API Reconciliation Pipeline
Warner Chappell Music · Task 3

Stages:
  1. API Ingestion      — fetch tour JSON from URL or local file
  2. Catalog Load       — read internal catalog.csv
  3. Flatten Setlist    — explode nested JSON into rows
  4. Deterministic Pre-Processing — exact / normalized string matches (no LLM cost)
  5. Agentic Fuzzy Matching       — LLM for remaining unresolved tracks
  6. Output CSV                   — matched_setlists.csv
"""

import re
import json
import csv
import io
import urllib.request
import urllib.error
from typing import Optional
from pathlib import Path


# ── Match confidence levels ────────────────────────────────────────────────────
CONFIDENCE_EXACT   = "Exact"
CONFIDENCE_HIGH    = "High"
CONFIDENCE_REVIEW  = "Review"
CONFIDENCE_NONE    = "None"

OUTPUT_COLUMNS = [
    "show_date",
    "venue_name",
    "setlist_track_name",
    "matched_catalog_id",
    "match_confidence",
    "match_notes",
]


# ── Stage 1: API / Local JSON Ingestion ───────────────────────────────────────

def fetch_tour_data(source: str) -> dict:
    """
    Fetch tour JSON from a URL or local file path.
    Returns the parsed dict.
    Raises RuntimeError on network/parse failures.
    """
    source = source.strip()

    # Detect URL vs local file
    if source.startswith("http://") or source.startswith("https://"):
        try:
            req = urllib.request.Request(
                source,
                headers={"User-Agent": "WCM-Reconciliation-Agent/1.0"},
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"HTTP {e.code} fetching tour data: {e.reason}")
        except urllib.error.URLError as e:
            raise RuntimeError(f"Network error fetching tour data: {e.reason}")
        except Exception as e:
            raise RuntimeError(f"Failed to fetch tour data: {e}")
    else:
        # Local file
        try:
            with open(source, "r", encoding="utf-8") as f:
                raw = f.read()
        except FileNotFoundError:
            raise RuntimeError(f"Local file not found: {source}")
        except Exception as e:
            raise RuntimeError(f"Failed to read local file: {e}")

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON in tour data: {e}")

    if payload.get("status") != "success":
        raise RuntimeError(f"Tour API returned non-success status: {payload.get('status')}")

    return payload


# ── Stage 2: Catalog Load ──────────────────────────────────────────────────────

def load_catalog(catalog_file) -> list[dict]:
    """
    Load catalog.csv from a file path or file-like object.
    Returns list of dicts.
    """
    try:
        if hasattr(catalog_file, "read"):
            content = catalog_file.read()
            if isinstance(content, bytes):
                content = content.decode("utf-8")
            reader = csv.DictReader(io.StringIO(content))
        else:
            with open(catalog_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
            # Re-open since we closed it
            with open(catalog_file, "r", encoding="utf-8") as f:
                reader = list(csv.DictReader(f))
            return reader
        return list(reader)
    except Exception as e:
        raise RuntimeError(f"Failed to load catalog: {e}")


# ── Stage 3: Flatten Nested JSON ──────────────────────────────────────────────

def flatten_setlist(payload: dict) -> list[dict]:
    """
    Explode nested tour JSON into flat rows:
      { show_date, venue_name, city, setlist_track_name }
    """
    rows = []
    data = payload.get("data", {})
    artist = data.get("artist", "Unknown")
    tour = data.get("tour", "Unknown")

    for show in data.get("shows", []):
        date = show.get("date", "")
        venue = show.get("venue", "")
        city = show.get("city", "")

        for track in show.get("setlist", []):
            rows.append({
                "show_date": date,
                "venue_name": venue,
                "city": city,
                "artist": artist,
                "tour": tour,
                "setlist_track_name": track,
            })

    return rows


# ── Stage 4: Deterministic Pre-Processing ─────────────────────────────────────

def _normalize_str(s: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    s = s.lower()
    s = re.sub(r"[^\w\s]", "", s)          # remove punctuation
    s = re.sub(r"\s+", " ", s).strip()     # collapse whitespace
    return s


def _strip_qualifiers(s: str) -> str:
    """
    Remove common live-performance suffixes/prefixes that shouldn't affect matching:
      (Acoustic), (Extended Jam), (Live), (Radio Edit), (Remix), etc.
    """
    patterns = [
        r"\(acoustic\)",
        r"\(extended.*?\)",
        r"\(live.*?\)",
        r"\(radio.*?\)",
        r"\(remix.*?\)",
        r"\(feat.*?\)",
        r"\(ft\..*?\)",
        r"\(interlude\)",
        r"\(reprise\)",
    ]
    result = s.lower()
    for p in patterns:
        result = re.sub(p, "", result, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", result).strip()


def deterministic_match(track_name: str, catalog: list[dict]) -> Optional[dict]:
    """
    Try to match a setlist track to catalog entries without LLM.
    Returns the best match dict or None.

    Tries in order:
      1. Exact string match (case-sensitive)
      2. Normalized (lowercase + punctuation stripped)
      3. Qualifier-stripped normalized match
      4. Medley detection: if track contains ' / ', treat each part separately (exact/norm)

    Does NOT attempt fuzzy matching — that's for the LLM stage.
    """

    # ── 1. Exact ──────────────────────────────────────────────────────────────
    for entry in catalog:
        if entry["song_title"] == track_name:
            return {
                "matched_catalog_id": entry["catalog_id"],
                "match_confidence": CONFIDENCE_EXACT,
                "match_notes": "Exact string match",
                "matched_entries": [entry],
            }

    # ── 2. Normalized ─────────────────────────────────────────────────────────
    norm_track = _normalize_str(track_name)
    for entry in catalog:
        if _normalize_str(entry["song_title"]) == norm_track:
            return {
                "matched_catalog_id": entry["catalog_id"],
                "match_confidence": CONFIDENCE_EXACT,
                "match_notes": "Normalized exact match (case/punctuation diff)",
                "matched_entries": [entry],
            }

    # ── 3. Qualifier-stripped ─────────────────────────────────────────────────
    stripped_track = _strip_qualifiers(track_name)
    for entry in catalog:
        if _normalize_str(_strip_qualifiers(entry["song_title"])) == _normalize_str(stripped_track):
            return {
                "matched_catalog_id": entry["catalog_id"],
                "match_confidence": CONFIDENCE_HIGH,
                "match_notes": f"Qualifier-stripped match: '{track_name}' → '{entry['song_title']}'",
                "matched_entries": [entry],
            }

    # ── 4. Medley: split on ' / ' and try each part ───────────────────────────
    if " / " in track_name:
        parts = [p.strip() for p in track_name.split(" / ")]
        matched_parts = []
        for part in parts:
            norm_part = _normalize_str(_strip_qualifiers(part))
            for entry in catalog:
                if _normalize_str(_strip_qualifiers(entry["song_title"])) == norm_part:
                    matched_parts.append(entry)
                    break

        if matched_parts:
            ids = "; ".join(e["catalog_id"] for e in matched_parts)
            titles = "; ".join(e["song_title"] for e in matched_parts)
            return {
                "matched_catalog_id": ids,
                "match_confidence": CONFIDENCE_HIGH if len(matched_parts) == len(parts) else CONFIDENCE_REVIEW,
                "match_notes": f"Medley: matched {len(matched_parts)}/{len(parts)} parts → {titles}",
                "matched_entries": matched_parts,
                "is_medley": True,
            }

    return None  # Send to LLM


# ── Stage 5: LLM Fuzzy Matching ───────────────────────────────────────────────

FUZZY_SYSTEM_PROMPT = """You are a music rights reconciliation specialist at a major publisher.
Your job is to determine whether a live performance setlist track matches any song in our internal catalog.

You will receive:
  - setlist_track: the raw track name from a live setlist (may be abbreviated, misspelled, or garbled)
  - catalog: list of our controlled songs with their catalog_id and title

Your analysis must account for:
  ABBREVIATIONS: "Tokyo" might mean "Midnight in Tokyo" or "Tokyo Midnight"
  VARIATIONS: "Shattered Glass" might mean "Shatter"
  MEDLEYS: "Song A / Song B" means both songs were performed together; report each match separately
  GARBLED TEXT: "Smsls Lk Tn Sprt" is likely "Smells Like Teen Spirit" (not in our catalog)
  COVERS: A song not in our catalog is a cover or uncontrolled song — do NOT force a match

CRITICAL RULES:
1. Only match if you are genuinely confident. A vague similarity is NOT a match.
2. Never match a cover song to a catalog song with a similar-sounding title.
3. For medleys, return multiple match entries (one per matched catalog song).
4. If nothing in the catalog matches, return match_confidence = "None".
5. "Review" means: there is a possible match but a human should verify.

Return a JSON object with this exact structure:
{
  "matches": [
    {
      "catalog_id": "CAT-XXX or null",
      "song_title": "matched catalog title or null",
      "match_confidence": "Exact | High | Review | None",
      "reasoning": "brief explanation"
    }
  ]
}

Return ONLY the JSON object. No markdown fences, no extra text."""


def llm_fuzzy_match(track_name: str, catalog: list[dict], client, model: str) -> dict:
    """
    Use LLM to attempt fuzzy matching of a track to the catalog.
    Returns a result dict.
    """
    catalog_summary = [
        {"catalog_id": e["catalog_id"], "song_title": e["song_title"]}
        for e in catalog
    ]

    user_prompt = f"""setlist_track: "{track_name}"

catalog:
{json.dumps(catalog_summary, indent=2)}

Analyze this track and return your JSON result."""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": FUZZY_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=512,
        )
        raw = response.choices[0].message.content.strip()
    except Exception as e:
        return {
            "matched_catalog_id": None,
            "match_confidence": CONFIDENCE_NONE,
            "match_notes": f"LLM call failed: {e}",
        }

    # Parse JSON response
    try:
        cleaned = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
        cleaned = re.sub(r"\s*```$", "", cleaned.strip(), flags=re.MULTILINE).strip()
        parsed = json.loads(cleaned)
        matches = parsed.get("matches", [])
    except Exception as e:
        return {
            "matched_catalog_id": None,
            "match_confidence": CONFIDENCE_NONE,
            "match_notes": f"LLM response parse error: {e} | raw: {raw[:200]}",
        }

    if not matches:
        return {
            "matched_catalog_id": None,
            "match_confidence": CONFIDENCE_NONE,
            "match_notes": "LLM found no matches",
        }

    # Handle medleys (multiple matches returned)
    real_matches = [m for m in matches if m.get("catalog_id") and m.get("match_confidence") != CONFIDENCE_NONE]

    if not real_matches:
        # All returned None
        reasoning = matches[0].get("reasoning", "No match found") if matches else "No match found"
        return {
            "matched_catalog_id": None,
            "match_confidence": CONFIDENCE_NONE,
            "match_notes": reasoning,
        }

    if len(real_matches) == 1:
        m = real_matches[0]
        return {
            "matched_catalog_id": m.get("catalog_id"),
            "match_confidence": m.get("match_confidence", CONFIDENCE_REVIEW),
            "match_notes": m.get("reasoning", ""),
        }

    # Multiple matches (medley)
    ids = "; ".join(m["catalog_id"] for m in real_matches)
    titles = "; ".join(m.get("song_title", "") for m in real_matches)
    confidences = [m.get("match_confidence", CONFIDENCE_REVIEW) for m in real_matches]
    # Lowest confidence wins for the group
    conf_order = [CONFIDENCE_EXACT, CONFIDENCE_HIGH, CONFIDENCE_REVIEW, CONFIDENCE_NONE]
    group_conf = max(confidences, key=lambda c: conf_order.index(c) if c in conf_order else 3)
    reasonings = " | ".join(m.get("reasoning", "") for m in real_matches)

    return {
        "matched_catalog_id": ids,
        "match_confidence": group_conf,
        "match_notes": f"Medley: {titles} | {reasonings}",
    }


# ── Main Pipeline Orchestrator ─────────────────────────────────────────────────

def run_pipeline(
    tour_source: str,
    catalog_file,
    client,
    model: str,
    progress_callback=None,
) -> dict:
    """
    Full reconciliation pipeline.

    Args:
        tour_source:       URL string or local file path to tour JSON
        catalog_file:      Path or file-like object for catalog.csv
        client:            OpenAI-compatible LLM client
        model:             Model name string
        progress_callback: Optional callable(step, total, message)

    Returns dict with keys:
        tour_meta, catalog, flat_rows, results, stats, errors
    """

    def progress(step, total, msg):
        if progress_callback:
            progress_callback(step, total, msg)

    errors = []
    result = {
        "tour_meta": {},
        "catalog": [],
        "flat_rows": [],
        "results": [],
        "stats": {},
        "errors": errors,
    }

    total_steps = 5

    # ── Step 1: Fetch tour data ────────────────────────────────────────────────
    progress(1, total_steps, "Fetching tour data from API / local file…")
    try:
        payload = fetch_tour_data(tour_source)
        data = payload.get("data", {})
        result["tour_meta"] = {
            "artist": data.get("artist"),
            "tour": data.get("tour"),
            "show_count": len(data.get("shows", [])),
        }
    except Exception as e:
        errors.append(f"Tour data fetch failed: {e}")
        return result

    # ── Step 2: Load catalog ───────────────────────────────────────────────────
    progress(2, total_steps, "Loading internal song catalog…")
    try:
        catalog = load_catalog(catalog_file)
        result["catalog"] = catalog
    except Exception as e:
        errors.append(f"Catalog load failed: {e}")
        return result

    # ── Step 3: Flatten setlist ────────────────────────────────────────────────
    progress(3, total_steps, "Flattening nested tour JSON into rows…")
    flat_rows = flatten_setlist(payload)
    result["flat_rows"] = flat_rows

    # ── Step 4: Deterministic pre-processing ──────────────────────────────────
    progress(4, total_steps, "Running deterministic matching (exact + qualifier-strip + medley)…")

    pre_matched = []
    needs_llm = []

    for row in flat_rows:
        track = row["setlist_track_name"]
        match = deterministic_match(track, catalog)
        if match:
            pre_matched.append({**row, **match})
        else:
            needs_llm.append(row)

    # ── Step 5: LLM fuzzy matching ────────────────────────────────────────────
    llm_results = []
    if needs_llm:
        for i, row in enumerate(needs_llm):
            track = row["setlist_track_name"]
            progress(
                5, total_steps,
                f"AI fuzzy matching {i+1}/{len(needs_llm)}: \"{track}\"…"
            )
            match = llm_fuzzy_match(track, catalog, client, model)
            llm_results.append({**row, **match})
    else:
        progress(5, total_steps, "No tracks needed LLM matching — all resolved deterministically.")

    # ── Merge + build final output ─────────────────────────────────────────────
    all_results = []
    for row in pre_matched + llm_results:
        output_row = {
            "show_date":          row.get("show_date", ""),
            "venue_name":         row.get("venue_name", ""),
            "setlist_track_name": row.get("setlist_track_name", ""),
            "matched_catalog_id": row.get("matched_catalog_id") or "None",
            "match_confidence":   row.get("match_confidence", CONFIDENCE_NONE),
            "match_notes":        row.get("match_notes", ""),
        }
        all_results.append(output_row)

    # Sort by show_date, venue, then original setlist order
    all_results.sort(key=lambda r: (r["show_date"], r["venue_name"]))

    result["results"] = all_results

    # ── Stats ──────────────────────────────────────────────────────────────────
    total_tracks = len(all_results)
    n_exact   = sum(1 for r in all_results if r["match_confidence"] == CONFIDENCE_EXACT)
    n_high    = sum(1 for r in all_results if r["match_confidence"] == CONFIDENCE_HIGH)
    n_review  = sum(1 for r in all_results if r["match_confidence"] == CONFIDENCE_REVIEW)
    n_none    = sum(1 for r in all_results if r["match_confidence"] == CONFIDENCE_NONE)
    n_llm     = len(llm_results)
    n_pre     = len(pre_matched)

    result["stats"] = {
        "total_tracks":   total_tracks,
        "exact_matches":  n_exact,
        "high_matches":   n_high,
        "review_matches": n_review,
        "no_matches":     n_none,
        "deterministic":  n_pre,
        "llm_resolved":   n_llm,
        "llm_savings_pct": round((n_pre / total_tracks * 100) if total_tracks else 0, 1),
    }

    return result


# ── CSV Builder ────────────────────────────────────────────────────────────────

def build_output_csv(results: list[dict]) -> str:
    """Build the output CSV string from result rows."""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(results)
    return buf.getvalue()
