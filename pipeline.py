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

from __future__ import annotations

import csv
import io
import json
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

# Engine version (helpful to confirm you’re on the new pipeline)
ENGINE_VERSION = "v1.1.1"

# ── Match confidence levels ────────────────────────────────────────────────────
CONFIDENCE_EXACT = "Exact"
CONFIDENCE_HIGH = "High"
CONFIDENCE_REVIEW = "Review"
CONFIDENCE_NONE = "None"

OUTPUT_COLUMNS = [
    "show_date",
    "venue_name",
    "setlist_track_name",
    "matched_catalog_id",
    "match_confidence",
    "match_notes",
]


# ── Stage 1: API / Local JSON Ingestion ───────────────────────────────────────


def fetch_tour_data(source: str) -> Dict[str, Any]:
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
            raw = Path(source).read_text(encoding="utf-8")
        except FileNotFoundError:
            raise RuntimeError(f"Local file not found: {source}")
        except Exception as e:
            raise RuntimeError(f"Failed to read local file: {e}")

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON in tour data: {e}")

    if payload.get("status") != "success":
        raise RuntimeError(
            f"Tour API returned non-success status: {payload.get('status')}"
        )

    return payload


# ── Stage 2: Catalog Load ─────────────────────────────────────────────────────


def _parse_warner_style_catalog(raw: str) -> List[Dict[str, str]]:
    """
    Handle the provided sample catalog where each *line* is a quoted,
    comma-separated string, e.g.:

        "catalog_id,title,writers,controlled_percentage"
        "CAT-001,Neon Dreams,Alex Park,100"
    """
    reader = csv.reader(io.StringIO(raw))
    rows = list(reader)
    if not rows:
        return []

    header_cell = rows[0][0].strip().strip('"').lstrip("\ufeff")
    header = [h.strip() for h in header_cell.split(",")]

    records: List[Dict[str, str]] = []
    for r in rows[1:]:
        if not r:
            continue
        cell = r[0].strip().strip('"')
        if not cell:
            continue
        parts = [p.strip() for p in cell.split(",")]
        if len(parts) < len(header):
            parts += [""] * (len(header) - len(parts))
        rec = dict(zip(header, parts[: len(header)]))
        records.append(rec)
    return records


def load_catalog(catalog_file) -> List[Dict[str, str]]:
    """
    Load catalog.csv from:
      - a file path string, or
      - a file-like object (e.g. Streamlit upload)

    Normalises keys so every entry has at least:
      - catalog_id
      - song_title
    """
    try:
        if hasattr(catalog_file, "read"):
            raw = catalog_file.read()
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
        else:
            raw = Path(catalog_file).read_text(encoding="utf-8")
    except Exception as e:
        raise RuntimeError(f"Failed to load catalog: {e}")

    # Normalise BOM + newlines
    raw = raw.lstrip("\ufeff")
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")

    # Decide how to parse:
    #   - If csv.reader sees exactly one column in the first row whose value
    #     contains commas, fall back to the special parser above.
    reader = csv.reader(io.StringIO(raw))
    first_row = next(reader, None)
    if first_row is not None and len(first_row) == 1 and "," in first_row[0]:
        parsed_rows = _parse_warner_style_catalog(raw)
    else:
        parsed_rows = list(csv.DictReader(io.StringIO(raw)))

    normalised: List[Dict[str, str]] = []
    for row in parsed_rows:
        if not row:
            continue
        # Strip whitespace from keys/values
        row = {(k or "").strip(): (v or "").strip() for k, v in row.items()}

        # Canonical keys
        catalog_id = (
            row.get("catalog_id")
            or row.get("CatalogID")
            or row.get("id")
            or row.get("ID")
        )
        title = (
            row.get("song_title")
            or row.get("title")
            or row.get("Title")
            or row.get("name")
        )

        if not catalog_id or not title:
            row["catalog_id"] = catalog_id or ""
            row["song_title"] = title or ""
        else:
            row["catalog_id"] = catalog_id
            row["song_title"] = title

        normalised.append(row)

    return normalised


# ── Stage 3: Flatten Nested JSON ──────────────────────────────────────────────


def flatten_setlist(payload: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Explode nested tour JSON into flat rows:
      { show_date, venue_name, city, setlist_track_name }
    """
    rows: List[Dict[str, str]] = []
    data = payload.get("data", {})
    artist = data.get("artist", "Unknown")
    tour = data.get("tour", "Unknown")

    for show in data.get("shows", []):
        date = show.get("date", "")
        venue = show.get("venue", "")
        city = show.get("city", "")

        for track in show.get("setlist", []):
            rows.append(
                {
                    "show_date": date,
                    "venue_name": venue,
                    "city": city,
                    "artist": artist,
                    "tour": tour,
                    "setlist_track_name": track,
                }
            )

    return rows


# ── Stage 4: Deterministic Pre-Processing ─────────────────────────────────────


def _normalize_str(s: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    s = s.lower()
    s = re.sub(r"[^\w\s]", "", s)  # remove punctuation
    s = re.sub(r"\s+", " ", s).strip()  # collapse whitespace
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
    result = s
    for p in patterns:
        result = re.sub(p, "", result, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", result).strip()


def deterministic_match(
    track_name: str, catalog: List[Dict[str, str]]
) -> Optional[Dict[str, Any]]:
    """
    Try to match a setlist track to catalog entries without LLM.
    Returns the best match dict or None.

    Tries in order:
      1. Exact string match (case-sensitive)
      2. Normalized (lowercase + punctuation stripped)
      3. Qualifier-stripped normalized match
      4. Medley detection: if track contains ' / ', treat each part separately
      5. Conservative substring heuristic for abbreviations (e.g. "Tokyo" -> "Midnight in Tokyo")

    Does NOT attempt true fuzzy matching — that's for the LLM stage.
    """

    # ── 1. Exact (case-sensitive) ─────────────────────────────────────────────
    for entry in catalog:
        title = entry.get("song_title") or ""
        if title == track_name:
            return {
                "matched_catalog_id": entry.get("catalog_id", ""),
                "match_confidence": CONFIDENCE_EXACT,
                "match_notes": "Exact string match",
                "matched_entries": [entry],
            }

    # ── 2. Normalized ─────────────────────────────────────────────────────────
    norm_track = _normalize_str(track_name)
    for entry in catalog:
        title = entry.get("song_title") or ""
        if _normalize_str(title) == norm_track:
            return {
                "matched_catalog_id": entry.get("catalog_id", ""),
                "match_confidence": CONFIDENCE_EXACT,
                "match_notes": "Normalized exact match (case/punctuation diff)",
                "matched_entries": [entry],
            }

    # ── 3. Qualifier-stripped ─────────────────────────────────────────────────
    stripped_track = _strip_qualifiers(track_name)
    norm_stripped_track = _normalize_str(stripped_track)
    for entry in catalog:
        title = entry.get("song_title") or ""
        if _normalize_str(_strip_qualifiers(title)) == norm_stripped_track:
            return {
                "matched_catalog_id": entry.get("catalog_id", ""),
                "match_confidence": CONFIDENCE_HIGH,
                "match_notes": f"Qualifier-stripped match: '{track_name}' → '{title}'",
                "matched_entries": [entry],
            }

    # ── 4. Medley: split on ' / ' and try each part ───────────────────────────
    if " / " in track_name:
        parts = [p.strip() for p in track_name.split(" / ")]
        matched_parts: List[Dict[str, str]] = []
        for part in parts:
            norm_part = _normalize_str(_strip_qualifiers(part))
            for entry in catalog:
                title = entry.get("song_title") or ""
                if _normalize_str(_strip_qualifiers(title)) == norm_part:
                    matched_parts.append(entry)
                    break

        if matched_parts:
            ids = "; ".join(e.get("catalog_id", "") for e in matched_parts)
            titles = "; ".join(e.get("song_title", "") for e in matched_parts)
            return {
                "matched_catalog_id": ids,
                "match_confidence": CONFIDENCE_HIGH
                if len(matched_parts) == len(parts)
                else CONFIDENCE_REVIEW,
                "match_notes": f"Medley: matched {len(matched_parts)}/{len(parts)} parts → {titles}",
                "matched_entries": matched_parts,
                "is_medley": True,
            }

    # ── 5. Conservative substring heuristic (abbreviations) ───────────────────
    # Example: "Tokyo (Acoustic)" → "Midnight in Tokyo"
    base = _normalize_str(_strip_qualifiers(track_name))
    # Only consider reasonably informative tokens (length >= 4)
    if base and len(base) >= 4:
        candidates: List[Dict[str, str]] = []
        for entry in catalog:
            title = entry.get("song_title") or ""
            title_norm = _normalize_str(_strip_qualifiers(title))
            if title_norm == base:
                continue  # would have matched above
            # Abbreviation-like containment either way
            if base in title_norm or title_norm in base:
                candidates.append(entry)

        if len(candidates) == 1:
            e = candidates[0]
            return {
                "matched_catalog_id": e.get("catalog_id", ""),
                "match_confidence": CONFIDENCE_REVIEW,
                "match_notes": (
                    f"Substring heuristic match: '{track_name}' ~ '{e.get('song_title','')}'"
                ),
                "matched_entries": [e],
            }

    return None  # Send to LLM


# ── Stage 5: LLM Fuzzy Matching ───────────────────────────────────────────────

FUZZY_SYSTEM_PROMPT = """You are a music rights reconciliation specialist at a major publisher.
Your job is to determine whether a live performance setlist track matches any song in our internal catalog.

You will receive:
  - setlist_track: the raw track name from a live setlist (may be abbreviated, misspelled, or garbled)
  - catalog: list of our controlled songs with their catalog_id and title

Your analysis must account for:
  ABBREVIATIONS: "Tokyo" might mean "Midnight in Tokyo"
  VARIATIONS: small wording differences
  MEDLEYS: "Song A / Song B" means both songs were performed together; report each match separately
  GARBLED TEXT: "Smsls Lk Tn Sprt" is likely "Smells Like Teen Spirit" (not in our catalog)
  COVERS: A song not in our catalog is a cover or uncontrolled song — do NOT force a match

CRITICAL RULES:
1. Only match if you are genuinely confident. A vague similarity is NOT a match.
2. Never match a cover song to a catalog song with a similar-sounding title.
3. For medleys, return multiple match entries (one per matched catalog song).
4. If nothing in the catalog matches, return match_confidence = "None".
5. "Review" means: there is a possible match but a human should verify.
6. You MUST only use catalog_id values that appear in the provided catalog list.
   If you are unsure, use catalog_id = null and match_confidence = "None" or "Review".

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


def llm_fuzzy_match(
    track_name: str, catalog: List[Dict[str, str]], client, model: str
) -> Dict[str, Any]:
    """
    Use LLM to attempt fuzzy matching of a track to the catalog.
    Returns a result dict.
    """
    catalog_summary = [
        {"catalog_id": e.get("catalog_id", ""), "song_title": e.get("song_title", "")}
        for e in catalog
        if e.get("catalog_id") and e.get("song_title")
    ]
    valid_ids = {e["catalog_id"] for e in catalog_summary}

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
            "match_notes": "LLM returned no matches",
        }

    # Sanitize and filter matches:
    #   - only keep catalog_ids that actually exist in the catalog
    #   - normalise confidence labels
    real_matches: List[Dict[str, Any]] = []
    for m in matches:
        cid = m.get("catalog_id")
        if not cid:
            continue
        cid = str(cid).strip()
        if cid not in valid_ids:
            # Ignore hallucinated IDs
            continue

        conf = (m.get("match_confidence") or "").strip().title()
        if conf not in {
            CONFIDENCE_EXACT,
            CONFIDENCE_HIGH,
            CONFIDENCE_REVIEW,
            CONFIDENCE_NONE,
        }:
            conf = CONFIDENCE_REVIEW

        if conf == CONFIDENCE_NONE:
            continue

        real_matches.append(
            {
                "catalog_id": cid,
                "song_title": m.get("song_title", ""),
                "match_confidence": conf,
                "reasoning": m.get("reasoning", ""),
            }
        )

    if not real_matches:
        # All returned None or invalid; surface the first reasoning if available
        first_reason = (
            matches[0].get("reasoning", "No match found") if matches else "No match found"
        )
        return {
            "matched_catalog_id": None,
            "match_confidence": CONFIDENCE_NONE,
            "match_notes": first_reason,
        }

    if len(real_matches) == 1:
        m = real_matches[0]
        return {
            "matched_catalog_id": m["catalog_id"],
            "match_confidence": m["match_confidence"],
            "match_notes": m.get("reasoning", ""),
        }

    # Multiple matches (medley-style)
    ids = "; ".join(m["catalog_id"] for m in real_matches)
    titles = "; ".join(m.get("song_title", "") for m in real_matches)
    confidences = [m.get("match_confidence", CONFIDENCE_REVIEW) for m in real_matches]

    # Lowest confidence wins for the group (Exact > High > Review > None)
    order = {
        CONFIDENCE_EXACT: 0,
        CONFIDENCE_HIGH: 1,
        CONFIDENCE_REVIEW: 2,
        CONFIDENCE_NONE: 3,
    }
    group_conf = min(confidences, key=lambda c: order.get(c, 2))
    reasonings = " | ".join(m.get("reasoning", "") for m in real_matches)

    return {
        "matched_catalog_id": ids,
        "match_confidence": group_conf,
        "match_notes": f"Medley (LLM): {titles} | {reasonings}",
    }


# ── Main Pipeline Orchestrator ────────────────────────────────────────────────


def run_pipeline(
    tour_source: str,
    catalog_file,
    client,
    model: str,
    progress_callback=None,
) -> Dict[str, Any]:
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

    def progress(step: int, total: int, msg: str) -> None:
        if progress_callback:
            progress_callback(step, total, msg)

    errors: List[str] = []
    result: Dict[str, Any] = {
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
            "engine_version": ENGINE_VERSION,
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
    progress(
        4,
        total_steps,
        "Running deterministic matching (exact + qualifier-strip + medley + abbreviations)…",
    )

    pre_matched: List[Dict[str, Any]] = []
    needs_llm: List[Dict[str, Any]] = []

    for row in flat_rows:
        track = row["setlist_track_name"]
        match = deterministic_match(track, catalog)
        if match:
            pre_matched.append({**row, **match})
        else:
            needs_llm.append(row)

    # ── Step 5: LLM fuzzy matching ────────────────────────────────────────────
    llm_results: List[Dict[str, Any]] = []
    if needs_llm:
        for i, row in enumerate(needs_llm):
            track = row["setlist_track_name"]
            progress(
                5,
                total_steps,
                f'AI fuzzy matching {i + 1}/{len(needs_llm)}: "{track}"…',
            )
            match = llm_fuzzy_match(track, catalog, client, model)
            llm_results.append({**row, **match})
    else:
        progress(
            5,
            total_steps,
            "No tracks needed LLM matching — all resolved deterministically.",
        )

    # ── Merge + build final output ─────────────────────────────────────────────
    all_results: List[Dict[str, Any]] = []
    for row in pre_matched + llm_results:
        output_row = {
            "show_date": row.get("show_date", ""),
            "venue_name": row.get("venue_name", ""),
            "setlist_track_name": row.get("setlist_track_name", ""),
            "matched_catalog_id": row.get("matched_catalog_id") or "None",
            "match_confidence": row.get("match_confidence", CONFIDENCE_NONE),
            "match_notes": row.get("match_notes", ""),
        }
        all_results.append(output_row)

    # Sort by show_date, venue, then original setlist order
    all_results.sort(key=lambda r: (r["show_date"], r["venue_name"]))

    result["results"] = all_results

    # ── Stats ──────────────────────────────────────────────────────────────────
    total_tracks = len(all_results)
    n_exact = sum(1 for r in all_results if r["match_confidence"] == CONFIDENCE_EXACT)
    n_high = sum(1 for r in all_results if r["match_confidence"] == CONFIDENCE_HIGH)
    n_review = sum(1 for r in all_results if r["match_confidence"] == CONFIDENCE_REVIEW)
    n_none = sum(1 for r in all_results if r["match_confidence"] == CONFIDENCE_NONE)
    n_llm = len(llm_results)
    n_pre = len(pre_matched)

    result["stats"] = {
        "total_tracks": total_tracks,
        "exact_matches": n_exact,
        "high_matches": n_high,
        "review_matches": n_review,
        "no_matches": n_none,
        "deterministic": n_pre,
        "llm_resolved": n_llm,
        "llm_savings_pct": round((n_pre / total_tracks * 100) if total_tracks else 0, 1),
        "engine_version": ENGINE_VERSION,
    }

    return result


# ── CSV Builder ────────────────────────────────────────────────────────────────


def build_output_csv(results: List[Dict[str, Any]]) -> str:
    """Build the output CSV string from result rows."""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(results)
    return buf.getvalue()
