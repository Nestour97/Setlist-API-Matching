"""
pipeline.py  —  Setlist API Reconciliation Pipeline v3
Warner Chappell Music · Task 3

Root-cause fixes in this version:

  FIX 1  Catalog column flexibility
         Real catalog uses 'title' not 'song_title', and 'controlled_percentage'
         not 'controlled'. A helper get_title() checks both column names so the
         pipeline works with any catalog CSV column naming convention.

  FIX 2  Correct bundled catalog
         catalog.csv now matches the real assessment file exactly:
           CAT-001=Neon Dreams, CAT-002=Midnight in Tokyo,
           CAT-003=Shattered Glass, CAT-004=Desert Rain, CAT-005=Ocean Avenue,
           CAT-006=Golden Gate, CAT-007=Velocity, CAT-013=The Glass House, etc.

  FIX 3  Deterministic matching now works correctly
         Because titles are read correctly, exact and normalised matching fire
         before any heuristics:
           "Shattered Glass"  → CAT-003 Exact  (no LLM)
           "Midnight In Tokyo"→ CAT-002 Exact  (case-normalised)
           "Golden Gate"      → CAT-006 Exact
           "Desert Rain / Ocean Avenue" → CAT-004; CAT-005 High  (medley)
           "Velocity (Extended Jam)"    → CAT-007 High  (qualifier-stripped)

  FIX 4  LLM prompt anti-medley guard
         The prompt now explicitly states: "Only treat a track as a medley if it
         contains ' / '. A qualifier like '(Acoustic)' or '(Extended)' does NOT
         make a medley — it is still one song."
         This prevents Tokyo (Acoustic) being returned as two matches.

  FIX 5  LLM catalog grounding + post-validation
         Every catalog_id the LLM returns is validated against the loaded catalog.
         Any ID not in the catalog is silently dropped, preventing hallucinated IDs.
"""

from __future__ import annotations

import re
import json
import csv
import io
import urllib.request
import urllib.error
from typing import Optional

# ── Confidence levels ──────────────────────────────────────────────────────────
EXACT  = "Exact"
HIGH   = "High"
REVIEW = "Review"
NONE   = "None"

OUTPUT_COLUMNS = [
    "show_date", "venue_name", "setlist_track_name",
    "matched_catalog_id", "match_confidence", "match_notes",
]

_QUALIFIER_PATTERNS = [
    r"\s*\(acoustic[^)]*\)",
    r"\s*\(extended[^)]*\)",
    r"\s*\(live[^)]*\)",
    r"\s*\(radio[^)]*\)",
    r"\s*\(remix[^)]*\)",
    r"\s*\(feat[^)]*\)",
    r"\s*\(ft\.[^)]*\)",
    r"\s*\(interlude\)",
    r"\s*\(reprise\)",
    r"\s*\(version[^)]*\)",
    r"\s*\(remaster[^)]*\)",
    r"\s*-\s*(acoustic|live|remix|radio edit)$",
]


# ══════════════════════════════════════════════════════════════════════════════
# Catalog helpers — flexible column detection
# ══════════════════════════════════════════════════════════════════════════════

def get_title(entry: dict) -> str:
    """
    Return the song title from a catalog row, regardless of column name.
    Checks 'title', 'song_title', 'name', 'track_title' in that order.
    """
    for col in ("title", "song_title", "name", "track_title"):
        v = entry.get(col, "").strip()
        if v:
            return v
    return ""


# ══════════════════════════════════════════════════════════════════════════════
# Stage 1  API / local file ingestion
# ══════════════════════════════════════════════════════════════════════════════

def fetch_tour_data(source: str) -> dict:
    """Fetch tour JSON from URL or local file path."""
    source = source.strip()
    if source.startswith("http://") or source.startswith("https://"):
        try:
            req = urllib.request.Request(
                source, headers={"User-Agent": "WCM-Reconciliation/3.0"}
            )
            with urllib.request.urlopen(req, timeout=15) as r:
                raw = r.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"HTTP {e.code}: {e.reason}")
        except urllib.error.URLError as e:
            raise RuntimeError(f"Network error: {e.reason}")
        except Exception as e:
            raise RuntimeError(f"Request failed: {e}")
    else:
        try:
            with open(source, encoding="utf-8") as f:
                raw = f.read()
        except FileNotFoundError:
            raise RuntimeError(f"File not found: {source}")
        except Exception as e:
            raise RuntimeError(f"File read failed: {e}")

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON: {e}")

    if payload.get("status") != "success":
        raise RuntimeError(f"API returned status: {payload.get('status')!r}")
    return payload


# ══════════════════════════════════════════════════════════════════════════════
# Stage 2  Catalog load
# ══════════════════════════════════════════════════════════════════════════════

def load_catalog(catalog_file) -> list[dict]:
    """Load catalog CSV from path or file-like object."""
    try:
        if hasattr(catalog_file, "read"):
            content = catalog_file.read()
            if isinstance(content, bytes):
                content = content.decode("utf-8")
            catalog_file.seek(0)
            return list(csv.DictReader(io.StringIO(content)))
        with open(catalog_file, encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except Exception as e:
        raise RuntimeError(f"Catalog load failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# Stage 3  Flatten nested JSON
# ══════════════════════════════════════════════════════════════════════════════

def flatten_setlist(payload: dict) -> list[dict]:
    """Explode shows[].setlist[] into one dict per track, preserving order."""
    rows = []
    data = payload.get("data", {})
    for show in data.get("shows", []):
        for track in show.get("setlist", []):
            rows.append({
                "show_date":          show.get("date", ""),
                "venue_name":         show.get("venue", ""),
                "city":               show.get("city", ""),
                "setlist_track_name": track,
            })
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# Stage 4  Deterministic pre-processing
# ══════════════════════════════════════════════════════════════════════════════

def _norm(s: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    return re.sub(r"\s+", " ",
                  re.sub(r"[^\w\s]", "", s.lower())).strip()


def _strip_qualifiers(s: str) -> str:
    """Remove live-performance qualifiers from a track name."""
    result = s
    for pat in _QUALIFIER_PATTERNS:
        result = re.sub(pat, "", result, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", result).strip()


def _match_single(track: str, catalog: list[dict]) -> Optional[dict]:
    """
    Attempt deterministic strategies for a single (non-medley) track.
    Returns a result dict or None (→ send to LLM).

    Strategy order:
      1. Exact string match
      2. Normalised exact (case + punctuation insensitive)
      3. Qualifier-stripped normalised exact
      (No substring heuristics — those produce too many false positives.
       Ambiguous cases go to the LLM instead.)
    """
    norm_track    = _norm(track)
    stripped      = _strip_qualifiers(track)
    norm_stripped = _norm(stripped)

    for entry in catalog:
        title     = get_title(entry)
        norm_cat  = _norm(title)
        stripped_cat = _norm(_strip_qualifiers(title))

        # 1. Exact
        if title == track:
            return _make_result(entry, EXACT, "Exact string match")

        # 2. Normalised exact (handles case differences, punctuation)
        if norm_cat == norm_track and norm_track:
            return _make_result(
                entry, EXACT,
                f"Normalised exact match (case/punctuation difference)"
            )

        # 3. Qualifier-stripped normalised exact
        if stripped_cat == norm_stripped and norm_stripped:
            return _make_result(
                entry, HIGH,
                f"Qualifier-stripped match: '{track}' → '{title}'"
            )

    return None  # Nothing found deterministically → send to LLM


def _make_result(entry: dict, confidence: str, notes: str) -> dict:
    return {
        "matched_catalog_id": entry["catalog_id"],
        "match_confidence":   confidence,
        "match_notes":        notes,
        "matched_entries":    [entry],
    }


def deterministic_match(track: str, catalog: list[dict]) -> Optional[dict]:
    """
    Full deterministic match including medley splitting.
    Returns a result dict or None.
    """
    # Medley: only split on explicit " / " separator
    if " / " in track:
        parts   = [p.strip() for p in track.split(" / ")]
        matched: list[dict] = []
        unmatched: list[str] = []

        for part in parts:
            m = _match_single(part, catalog)
            if m:
                matched.append(m["matched_entries"][0])
            else:
                unmatched.append(part)

        if matched:
            ids   = "; ".join(e["catalog_id"]   for e in matched)
            names = "; ".join(get_title(e) for e in matched)
            conf  = HIGH if not unmatched else REVIEW
            note  = (
                f"Medley: matched {len(matched)}/{len(parts)} parts → {names}"
                + (f" | Unmatched: {', '.join(unmatched)}" if unmatched else "")
            )
            return {
                "matched_catalog_id": ids,
                "match_confidence":   conf,
                "match_notes":        note,
                "matched_entries":    matched,
                "is_medley":          True,
            }
        return None  # All medley parts unmatched → LLM

    return _match_single(track, catalog)


# ══════════════════════════════════════════════════════════════════════════════
# Stage 5  LLM fuzzy matching
# ══════════════════════════════════════════════════════════════════════════════

_LLM_SYSTEM_PROMPT = """You are a music rights reconciliation specialist.
Determine whether a live setlist track matches any song in the provided internal catalog.

You receive:
  setlist_track: the raw track name from a live performance setlist
  catalog: our complete list of controlled songs (catalog_id, title)

MATCHING RULES:
  ABBREVIATIONS : "Tokyo" by itself may be a shortened reference to a catalog title
                  containing "Tokyo", e.g. "Midnight in Tokyo"
  QUALIFIERS    : "(Acoustic)", "(Extended Jam)" etc. do NOT change the underlying
                  song — strip them and match the core title
  COVERS        : Songs not in this catalog (e.g. "Wonderwall") must NOT be matched.
                  Return confidence "None"
  GARBLED TEXT  : "Smsls Lk Tn Sprt" style → decode if confident it is a catalog song,
                  otherwise return "None"

MEDLEY RULE — VERY IMPORTANT:
  A setlist entry is a medley ONLY if it contains " / " (slash with spaces),
  e.g. "Desert Rain / Ocean Avenue".
  A track like "Tokyo (Acoustic)" is ONE song with a qualifier — NOT a medley.
  Never return more than one match for a track that does not contain " / ".

CONFIDENCE LEVELS:
  Exact  — title matches exactly (after normalisation/qualifier stripping)
  High   — very confident match (e.g. clear abbreviation or qualifier variation)
  Review — possible match but human verification recommended
  None   — not in catalog

CRITICAL RULES:
  1. catalog_id must be EXACTLY as shown — copy it character-for-character.
  2. Only use catalog_ids from the provided list. Never invent IDs.
  3. match_notes must only reference titles that exist in the provided catalog.
  4. For a non-medley track, return exactly ONE match in the "matches" array.

Return a JSON object:
{
  "matches": [
    {
      "catalog_id": "CAT-XXX",
      "catalog_title": "exact title from catalog",
      "match_confidence": "Exact | High | Review | None",
      "reasoning": "brief explanation using only catalog titles"
    }
  ]
}

For no match: one entry with catalog_id=null, match_confidence="None".
Return ONLY the JSON object. No markdown fences, no extra text."""


def llm_fuzzy_match(
    track: str, catalog: list[dict], client, model: str
) -> dict:
    """LLM fuzzy match for one unresolved track."""
    # Build catalog summary for the prompt using the correct title column
    catalog_list = [
        {"catalog_id": e["catalog_id"], "title": get_title(e)}
        for e in catalog
        if get_title(e)
    ]
    valid_ids = {e["catalog_id"] for e in catalog}

    user_prompt = (
        f'setlist_track: "{track}"\n\n'
        f"catalog:\n{json.dumps(catalog_list, indent=2)}\n\n"
        f"Analyze this setlist track. "
        f"Is it a medley (contains ' / ')? No. So return exactly ONE match. "
        f"Only use catalog_id values from the list above."
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _LLM_SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=512,
        )
        raw = resp.choices[0].message.content.strip()
    except Exception as e:
        return {
            "matched_catalog_id": None,
            "match_confidence":   NONE,
            "match_notes":        f"LLM call failed: {e}",
        }

    # Parse
    try:
        cleaned = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
        cleaned = re.sub(r"\s*```$", "", cleaned.strip(), flags=re.MULTILINE).strip()
        parsed  = json.loads(cleaned)
        matches = parsed.get("matches", [])
    except Exception as e:
        return {
            "matched_catalog_id": None,
            "match_confidence":   NONE,
            "match_notes":        f"LLM parse error: {e} | raw: {raw[:200]}",
        }

    if not matches:
        return {
            "matched_catalog_id": None,
            "match_confidence":   NONE,
            "match_notes":        "LLM returned no matches",
        }

    # Validate: only accept IDs that actually exist in our catalog
    valid_matches = [
        m for m in matches
        if m.get("catalog_id") in valid_ids
        and m.get("match_confidence", NONE) != NONE
    ]

    if not valid_matches:
        reasoning = (matches[0].get("reasoning") or "No catalog match found") if matches else "No catalog match"
        return {
            "matched_catalog_id": None,
            "match_confidence":   NONE,
            "match_notes":        reasoning,
        }

    # Take the single best match (LLM is told not to return multiples for non-medleys)
    # If it returns multiple anyway, take the highest-confidence one
    conf_order = {EXACT: 0, HIGH: 1, REVIEW: 2, NONE: 3}
    best = sorted(valid_matches,
                  key=lambda m: conf_order.get(m.get("match_confidence", NONE), 3))[0]

    return {
        "matched_catalog_id": best["catalog_id"],
        "match_confidence":   best.get("match_confidence", REVIEW),
        "match_notes":        best.get("reasoning", ""),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Orchestrator
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(
    tour_source:       str,
    catalog_file,
    client,
    model:             str,
    progress_callback=None,
) -> dict:
    TOTAL  = 5
    errors: list[str] = []
    result = dict(
        tour_meta={}, catalog=[], flat_rows=[],
        results=[], stats={}, errors=errors,
    )

    def step(n, msg):
        if progress_callback:
            progress_callback(n, TOTAL, msg)

    # Stage 1
    step(1, "Fetching tour data…")
    try:
        payload = fetch_tour_data(tour_source)
        d = payload.get("data", {})
        result["tour_meta"] = {
            "artist":     d.get("artist"),
            "tour":       d.get("tour"),
            "show_count": len(d.get("shows", [])),
        }
    except Exception as e:
        errors.append(str(e))
        return result

    # Stage 2
    step(2, "Loading catalog…")
    try:
        catalog = load_catalog(catalog_file)
        result["catalog"] = catalog
    except Exception as e:
        errors.append(str(e))
        return result

    # Stage 3
    step(3, "Flattening setlist JSON…")
    flat = flatten_setlist(payload)
    result["flat_rows"] = flat

    # Stage 4 — Deterministic
    step(4, "Deterministic matching (exact → normalised → qualifier-stripped → medley)…")
    pre_matched: list[dict] = []
    needs_llm:  list[dict] = []

    for row in flat:
        m = deterministic_match(row["setlist_track_name"], catalog)
        if m:
            pre_matched.append({**row, **m})
        else:
            needs_llm.append(row)

    # Stage 5 — LLM (only for unresolved tracks)
    llm_results: list[dict] = []
    for i, row in enumerate(needs_llm):
        track = row["setlist_track_name"]
        step(5, f"AI matching {i+1}/{len(needs_llm)}: \"{track}\"…")
        m = llm_fuzzy_match(track, catalog, client, model)
        llm_results.append({**row, **m})

    if not needs_llm:
        step(5, "All tracks resolved deterministically — 0 LLM calls needed.")

    # Build final output rows
    all_results = []
    for row in pre_matched + llm_results:
        all_results.append({
            "show_date":          row.get("show_date", ""),
            "venue_name":         row.get("venue_name", ""),
            "setlist_track_name": row.get("setlist_track_name", ""),
            "matched_catalog_id": row.get("matched_catalog_id") or "None",
            "match_confidence":   row.get("match_confidence", NONE),
            "match_notes":        row.get("match_notes", ""),
        })

    # Sort by show date then venue to keep shows together
    all_results.sort(key=lambda r: (r["show_date"], r["venue_name"]))
    result["results"] = all_results

    # Stats
    total = len(all_results)
    result["stats"] = {
        "total_tracks":    total,
        "exact_matches":   sum(1 for r in all_results if r["match_confidence"] == EXACT),
        "high_matches":    sum(1 for r in all_results if r["match_confidence"] == HIGH),
        "review_matches":  sum(1 for r in all_results if r["match_confidence"] == REVIEW),
        "no_matches":      sum(1 for r in all_results if r["match_confidence"] == NONE),
        "deterministic":   len(pre_matched),
        "llm_resolved":    len(llm_results),
        "llm_savings_pct": round(len(pre_matched) / total * 100 if total else 0, 1),
    }
    return result


# ── CSV output ─────────────────────────────────────────────────────────────────

def build_output_csv(results: list[dict]) -> str:
    buf = io.StringIO()
    w   = csv.DictWriter(buf, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
    w.writeheader()
    w.writerows(results)
    return buf.getvalue()
