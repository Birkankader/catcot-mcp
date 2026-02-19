"""Catcot savings tracker — records search usage and calculates token savings."""

import json
import os
import time
from datetime import datetime, date
from pathlib import Path

SAVINGS_DIR = os.path.expanduser("~/.code-rag-mcp")
SAVINGS_FILE = os.path.join(SAVINGS_DIR, "savings.json")

# Token estimation: ~4 characters = 1 token
CHARS_PER_TOKEN = 4

# Claude pricing per million input tokens (USD)
PRICING = {
    "opus": 15.0,
    "sonnet": 3.0,
    "haiku": 0.80,
}

# Default model for cost estimation
DEFAULT_MODEL = "sonnet"


def _load_data() -> dict:
    """Load savings data from disk."""
    if not os.path.exists(SAVINGS_FILE):
        return {
            "searches": [],
            "totals": {
                "total_searches": 0,
                "tokens_saved": 0,
                "tokens_used": 0,
                "dollars_saved": 0.0,
            },
        }
    try:
        with open(SAVINGS_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {
            "searches": [],
            "totals": {
                "total_searches": 0,
                "tokens_saved": 0,
                "tokens_used": 0,
                "dollars_saved": 0.0,
            },
        }


def _save_data(data: dict) -> None:
    """Persist savings data to disk."""
    os.makedirs(SAVINGS_DIR, exist_ok=True)
    with open(SAVINGS_FILE, "w") as f:
        json.dump(data, f, indent=2)


def _estimate_full_read_tokens(project_path: str) -> int:
    """Estimate how many tokens it would take to read all indexed files in a project."""
    # Walk project directory and sum up file sizes for supported extensions
    supported = {
        ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".kt", ".kts",
        ".go", ".rs", ".c", ".cpp", ".h", ".hpp", ".cs", ".rb",
        ".php", ".swift", ".sql", ".yaml", ".yml", ".json", ".md",
        ".html", ".css", ".scss", ".xml", ".toml", ".cfg", ".ini",
    }
    total_chars = 0
    if not project_path or not os.path.isdir(project_path):
        return 50000  # fallback estimate: ~50k chars = 12.5k tokens

    for root, dirs, files in os.walk(project_path):
        # Skip hidden dirs and common non-code dirs
        dirs[:] = [
            d for d in dirs
            if not d.startswith(".") and d not in {
                "node_modules", "__pycache__", "venv", ".venv", "dist",
                "build", ".git", ".svn", "vendor",
            }
        ]
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in supported:
                try:
                    total_chars += os.path.getsize(os.path.join(root, f))
                except OSError:
                    pass

    return max(total_chars // CHARS_PER_TOKEN, 1000)


def record_search(
    query: str,
    results: list[dict],
    project_path: str = "",
) -> dict:
    """Record a search and calculate savings.

    Returns savings info for this search.
    """
    data = _load_data()

    # Calculate tokens used (what Catcot actually returned)
    returned_chars = sum(len(r.get("content", "")) for r in results)
    tokens_used = returned_chars // CHARS_PER_TOKEN

    # Calculate tokens that would have been used reading all files
    tokens_full_read = _estimate_full_read_tokens(project_path)

    # Savings = full read - what we actually returned
    tokens_saved = max(tokens_full_read - tokens_used, 0)

    # Dollar savings (using default model pricing)
    price_per_token = PRICING[DEFAULT_MODEL] / 1_000_000
    dollars_saved = tokens_saved * price_per_token

    search_record = {
        "query": query,
        "timestamp": time.time(),
        "date": date.today().isoformat(),
        "results_count": len(results),
        "tokens_used": tokens_used,
        "tokens_saved": tokens_saved,
        "dollars_saved": round(dollars_saved, 6),
        "project_path": project_path or "all",
    }

    data["searches"].append(search_record)

    # Update totals
    data["totals"]["total_searches"] += 1
    data["totals"]["tokens_saved"] += tokens_saved
    data["totals"]["tokens_used"] += tokens_used
    data["totals"]["dollars_saved"] = round(
        data["totals"]["dollars_saved"] + dollars_saved, 6
    )

    # Keep only last 1000 searches to prevent unbounded growth
    if len(data["searches"]) > 1000:
        data["searches"] = data["searches"][-1000:]

    _save_data(data)

    return {
        "tokens_used": tokens_used,
        "tokens_saved": tokens_saved,
        "dollars_saved": round(dollars_saved, 4),
    }


def get_savings_summary() -> dict:
    """Get cumulative savings summary + last 7 days trend."""
    data = _load_data()
    totals = data["totals"]

    # Build last 7 days trend
    today = date.today()
    daily = {}
    for s in data["searches"]:
        d = s.get("date", "")
        if d:
            daily.setdefault(d, {"tokens_saved": 0, "searches": 0, "dollars_saved": 0.0})
            daily[d]["tokens_saved"] += s.get("tokens_saved", 0)
            daily[d]["searches"] += 1
            daily[d]["dollars_saved"] += s.get("dollars_saved", 0.0)

    trend = []
    for i in range(6, -1, -1):
        from datetime import timedelta
        d = (today - timedelta(days=i)).isoformat()
        entry = daily.get(d, {"tokens_saved": 0, "searches": 0, "dollars_saved": 0.0})
        trend.append({
            "date": d,
            "tokens_saved": entry["tokens_saved"],
            "searches": entry["searches"],
            "dollars_saved": round(entry["dollars_saved"], 4),
        })

    return {
        "total_searches": totals["total_searches"],
        "tokens_saved": totals["tokens_saved"],
        "tokens_used": totals["tokens_used"],
        "dollars_saved": round(totals["dollars_saved"], 4),
        "model": DEFAULT_MODEL,
        "pricing": PRICING,
        "trend": trend,
    }
