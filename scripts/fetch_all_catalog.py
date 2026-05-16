"""
Fetch ALL Individual Test Solutions from the SHL catalog by paginating through
?start=0, ?start=12, ?start=24, ... until no more items are found.

Then enrich each item with data from its individual product page.

Usage:
    python scripts/fetch_all_catalog.py
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup

CATALOG_BASE = "https://www.shl.com/solutions/products/product-catalog/"
BASE_URL = "https://www.shl.com"
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "catalog.json"
PAGE_SIZE = 12

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
}

TEST_TYPE_LABELS = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behavior",
    "S": "Simulations",
}


def fetch_html(url: str, retries: int = 3) -> Optional[BeautifulSoup]:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=25, verify=False)
            r.raise_for_status()
            return BeautifulSoup(r.text, "lxml")
        except Exception as exc:
            wait = 2 ** attempt
            print(f"  [attempt {attempt+1}] Error fetching {url}: {exc}. Waiting {wait}s...")
            time.sleep(wait)
    return None


def parse_individual_tests_from_page(soup: BeautifulSoup) -> list[dict]:
    """
    Extract Individual Test Solutions from one catalog page.
    Returns empty list if the table has no rows (end of pagination).
    """
    items = []
    tables = soup.find_all("table")

    individual_table = None
    for table in tables:
        header = table.find_previous_sibling()
        text = table.get_text(" ", strip=True)
        if "Individual Test Solutions" in text:
            individual_table = table
            break

    if individual_table is None:
        # Try to find table that is NOT the pre-packaged table
        for table in tables:
            if "Individual Test Solutions" in table.get_text(" "):
                individual_table = table
                break

    if individual_table is None:
        return []

    for row in individual_table.find_all("tr"):
        cells = row.find_all("td")
        if not cells:
            continue
        link = cells[0].find("a")
        if not link:
            continue

        name = link.get_text(strip=True)
        href = link.get("href", "")
        if not href.startswith("http"):
            href = BASE_URL + href

        remote = _cell_has_check(cells[1]) if len(cells) > 1 else False
        adaptive = _cell_has_check(cells[2]) if len(cells) > 2 else False

        type_text = cells[3].get_text(" ", strip=True) if len(cells) > 3 else ""
        codes = [c for c in re.findall(r"\b([ABCDEKPS])\b", type_text) if c in TEST_TYPE_LABELS]
        # Fallback: single letters anywhere in the cell
        if not codes:
            codes = [c for c in re.findall(r"[ABCDEKPS]", type_text) if c in TEST_TYPE_LABELS]

        items.append({
            "name": name,
            "url": href,
            "test_types": codes,
            "remote_testing": remote,
            "adaptive_irt": adaptive,
        })

    return items


def _cell_has_check(cell) -> bool:
    text = cell.get_text(strip=True)
    if any(c in text for c in ["✓", "✔", "☑", "●", "•"]):
        return True
    for img in cell.find_all("img"):
        alt = img.get("alt", "").lower()
        src = img.get("src", "").lower()
        if any(w in alt or w in src for w in ["check", "tick", "yes", "true"]):
            return True
    return False


def scrape_product_page(url: str) -> dict:
    """Scrape an individual product page for description, job levels, duration, languages."""
    soup = fetch_html(url)
    if soup is None:
        return {}

    data: dict = {}
    full_text = soup.get_text(" ", strip=True)

    # Description
    for sel in [
        "div.product-description", "div.tab-content", "section.description",
        "div.field--name-body", "article", "main",
    ]:
        elem = soup.select_one(sel)
        if elem:
            text = elem.get_text(" ", strip=True)
            if len(text) > 50:
                data["description"] = text[:600]
                break

    # Fallback description from page text
    if "description" not in data:
        # Look for a paragraph after the title
        h1 = soup.find("h1")
        if h1:
            sibling = h1.find_next_sibling(["p", "div"])
            if sibling:
                text = sibling.get_text(" ", strip=True)
                if len(text) > 20:
                    data["description"] = text[:400]

    # Duration
    dur_match = re.search(r"(?:Approximate Completion Time[^=\d]*=?\s*)?(\d+)\s*(?:min|minute)", full_text, re.IGNORECASE)
    if dur_match:
        data["duration_minutes"] = int(dur_match.group(1))

    # Job levels
    jl_match = re.search(r"Job [Ll]evels?\s*[:]*\s*([A-Za-z ,\-]+?)(?:\n|Test Type|Language|Download)", full_text)
    if jl_match:
        data["job_levels"] = jl_match.group(1).strip().rstrip(",")

    # Languages
    lang_match = re.search(r"Language[s]?\s*[:]*\s*([A-Za-z ,()\-]+?)(?:\n|Assessment|Test Type|Download)", full_text)
    if lang_match:
        raw_lang = lang_match.group(1).strip().rstrip(",")
        if len(raw_lang) < 300:
            data["languages"] = raw_lang

    return data


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    all_items: list[dict] = []
    seen_names: set[str] = set()
    start = 0

    print("Paginating through Individual Test Solutions catalog...")
    while True:
        url = f"{CATALOG_BASE}?start={start}&type=1"
        print(f"  Fetching page start={start}: {url}")
        soup = fetch_html(url)
        if soup is None:
            print("  Failed to fetch page. Stopping pagination.")
            break

        items = parse_individual_tests_from_page(soup)
        if not items:
            print(f"  No items found on page start={start}. End of catalog.")
            break

        new_items = [i for i in items if i["name"] not in seen_names]
        if not new_items:
            print(f"  All items already seen at start={start}. End of catalog.")
            break

        for i in new_items:
            seen_names.add(i["name"])
        all_items.extend(new_items)
        print(f"  Got {len(new_items)} new items (total: {len(all_items)})")

        start += PAGE_SIZE
        time.sleep(0.8)

    print(f"\nTotal assessments found: {len(all_items)}")

    # Save bare catalog before enrichment (fallback)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_items, f, ensure_ascii=False, indent=2)
    print(f"Saved bare catalog to {OUTPUT_PATH}")

    # Enrich each assessment from its product page
    print("\nEnriching from individual product pages...")
    for i, item in enumerate(all_items):
        print(f"  [{i+1}/{len(all_items)}] {item['name']}")
        extra = scrape_product_page(item["url"])
        item.update(extra)

        # Also compute human-readable type info
        codes = item.get("test_types", [])
        item["test_type"] = " ".join(codes)
        item["test_type_labels"] = [TEST_TYPE_LABELS.get(c, c) for c in codes]

        # Save incrementally every 20 items
        if (i + 1) % 20 == 0:
            with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
                json.dump(all_items, f, ensure_ascii=False, indent=2)
            print(f"  [checkpoint] Saved {len(all_items)} items")

        time.sleep(0.5)

    # Final save
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_items, f, ensure_ascii=False, indent=2)
    print(f"\nDone. Saved {len(all_items)} enriched assessments to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
