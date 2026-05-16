"""
Scrape the SHL Individual Test Solutions catalog and save to data/catalog.json.
Run once before building the vector index:
    python scripts/scrape_catalog.py
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup

CATALOG_URL = "https://www.shl.com/solutions/products/product-catalog/"
BASE_URL = "https://www.shl.com"
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "catalog.json"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
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


def fetch_page(url: str, retries: int = 3) -> Optional[BeautifulSoup]:
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=HEADERS, timeout=20)
            response.raise_for_status()
            return BeautifulSoup(response.text, "lxml")
        except Exception as exc:
            print(f"  Attempt {attempt + 1} failed for {url}: {exc}")
            time.sleep(2 ** attempt)
    return None


def parse_individual_tests(soup: BeautifulSoup) -> list[dict]:
    """Extract all Individual Test Solutions rows from the catalog page."""
    assessments: list[dict] = []

    # The page has two tables: Pre-packaged Job Solutions and Individual Test Solutions.
    # Find all tables and pick the one whose header contains "Individual Test Solutions".
    tables = soup.find_all("table")
    individual_table = None
    for table in tables:
        header_text = table.get_text(" ", strip=True)
        if "Individual Test Solutions" in header_text:
            individual_table = table
            break

    if individual_table is None:
        # Fall back: look for rows with links to /product-catalog/view/ after the marker
        print("  Warning: could not find Individual Test Solutions table, using heuristic")
        rows = soup.select("a[href*='/product-catalog/view/']")
        for a_tag in rows:
            href = a_tag["href"]
            if not href.startswith("http"):
                href = BASE_URL + href
            name = a_tag.get_text(strip=True)
            if name:
                assessments.append({"name": name, "url": href, "test_types": [], "remote_testing": False, "adaptive_irt": False})
        return assessments

    for row in individual_table.find_all("tr"):
        cells = row.find_all("td")
        if not cells:
            continue

        link_tag = cells[0].find("a")
        if not link_tag:
            continue

        name = link_tag.get_text(strip=True)
        href = link_tag.get("href", "")
        if not href.startswith("http"):
            href = BASE_URL + href

        # Remote Testing: look for check mark icons in cells 1 & 2
        remote_testing = _has_check(cells[1]) if len(cells) > 1 else False
        adaptive_irt = _has_check(cells[2]) if len(cells) > 2 else False

        # Test types: last cell contains letter codes
        test_type_str = ""
        if len(cells) > 3:
            test_type_str = cells[3].get_text(" ", strip=True)
        else:
            # Sometimes type codes appear in the row text after the name
            row_text = row.get_text(" ", strip=True)
            test_type_str = row_text.replace(name, "").strip()

        type_codes = [c for c in re.findall(r"[A-Z]", test_type_str) if c in TEST_TYPE_LABELS]

        assessments.append({
            "name": name,
            "url": href,
            "test_types": type_codes,
            "remote_testing": remote_testing,
            "adaptive_irt": adaptive_irt,
        })

    return assessments


def _has_check(cell) -> bool:
    """Return True if a table cell contains a check mark (tick) indicator."""
    text = cell.get_text(strip=True)
    if any(ch in text for ch in ["✓", "✔", "☑", "●"]):
        return True
    imgs = cell.find_all("img")
    for img in imgs:
        alt = img.get("alt", "").lower()
        src = img.get("src", "").lower()
        if "check" in alt or "tick" in alt or "yes" in alt or "check" in src or "tick" in src:
            return True
    return False


def scrape_product_page(url: str) -> dict:
    """Visit an individual assessment page and extract description + metadata."""
    soup = fetch_page(url)
    if soup is None:
        return {}

    data: dict = {}

    # Description: look for common description containers
    for sel in ["div.product-description", "div.description", "div.content", "main p", "article p"]:
        elem = soup.select_one(sel)
        if elem:
            text = elem.get_text(" ", strip=True)
            if len(text) > 30:
                data["description"] = text[:800]
                break

    # Duration
    duration_match = re.search(r"(\d+)\s*(min|minute)", soup.get_text(), re.IGNORECASE)
    if duration_match:
        data["duration_minutes"] = int(duration_match.group(1))

    # Languages
    lang_section = soup.find(string=re.compile(r"language", re.IGNORECASE))
    if lang_section:
        parent = lang_section.find_parent()
        if parent:
            data["languages_raw"] = parent.get_text(" ", strip=True)[:200]

    return data


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Fetching catalog page...")
    soup = fetch_page(CATALOG_URL)
    if soup is None:
        print("Failed to fetch catalog page. Aborting.")
        return

    print("Parsing Individual Test Solutions...")
    assessments = parse_individual_tests(soup)
    print(f"  Found {len(assessments)} assessments in catalog listing.")

    # Enrich each assessment with data from its product page
    print("Enriching each assessment from product pages (may take a few minutes)...")
    for i, item in enumerate(assessments):
        print(f"  [{i+1}/{len(assessments)}] {item['name']}")
        extra = scrape_product_page(item["url"])
        item.update(extra)
        time.sleep(0.5)  # be polite

    # Compute a human-readable test_type string
    for item in assessments:
        codes = item.get("test_types", [])
        item["test_type"] = " ".join(codes)
        item["test_type_labels"] = [TEST_TYPE_LABELS.get(c, c) for c in codes]

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(assessments, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(assessments)} assessments to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
