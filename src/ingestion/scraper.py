"""
src/ingestion/scraper.py
Scrape articles từ discovered_urls.json

Chạy:
    python -m src.ingestion.scraper --limit 5 --dry-run
    python -m src.ingestion.scraper --limit 50
"""
from __future__ import annotations

import argparse
import json
import re
import time
import random
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from src.utils.logger import get_logger

logger = get_logger(__name__, log_file="logs/scraper.log")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

URL_FILE  = "data/discovered_urls.json"
OUTPUT_DIR = "data/raw"


# ── Data classes ──────────────────────────────────────────────

@dataclass
class ArticleMetadata:
    url: str
    article_id: str
    title: str
    category: str
    section: str
    last_updated: str
    scraped_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class RawArticle:
    metadata: ArticleMetadata
    plain_text: str
    headings: list[str]
    symptoms: list[str]
    causes: list[str]
    resolution_steps: list[str]
    error_codes: list[str]


# ── Load URLs ─────────────────────────────────────────────────

def load_urls(url_file: str = URL_FILE) -> dict[str, list[str]]:
    """Load discovered_urls.json → dict {category: [urls]}"""
    data = json.loads(Path(url_file).read_text(encoding="utf-8"))
    total = sum(len(v) for v in data.values())
    logger.info(f"Loaded {total} URLs từ {url_file}")
    return data


# ── Session ───────────────────────────────────────────────────

def create_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)
    return s


# ── Scrape một article ────────────────────────────────────────

def scrape_article(
    session: requests.Session,
    url: str,
    category: str,
) -> RawArticle | None:
    time.sleep(random.uniform(1.0, 2.5))

    try:
        resp = session.get(url, timeout=20)
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.warning(f"Lỗi fetch: {url} — {e}")
        return None

    soup = BeautifulSoup(resp.text, "lxml")

    title = _get_title(soup)
    if not title:
        return None

    content = _get_content(soup)
    if not content:
        return None

    return RawArticle(
        metadata=ArticleMetadata(
            url=url,
            article_id=_get_article_id(url, soup),
            title=title,
            category=category,
            section=_get_section(url),
            last_updated=_get_last_updated(soup),
        ),
        plain_text=_to_text(content),
        headings=_get_headings(content),
        symptoms=_get_section_text(content, ["symptoms", "symptom"]),
        causes=_get_section_text(content, ["cause", "causes"]),
        resolution_steps=_get_section_text(content, ["resolution", "solution", "fix"]),
        error_codes=_get_error_codes(_to_text(content)),
    )


# ── HTML helpers ──────────────────────────────────────────────

def _get_title(soup: BeautifulSoup) -> str:
    h1 = soup.find("h1")
    if h1:
        return h1.get_text(strip=True)
    tag = soup.find("title")
    if tag:
        return re.sub(r"\s*\|\s*Microsoft Learn.*$", "", tag.get_text(strip=True)).strip()
    return ""


def _get_content(soup: BeautifulSoup):
    for sel in ["main", "article", {"role": "main"}]:
        el = soup.find(sel) if isinstance(sel, str) else soup.find(attrs=sel)
        if el:
            return el
    return None


def _to_text(el) -> str:
    for tag in el.find_all(["script", "style", "nav"]):
        tag.decompose()
    lines = [l.strip() for l in el.get_text(separator="\n").splitlines() if l.strip()]
    return "\n".join(lines)


def _get_headings(el) -> list[str]:
    return [
        f"{t.name.upper()}: {t.get_text(strip=True)}"
        for t in el.find_all(["h1", "h2", "h3"])
        if t.get_text(strip=True)
    ]


def _get_section_text(el, keywords: list[str]) -> list[str]:
    results = []
    headings = el.find_all(["h2", "h3"])
    for i, h in enumerate(headings):
        if any(kw in h.get_text(strip=True).lower() for kw in keywords):
            next_h = headings[i + 1] if i + 1 < len(headings) else None
            sib = h.find_next_sibling()
            while sib and sib != next_h:
                if sib.name in ["p", "li", "pre"]:
                    t = sib.get_text(strip=True)
                    if t:
                        results.append(t)
                sib = sib.find_next_sibling()
    return results


def _get_error_codes(text: str) -> list[str]:
    patterns = [
        r"\b0x[0-9A-Fa-f]{4,8}\b",
        r"\bERROR_[A-Z_]+\b",
        r"\bEvent\s+ID\s+\d+\b",
        r"\bKB\d{6,7}\b",
    ]
    seen, codes = set(), []
    for p in patterns:
        for m in re.findall(p, text, re.IGNORECASE):
            if m not in seen:
                seen.add(m)
                codes.append(m)
    return codes


def _get_article_id(url: str, soup: BeautifulSoup) -> str:
    text = soup.get_text()
    m = re.search(r"\b(KB\d{6,7})\b", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return urlparse(url).path.rstrip("/").split("/")[-1]


def _get_section(url: str) -> str:
    parts = urlparse(url).path.strip("/").split("/")
    return parts[-2] if len(parts) >= 2 else "general"


def _get_last_updated(soup: BeautifulSoup) -> str:
    tag = soup.find("time")
    if tag:
        return tag.get("datetime", tag.get_text(strip=True))
    return ""


# ── Lưu file ──────────────────────────────────────────────────

def save_article(article: RawArticle, output_dir: str = OUTPUT_DIR) -> Path:
    out = Path(output_dir) / article.metadata.category
    out.mkdir(parents=True, exist_ok=True)
    filepath = out / f"{article.metadata.article_id}.json"
    filepath.write_text(
        json.dumps(asdict(article), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return filepath


# ── Pipeline ──────────────────────────────────────────────────

def run(limit: int | None = None, dry_run: bool = False) -> None:
    url_data = load_urls()
    session  = create_session()
    saved, errors = 0, 0

    # Gom tất cả URLs thành list [(category, url)]
    all_items = [
        (category, url)
        for category, urls in url_data.items()
        for url in urls
    ]

    if limit:
        all_items = all_items[:limit]

    total = len(all_items)
    logger.info(f"Bắt đầu scrape {total} articles...")

    for i, (category, url) in enumerate(all_items, 1):
        logger.info(f"[{i}/{total}] [{category}] {url.split('/')[-1]}")

        article = scrape_article(session, url, category)

        if article is None:
            errors += 1
            continue

        if dry_run:
            logger.info(
                f"  [DRY RUN] {article.metadata.title[:60]} | "
                f"steps={len(article.resolution_steps)} | "
                f"errors={article.error_codes[:2]}"
            )
        else:
            path = save_article(article)
            logger.info(f"  Saved → {path}")
            saved += 1

    logger.info(f"=== Xong: {saved} saved, {errors} errors ===")


# ── CLI ───────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit",   type=int, default=None, help="Giới hạn số article")
    parser.add_argument("--dry-run", action="store_true",    help="Chỉ log, không lưu")
    args = parser.parse_args()

    run(limit=args.limit, dry_run=args.dry_run)