"""
scripts/scrape_qa.py
Scrape câu hỏi thật từ Microsoft Q&A (learn.microsoft.com/en-us/answers/)
để tạo test set chuẩn cho hệ thống KBQA.

Microsoft Q&A nhúng toàn bộ Q&A data dưới dạng JSON-LD (schema.org QAPage)
trong thẻ <script type="application/ld+json"> — parse trực tiếp từ đó,
không phụ thuộc vào HTML class selectors (vốn thay đổi thường xuyên).

Cách dùng:
    python scripts/scrape_qa.py --dry-run --limit 20
    python scripts/scrape_qa.py --limit 100
    python scripts/scrape_qa.py --limit 100 --min-votes 1
"""
from __future__ import annotations

import argparse
import json
import re
import time
import random
import os
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

# ── Logging ───────────────────────────────────────────────────
try:
    from src.utils.logger import get_logger
    logger = get_logger(__name__, log_file="logs/scrape_qa.log")
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
DEFAULT_OUTPUT = os.path.join(PROJECT_ROOT, "data", "qa_testset_raw.json")
BASE_URL = "https://learn.microsoft.com"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# ── Tag URLs — format: /en-us/answers/tags/{id}/{slug} ────────
CATEGORY_TAGS: dict[str, list[str]] = {
    "DeviceMgmt": [
        "https://learn.microsoft.com/en-us/answers/tags/409/microsoft-security-intune-enrollment",
        "https://learn.microsoft.com/en-us/answers/tags/415/intune",
        "https://learn.microsoft.com/en-us/answers/tags/416/microsoft-security-intune-configuration",
        # Error-code-heavy: compliance failures and troubleshooting logs
        "https://learn.microsoft.com/en-us/answers/tags/408/microsoft-security-intune-compliance",
        "https://learn.microsoft.com/en-us/answers/tags/411/microsoft-security-intune-troubleshooting",
    ],
    "Teams": [
        "https://learn.microsoft.com/en-us/answers/tags/110/microsoft-teams",
        "https://learn.microsoft.com/en-us/answers/tags/436/microsoft-teams-rooms",
        # Calling/meeting errors tend to include error codes and version info
        "https://learn.microsoft.com/en-us/answers/tags/112/microsoft-teams-calling",
        "https://learn.microsoft.com/en-us/answers/tags/114/microsoft-teams-meetings",
    ],
    "Identity": [
        "https://learn.microsoft.com/en-us/answers/tags/455/microsoft-security-entra-entra-id",
        "https://learn.microsoft.com/en-us/answers/tags/456/microsoft-security-entra-conditional-access",
        # Auth failures frequently surface AADSTS/0x error codes; hybrid join uses event IDs
        "https://learn.microsoft.com/en-us/answers/tags/427/microsoft-security-entra-authentication",
        "https://learn.microsoft.com/en-us/answers/tags/428/microsoft-security-entra-hybrid-join",
    ],
    "Network": [
        "https://learn.microsoft.com/en-us/answers/tags/22/windows-server-networking",
        "https://learn.microsoft.com/en-us/answers/tags/23/windows-10-networking",
        # Active Directory: event IDs and LDAP errors; Windows 11: version/update questions
        "https://learn.microsoft.com/en-us/answers/tags/25/active-directory",
        "https://learn.microsoft.com/en-us/answers/tags/3/windows-11",
    ],
}

CATEGORY_LIMITS: dict[str, int] = {
    "DeviceMgmt": 80,
    "Teams":      70,
    "Identity":   70,
    "Network":    60,
}


# ── Data class ────────────────────────────────────────────────

@dataclass
class QAItem:
    id: str
    question: str
    question_body: str
    accepted_answer: str
    top_answer: str
    answer_count: int
    votes: int
    tags: list[str]
    category: str
    source_tag_url: str
    url: str
    has_accepted_answer: bool
    scraped_at: str = field(default_factory=lambda: datetime.now().isoformat())
    matched_article_id: str = ""
    expected_tool: str = ""
    query_type: str = ""
    difficulty: str = ""
    ground_truth_answer: str = ""


# ── Session ───────────────────────────────────────────────────

def create_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)
    return s


# ── JSON-LD extraction (core method) ─────────────────────────

def _extract_jsonld(soup: BeautifulSoup) -> dict | None:
    """
    Trích JSON-LD schema.org QAPage từ <script type="application/ld+json">.
    Microsoft Q&A nhúng toàn bộ Q&A data ở đây — đây là cách parse đáng tin nhất.

    Structure:
    {
      "@type": "QAPage",
      "mainEntity": {
        "@type": "Question",
        "name": "...",          ← question title
        "text": "...",          ← question body (HTML encoded)
        "upvoteCount": N,
        "acceptedAnswer": {     ← accepted answer nếu có
          "@type": "Answer",
          "text": "...",
          "upvoteCount": N,
          "author": "..."
        },
        "suggestedAnswer": [    ← các answers khác
          {"@type": "Answer", "text": "...", ...}
        ]
      }
    }
    """
    for script in soup.find_all("script"):
        # Tìm script chứa JSON-LD
        script_type = script.get("type", "")
        content = script.string or ""

        if not content:
            continue

        # Microsoft Q&A có thể không set type attribute, check content thay thế
        if script_type != "application/ld+json" and '"QAPage"' not in content:
            continue

        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            continue

        # Handle cả single object và array
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and item.get("@type") == "QAPage":
                    return item.get("mainEntity", {})
        elif isinstance(data, dict):
            if data.get("@type") == "QAPage":
                return data.get("mainEntity", {})
            # Đôi khi mainEntity nằm trực tiếp
            if data.get("@type") == "Question":
                return data

    return None


def _clean_text(text: str) -> str:
    """Fix encoding artifacts như â, â, â (Windows-1252 misread as UTF-8)."""
    replacements = {
        'â': "'",   # right single quote
        'â': '"',   # left double quote
        'â': '"',   # right double quote
        'â': '-',    # en dash
        'â': '--',   # em dash
        'â¦': '...',  # ellipsis
        'â': '',     # zero-width space
        'Â': '',             # stray Â
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    # Fallback: encode/decode để loại ký tự không hợp lệ
    text = text.encode('utf-8', errors='ignore').decode('utf-8')
    return text


def _html_to_text(html_str: str) -> str:
    """
    Convert HTML string từ JSON-LD thành plain text.
    json.loads() đã tự decode unicode escapes nên KHÔNG cần unicode_escape.
    """
    if not html_str:
        return ""
    soup = BeautifulSoup(html_str, "lxml")
    text = soup.get_text(separator=" ", strip=True)
    return _clean_text(text)


def _parse_answers_from_schema(schema: dict) -> tuple[str, str, int]:
    """
    Parse accepted answer và top answer từ JSON-LD schema.

    Thực tế Microsoft Q&A:
      acceptedAnswer: []       → không có accepted answer
      acceptedAnswer: [{...}]  → có accepted answer
      suggestedAnswer: [...]   → các answers khác

    Returns: (accepted_answer, top_answer, answer_count)
    """
    accepted = ""
    top      = ""
    count    = 0

    # acceptedAnswer — [] khi không có, [{...}] khi có
    accepted_raw = schema.get("acceptedAnswer", [])
    if isinstance(accepted_raw, dict):
        accepted_raw = [accepted_raw]
    if isinstance(accepted_raw, list) and accepted_raw:
        first = accepted_raw[0]
        if isinstance(first, dict):
            text = _html_to_text(first.get("text", ""))
            if text:
                accepted = text
                count += 1

    # suggestedAnswer — luôn là list
    suggested = schema.get("suggestedAnswer", [])
    if isinstance(suggested, dict):
        suggested = [suggested]
    if not isinstance(suggested, list):
        suggested = []
    count += len(suggested)

    # top = suggested answer có upvoteCount cao nhất
    if suggested:
        best = max(
            suggested,
            key=lambda a: a.get("upvoteCount", 0) if isinstance(a, dict) else 0
        )
        top = _html_to_text(best.get("text", ""))

    return accepted[:1500], top[:1500], count


# ── Listing page parser ───────────────────────────────────────

def fetch_questions_by_tag_url(
    session: requests.Session,
    tag_url: str,
    max_pages: int = 4,
) -> list[dict]:
    """
    Lấy danh sách URLs từ trang tag.
    HTML structure: <h2><a href="/en-us/answers/questions/{id}/{slug}">Title</a></h2>
    Filter: filterby=answered — chỉ lấy câu đã có answer
    Sort:   orderby=createdat — đa dạng về thời gian
    """
    questions = []
    seen_urls: set[str] = set()

    for page in range(1, max_pages + 1):
        url = f"{tag_url}?filterby=answered&orderby=createdat&page={page}"
        logger.info(f"  Fetching page={page}: .../{tag_url.split('/')[-1]}")
        time.sleep(random.uniform(1.5, 3.0))

        try:
            resp = session.get(url, timeout=20)
            if resp.status_code == 404:
                logger.warning(f"  Tag không tồn tại (404): {tag_url}")
                break
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.warning(f"  Lỗi fetch: {e}")
            break

        soup = BeautifulSoup(resp.text, "lxml")
        found = _parse_question_links(soup)

        new_count = 0
        for q in found:
            if q["url"] not in seen_urls:
                seen_urls.add(q["url"])
                questions.append(q)
                new_count += 1

        logger.info(f"  → Page {page}: {new_count} câu mới (tổng: {len(questions)})")

        if new_count == 0:
            break

    return questions


def _parse_question_links(soup: BeautifulSoup) -> list[dict]:
    """Parse <h2><a href="/answers/questions/ID/slug"> từ listing page."""
    results = []

    # Primary: h2 > a với href khớp pattern
    for h2 in soup.find_all("h2"):
        link = h2.find("a", href=re.compile(r"/answers/questions/\d+/"))
        if not link:
            continue
        title = link.get_text(strip=True)
        href  = link.get("href", "")
        if not title or len(title) < 10:
            continue
        results.append({"url": urljoin(BASE_URL, href), "title": title})

    # Fallback: tìm tất cả links nếu h2 không có
    if not results:
        for link in soup.find_all("a", href=re.compile(r"/en-us/answers/questions/\d+/")):
            title = link.get_text(strip=True)
            if not title or len(title) < 10:
                continue
            parents = [p.name for p in link.parents]
            if any(p in parents for p in ["nav", "footer", "aside"]):
                continue
            results.append({"url": urljoin(BASE_URL, link["href"]), "title": title})

    return results


# ── Scrape chi tiết một câu hỏi ───────────────────────────────

def scrape_question_detail(
    session: requests.Session,
    url: str,
    category: str,
    source_tag_url: str,
) -> QAItem | None:
    time.sleep(random.uniform(1.0, 2.5))

    try:
        resp = session.get(url, timeout=20)
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.warning(f"Lỗi fetch detail {url}: {e}")
        return None

    soup = BeautifulSoup(resp.text, "lxml")

    # ── Parse từ JSON-LD (schema.org QAPage) ──────────────────
    schema = _extract_jsonld(soup)

    if schema:
        question_title = schema.get("name", "").strip()
        question_body  = _html_to_text(schema.get("text", ""))
        accepted, top, ans_count = _parse_answers_from_schema(schema)
        votes = schema.get("upvoteCount", 0) or 0
        logger.debug(f"  JSON-LD OK: accepted={bool(accepted)} top={bool(top)} answers={ans_count}")
    else:
        # Fallback HTML parsing
        logger.warning(f"  JSON-LD không tìm thấy, fallback HTML: {url}")
        question_title = _get_question_title_html(soup)
        question_body  = _get_question_body_html(soup)
        accepted, top, ans_count = "", "", 0
        votes = 0

    # Nếu title quá dài (>200 chars) thì đang lấy nhầm answer text — fallback về HTML h1
    if len(question_title) > 200:
        question_title = _get_question_title_html(soup)

    if not question_title or len(question_title) < 15:
        return None

    if not accepted and not top:
        logger.info(f"  Bỏ qua (không có answer): {question_title[:50]}")
        return None

    tags      = _get_tags(soup)
    full_text = question_title + " " + question_body

    return QAItem(
        id=_extract_id(url),
        question=question_title,
        question_body=question_body[:1500],
        accepted_answer=accepted,
        top_answer=top,
        answer_count=ans_count,
        votes=votes,
        tags=tags,
        category=category,
        source_tag_url=source_tag_url,
        url=url,
        has_accepted_answer=bool(accepted),
        expected_tool=_detect_expected_tool(full_text),
        query_type=_detect_query_type(full_text),
        difficulty=_estimate_difficulty(question_body, ans_count),
    )


# ── HTML fallback helpers ─────────────────────────────────────

def _get_question_title_html(soup: BeautifulSoup) -> str:
    h1 = soup.find("h1")
    if h1:
        return h1.get_text(strip=True)
    tag = soup.find("title")
    if tag:
        return re.sub(r"\s*[-|]\s*Microsoft Q&A.*$", "", tag.get_text(strip=True)).strip()
    return ""


def _get_question_body_html(soup: BeautifulSoup) -> str:
    main = soup.find("main") or soup.find(id="main-content")
    if not main:
        return ""
    parts = [p.get_text(strip=True) for p in main.find_all("p")[:8] if len(p.get_text(strip=True)) > 30]
    return " ".join(parts)[:1500]


def _get_tags(soup: BeautifulSoup) -> list[str]:
    tags = []
    for link in soup.find_all("a", href=re.compile(r"/answers/tags/")):
        t = link.get_text(strip=True).lower()
        if 2 < len(t) < 60:
            tags.append(t)
    return list(dict.fromkeys(tags))[:10]


def _get_question_votes(soup: BeautifulSoup) -> int:
    for sel in ["span.vote-count", "span[class*='vote']"]:
        el = soup.select_one(sel)
        if el:
            try:
                return int(re.search(r"-?\d+", el.get_text()).group())
            except (AttributeError, ValueError):
                pass
    return 0


def _extract_id(url: str) -> str:
    m = re.search(r"/questions/(\d+)/", url)
    return m.group(1) if m else url.rstrip("/").split("/")[-1]


# ── Auto-labeling ─────────────────────────────────────────────

def _detect_expected_tool(text: str) -> str:
    tl = text.lower()
    # CYPHER: error/event codes that map to KG entities
    if re.search(
        r"0x[0-9a-f]{4,8}"             # hex: 0x80180014
        r"|(0x[0-9a-f]{4,8})"          # parenthesised: (0x80070002)
        r"|hresult\s*[=:]\s*0x"       # HRESULT: 0x80070005
        r"|kb[0-9]{6,7}"               # KB articles: KB5034441
        r"|error\s+code\s+[0-9]{5,}" # decimal: error code 800180014
        r"|aadsts[0-9]{5,}"            # Azure AD: AADSTS70011
        r"|event\s+id\s+[0-9]+"      # Event ID 1234
        r"|error_[a-z_]+",             # ERROR_ACCESS_DENIED
        tl,
    ):
        return "CYPHER"
    # BFS: relationship / causality questions
    if re.search(
        r"relationship\s+between"
        r"|how\s+does\s+.{3,40}\s+affect"
        r"|depends?\s+on"
        r"|cause\s+of|what\s+causes"
        r"|difference\s+between"
        r"|how\s+is\s+.{3,40}\s+related",
        tl,
    ):
        return "BFS"
    # WEBSEARCH: version / recency questions
    if re.search(
        r"latest|recent\s+update|after\s+(the\s+)?update"
        r"|windows\s+1[12]\s+[0-9]{2}h[0-9]"
        r"|new\s+version|just\s+updated"
        r"|since\s+(the\s+)?update|after\s+upgrading",
        tl,
    ):
        return "WEBSEARCH"
    return "EMBEDDING"


def _detect_query_type(text: str) -> str:
    tl = text.lower()
    if re.search(r"0x[0-9a-f]{4,8}|[0-9]{8,10}", tl): return "error_code"
    if re.search(r"kb[0-9]{6,7}", tl):                       return "kb_number"
    if re.search(r"relationship|cause|affect", tl):           return "relationship"
    if re.search(r"latest|recent|after update", tl):          return "recent_version"
    return "symptom_vague"


def _estimate_difficulty(body: str, answer_count: int) -> str:
    words = len(body.split())
    if words > 200 or answer_count > 5: return "hard"
    if words > 80  or answer_count > 2: return "medium"
    return "easy"


# ── Quality filter ────────────────────────────────────────────

def is_good_qa(item: QAItem, min_votes: int = 0) -> bool:
    if len(item.question) < 20:
        return False
    # Title không được quá dài (tránh lấy nhầm answer text)
    if len(item.question) > 200:
        return False
    best = item.accepted_answer or item.top_answer
    # Answer phải đủ dài và có nội dung kỹ thuật
    if not best or len(best.split()) < 25:
        return False
    if item.votes < min_votes:
        return False
    # Bỏ câu hỏi không phải tiếng Anh
    non_ascii = sum(1 for c in item.question if ord(c) > 127)
    if non_ascii / max(len(item.question), 1) > 0.3:
        return False
    # Bỏ answer chỉ là 1 câu ngắn không có hướng dẫn cụ thể
    if best.count(".") < 2 and len(best.split()) < 40:
        return False
    return True


# ── Convert sang test set format ──────────────────────────────

def to_testset_entry(item: QAItem) -> dict:
    best = item.accepted_answer or item.top_answer
    return {
        # Compatible với test_set.json cũ
        "question":            item.question,
        "answer":              best[:500],
        "article_id":          item.matched_article_id or f"qa_{item.id}",
        "category":            item.category,
        "expected_tool":       item.expected_tool,
        # Fields mới
        "id":                  item.id,
        "query_type":          item.query_type,
        "difficulty":          item.difficulty,
        "has_accepted_answer": item.has_accepted_answer,
        "source":              "microsoft_qa",
        "source_url":          item.url,
        "question_body":       item.question_body[:300],
        "tags":                item.tags,
        "answer_keywords":     _extract_keywords(best),
        # Fill thủ công sau khi map về KG
        "ground_truth_answer": "",
        "matched_article_id":  "",
    }


def _extract_keywords(text: str) -> list[str]:
    kws  = re.findall(r"0x[0-9A-Fa-f]{4,8}|KB\d{6,7}|ERROR_[A-Z_]+", text)
    kws += re.findall(r"[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){1,4}", text)[:5]
    kws += re.findall(r"[A-Z][a-z]+-[A-Z][a-zA-Z]+", text)
    return list(dict.fromkeys(kws))[:10]


# ── Main pipeline ─────────────────────────────────────────────

def run(
    limit: int | None = None,
    dry_run: bool = False,
    output: str = DEFAULT_OUTPUT,
    min_votes: int = 0,
) -> None:
    session   = create_session()
    all_items: list[QAItem] = []
    seen_ids:  set[str]     = set()

    logger.info("=" * 60)
    logger.info("Scrape Microsoft Q&A → Test set")
    logger.info("=" * 60)

    for category, tag_urls in CATEGORY_TAGS.items():
        cat_limit = CATEGORY_LIMITS[category]
        cat_items: list[QAItem] = []

        logger.info(f"\n[{category}] Target: {cat_limit} câu hỏi")

        for tag_url in tag_urls:
            if len(cat_items) >= cat_limit:
                break

            logger.info(f"  Tag: .../{tag_url.split('/')[-1]}")
            q_list = fetch_questions_by_tag_url(session, tag_url, max_pages=4)
            logger.info(f"  → Listing: {len(q_list)} câu hỏi")

            for q in q_list:
                if len(cat_items) >= cat_limit:
                    break
                url   = q.get("url", "")
                qa_id = _extract_id(url)
                if not url or qa_id in seen_ids:
                    continue

                item = scrape_question_detail(session, url, category, tag_url)
                if item is None:
                    continue
                if not is_good_qa(item, min_votes):
                    logger.info(f"  ⬡ Skip: {item.question[:50]}")
                    continue

                seen_ids.add(qa_id)
                cat_items.append(item)

                prefix = "[DRY RUN] " if dry_run else ""
                logger.info(
                    f"  {prefix}✅ [{len(cat_items)}/{cat_limit}] "
                    f"{item.question[:60]}\n"
                    f"       tool={item.expected_tool} | "
                    f"type={item.query_type} | "
                    f"accepted={item.has_accepted_answer} | "
                    f"votes={item.votes}"
                )

        logger.info(f"  [{category}] Thu được: {len(cat_items)}")
        all_items.extend(cat_items)

        if limit and len(all_items) >= limit:
            all_items = all_items[:limit]
            break

    # ── Stats ──────────────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info(f"Tổng: {len(all_items)} câu hỏi")

    tool_dist = {}
    for i in all_items:
        tool_dist[i.expected_tool] = tool_dist.get(i.expected_tool, 0) + 1
    logger.info(f"Tool distribution   : {tool_dist}")
    logger.info(f"Accepted answers    : {sum(1 for i in all_items if i.has_accepted_answer)}/{len(all_items)}")

    if dry_run:
        logger.info("[DRY RUN] Không lưu file.")
        return

    if not all_items:
        logger.warning("Không thu được câu hỏi nào!")
        return

    # ── Lưu raw ───────────────────────────────────────────────
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    Path(output).write_text(
        json.dumps([asdict(i) for i in all_items], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(f"\nRaw      → {output}")

    # ── Lưu test set format ────────────────────────────────────
    ts_path = re.sub(r"_raw\.json$", ".json", output)
    if ts_path == output:
        ts_path = output.replace(".json", "_testset.json")
    Path(ts_path).write_text(
        json.dumps([to_testset_entry(i) for i in all_items], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(f"Test set → {ts_path}")
    logger.info("""
⚠️  REVIEW THỦ CÔNG TRƯỚC KHI DÙNG:
   1. Kiểm tra expected_tool (detect bằng regex, có thể sai)
   2. Fill matched_article_id → article_id trong KG
   3. Xóa câu ngoài 4 domains: DeviceMgmt/Teams/Identity/Network
   4. Xóa câu trùng với test_set.json cũ
""")


# ── CLI ───────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape Microsoft Q&A → test set")
    parser.add_argument("--limit",     type=int, default=None,          help="Tổng số câu tối đa")
    parser.add_argument("--dry-run",   action="store_true",             help="Chỉ log, không lưu")
    parser.add_argument("--output",    type=str, default=DEFAULT_OUTPUT, help="Output path")
    parser.add_argument("--min-votes", type=int, default=0,             help="Votes tối thiểu")
    args = parser.parse_args()
    run(limit=args.limit, dry_run=args.dry_run, output=args.output, min_votes=args.min_votes)