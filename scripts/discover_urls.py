import requests
import json
import time
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
OUTPUT_FILE = os.path.join(PROJECT_ROOT, 'data', 'discovered_urls.json')

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

BASE = "https://learn.microsoft.com"

TOC_URLS = {
    "Network": [
        "https://learn.microsoft.com/en-us/troubleshoot/windows-client/toc.json",
        "https://learn.microsoft.com/en-us/troubleshoot/windows-server/toc.json",
    ],
    "Teams": [
        "https://learn.microsoft.com/en-us/troubleshoot/microsoftteams/toc.json",
    ],
    "Identity": [
        "https://learn.microsoft.com/en-us/troubleshoot/entra/toc.json",
    ],
    "DeviceMgmt": [
        "https://learn.microsoft.com/en-us/troubleshoot/mem/intune/toc.json",
        "https://learn.microsoft.com/en-us/troubleshoot/mem/configmgr/toc.json",
    ],
}

CATEGORY_LIMITS = {
    "Network":    100,
    "Teams":      78,
    "Identity":   100,
    "DeviceMgmt": 100,
}

IT_KEYWORDS = [
    "network", "wifi", "wireless", "connection", "internet",
    "printer", "print", "driver", "hardware", "performance",
    "slow", "crash", "freeze", "update", "install",
    "login", "password", "account", "authentication", "sign-in",
    "email", "outlook", "mail", "teams", "meeting",
    "sharepoint", "onedrive", "office", "microsoft-365",
    "error", "fix", "troubleshoot", "cannot", "failed",
    "remote", "vpn", "firewall", "proxy", "dns",
    "memory", "disk", "storage", "cpu", "boot",
    "device", "enroll", "policy", "compliance", "intune",
    "identity", "access", "permission", "license", "token",
]

SKIP_PATTERNS = [
    "welcome", "overview", "toc", "release-notes",
    "known-issues-overview", "experts-welcome",
]


def is_relevant_article(url: str) -> bool:
    if not url or 'learn.microsoft.com' not in url:
        return False
    path = url.rstrip('/').split('?')[0]
    last = path.split('/')[-1].lower()
    if len(last) < 10:
        return False
    if any(s in last for s in SKIP_PATTERNS):
        return False
    return any(kw in last for kw in IT_KEYWORDS)


def extract_urls_from_toc(toc_data, base_path: str) -> list:
    urls = []

    def recurse(node):
        href = node.get('href', '')
        if href:
            if href.startswith('http'):
                full_url = href
            elif href.startswith('/'):
                full_url = BASE + href
            else:
                full_url = base_path + href
            full_url = full_url.split('?')[0].split('#')[0]
            if is_relevant_article(full_url) and full_url not in urls:
                urls.append(full_url)

        for child in node.get('items', []):
            recurse(child)
        for child in node.get('children', []):
            recurse(child)

    if isinstance(toc_data, dict):
        for item in toc_data.get('items', []):
            recurse(item)
    elif isinstance(toc_data, list):
        for item in toc_data:
            recurse(item)

    return urls


def fetch_toc(toc_url: str) -> list:
    base_path = toc_url.rsplit('/', 1)[0] + '/'
    print(f"  → {toc_url.split('troubleshoot/')[-1]}...")
    try:
        resp = requests.get(toc_url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            print(f"     ❌ HTTP {resp.status_code}")
            return []
        toc_data = resp.json()
        urls = extract_urls_from_toc(toc_data, base_path)
        print(f"     ✅ {len(urls)} bài IT phù hợp")
        return urls
    except Exception as e:
        print(f"     ❌ {str(e)[:60]}")
        return []


def discover_all():
    print("=" * 60)
    print("Discover URLs từ Microsoft Learn")
    print("=" * 60)

    all_urls = {}
    total = 0

    for category, toc_pages in TOC_URLS.items():
        print(f"\n[{category}]")
        cat_urls = []
        for toc_url in toc_pages:
            urls = fetch_toc(toc_url)
            for u in urls:
                if u not in cat_urls:
                    cat_urls.append(u)
            time.sleep(1)

        limit = CATEGORY_LIMITS.get(category, 100)
        cat_urls = cat_urls[:limit]
        all_urls[category] = cat_urls
        total += len(cat_urls)
        print(f"   → Chọn {len(cat_urls)} bài")

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_urls, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Tong: {total} URLs")
    print(f"Saved: {OUTPUT_FILE}")
    print("\nPreview:")
    for cat, urls in all_urls.items():
        print(f"  {cat} ({len(urls)}):")
        for u in urls[:2]:
            print(f"    - {u.split('/')[-1][:60]}")

    return all_urls


if __name__ == "__main__":
    discover_all()