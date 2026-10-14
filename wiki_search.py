"""
Free, LLM-free candidate image search for the curation pipeline.

Queries two MediaWiki sites whose structure already labels images by person
(the "label" step in curation_agent.py), so no API credits are spent:

  - Wikimedia Commons: the idol's person category (e.g. "Category:An Yu-jin"),
    crawled a few levels into subcategories like "... by year". Images are
    freely licensed.
  - Kpop Fandom wiki: the idol's "<Page>/Gallery" page. Only files whose
    name contains the idol's page title are kept, which drops group shots
    and unrelated images on the page.

The group is used to disambiguate stage names (e.g. IVE's Yujin vs
Kep1er's Yujin): a Commons category must be filed under "Members of <group>"
and a Fandom page must be in the "<group>" category. Without --group the
first search hit is taken, which can be the wrong person.

Results are cached as JSON in data/candidates/<label>.json (one entry per
image, with source page for provenance). These are still only *candidates*:
detect/verify in curation_agent.py should gate them before training.

Usage:
  python wiki_search.py yujin --group ive
  python wiki_search.py wonyoung --group ive --source fandom --limit 100
  python wiki_search.py yujin --group ive --refresh   # ignore cache
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import requests

BASE_DIR = Path(__file__).resolve().parent
CANDIDATES_DIR = BASE_DIR / "data" / "candidates"

COMMONS_API = "https://commons.wikimedia.org/w/api.php"
FANDOM_API = "https://kpop.fandom.com/api.php"
# Wikimedia asks API clients for a descriptive User-Agent.
USER_AGENT = "kpop-classifier-curation/0.1 (personal research; python-requests)"

IMAGE_MIMES = {"image/jpeg", "image/png", "image/webp"}
MIN_SIDE = 256          # smaller images rarely give a usable face crop
THUMB_WIDTH = 1024      # download a resized thumbnail, not the multi-MB original
COMMONS_MAX_DEPTH = 3   # person -> "by year" -> "in 2023" is depth 2

MIN_INTERVAL = 1.0      # seconds between requests; Commons rate-limits bursts
MAX_RETRIES = 5

session = requests.Session()
session.headers["User-Agent"] = USER_AGENT
_last_request = 0.0


def polite_get(url: str, params: dict) -> requests.Response:
    """GET with a fixed delay between requests and backoff on 429/503,
    honoring Retry-After when the server sends one."""
    global _last_request
    for attempt in range(MAX_RETRIES):
        time.sleep(max(0.0, _last_request + MIN_INTERVAL - time.monotonic()))
        _last_request = time.monotonic()
        resp = session.get(url, params=params, timeout=30)
        if resp.status_code not in (429, 503):
            resp.raise_for_status()
            return resp
        retry_after = resp.headers.get("Retry-After", "")
        time.sleep(int(retry_after) if retry_after.isdigit() else 5 * 2 ** attempt)
    resp.raise_for_status()
    return resp


def norm(text: str) -> str:
    """Lowercase alphanumerics only: "IZ*ONE" -> "izone", "An Yu-jin" -> "anyujin"."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def api_query(api: str, **params) -> list[dict]:
    """Run a MediaWiki action=query, following continuation, and return all
    `query.pages` entries (or list results when no generator is used)."""
    params = {"action": "query", "format": "json", "formatversion": "2", **params}
    out = []
    while True:
        data = polite_get(api, params).json()
        query = data.get("query", {})
        for key in ("pages", "search", "categorymembers"):
            out.extend(query.get(key, []))
        if "continue" not in data:
            return out
        params.update(data["continue"])


def image_entries(pages: list[dict], source: str, source_page: str) -> list[dict]:
    entries = []
    for page in pages:
        info = (page.get("imageinfo") or [{}])[0]
        if info.get("mime") not in IMAGE_MIMES:
            continue
        if min(info.get("width", 0), info.get("height", 0)) < MIN_SIDE:
            continue
        entries.append({
            "image_url": info.get("thumburl") or info["url"],
            "file_title": page["title"],
            "width": info["width"],
            "height": info["height"],
            "source": source,
            "source_page": source_page,
        })
    return entries


IMAGEINFO = {"prop": "imageinfo", "iiprop": "url|size|mime", "iiurlwidth": THUMB_WIDTH}


# --- Wikimedia Commons -------------------------------------------------------

def find_commons_category(person: str, group: str | None) -> str | None:
    query = f"{person} {group}" if group else person
    hits = api_query(COMMONS_API, list="search", srsearch=query, srnamespace=14, srlimit=10)
    for hit in hits:
        title = hit["title"]
        if group and norm(title) == norm(f"Category:{group}"):
            continue  # the group's own category
        if not group:
            return title
        parents = api_query(COMMONS_API, titles=title, prop="categories", cllimit="max")
        parent_names = {norm(c["title"].split(":", 1)[1]) for p in parents for c in p.get("categories", [])}
        if norm(f"members of {group}") in parent_names:
            return title
    return None


def search_commons(person: str, group: str | None) -> list[dict]:
    root = find_commons_category(person, group)
    if not root:
        print(f"  commons: no person category found for {person!r} ({group})")
        return []
    print(f"  commons: using {root}")
    url = f"https://commons.wikimedia.org/wiki/{root.replace(' ', '_')}"

    entries, seen_cats, frontier = [], {root}, [root]
    for _ in range(COMMONS_MAX_DEPTH):
        next_frontier = []
        for cat in frontier:
            files = api_query(COMMONS_API, generator="categorymembers", gcmtitle=cat,
                              gcmtype="file", gcmlimit="max", **IMAGEINFO)
            entries.extend(image_entries(files, "commons", url))
            for sub in api_query(COMMONS_API, list="categorymembers", cmtitle=cat,
                                 cmtype="subcat", cmlimit="max"):
                if sub["title"] not in seen_cats:
                    seen_cats.add(sub["title"])
                    next_frontier.append(sub["title"])
        frontier = next_frontier
    return entries


# --- Kpop Fandom wiki --------------------------------------------------------

def find_fandom_page(person: str, group: str | None) -> str | None:
    query = f"{person} {group}" if group else person
    hits = api_query(FANDOM_API, list="search", srsearch=query, srnamespace=0, srlimit=10)
    for hit in hits:
        title = hit["title"]
        if "/" in title or (group and norm(title) == norm(group)):
            continue  # subpages and the group's own page
        if not group:
            return title
        pages = api_query(FANDOM_API, titles=title, prop="categories", cllimit="max")
        cats = {norm(c["title"].split(":", 1)[1]) for p in pages for c in p.get("categories", [])}
        if norm(group) in cats:
            return title
    return None


def search_fandom(person: str, group: str | None) -> list[dict]:
    page = find_fandom_page(person, group)
    if not page:
        print(f"  fandom: no page found for {person!r} ({group})")
        return []
    gallery = f"{page}/Gallery"
    print(f"  fandom: using {gallery}")
    files = api_query(FANDOM_API, generator="images", titles=gallery, gimlimit="max", **IMAGEINFO)
    # Gallery pages also embed group photos and icons; keep files named after
    # the idol. Drop disambiguators like "(IVE)" / "(singer)" from the title,
    # since filenames read "IVE Liz ..." or "HyunA ...".
    names = {norm(re.sub(r"\(.*?\)", "", page)), norm(person)} - {""}
    files = [f for f in files if any(n in norm(f["title"]) for n in names)]
    return image_entries(files, "fandom", f"https://kpop.fandom.com/wiki/{gallery.replace(' ', '_')}")


SOURCES = {"commons": search_commons, "fandom": search_fandom}


def search(label: str, person: str, group: str | None, sources: list[str],
           refresh: bool = False) -> list[dict]:
    """Candidates for one class, from cache unless `refresh`. Deduplicated by
    image URL. Cache key is the training label so it lines up with
    combined_train.csv."""
    cache = CANDIDATES_DIR / f"{label}.json"
    if cache.exists() and not refresh:
        cached = json.loads(cache.read_text())
        if set(sources) <= set(cached["sources"]):
            print(f"  cached: {cache.relative_to(BASE_DIR)} (use --refresh to re-query)")
            return [c for c in cached["candidates"] if c["source"] in sources]

    seen, candidates, succeeded = set(), [], []
    for name in sources:
        try:
            found = SOURCES[name](person, group)
        except requests.RequestException as e:
            # Not recorded as searched, so the next run retries this source.
            print(f"  {name}: request failed ({e}); skipping")
            continue
        succeeded.append(name)
        for c in found:
            if c["image_url"] not in seen:
                seen.add(c["image_url"])
                candidates.append(c)

    CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(
        {"label": label, "person": person, "group": group, "sources": succeeded, "candidates": candidates},
        indent=2,
    ))
    return candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("label", help="Training label, e.g. yujin or ive_yujin")
    parser.add_argument("--group", help="Group name (strongly recommended; disambiguates stage names)")
    parser.add_argument("--person", help="Search name if different from the label, e.g. 'An Yujin'")
    parser.add_argument("--source", choices=[*SOURCES, "all"], default="all")
    parser.add_argument("--limit", type=int, default=20, help="How many results to print (all are cached)")
    parser.add_argument("--refresh", action="store_true", help="Ignore the cache and re-query")
    args = parser.parse_args()

    group, person = args.group, args.person or args.label
    if not group and "_" in args.label:
        group, person = args.label.split("_", 1)
    if not group:
        print("warning: no --group given; the first search hit may be a different idol", file=sys.stderr)

    sources = list(SOURCES) if args.source == "all" else [args.source]
    print(f"Searching {', '.join(sources)} for {person!r} ({group or 'no group'})...")
    candidates = search(args.label, person, group, sources, refresh=args.refresh)

    by_source = {s: sum(c["source"] == s for c in candidates) for s in sources}
    print(f"\n{len(candidates)} candidates ({', '.join(f'{s}: {n}' for s, n in by_source.items())})")
    for c in candidates[:args.limit]:
        print(f"  [{c['source']}] {c['file_title']}  {c['width']}x{c['height']}")
    if len(candidates) > args.limit:
        print(f"  ... {len(candidates) - args.limit} more in data/candidates/{args.label}.json")


if __name__ == "__main__":
    main()
