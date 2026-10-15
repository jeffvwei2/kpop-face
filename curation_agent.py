"""
Training-data curation pipeline. `curate_class` (--label) runs search ->
download -> single-face crop -> save, using wiki_search.py for free candidates.
The embedding check against gold anchors (step 4) is still a TODO.

Goal: grow data/Kpop_Faces for classes that have too few images (127 classes,
median 6 images/class, some with just 1 -- see list_underrepresented_classes)
without introducing mislabeled images, which would make things worse, not
better.

Pipeline design (see project notes for the full rationale):
  1. search  - find candidate source pages for a named idol (web search)
  2. label   - only accept images whose *source* explicitly credits the idol
               by name (captions, filenames, gallery structure). We do not
               ask an LLM -- or KpopClassifier -- to visually recognize which
               of ~127 similar faces is in an unlabeled photo. That's an
               unreliable task for a general vision model and circular for
               our own under-trained classifier.
  3. detect  - confirm exactly one face at usable resolution/pose. Reuse the
               mediapipe face detector already set up in api.py
               (crop_to_face) rather than duplicating that logic here.
  4. verify  - embed the candidate face with a *pretrained* general face
               recognition model (e.g. insightface buffalo_l) -- NOT
               KpopClassifier, which isn't accurate enough to gate its own
               training data -- and require it to be similar to a hand-picked
               "gold" anchor set for that class.
  5. save    - resize to match the existing pipeline, append to a CSV, keep
               provenance (source URL) so bad labels can be traced and pulled
               later if needed.

Needed once this moves past scaffolding:
  pip install insightface onnxruntime requests

The search step is provider-agnostic (see SearchProvider below). Pick one
with --provider or CURATION_PROVIDER=openai|anthropic; if neither is set,
whichever of OPENAI_API_KEY / ANTHROPIC_API_KEY is present is used (OpenAI
first). Override the model with --model or CURATION_MODEL. Only the chosen
provider's SDK needs to be installed (`openai` or `anthropic`). Keys can go
in .env (already gitignored).
"""

import csv
import hashlib
import io
import os
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd
from PIL import Image

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

BASE_DIR = Path(__file__).resolve().parent
TRAIN_CSV = BASE_DIR / "data" / "combined_train.csv"
KPOP_FACES_DIR = BASE_DIR / "data" / "Kpop_Faces"
GOLD_ANCHORS_DIR = BASE_DIR / "data" / "gold_anchors"
CURATED_CSV = BASE_DIR / "data" / "curated_train.csv"  # new images land here first,
# kept separate from combined_train.csv until spot-checked and merged manually.
CURATION_LOG = BASE_DIR / "data" / "curation_log.csv"  # every processed URL and its outcome,
# so re-runs skip images already accepted or rejected instead of downloading them again.

CROP_SIZE = 256          # training resizes to 256 anyway; storing larger only wastes space
JPEG_QUALITY = 90        # ~15-25 KB per crop
FACE_PAD_RATIO = 0.4     # matches the framing of the existing Kpop_Faces images
MIN_FACE_PX = 80         # smaller faces upscale into blurry crops
MIN_FACE_SCORE = 0.6     # detections below this are ignored when counting faces
MAX_DOWNLOAD_BYTES = 15_000_000

DEFAULT_MIN_IMAGES = 30
DEFAULT_SIMILARITY_THRESHOLD = 0.5


def list_underrepresented_classes(min_images: int = DEFAULT_MIN_IMAGES) -> list[tuple[str, int]]:
    """Classes in the current training set with fewer than `min_images`
    images, sorted worst-first. This is the real starting point for the
    agent -- run it to decide which labels to curate first."""
    df = pd.read_csv(TRAIN_CSV)
    counts = df["name"].value_counts()
    short = counts[counts < min_images].sort_values()
    return list(short.items())


def label_to_group_and_person(label: str) -> tuple[str | None, str]:
    """Splits a combined_train.csv label back into (group, person).
    Labels are disambiguated as "<group>_<person>" only when the person name
    is ambiguous across groups (see build_kpop_faces_csv.py); otherwise the
    label is just the person name and the group is unknown from the label
    alone.

    For those, the group is looked up from the data/Kpop_Faces/<group>/<person>
    folders. Returns group None when there's no folder (idols only in the
    older kid_f dataset) or more than one.
    """
    if "_" in label:
        group, person = label.split("_", 1)
        if (KPOP_FACES_DIR / group).is_dir():
            return group, person
    groups = [d.parent.name for d in KPOP_FACES_DIR.glob(f"*/{label}") if d.is_dir()]
    return (groups[0] if len(groups) == 1 else None), label


SEARCH_PROMPT = (
    "Find web pages with photos of the K-pop idol {who}, where each photo is "
    "explicitly credited to them by name: official group/agency galleries, "
    "fan wiki profile pages, and press/news photo sets. Prefer solo photos "
    "over group photos. Avoid pages about other idols with the same stage "
    "name. Cite every page you recommend."
)


class SearchProvider:
    """Web-search backend for search_candidate_sources. Subclasses wrap one
    LLM API's hosted web search tool and return the pages it cited as
    [{"url": ..., "title": ...}, ...]. Add a new backend by subclassing and
    registering it in PROVIDERS."""

    env_key: str = ""
    default_model: str = ""

    def __init__(self, model: str | None = None):
        self.model = model or os.environ.get("CURATION_MODEL") or self.default_model

    def search(self, prompt: str) -> list[dict]:
        raise NotImplementedError


class OpenAISearchProvider(SearchProvider):
    env_key = "OPENAI_API_KEY"
    default_model = "gpt-5"

    def search(self, prompt: str) -> list[dict]:
        from openai import OpenAI

        response = OpenAI().responses.create(
            model=self.model,
            tools=[{"type": "web_search"}],
            input=prompt,
        )
        results = []
        for item in response.output:
            if item.type != "message":
                continue
            for part in item.content:
                for ann in getattr(part, "annotations", None) or []:
                    if ann.type == "url_citation":
                        results.append({"url": ann.url, "title": ann.title})
        return results


class AnthropicSearchProvider(SearchProvider):
    env_key = "ANTHROPIC_API_KEY"
    default_model = "claude-opus-5-5"

    def search(self, prompt: str) -> list[dict]:
        import anthropic

        client = anthropic.Anthropic()
        messages = [{"role": "user", "content": prompt}]
        results = []
        # Server-side search can pause long turns; resume up to a few times.
        for _ in range(5):
            response = client.beta.messages.create(
                model=self.model,
                max_tokens=16000,
                tools=[{"type": "web_search_20260209", "name": "web_search", "max_uses": 5}],
                # Retry on a fallback model if the request is declined.
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                messages=messages,
            )
            for block in response.content:
                # An error result's content is an object, not a list.
                if block.type == "web_search_tool_result" and isinstance(block.content, list):
                    results.extend({"url": r.url, "title": r.title} for r in block.content)
            if response.stop_reason != "pause_turn":
                break
            messages.append({"role": "assistant", "content": response.content})
        return results


PROVIDERS: dict[str, type[SearchProvider]] = {
    "openai": OpenAISearchProvider,
    "anthropic": AnthropicSearchProvider,
}


def get_search_provider(name: str | None = None, model: str | None = None) -> SearchProvider:
    """Resolve the provider from the explicit name, then CURATION_PROVIDER,
    then whichever API key is set."""
    name = name or os.environ.get("CURATION_PROVIDER")
    if not name:
        name = next((n for n, cls in PROVIDERS.items() if os.environ.get(cls.env_key)), None)
    if name not in PROVIDERS:
        keys = " or ".join(cls.env_key for cls in PROVIDERS.values())
        raise RuntimeError(
            f"No search provider available (got {name!r}). Set {keys}, "
            f"or pass --provider {'|'.join(PROVIDERS)}."
        )
    return PROVIDERS[name](model)


def search_candidate_sources(
    label: str,
    group: str | None = None,
    max_results: int = 10,
    provider: SearchProvider | None = None,
) -> list[dict]:
    """Find web pages likely to contain photos of this idol, explicitly
    credited to them by name (official galleries, fan wikis, press photos).
    Returns [{"url": ..., "title": ...}, ...], deduplicated by URL. The group
    name is included when known to disambiguate common stage names."""
    provider = provider or get_search_provider()
    who = f"{label} of {group}" if group else label
    seen, results = set(), []
    for r in provider.search(SEARCH_PROMPT.format(who=who)):
        if r["url"] not in seen:
            seen.add(r["url"])
            results.append(r)
    return results[:max_results]


def extract_labeled_image_urls(source: dict, label: str) -> list[str]:
    """From a candidate source page, extract image URLs whose caption,
    alt-text, filename, or surrounding page structure explicitly credits
    `label`. This is the identity signal -- named context from the source,
    not visual recognition.

    TODO: fetch the page (web_fetch tool or requests), parse for images with
    matching credited text. Skip anything ambiguous (group photos without
    per-person tagging, uncredited galleries) rather than guessing.
    """
    raise NotImplementedError


MAX_RATE_LIMIT_WAIT = 30  # seconds; longer waits skip the host for the rest of the run


def download_image(url: str) -> Image.Image | None:
    """Download into memory (never to disk) and decode as RGB. Returns None
    if the file isn't a usable image. Raises wiki_search.RateLimited if the
    host wants a long back-off, and requests.RequestException on network
    errors -- both transient, so the caller shouldn't mark the URL as tried."""
    from wiki_search import polite_get  # same User-Agent and rate limiting as the search

    resp = polite_get(url, {}, max_wait=MAX_RATE_LIMIT_WAIT)
    if not resp.headers.get("Content-Type", "").startswith("image/"):
        return None
    if len(resp.content) > MAX_DOWNLOAD_BYTES:
        return None
    try:
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception:
        return None


def detect_single_face(image: Image.Image) -> tuple[Image.Image | None, str]:
    """Square face crop resized to CROP_SIZE, or (None, reason). Rejects images
    with zero or several faces, since group and candid photos aren't safe to
    label automatically, and faces too small to give a sharp crop."""
    from face_crop import crop_box, detect_faces

    faces = [d for d in detect_faces(image) if d.categories[0].score >= MIN_FACE_SCORE]
    if not faces:
        return None, "no face"
    if len(faces) > 1:
        return None, f"{len(faces)} faces"
    box = faces[0].bounding_box
    if min(box.width, box.height) < MIN_FACE_PX:
        return None, "face too small"
    crop = crop_box(image, box, FACE_PAD_RATIO, square=True)
    return crop.resize((CROP_SIZE, CROP_SIZE), Image.LANCZOS), "ok"


def embed_face(image: Image.Image):
    """Embed a cropped face with a pretrained, general-purpose face
    recognition model -- NOT KpopClassifier.

    TODO: load a pretrained insightface (buffalo_l) or similar model once at
    module import time, run inference here, return the embedding vector.
    """
    raise NotImplementedError


def load_gold_anchor_embeddings(label: str):
    """Embeddings for the hand-picked reference images at
    data/gold_anchors/<label>/*.jpg -- the trusted anchor set a candidate
    image must match to be accepted.

    TODO: embed each image in the folder (via embed_face), cache to disk
    (e.g. data/gold_anchors/<label>/.embeddings.npy) so this isn't redone
    every run. Raise a clear error if the folder doesn't exist yet --
    curation for a class shouldn't silently proceed without anchors.

    Setup note: this folder does not exist yet. Before curating a class, pick
    5-10 images you're confident are correctly labeled (ideally from official
    sources) and drop them in data/gold_anchors/<label>/.
    """
    raise NotImplementedError


def is_verified_match(candidate_embedding, anchor_embeddings, threshold: float = DEFAULT_SIMILARITY_THRESHOLD) -> bool:
    """TODO: cosine similarity between candidate_embedding and each anchor
    embedding; accept if similarity to the anchor set (max or mean -- decide
    after looking at real score distributions) clears `threshold`."""
    raise NotImplementedError


class CurationReport:
    """Counts through the pipeline plus why each candidate was rejected."""

    def __init__(self, label: str):
        self.label = label
        self.found = 0
        self.skipped_seen = 0
        self.downloaded = 0
        self.passed_face_detect = 0
        self.saved = 0
        self.rejected: dict[str, int] = defaultdict(int)

    def summary(self) -> str:
        reasons = ", ".join(f"{r}: {n}" for r, n in sorted(self.rejected.items(), key=lambda x: -x[1]))
        return (f"{self.label}: {self.found} candidates, {self.skipped_seen} already processed, "
                f"{self.downloaded} downloaded, {self.passed_face_detect} single-face, "
                f"{self.saved} saved" + (f"\n  rejected - {reasons}" if reasons else ""))


def _append_csv(path: Path, row: dict):
    is_new = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def _seen_urls() -> set[str]:
    if not CURATION_LOG.exists():
        return set()
    with open(CURATION_LOG, newline="") as f:
        return {row["image_url"] for row in csv.DictReader(f)}


def _curated_count(label: str) -> int:
    if not CURATED_CSV.exists():
        return 0
    with open(CURATED_CSV, newline="") as f:
        return sum(row["name"] == label for row in csv.DictReader(f))


def curate_class(label: str, target_new_images: int = 20, group: str | None = None) -> CurationReport:
    """Search -> download -> single-face crop -> save, for one class, until it
    has target_new_images curated images in total (earlier runs count).

    Candidates come from wiki_search.py (free; labeled by the wiki's own
    structure). Images are downloaded into memory and only the 256x256 face
    crop is written, under data/Kpop_Faces/<group>/<person>/curated_*.jpg.
    Accepted rows go to CURATED_CSV, not combined_train.csv, so nothing
    reaches training until reviewed and merged.

    Not done yet: the embedding check against data/gold_anchors (embed_face /
    is_verified_match). Rows are saved with verified=False so they can be
    filtered once that exists.
    """
    import requests
    import wiki_search

    found_group, person = label_to_group_and_person(label)
    group = group or found_group
    if not group:
        raise SystemExit(f"Can't tell which group {label!r} is in; pass --group.")
    out_dir = KPOP_FACES_DIR / group / person
    out_dir.mkdir(parents=True, exist_ok=True)

    report = CurationReport(label)
    candidates = wiki_search.search(label, person, group, list(wiki_search.SOURCES))
    report.found = len(candidates)
    # Galleries list shoots in order, so taking the first N would mean N shots
    # from one photoshoot. Shuffle (fixed seed, so re-runs see the same order).
    random.Random(label).shuffle(candidates)

    seen = _seen_urls()
    already = _curated_count(label)
    if already >= target_new_images:
        print(f"{label}: already has {already} curated images (target {target_new_images}); nothing to do")
        return report
    blocked_hosts = set()
    for c in candidates:
        if already + report.saved >= target_new_images:
            break
        url = c["image_url"]
        if url in seen:
            report.skipped_seen += 1
            continue
        host = requests.utils.urlparse(url).netloc
        if host in blocked_hosts:
            report.rejected["host rate-limited (retry later)"] += 1
            continue

        try:
            image = download_image(url)
        except wiki_search.RateLimited as e:
            print(f"  {e}; skipping its images for the rest of this run")
            blocked_hosts.add(host)
            report.rejected["host rate-limited (retry later)"] += 1
            continue  # not logged, so a later run retries it
        except requests.RequestException:
            report.rejected["network error (retry later)"] += 1
            continue
        if image is None:
            status, file_name = "not an image", ""
        else:
            report.downloaded += 1
            crop, status = detect_single_face(image)
            file_name = ""
            if crop is not None:
                report.passed_face_detect += 1
                path = out_dir / f"curated_{hashlib.sha1(url.encode()).hexdigest()[:12]}.jpg"
                crop.save(path, "JPEG", quality=JPEG_QUALITY)
                file_name = str(path.relative_to(BASE_DIR / "data"))  # same form as combined_train.csv
                _append_csv(CURATED_CSV, {
                    "file_name": file_name, "name": label, "source_url": url,
                    "source_page": c["source_page"], "verified": False,
                })
                report.saved += 1
        if status != "ok":
            report.rejected[status] += 1
        _append_csv(CURATION_LOG, {"label": label, "image_url": url, "status": status, "file_name": file_name})

    print(report.summary())
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--list-underrepresented",
        action="store_true",
        help="Print classes with fewer than --min-images training images and exit.",
    )
    parser.add_argument("--min-images", type=int, default=DEFAULT_MIN_IMAGES)
    parser.add_argument("--label", help="Curate a single class, e.g. --label yujin")
    parser.add_argument("--target", type=int, default=20, help="Total curated images to reach for the class (earlier runs count)")
    parser.add_argument("--group", help="Group, if it can't be found from data/Kpop_Faces folders")
    parser.add_argument("--provider", choices=sorted(PROVIDERS), help="Search backend (default: CURATION_PROVIDER or whichever API key is set)")
    parser.add_argument("--model", help="Override the provider's default model (or set CURATION_MODEL)")
    parser.add_argument("--search-only", action="store_true", help="With --label, just print candidate source pages and exit")
    args = parser.parse_args()

    if args.list_underrepresented or not args.label:
        short = list_underrepresented_classes(args.min_images)
        print(f"{len(short)} classes with fewer than {args.min_images} images:")
        for name, count in short:
            print(f"  {name}: {count}")
    elif args.search_only:
        provider = get_search_provider(args.provider, args.model)
        group, person = label_to_group_and_person(args.label)
        print(f"Searching with {type(provider).__name__} ({provider.model})...")
        for r in search_candidate_sources(person, group, provider=provider):
            print(f"  {r['title']}\n    {r['url']}")
    else:
        curate_class(args.label, target_new_images=args.target, group=args.group)
