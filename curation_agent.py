"""
Scaffolding for the training-data curation agent. Nothing here runs an actual
web search or downloads anything yet -- see the TODOs. `list_underrepresented_classes`
is real and works today.

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

import os
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

    TODO: this currently can't recover the group for non-ambiguous labels
    (e.g. "yujin" could be Ive's Yujin or IZ*ONE's Yujin). Cross-reference
    against data/Kpop_Faces/*/<person> directories to disambiguate, or store
    group alongside name in the CSV going forward.
    """
    if "_" in label:
        group, person = label.split("_", 1)
        if (KPOP_FACES_DIR / group).is_dir():
            return group, person
    return None, label


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


def download_image(url: str) -> Image.Image | None:
    """TODO: download with a timeout, verify it's actually image content,
    return None (not raise) on any failure so one bad URL doesn't kill a
    curation run."""
    raise NotImplementedError


def detect_single_face(image: Image.Image) -> Image.Image | None:
    """Confirm exactly one face is present and crop to it.

    TODO: reuse api.py's crop_to_face / the mediapipe blaze_face detector
    (models/blaze_face_short_range.tflite) instead of reimplementing face
    detection here. Factor that out of api.py into a shared module if it's
    still only defined inline there. Return None if zero or >1 faces found
    (group/candid photos aren't safe to auto-label).
    """
    raise NotImplementedError


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
    """TODO: tally counts through the pipeline (found / downloaded /
    passed face-detect / passed embedding-gate / saved) plus per-image
    provenance (source URL) for later spot-checking. Print or return a
    summary from curate_class."""

    def __init__(self, label: str):
        self.label = label
        self.found = 0
        self.downloaded = 0
        self.passed_face_detect = 0
        self.passed_embedding_gate = 0
        self.saved = 0
        self.rejected: list[dict] = []  # TODO: record why each rejection happened


def curate_class(label: str, target_new_images: int = 20) -> CurationReport:
    """Orchestrates the full pipeline for one class: search -> label-filter
    -> download -> face-detect -> embedding-verify -> save.

    TODO: wire the functions above together. Stop once target_new_images are
    accepted or candidates run out. Append accepted rows to CURATED_CSV
    (file_name, name, source_url) -- do not write directly into
    combined_train.csv. Save image files under
    data/Kpop_Faces/<group>/<person>/ following the existing layout.
    """
    group, person = label_to_group_and_person(label)
    report = CurationReport(label)
    raise NotImplementedError(f"curate_class scaffolding only -- label={label!r} group={group!r}")


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
    parser.add_argument("--target", type=int, default=20, help="New images to accept before stopping")
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
        curate_class(args.label, target_new_images=args.target)
