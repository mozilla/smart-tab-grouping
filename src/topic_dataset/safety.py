"""Safety labels shared by topic-model training and evaluation."""

from __future__ import annotations

import re


DISSIMILAR_LABEL = "None"
ADULT_LABEL = "Adult Content"
ABSTENTION_LABELS = frozenset({DISSIMILAR_LABEL.casefold(), ADULT_LABEL.casefold()})

# These are explicit labels already present in Mozilla's Common Crawl topic data.
# Keep this list label-based so benign pages containing an ambiguous keyword are
# not converted to adult-content examples.
ADULT_SOURCE_LABELS = frozenset(
    {
        "Adult Content",
        "Adult Entertainment",
        "Adult Live Cams",
        "Adult Live Streams",
        "Adult Movies",
        "Adult Products",
        "Adult Services",
        "Adult Toys",
        "Adult Webcams",
        "Escort Services",
        "Escort Services in Tynemouth",
        "Gay Adult Content",
        "Live Cam Chat",
        "Sex",
        "Sex Toy Materials",
    }
)


def normalize_label(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip().casefold()


def is_abstention_label(value: str) -> bool:
    return normalize_label(value) in ABSTENTION_LABELS


def firefox_visible_label(value: str) -> str:
    """Apply Firefox's sentinel post-processing to a raw model prediction."""

    cleaned = re.sub(r"\s+", " ", value or "").strip()
    return "" if is_abstention_label(cleaned) else cleaned


def canonical_safety_target(value: str) -> str | None:
    """Return a Firefox sentinel for explicitly curated safety-source labels."""

    cleaned = re.sub(r"\s+", " ", value or "").strip()
    if normalize_label(cleaned) == DISSIMILAR_LABEL.casefold():
        return DISSIMILAR_LABEL
    if cleaned in ADULT_SOURCE_LABELS:
        return ADULT_LABEL
    return None
