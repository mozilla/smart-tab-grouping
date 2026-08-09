"""Build a leakage-safe, Firefox-compatible topic generation dataset.

The synthetic source data describes whole browsing windows and task groups.
This builder recovers those full groups, creates realistic group-size variants,
recomputes the exact keywords Firefox will provide, and splits by browsing
window. It also adds curated single-tab labels and synthetic unrelated groups.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

from topic_dataset.firefox_text import create_model_input, extract_group_keywords
from topic_dataset.safety import (
    ADULT_LABEL,
    DISSIMILAR_LABEL,
    canonical_safety_target,
)


MAX_LABEL_CHARACTERS = 35
MAX_LABEL_WORDS = 5
DEFAULT_SEED = 42

GENERIC_TASK_LABELS = {
    "Browse Social Media": "Social Media",
    "Social Media": "Social Media",
    "Using web based work applications": "Work Apps",
    "Online banking and Financial Management": "Banking and Finance",
    "Participating in online forums and communities": "Online Forums",
    "Email Management": "Email",
    "Streaming Videos": "Videos",
    "Researching or Booking Travel": "Travel",
    "Reading News": "News",
    "Reading Articles": "Articles",
    "Online Shopping": "Shopping",
}

TASK_PREFIXES = (
    "Reading Articles about ",
    "Streaming Videos about ",
    "Participating in online forums and communities about ",
    "Online Shopping for ",
    "Online research for ",
    "Researching or Booking Travel for ",
)

LABEL_REPLACEMENTS = {
    "Diplomacy and International Relations": "International Relations",
    "Air Conditioners and Heaters": "Heating and Cooling",
    "Audio Equipment": "Audio Equipment",
    "Gaming Consoles and Accessories": "Gaming Consoles",
    "Online Purchases and Deliveries": "Orders and Deliveries",
    "Politics and Government": "Politics",
    "Startups and Entrepreneurship": "Startups",
    "Visa and Passport Updates": "Visas and Passports",
    "Webinar Links and Study Resources": "Webinar Resources",
}

REFUSAL_PATTERNS = (
    "i cannot",
    "i can't",
    "i am unable",
    "i'm unable",
    "as an ai",
    "sorry",
)

CSV_FIELDS = (
    "example_id",
    "base_group_id",
    "split_key",
    "split",
    "source",
    "variant",
    "tab_count",
    "input_titles",
    "input_keywords",
    "input_text",
    "output",
    "original_task",
)


@dataclass(frozen=True)
class SyntheticGroup:
    source_file: str
    window_id: str
    task_id: str
    task: str
    titles: tuple[str, ...]

    @property
    def base_group_id(self) -> str:
        return f"synthetic:{self.source_file}:{self.task_id}"

    @property
    def split_key(self) -> str:
        return f"synthetic-window:{self.source_file}:{self.window_id}"


@dataclass(frozen=True)
class TopicExample:
    example_id: str
    base_group_id: str
    split_key: str
    split: str
    source: str
    variant: str
    tab_count: int
    input_titles: str
    input_keywords: str
    input_text: str
    output: str
    original_task: str


def normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def normalize_title(value: str) -> str:
    return normalize_space(value).casefold()


def stable_number(value: str, seed: int = DEFAULT_SEED) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def assign_split(split_key: str, seed: int = DEFAULT_SEED) -> str:
    bucket = stable_number(split_key, seed) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "dev"
    return "test"


def compact_label(value: str) -> str:
    """Normalize a label to the deployed model's intentionally small surface."""

    label = normalize_space(value)
    label = re.sub(r"\s*\([^)]*\)\s*", " ", label)
    label = normalize_space(label).strip(" .,:;-_")
    label = LABEL_REPLACEMENTS.get(label, label)

    if len(label) <= MAX_LABEL_CHARACTERS and len(label.split()) <= MAX_LABEL_WORDS:
        return label

    words = label.split()
    while words and words[-1].casefold() in {"updates", "management", "resources"}:
        words.pop()
    words = words[:MAX_LABEL_WORDS]
    while words and len(" ".join(words)) > MAX_LABEL_CHARACTERS:
        words.pop()
    return " ".join(words)


def canonical_task_label(task: str) -> str:
    task = normalize_space(task)
    if task in GENERIC_TASK_LABELS:
        return GENERIC_TASK_LABELS[task]

    for prefix in TASK_PREFIXES:
        if task.startswith(prefix):
            subject = task[len(prefix) :]
            if prefix == "Researching or Booking Travel for ":
                subject = subject.split(",", 1)[0]
            return compact_label(subject)
    return compact_label(task)


def clean_supervised_label(label: str) -> str | None:
    label = normalize_space(label)
    if not label:
        return None
    if label in {DISSIMILAR_LABEL, ADULT_LABEL}:
        return label
    lowered = label.casefold()
    if any(pattern in lowered for pattern in REFUSAL_PATTERNS):
        return None
    if re.search(r"\b(?:19|20)\d?$", label):
        return None
    label = compact_label(label)
    if not label or len(label) > MAX_LABEL_CHARACTERS:
        return None
    if len(label.split()) > MAX_LABEL_WORDS:
        return None
    return label


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def load_synthetic_groups(data_dir: Path) -> list[SyntheticGroup]:
    groups: list[SyntheticGroup] = []
    for path in sorted(data_dir.glob("gen_test_set_*.csv")):
        grouped_rows: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
        for row in read_csv(path):
            window_id = normalize_space(row.get("test_set_id", ""))
            task_id = normalize_space(row.get("task_id", ""))
            title = normalize_space(row.get("title", ""))
            if window_id and task_id and title:
                grouped_rows[(window_id, task_id)].append(row)

        for (window_id, task_id), rows in grouped_rows.items():
            titles = tuple(normalize_space(row["title"]) for row in rows[:10])
            task = normalize_space(rows[0].get("task", ""))
            if titles and task:
                groups.append(
                    SyntheticGroup(path.stem, window_id, task_id, task, titles)
                )
    return groups


def make_example(
    *,
    base_group_id: str,
    split_key: str,
    source: str,
    variant: str,
    titles: Iterable[str],
    output: str,
    original_task: str = "",
    other_titles: Iterable[str] = (),
    keywords: Iterable[str] | None = None,
    split: str | None = None,
    seed: int = DEFAULT_SEED,
) -> TopicExample:
    title_list = [normalize_space(title) for title in titles if normalize_space(title)][:10]
    if not title_list:
        raise ValueError("an example must contain at least one title")
    keyword_list = (
        list(keywords)
        if keywords is not None
        else extract_group_keywords(title_list, list(other_titles))
    )
    input_text = create_model_input(keyword_list, title_list)
    identity = f"{base_group_id}:{variant}:{input_text}:{output}"
    return TopicExample(
        example_id=hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20],
        base_group_id=base_group_id,
        split_key=split_key,
        split=split or assign_split(split_key, seed),
        source=source,
        variant=variant,
        tab_count=len(title_list),
        input_titles="\n".join(title_list),
        input_keywords=",".join(keyword_list),
        input_text=input_text,
        output=output,
        original_task=original_task,
    )


def group_variants(group: SyntheticGroup, seed: int) -> list[tuple[str, tuple[str, ...]]]:
    variants: list[tuple[str, tuple[str, ...]]] = [("full", group.titles)]
    if len(group.titles) > 3:
        variants.append(("prefix3", group.titles[:3]))

    if len(group.titles) >= 4:
        rng = random.Random(stable_number(group.base_group_id, seed))
        sample_size = rng.randint(2, min(6, len(group.titles)))
        indices = sorted(rng.sample(range(len(group.titles)), sample_size))
        variants.append(("subset", tuple(group.titles[index] for index in indices)))

    result: list[tuple[str, tuple[str, ...]]] = []
    seen: set[tuple[str, ...]] = set()
    for variant, titles in variants:
        if titles not in seen:
            result.append((variant, titles))
            seen.add(titles)
    return result


def build_synthetic_examples(
    groups: list[SyntheticGroup], seed: int
) -> list[TopicExample]:
    examples: list[TopicExample] = []
    by_window: dict[tuple[str, str], list[SyntheticGroup]] = defaultdict(list)
    for group in groups:
        by_window[(group.source_file, group.window_id)].append(group)

    for group in groups:
        window_groups = by_window[(group.source_file, group.window_id)]
        other_titles = [
            title
            for other_group in window_groups
            if other_group.task_id != group.task_id
            for title in other_group.titles
        ]
        output = canonical_task_label(group.task)
        if not output:
            continue
        for variant, titles in group_variants(group, seed):
            examples.append(
                make_example(
                    base_group_id=group.base_group_id,
                    split_key=group.split_key,
                    source="synthetic_group",
                    variant=variant,
                    titles=titles,
                    other_titles=other_titles,
                    output=output,
                    original_task=group.task,
                    seed=seed,
                )
            )

        if stable_number(f"keyword-dropout:{group.base_group_id}", seed) % 5 == 0:
            examples.append(
                make_example(
                    base_group_id=group.base_group_id,
                    split_key=group.split_key,
                    source="synthetic_group",
                    variant="keyword_dropout",
                    titles=group.titles,
                    keywords=[],
                    output=output,
                    original_task=group.task,
                    seed=seed,
                )
            )
    return examples


def build_negative_examples(
    groups: list[SyntheticGroup], seed: int
) -> list[TopicExample]:
    by_window: dict[tuple[str, str], list[SyntheticGroup]] = defaultdict(list)
    for group in groups:
        by_window[(group.source_file, group.window_id)].append(group)

    examples: list[TopicExample] = []
    for (source_file, window_id), window_groups in sorted(by_window.items()):
        if len(window_groups) < 2:
            continue
        rng = random.Random(stable_number(f"negative:{source_file}:{window_id}", seed))
        count = min(4, len(window_groups))
        selected_groups = rng.sample(window_groups, count)
        selected_titles = [rng.choice(group.titles) for group in selected_groups]
        selected_keys = {(group.task_id, title) for group, title in zip(selected_groups, selected_titles)}
        other_titles = [
            title
            for group in window_groups
            for title in group.titles
            if (group.task_id, title) not in selected_keys
        ]
        split_key = f"synthetic-window:{source_file}:{window_id}"
        examples.append(
            make_example(
                base_group_id=f"negative:{source_file}:{window_id}",
                split_key=split_key,
                source="synthetic_unrelated",
                variant="cross_task",
                titles=selected_titles,
                other_titles=other_titles,
                output="None",
                seed=seed,
            )
        )
    return examples


def hostname_key(url: str, fallback: str) -> str:
    hostname = (urlparse(url).hostname or "").casefold()
    return hostname or normalize_title(fallback)


def sample_deterministically(
    rows: list[dict[str, str]], limit: int, identity_field: str, seed: int
) -> list[dict[str, str]]:
    return sorted(
        rows,
        key=lambda row: stable_number(normalize_space(row.get(identity_field, "")), seed),
    )[:limit]


def assign_safety_splits(
    rows: list[tuple[dict[str, str], str, str]], seed: int
) -> dict[str, str]:
    """Create larger held-out safety slices while keeping host keys together."""

    split_by_key: dict[str, str] = {}
    keys_by_target: dict[str, set[str]] = defaultdict(set)
    for _, target, split_key in rows:
        keys_by_target[target].add(split_key)

    for target, keys in keys_by_target.items():
        ordered = sorted(
            keys,
            key=lambda key: stable_number(f"safety:{target}:{key}", seed),
        )
        train_end = int(len(ordered) * 0.70)
        dev_end = int(len(ordered) * 0.85)
        for index, key in enumerate(ordered):
            split_by_key[key] = (
                "train" if index < train_end else "dev" if index < dev_end else "test"
            )
    return split_by_key


def build_multi_tab_safety_examples(
    rows: list[tuple[dict[str, str], str, str]],
    split_by_key: dict[str, str],
    seed: int,
) -> list[TopicExample]:
    adult_rows_by_split: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row, target, split_key in rows:
        if target == ADULT_LABEL:
            adult_rows_by_split[split_by_key[split_key]].append(row)

    examples: list[TopicExample] = []
    for split, adult_rows in adult_rows_by_split.items():
        ordered = sorted(
            adult_rows,
            key=lambda row: stable_number(
                f"safety-bundle:{normalize_title(row.get('input_titles', ''))}", seed
            ),
        )
        for offset in range(0, len(ordered), 3):
            titles = [
                normalize_space(row.get("input_titles", ""))
                for row in ordered[offset : offset + 3]
            ]
            if len(titles) < 2:
                continue
            examples.append(
                make_example(
                    base_group_id=f"trust_safety_bundle:{split}:{offset // 3}",
                    split_key=f"trust-safety-bundle:{split}:{offset // 3}",
                    source="trust_safety_multi",
                    variant="multi_safety",
                    titles=titles,
                    output=ADULT_LABEL,
                    original_task=ADULT_LABEL,
                    split=split,
                    seed=seed,
                )
            )
    return examples


def build_single_tab_examples(
    gcs_dir: Path,
    repo_data_dir: Path,
    seed: int,
    common_crawl_limit: int,
    noise_limit: int,
) -> list[TopicExample]:
    validation_path = gcs_dir / "single_tab_validation.csv"
    validation_rows = read_csv(validation_path) if validation_path.exists() else []
    held_out_titles = {
        normalize_title(row.get("input_titles", "")) for row in validation_rows
    }

    url_by_title: dict[str, str] = {}
    raw_common_crawl_path = repo_data_dir / "external" / "common_crawl.csv"
    if raw_common_crawl_path.exists():
        for row in read_csv(raw_common_crawl_path):
            url_by_title.setdefault(normalize_title(row.get("title", "")), row.get("url", ""))

    candidates: list[tuple[str, Path, int]] = [
        (
            "common_crawl_single",
            gcs_dir / "fine_tuning_data__common_crawl_2025-02-23_08-18__filtered_fashion_edit.csv",
            common_crawl_limit,
        ),
        ("search_single", gcs_dir / "search_simplified.csv", 10_000),
        ("legacy_none_single", gcs_dir / "common_corpus_noise_none_3_12.csv", noise_limit),
    ]

    examples: list[TopicExample] = []
    seen_titles: set[str] = set(held_out_titles)
    safety_split_by_key: dict[str, str] = {}

    safety_filename = "fine_tuning_data__with_none__common_crawl_2025-02-23_08-18.csv"
    safety_path = next(
        (
            path
            for path in (
                repo_data_dir / "extract" / safety_filename,
                gcs_dir / safety_filename,
            )
            if path.exists()
        ),
        None,
    )
    if safety_path is not None:
        safety_rows: list[tuple[dict[str, str], str, str]] = []
        for row in read_csv(safety_path):
            title = normalize_space(row.get("input_titles", ""))
            normalized = normalize_title(title)
            target = canonical_safety_target(row.get("output", ""))
            if title and normalized not in seen_titles and target:
                url = row.get("url", "") or url_by_title.get(normalized, "")
                split_key = f"single-host:{hostname_key(url, title)}"
                safety_rows.append((row, target, split_key))
                seen_titles.add(normalized)

        safety_split_by_key = assign_safety_splits(safety_rows, seed)
        for row, target, split_key in safety_rows:
            title = normalize_space(row["input_titles"])
            examples.append(
                make_example(
                    base_group_id=f"trust_safety:{normalize_title(title)}",
                    split_key=split_key,
                    source="trust_safety_single",
                    variant="single_safety",
                    titles=[title],
                    keywords=[],
                    output=target,
                    original_task=row.get("output", ""),
                    split=safety_split_by_key[split_key],
                    seed=seed,
                )
            )
        examples.extend(
            build_multi_tab_safety_examples(safety_rows, safety_split_by_key, seed)
        )

    for source, path, limit in candidates:
        if not path.exists():
            continue
        usable_rows: list[dict[str, str]] = []
        for row in read_csv(path):
            title = normalize_space(row.get("input_titles", ""))
            normalized = normalize_title(title)
            label = clean_supervised_label(row.get("output", ""))
            if title and normalized not in seen_titles and label:
                usable_rows.append(row)
                seen_titles.add(normalized)

        for row in sample_deterministically(usable_rows, limit, "input_titles", seed):
            title = normalize_space(row["input_titles"])
            label = clean_supervised_label(row["output"])
            assert label is not None
            url = row.get("url", "") or url_by_title.get(normalize_title(title), "")
            split_key = f"single-host:{hostname_key(url, title)}"
            examples.append(
                make_example(
                    base_group_id=f"{source}:{normalize_title(title)}",
                    split_key=split_key,
                    source=source,
                    variant="single",
                    titles=[title],
                    keywords=[],
                    output=label,
                    split=safety_split_by_key.get(split_key),
                    seed=seed,
                )
            )

    for row in validation_rows:
        title = normalize_space(row.get("input_titles", ""))
        label = clean_supervised_label(row.get("output", ""))
        if title and label:
            examples.append(
                make_example(
                    base_group_id=f"legacy_validation:{normalize_title(title)}",
                    split_key=f"legacy-validation:{normalize_title(title)}",
                    source="legacy_validation",
                    variant="single",
                    titles=[title],
                    keywords=[],
                    output=label,
                    split="test",
                    seed=seed,
                )
            )
    return examples


def deduplicate_examples(examples: Iterable[TopicExample]) -> list[TopicExample]:
    by_prompt: dict[str, TopicExample] = {}
    for example in examples:
        existing = by_prompt.get(example.input_text)
        if existing is None:
            by_prompt[example.input_text] = example
        elif existing.output != example.output:
            # Conflicting supervision is more damaging than losing one example.
            by_prompt[example.input_text] = TopicExample(
                **{**asdict(existing), "output": "__CONFLICT__"}
            )
    return sorted(
        (example for example in by_prompt.values() if example.output != "__CONFLICT__"),
        key=lambda example: (example.split, example.source, example.example_id),
    )


def audit_examples(examples: list[TopicExample]) -> dict[str, object]:
    split_by_base: dict[str, set[str]] = defaultdict(set)
    split_by_key: dict[str, set[str]] = defaultdict(set)
    for example in examples:
        split_by_base[example.base_group_id].add(example.split)
        split_by_key[example.split_key].add(example.split)

    leakage_base = sorted(key for key, splits in split_by_base.items() if len(splits) > 1)
    leakage_keys = sorted(key for key, splits in split_by_key.items() if len(splits) > 1)
    invalid_labels = sorted(
        {
            example.output
            for example in examples
            if len(example.output) > MAX_LABEL_CHARACTERS
            or len(example.output.split()) > MAX_LABEL_WORDS
        }
    )
    if leakage_base or leakage_keys or invalid_labels:
        raise ValueError(
            f"dataset audit failed: base leakage={len(leakage_base)}, "
            f"split-key leakage={len(leakage_keys)}, invalid labels={invalid_labels}"
        )

    return {
        "total_examples": len(examples),
        "split_counts": dict(sorted(Counter(row.split for row in examples).items())),
        "source_counts": dict(sorted(Counter(row.source for row in examples).items())),
        "variant_counts": dict(sorted(Counter(row.variant for row in examples).items())),
        "tab_count_counts": dict(
            sorted(Counter(str(row.tab_count) for row in examples).items(), key=lambda item: int(item[0]))
        ),
        "multi_tab_examples": sum(row.tab_count > 1 for row in examples),
        "none_examples": sum(row.output == DISSIMILAR_LABEL for row in examples),
        "adult_content_examples": sum(row.output == ADULT_LABEL for row in examples),
        "trust_safety_examples": sum(
            row.source.startswith("trust_safety_") for row in examples
        ),
        "unique_labels": len({row.output for row in examples}),
        "duplicate_prompts": len(examples) - len({row.input_text for row in examples}),
        "base_group_split_leaks": leakage_base,
        "split_key_leaks": leakage_keys,
        "invalid_labels": invalid_labels,
        "builder": {
            "seed": DEFAULT_SEED,
            "max_label_characters": MAX_LABEL_CHARACTERS,
            "max_label_words": MAX_LABEL_WORDS,
            "firefox_max_documents": 10,
            "firefox_keyword_documents": 3,
            "firefox_generation_max_length": 6,
        },
    }


def write_dataset(examples: list[TopicExample], output_dir: Path, audit: dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "dev", "test"):
        write_csv(output_dir / f"{split}.csv", [row for row in examples if row.split == split])
    write_csv(output_dir / "all.csv", examples)
    with (output_dir / "audit.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_csv(path: Path, examples: Iterable[TopicExample]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for example in examples:
            writer.writerow(asdict(example))


def build_dataset(
    repo_data_dir: Path,
    gcs_dir: Path,
    output_dir: Path,
    seed: int = DEFAULT_SEED,
    common_crawl_limit: int = 3000,
    noise_limit: int = 400,
) -> tuple[list[TopicExample], dict[str, object]]:
    groups = load_synthetic_groups(repo_data_dir / "synthetic_datasets")
    if not groups:
        raise FileNotFoundError(
            f"no synthetic groups found under {repo_data_dir / 'synthetic_datasets'}"
        )
    examples = build_synthetic_examples(groups, seed)
    examples.extend(build_negative_examples(groups, seed))
    examples.extend(
        build_single_tab_examples(
            gcs_dir, repo_data_dir, seed, common_crawl_limit, noise_limit
        )
    )
    examples = deduplicate_examples(examples)
    audit = audit_examples(examples)
    audit["builder"]["seed"] = seed
    audit["builder"]["common_crawl_limit"] = common_crawl_limit
    audit["builder"]["noise_limit"] = noise_limit
    audit["synthetic_base_groups"] = len(groups)
    write_dataset(examples, output_dir, audit)
    return examples, audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-data-dir", type=Path, default=Path("data"))
    parser.add_argument("--gcs-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("output/topic_dataset_v2"))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--common-crawl-limit", type=int, default=3000)
    parser.add_argument("--noise-limit", type=int, default=400)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, audit = build_dataset(
        args.repo_data_dir,
        args.gcs_dir,
        args.output_dir,
        args.seed,
        args.common_crawl_limit,
        args.noise_limit,
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
