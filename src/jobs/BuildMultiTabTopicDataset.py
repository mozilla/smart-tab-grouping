#!/usr/bin/env python3
"""Build a split-safe, Firefox-shaped, multi-tab-only naming dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path


FIELDS = (
    "example_id",
    "base_group_id",
    "split_key",
    "split",
    "source",
    "variant",
    "category",
    "tab_count",
    "input_titles",
    "input_keywords",
    "input_text",
    "output",
    "original_task",
)
SPLITS = ("train", "dev", "test")
ABSTENTION_TARGETS = {"None", "Adult Content"}
DEFAULT_REPO_DIR = Path(__file__).resolve().parents[2]
GCS_BUCKET = "stage-fx-tab-grouping"
GCS_TOPIC_FILES = (
    "common_corpus_noise_none_3_12.csv",
    "fine_tuning_data__common_crawl_2025-02-23_08-18__filtered_fashion_edit.csv",
    "search_simplified.csv",
    "single_tab_validation.csv",
)


def normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip().casefold()


def stable_number(value: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def title_list(row: dict[str, str]) -> list[str]:
    return [title.strip() for title in row["input_titles"].splitlines() if title.strip()]


def make_row(
    *,
    firefox_text: object,
    base_group_id: str,
    split: str,
    source: str,
    variant: str,
    category: str,
    titles: list[str],
    output: str,
    seed: int,
    other_titles: list[str] | None = None,
    keywords: list[str] | None = None,
    original_task: str = "",
) -> dict[str, str]:
    titles = [re.sub(r"\s+", " ", value).strip() for value in titles if value.strip()][:10]
    if len(titles) < 2:
        raise ValueError("multi-tab examples require at least two titles")
    if keywords is None:
        keywords = firefox_text.extract_group_keywords(titles, other_titles or [])
    prompt = firefox_text.create_model_input(keywords, titles)
    identity = f"{base_group_id}:{variant}:{prompt}:{output}"
    return {
        "example_id": hashlib.sha256(identity.encode()).hexdigest()[:20],
        "base_group_id": base_group_id,
        "split_key": base_group_id,
        "split": split,
        "source": source,
        "variant": variant,
        "category": category,
        "tab_count": str(len(titles)),
        "input_titles": "\n".join(titles),
        "input_keywords": ",".join(keywords),
        "input_text": prompt,
        "output": output,
        "original_task": original_task,
    }


def assign_exact_title_owners(rows: list[dict[str, str]]) -> dict[str, str]:
    """Prefer final-test ownership, then dev, when a title occurs in several splits."""

    priority = {"train": 0, "dev": 1, "test": 2}
    owners: dict[str, str] = {}
    for row in rows:
        for title in title_list(row):
            key = normalize(title)
            if key and priority[row["split"]] > priority.get(owners.get(key, "train"), 0):
                owners[key] = row["split"]
            elif key and key not in owners:
                owners[key] = row["split"]
    return owners


def owner_clean_rows(
    rows: list[dict[str, str]],
    owners: dict[str, str],
    forbidden_train_titles: set[str],
    fuzzy_forbidden_by_split: dict[str, set[str]] | None = None,
) -> tuple[list[dict[str, str]], int, int, int]:
    kept = []
    cross_split_dropped = 0
    golden_dropped = 0
    fuzzy_dropped = 0
    fuzzy_forbidden_by_split = fuzzy_forbidden_by_split or {}
    for row in rows:
        titles = {normalize(value) for value in title_list(row)}
        if any(owners.get(title) != row["split"] for title in titles):
            cross_split_dropped += 1
            continue
        if row["split"] == "train" and titles & forbidden_train_titles:
            golden_dropped += 1
            continue
        if titles & fuzzy_forbidden_by_split.get(row["split"], set()):
            fuzzy_dropped += 1
            continue
        kept.append(row)
    return kept, cross_split_dropped, golden_dropped, fuzzy_dropped


def fuzzy_lower_priority_titles(
    rows: list[dict[str, str]], golden: set[str], threshold: float = 0.95
) -> tuple[dict[str, set[str]], dict[str, object]]:
    """Find near-duplicate titles that must be removed from train or dev."""

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors

    titles_by_split: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        titles_by_split[row["split"]].update(normalize(title) for title in title_list(row))
    all_titles = sorted(set().union(*titles_by_split.values(), golden))
    vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True
    )
    matrix = vectorizer.fit_transform(all_titles)
    indices = {title: index for index, title in enumerate(all_titles)}
    forbidden: dict[str, set[str]] = defaultdict(set)
    comparisons: dict[str, object] = {}

    def compare(source_split: str, target_name: str, target_titles: set[str]) -> None:
        source_titles = sorted(titles_by_split[source_split])
        targets = sorted(target_titles)
        if not source_titles or not targets:
            return
        target_matrix = matrix[[indices[title] for title in targets]]
        neighbors = NearestNeighbors(n_neighbors=1, metric="cosine", algorithm="brute")
        neighbors.fit(target_matrix)
        distances, neighbor_indices = neighbors.kneighbors(
            matrix[[indices[title] for title in source_titles]]
        )
        matches = sorted(
            (
                1.0 - float(distances[index, 0]),
                title,
                targets[int(neighbor_indices[index, 0])],
            )
            for index, title in enumerate(source_titles)
        )
        flagged = [match for match in matches if match[0] >= threshold]
        forbidden[source_split].update(match[1] for match in flagged)
        comparisons[f"{source_split}_vs_{target_name}"] = {
            "threshold": threshold,
            "flagged_source_titles": len(flagged),
            "maximum_similarity": matches[-1][0] if matches else 0.0,
            "highest_matches": [
                {"similarity": score, "lower_priority": source, "held_out": target}
                for score, source, target in reversed(matches[-10:])
            ],
        }

    compare("train", "dev_test", titles_by_split["dev"] | titles_by_split["test"])
    compare("dev", "test", titles_by_split["test"])
    compare("train", "golden", golden)
    return dict(forbidden), comparisons


def convert_existing(row: dict[str, str], category: str) -> dict[str, str]:
    return {field: row.get(field, "") for field in FIELDS} | {"category": category}


def derive_same_label_groups(
    singles: list[dict[str, str]], firefox_text: object, seed: int
) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    split_pools: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in singles:
        if row["output"] not in ABSTENTION_TARGETS:
            grouped[(row["split"], row["output"])].append(row)
            split_pools[row["split"]].append(row)

    result: list[dict[str, str]] = []
    for (split, label), label_rows in sorted(grouped.items()):
        if len(label_rows) < 2:
            continue
        ordered = sorted(
            label_rows,
            key=lambda row: stable_number(f"same-label:{row['input_titles']}", seed),
        )
        rng = random.Random(stable_number(f"same-label-size:{split}:{label}", seed))
        offset = 0
        group_index = 0
        while offset < len(ordered):
            size = rng.choices((2, 3, 4, 5), weights=(35, 30, 22, 13), k=1)[0]
            selected = ordered[offset : offset + size]
            offset += size
            if len(selected) == 1:
                selected.append(ordered[0])
            titles = [row["input_titles"] for row in selected]
            context = [
                row["input_titles"]
                for row in split_pools[split]
                if row["output"] != label
            ][:30]
            result.append(
                make_row(
                    firefox_text=firefox_text,
                    base_group_id=f"same-label:{split}:{hashlib.sha256(label.encode()).hexdigest()[:12]}:{group_index}",
                    split=split,
                    source="same_label_group",
                    variant="same_label",
                    category="clean",
                    titles=titles,
                    other_titles=context,
                    output=label,
                    seed=seed,
                    original_task=label,
                )
            )
            group_index += 1
    return result


def donor_for(
    rows: list[dict[str, str]], source: dict[str, str], index: int, seed: int
) -> dict[str, str]:
    candidates = [row for row in rows if row["output"] != source["output"]]
    if not candidates:
        raise ValueError("noise construction needs at least two labels in a split")
    return candidates[stable_number(f"donor:{source['example_id']}:{index}", seed) % len(candidates)]


def build_noise_rows(
    clean: list[dict[str, str]], target: int, firefox_text: object, seed: int
) -> list[dict[str, str]]:
    existing = []
    result = list(existing)
    prompts: set[str] = set()
    ordered = sorted(clean, key=lambda row: stable_number(f"noise:{row['example_id']}", seed))
    index = 0
    attempts = 0
    while len(result) < target and attempts < target * 20:
        source = ordered[index % len(ordered)]
        donor = donor_for(ordered, source, index, seed)
        titles = title_list(source)
        donor_titles = title_list(donor)
        mode = (index // len(ordered) + index) % 3
        if mode == 0:
            variant = "keyword_dropout"
            keywords: list[str] | None = []
        elif mode == 1:
            variant = "keyword_corruption"
            source_keywords = [value for value in source["input_keywords"].split(",") if value]
            donor_keywords = [value for value in donor["input_keywords"].split(",") if value]
            if not donor_keywords:
                donor_keywords = firefox_text.extract_group_keywords(donor_titles, titles)
            keywords = source_keywords[:2] + donor_keywords[:1]
        else:
            variant = "title_distractor"
            distractor = donor_titles[stable_number(f"distractor:{index}", seed) % len(donor_titles)]
            titles = titles[:9] + [distractor]
            keywords = None
        row = make_row(
            firefox_text=firefox_text,
            base_group_id=source["base_group_id"],
            split=source["split"],
            source="augmented_coherent",
            variant=variant,
            category="noise",
            titles=titles,
            other_titles=donor_titles,
            keywords=keywords,
            output=source["output"],
            seed=seed,
            original_task=source["original_task"],
        )
        if row["input_text"] not in prompts and row["input_text"] != source["input_text"]:
            result.append(row)
            prompts.add(row["input_text"])
        index += 1
        attempts += 1
    if len(result) != target:
        raise RuntimeError(f"could only construct {len(result)}/{target} noise rows")
    return result


def build_none_rows(
    clean: list[dict[str, str]], existing: list[dict[str, str]], target: int, firefox_text: object, seed: int
) -> list[dict[str, str]]:
    result = list(existing[:target])
    prompts = {row["input_text"] for row in result}
    ordered = sorted(clean, key=lambda row: stable_number(f"none:{row['example_id']}", seed))
    index = 0
    attempts = 0
    while len(result) < target and attempts < target * 30:
        size = (2, 3, 4, 5)[stable_number(f"none-size:{index}", seed) % 4]
        chosen: list[dict[str, str]] = []
        labels: set[str] = set()
        cursor = stable_number(f"none-start:{index}", seed) % len(ordered)
        for step in range(len(ordered)):
            candidate = ordered[(cursor + step * 37) % len(ordered)]
            if candidate["output"] not in labels:
                chosen.append(candidate)
                labels.add(candidate["output"])
            if len(chosen) == size:
                break
        titles = [title_list(row)[stable_number(f"none-title:{index}:{i}", seed) % len(title_list(row))] for i, row in enumerate(chosen)]
        context = [title for row in chosen for title in title_list(row) if title not in titles]
        row = make_row(
            firefox_text=firefox_text,
            base_group_id=f"unrelated:{chosen[0]['split']}:{index}",
            split=chosen[0]["split"],
            source="synthetic_unrelated_v3",
            variant="cross_group",
            category="none",
            titles=titles,
            other_titles=context,
            output="None",
            seed=seed,
        )
        if row["input_text"] not in prompts:
            result.append(row)
            prompts.add(row["input_text"])
        index += 1
        attempts += 1
    if len(result) != target:
        raise RuntimeError(f"could only construct {len(result)}/{target} None rows")
    return result


def build_adult_rows(
    adult_singles: list[dict[str, str]], existing: list[dict[str, str]], target: int, firefox_text: object, seed: int
) -> list[dict[str, str]]:
    result = list(existing[:target])
    prompts = {row["input_text"] for row in result}
    titles = [row["input_titles"] for row in adult_singles]
    if len(titles) < 2 and len(result) < target:
        raise ValueError("not enough split-local adult titles to build multi-tab safety rows")
    ordered = sorted(titles, key=lambda value: stable_number(f"adult:{value}", seed))
    index = 0
    attempts = 0
    while len(result) < target and attempts < target * 30:
        size = (2, 3, 4, 5)[stable_number(f"adult-size:{index}", seed) % 4]
        start = stable_number(f"adult-start:{index}", seed) % len(ordered)
        selected = [ordered[(start + step * (index % 5 + 1)) % len(ordered)] for step in range(size)]
        selected = list(dict.fromkeys(selected))
        if len(selected) < 2:
            index += 1
            attempts += 1
            continue
        row = make_row(
            firefox_text=firefox_text,
            base_group_id=f"adult-bundle:{adult_singles[0]['split']}:{index}",
            split=adult_singles[0]["split"],
            source="trust_safety_multi_v3",
            variant="adult_bundle",
            category="adult",
            titles=selected,
            output="Adult Content",
            seed=seed,
            original_task="Adult Content",
        )
        if row["input_text"] not in prompts:
            result.append(row)
            prompts.add(row["input_text"])
        index += 1
        attempts += 1
    if len(result) != target:
        raise RuntimeError(f"could only construct {len(result)}/{target} adult rows")
    return result


def deduplicate(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_prompt: dict[str, dict[str, str] | None] = {}
    for row in rows:
        current = by_prompt.get(row["input_text"])
        if current is None and row["input_text"] not in by_prompt:
            by_prompt[row["input_text"]] = row
        elif current is not None and current["output"] != row["output"]:
            by_prompt[row["input_text"]] = None
    return [row for row in by_prompt.values() if row is not None]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_gcs_inputs(
    destination: Path,
    download: Callable[[str, str, str], None],
    *,
    refresh: bool = False,
) -> dict[str, Path]:
    """Download the canonical GCS inputs into the ignored output tree."""

    destination.mkdir(parents=True, exist_ok=True)
    downloaded: dict[str, Path] = {}
    for filename in GCS_TOPIC_FILES:
        path = destination / filename
        if refresh or not path.is_file():
            temporary = destination / f".{filename}.part"
            if temporary.exists():
                temporary.unlink()
            download(GCS_BUCKET, f"topic/{filename}", str(temporary))
            if not temporary.is_file() or temporary.stat().st_size == 0:
                raise RuntimeError(f"GCS download was empty: topic/{filename}")
            temporary.replace(path)
        downloaded[filename] = path
    return downloaded


def audit(rows: list[dict[str, str]], forbidden_train_titles: set[str]) -> dict[str, object]:
    title_splits: dict[str, set[str]] = defaultdict(set)
    base_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if not 2 <= int(row["tab_count"]) <= 10:
            raise ValueError(f"invalid tab count: {row}")
        base_splits[row["base_group_id"]].add(row["split"])
        for title in title_list(row):
            title_splits[normalize(title)].add(row["split"])
    title_leaks = sorted(title for title, splits in title_splits.items() if len(splits) > 1)
    base_leaks = sorted(base for base, splits in base_splits.items() if len(splits) > 1)
    train_titles = {
        normalize(title)
        for row in rows
        if row["split"] == "train"
        for title in title_list(row)
    }
    if title_leaks or base_leaks or train_titles & forbidden_train_titles:
        raise ValueError(
            f"leakage audit failed: titles={len(title_leaks)}, bases={len(base_leaks)}, "
            f"golden_titles={len(train_titles & forbidden_train_titles)}"
        )
    return {
        "total_examples": len(rows),
        "all_examples_are_multi_tab": all(int(row["tab_count"]) >= 2 for row in rows),
        "split_counts": dict(Counter(row["split"] for row in rows)),
        "category_counts": dict(Counter(row["category"] for row in rows)),
        "source_counts": dict(Counter(row["source"] for row in rows)),
        "variant_counts": dict(Counter(row["variant"] for row in rows)),
        "tab_count_counts": dict(Counter(row["tab_count"] for row in rows)),
        "unique_labels": len({row["output"] for row in rows}),
        "duplicate_prompts": len(rows) - len({row["input_text"] for row in rows}),
        "exact_title_cross_split_leaks": title_leaks,
        "base_group_cross_split_leaks": base_leaks,
        "train_exact_golden_title_overlaps": sorted(train_titles & forbidden_train_titles),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, default=DEFAULT_REPO_DIR)
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=DEFAULT_REPO_DIR / "output" / "multitab-training-inputs" / "gcs",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--refresh-downloads", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    source_dir = str(args.repo_dir.resolve() / "src")
    if source_dir not in sys.path:
        sys.path.insert(0, source_dir)
    from topic_dataset import firefox_text
    from topic_dataset.build_v2 import build_dataset
    from util.storage import download_bucket_to_file

    gcs_inputs = download_gcs_inputs(
        args.download_dir,
        download_bucket_to_file,
        refresh=args.refresh_downloads,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base_dir = args.output_dir / "base_v2"
    _, base_audit = build_dataset(
        args.repo_dir / "data",
        args.download_dir,
        base_dir,
        args.seed,
    )
    raw_rows = read_csv(base_dir / "all.csv")
    forbidden: set[str] = set()
    owners = assign_exact_title_owners(raw_rows)
    preliminary_rows, cross_split_dropped, golden_dropped, _ = owner_clean_rows(
        raw_rows, owners, forbidden
    )
    fuzzy_forbidden, fuzzy_audit = fuzzy_lower_priority_titles(
        preliminary_rows, forbidden
    )
    clean_rows, _, _, fuzzy_dropped = owner_clean_rows(
        preliminary_rows,
        owners,
        forbidden,
        fuzzy_forbidden_by_split=fuzzy_forbidden,
    )

    clean: list[dict[str, str]] = []
    original_noise: list[dict[str, str]] = []
    original_none: list[dict[str, str]] = []
    original_adult: list[dict[str, str]] = []
    singles: list[dict[str, str]] = []
    adult_singles_by_split: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in clean_rows:
        tabs = int(row["tab_count"])
        if tabs == 1:
            singles.append(row)
            if row["output"] == "Adult Content":
                adult_singles_by_split[row["split"]].append(row)
        elif row["output"] == "Adult Content":
            original_adult.append(convert_existing(row, "adult"))
        elif row["output"] == "None":
            original_none.append(convert_existing(row, "none"))
        elif row["variant"] == "keyword_dropout":
            original_noise.append(convert_existing(row, "noise"))
        else:
            clean.append(convert_existing(row, "clean"))

    clean.extend(derive_same_label_groups(singles, firefox_text, args.seed))
    final_rows: list[dict[str, str]] = []
    target_distribution = {"clean": 0.60, "noise": 0.20, "none": 0.15, "adult": 0.05}
    for split in SPLITS:
        split_clean = [row for row in clean if row["split"] == split]
        total_target = math.ceil(len(split_clean) / target_distribution["clean"])
        noise_target = round(total_target * target_distribution["noise"])
        none_target = round(total_target * target_distribution["none"])
        adult_target = max(1, total_target - len(split_clean) - noise_target - none_target)
        split_original_noise = [row for row in original_noise if row["split"] == split]
        noise_seed = split_original_noise[:noise_target]
        if len(noise_seed) < noise_target:
            noise_seed.extend(
                build_noise_rows(split_clean, noise_target - len(noise_seed), firefox_text, args.seed + 101)
            )
        split_none = build_none_rows(
            split_clean,
            [row for row in original_none if row["split"] == split],
            none_target,
            firefox_text,
            args.seed + 202,
        )
        split_adult = build_adult_rows(
            adult_singles_by_split[split],
            [row for row in original_adult if row["split"] == split],
            adult_target,
            firefox_text,
            args.seed + 303,
        )
        final_rows.extend(split_clean + noise_seed + split_none + split_adult)

    final_rows = sorted(
        deduplicate(final_rows),
        key=lambda row: (SPLITS.index(row["split"]), row["category"], row["example_id"]),
    )
    post_fuzzy_forbidden, post_fuzzy_audit = fuzzy_lower_priority_titles(
        final_rows, forbidden
    )
    if any(post_fuzzy_forbidden.values()):
        raise ValueError(
            "fuzzy title leakage remained after augmentation: "
            + ", ".join(
                f"{split}={len(titles)}"
                for split, titles in sorted(post_fuzzy_forbidden.items())
                if titles
            )
        )
    report = audit(final_rows, forbidden)
    report.update(
        {
            "seed": args.seed,
            "target_distribution": target_distribution,
            "base_v2_audit": base_audit,
            "base_rows_dropped_for_exact_cross_split_titles": cross_split_dropped,
            "base_train_rows_dropped_for_exact_golden_titles": golden_dropped,
            "base_rows_dropped_for_fuzzy_cross_split_or_golden_titles": fuzzy_dropped,
            "fuzzy_title_audit_before_augmentation": fuzzy_audit,
            "fuzzy_title_audit_after_augmentation": post_fuzzy_audit,
            "input_files": {
                "gcs": {
                    f"gs://{GCS_BUCKET}/topic/{filename}": {
                        "local_path": str(path),
                        "sha256": sha256(path),
                    }
                    for filename, path in sorted(gcs_inputs.items())
                },
                "tracked_repo_data": {
                    str(path.relative_to(args.repo_dir)): sha256(path)
                    for path in (
                        args.repo_dir / "data" / "external" / "common_crawl.csv",
                        args.repo_dir
                        / "data"
                        / "extract"
                        / "fine_tuning_data__with_none__common_crawl_2025-02-23_08-18.csv",
                    )
                },
            },
        }
    )
    for split in SPLITS:
        write_csv(args.output_dir / f"{split}.csv", [row for row in final_rows if row["split"] == split])
    write_csv(args.output_dir / "all.csv", final_rows)
    (args.output_dir / "audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
