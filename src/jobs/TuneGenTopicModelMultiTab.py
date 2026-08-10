#!/usr/bin/env python3
"""Resumable online T5 distillation for Firefox multi-tab topic naming."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any


ABSTENTION_TARGETS = {"none", "adult content"}
DEFAULT_REPO_DIR = Path(__file__).resolve().parents[2]
DEFAULT_TEACHER_ARTIFACT = "moso/tab_grouping/model-v40xoz3q:v0"
DEFAULT_GENERATION_CONFIG_URL = (
    "https://model-hub.mozilla.org/mozilla/smart-tab-topic/"
    "v0.7.4/generation_config.json"
)


def normalize_prediction(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def token_f1(prediction: str, reference: str) -> float:
    prediction_tokens = Counter(normalize_prediction(prediction).split())
    reference_tokens = Counter(normalize_prediction(reference).split())
    overlap = sum((prediction_tokens & reference_tokens).values())
    if not prediction_tokens or not reference_tokens or overlap == 0:
        return float(prediction_tokens == reference_tokens)
    precision = overlap / sum(prediction_tokens.values())
    recall = overlap / sum(reference_tokens.values())
    return 2 * precision * recall / (precision + recall)


def calculate_metrics(
    predictions: list[str], references: list[str]
) -> dict[str, float]:
    """Score naming quality and Firefox-visible abstention behavior."""

    from topic_dataset.safety import firefox_visible_label, is_abstention_label

    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length")
    if not predictions:
        return {}

    normalized_predictions = [normalize_prediction(value) for value in predictions]
    normalized_references = [normalize_prediction(value) for value in references]
    visible_predictions = [
        normalize_prediction(firefox_visible_label(value)) for value in predictions
    ]
    visible_references = [
        normalize_prediction(firefox_visible_label(value)) for value in references
    ]
    predicted_abstention = [is_abstention_label(value) for value in predictions]
    reference_abstention = [is_abstention_label(value) for value in references]
    true_positive = sum(
        predicted and reference
        for predicted, reference in zip(predicted_abstention, reference_abstention)
    )
    false_positive = sum(
        predicted and not reference
        for predicted, reference in zip(predicted_abstention, reference_abstention)
    )
    false_negative = sum(
        not predicted and reference
        for predicted, reference in zip(predicted_abstention, reference_abstention)
    )
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    adult_indices = [
        index
        for index, reference in enumerate(normalized_references)
        if reference == "adult content"
    ]
    non_abstention_indices = [
        index for index, reference in enumerate(reference_abstention) if not reference
    ]
    adult_abstentions = sum(predicted_abstention[index] for index in adult_indices)
    return {
        "exact_match": sum(
            prediction == reference
            for prediction, reference in zip(
                normalized_predictions, normalized_references
            )
        )
        / len(predictions),
        "firefox_visible_exact_match": sum(
            prediction == reference
            for prediction, reference in zip(
                visible_predictions, visible_references
            )
        )
        / len(predictions),
        "token_f1": sum(
            token_f1(prediction, reference)
            for prediction, reference in zip(predictions, references)
        )
        / len(predictions),
        "abstention_precision": precision,
        "abstention_recall": recall,
        "abstention_f1": (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        ),
        "false_abstention_rate": (
            sum(predicted_abstention[index] for index in non_abstention_indices)
            / len(non_abstention_indices)
            if non_abstention_indices
            else 0.0
        ),
        "adult_reference_examples": len(adult_indices),
        "adult_abstention_rate": (
            adult_abstentions / len(adult_indices) if adult_indices else 0.0
        ),
        "unsafe_adult_output_rate": (
            1.0 - adult_abstentions / len(adult_indices) if adult_indices else 0.0
        ),
        "mean_prediction_words": sum(len(value.split()) for value in predictions)
        / len(predictions),
    }


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def distillation_loss(
    student_logits: Any,
    teacher_logits: Any,
    labels: Any,
    *,
    alpha: float,
    temperature: float,
) -> tuple[Any, Any, Any]:
    import torch.nn.functional as functional

    vocabulary_size = student_logits.size(-1)
    ce_loss = functional.cross_entropy(
        student_logits.reshape(-1, vocabulary_size),
        labels.reshape(-1),
        ignore_index=-100,
    )
    student_log_probs = functional.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = functional.softmax(teacher_logits / temperature, dim=-1)
    token_kl = functional.kl_div(
        student_log_probs, teacher_probs, reduction="none"
    ).sum(dim=-1)
    valid = labels.ne(-100)
    kl_loss = token_kl.masked_select(valid).mean() * (temperature**2)
    return alpha * kl_loss + (1.0 - alpha) * ce_loss, kl_loss, ce_loss


def choose_device(require_mps: bool) -> Any:
    import torch

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if require_mps:
        raise RuntimeError("MPS is unavailable. Run this from the normal macOS Terminal.")
    return torch.device("cpu")


def download_teacher_artifact(
    artifact_name: str,
    destination: Path,
    api_factory: Callable[[], Any] | None = None,
) -> Path:
    """Download the configured W&B teacher unless it is already cached."""

    if (destination / "config.json").is_file():
        return destination
    if api_factory is None:
        import wandb

        api_factory = wandb.Api
    destination.parent.mkdir(parents=True, exist_ok=True)
    artifact = api_factory().artifact(artifact_name, type="model")
    resolved = Path(artifact.download(root=str(destination)))
    if not (resolved / "config.json").is_file():
        raise FileNotFoundError(
            f"W&B teacher artifact has no config.json: {artifact_name}"
        )
    return resolved


def download_generation_config(
    url: str,
    destination: Path,
    get: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Download and validate Firefox's deployed generation configuration."""

    if not destination.is_file():
        if get is None:
            import requests

            get = requests.get
        response = get(url, timeout=60)
        response.raise_for_status()
        payload = response.json()
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".json.part")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        temporary.replace(destination)
    payload = json.loads(destination.read_text(encoding="utf-8"))
    if not payload.get("bad_words_ids"):
        raise ValueError("downloaded generation config has no bad_words_ids")
    return payload


def apply_safety_config(model: Any, payload: dict[str, Any]) -> int:
    from transformers import GenerationConfig

    model.generation_config = GenerationConfig.from_dict(payload)
    return len(model.generation_config.bad_words_ids)


def deployable_rows(rows: list[dict[str, str]], tokenizer: Any, max_length: int) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    kept = []
    dropped = []
    for row in rows:
        if int(row["tab_count"]) < 2:
            raise ValueError(f"single-tab row reached multi-tab trainer: {row['example_id']}")
        count = len(tokenizer(row["output"], add_special_tokens=True)["input_ids"])
        enriched = {**row, "target_token_count": str(count)}
        (kept if count <= max_length else dropped).append(enriched)
    return kept, dropped


def confidence_from_generation(model: Any, generated: Any, tokenizer: Any) -> list[float]:
    import torch
    import torch.nn.functional as functional

    batch = generated.sequences.shape[0]
    log_sums = torch.zeros(batch, device=generated.sequences.device)
    counts = torch.zeros(batch, device=generated.sequences.device)
    finished = torch.zeros(batch, dtype=torch.bool, device=generated.sequences.device)
    for step, scores in enumerate(generated.scores):
        token_ids = generated.sequences[:, step + 1]
        token_log_probs = functional.log_softmax(scores, dim=-1).gather(1, token_ids[:, None]).squeeze(1)
        active = ~finished
        log_sums += token_log_probs * active
        counts += active
        finished |= token_ids.eq(tokenizer.eos_token_id) | token_ids.eq(tokenizer.pad_token_id)
    return torch.exp(log_sums / counts.clamp_min(1)).detach().cpu().tolist()


def sequence_kd_rows(
    rows: list[dict[str, str]],
    *,
    teacher: Any,
    tokenizer: Any,
    device: Any,
    cache_path: Path,
    ratio: float,
    confidence_threshold: float,
    batch_size: int,
    max_source_length: int,
    max_target_length: int,
    seed: int,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    if ratio <= 0:
        return [], {"enabled": False, "rows": 0}
    if cache_path.is_file():
        cached = read_rows(cache_path)
        return cached, {
            "enabled": True,
            "rows": len(cached),
            "confidence_threshold": confidence_threshold,
            "loaded_from_cache": True,
        }

    candidates = [
        row for row in rows if row["output"].strip().casefold() not in ABSTENTION_TARGETS
    ]
    candidates = sorted(candidates, key=lambda row: row["example_id"])
    rng = random.Random(seed + 9001)
    rng.shuffle(candidates)
    desired = min(len(candidates), math.ceil(len(rows) * ratio))
    accepted: list[dict[str, Any]] = []
    agreement = 0
    teacher.eval()
    for start in range(0, len(candidates), batch_size):
        batch_rows = candidates[start : start + batch_size]
        encoded = tokenizer(
            [row["input_text"] for row in batch_rows],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_source_length,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        import torch

        with torch.inference_mode():
            generated = teacher.generate(
                **encoded,
                max_length=max_target_length,
                num_beams=1,
                return_dict_in_generate=True,
                output_scores=True,
            )
        predictions = tokenizer.batch_decode(generated.sequences.cpu(), skip_special_tokens=True)
        confidences = confidence_from_generation(teacher, generated, tokenizer)
        for row, prediction, confidence in zip(batch_rows, predictions, confidences):
            prediction = " ".join(prediction.split())
            token_count = len(tokenizer(prediction, add_special_tokens=True)["input_ids"])
            if (
                prediction
                and prediction.casefold() not in ABSTENTION_TARGETS
                and token_count <= max_target_length
                and confidence >= confidence_threshold
            ):
                agreement += int(prediction.casefold() == row["output"].strip().casefold())
                accepted.append(
                    {
                        **row,
                        "example_id": f"{row['example_id']}-teacher",
                        "output": prediction,
                        "target_source": "teacher_sequence",
                        "teacher_confidence": f"{confidence:.8f}",
                        "hard_output": row["output"],
                    }
                )
                if len(accepted) >= desired:
                    break
        print(
            f"sequence-kd considered={min(start + batch_size, len(candidates))}/{len(candidates)} "
            f"accepted={len(accepted)}/{desired}",
            flush=True,
        )
        if len(accepted) >= desired:
            break
    fields = list(rows[0].keys()) + ["target_source", "teacher_confidence", "hard_output"]
    write_rows(cache_path, accepted, fields)
    return accepted, {
        "enabled": True,
        "rows": len(accepted),
        "desired_rows": desired,
        "confidence_threshold": confidence_threshold,
        "hard_target_exact_agreement": agreement / len(accepted) if accepted else 0.0,
        "loaded_from_cache": False,
    }


class TopicDataset:
    def __init__(self, rows: list[dict[str, str]], tokenizer: Any, max_source_length: int, max_target_length: int) -> None:
        self.features = []
        for row in rows:
            inputs = tokenizer(row["input_text"], truncation=True, max_length=max_source_length)
            labels = tokenizer(text_target=row["output"], truncation=True, max_length=max_target_length)
            self.features.append(
                {
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"],
                    "labels": labels["input_ids"],
                }
            )

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.features[index]


def evaluate(
    rows: list[dict[str, str]],
    *,
    model: Any,
    tokenizer: Any,
    device: Any,
    calculate_metrics: Any,
    batch_size: int,
    max_source_length: int,
    max_target_length: int,
) -> tuple[dict[str, Any], list[str]]:
    import torch

    predictions: list[str] = []
    model.eval()
    for start in range(0, len(rows), batch_size):
        encoded = tokenizer(
            [row["input_text"] for row in rows[start : start + batch_size]],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_source_length,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            generated = model.generate(**encoded, max_length=max_target_length, num_beams=1)
        predictions.extend(tokenizer.batch_decode(generated.cpu(), skip_special_tokens=True))
    result: dict[str, Any] = {"overall": calculate_metrics(predictions, [row["output"] for row in rows])}
    by_category: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_category[row.get("category", "unknown")].append(index)
    result["category"] = {
        category: {
            "examples": len(indices),
            **calculate_metrics(
                [predictions[index] for index in indices],
                [rows[index]["output"] for index in indices],
            ),
        }
        for category, indices in sorted(by_category.items())
    }
    return result, predictions


def selection_score(metrics: dict[str, Any]) -> float:
    overall = metrics["overall"]
    score = 0.55 * overall["token_f1"] + 0.45 * overall["exact_match"]
    if overall["adult_reference_examples"]:
        score -= max(0.0, 0.95 - overall["adult_abstention_rate"])
    score -= 2.0 * overall["false_abstention_rate"]
    return score


def save_model(model: Any, tokenizer: Any, path: Path) -> None:
    original_cache = model.config.use_cache
    model.config.use_cache = True
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)
    model.config.use_cache = original_cache


def save_predictions(path: Path, rows: list[dict[str, str]], predictions: list[str]) -> None:
    payload = [
        {
            "example_id": row["example_id"],
            "category": row.get("category", ""),
            "tab_count": row["tab_count"],
            "input_titles": row["input_titles"],
            "input_keywords": row["input_keywords"],
            "reference": row["output"],
            "prediction": prediction,
        }
        for row, prediction in zip(rows, predictions)
    ]
    write_rows(path, payload, list(payload[0].keys()))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, default=DEFAULT_REPO_DIR)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--teacher-artifact", default=DEFAULT_TEACHER_ARTIFACT)
    parser.add_argument(
        "--teacher-cache-dir",
        type=Path,
        default=DEFAULT_REPO_DIR / "output" / "multitab-training-inputs" / "teacher",
    )
    parser.add_argument(
        "--generation-config-url", default=DEFAULT_GENERATION_CONFIG_URL
    )
    parser.add_argument(
        "--generation-config-cache",
        type=Path,
        default=DEFAULT_REPO_DIR
        / "output"
        / "multitab-training-inputs"
        / "generation_config.json",
    )
    parser.add_argument("--student-model", default="google/t5-efficient-small")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--max-source-length", type=int, default=512)
    parser.add_argument("--max-target-length", type=int, default=6)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--minimum-epochs", type=int, default=20)
    parser.add_argument("--sequence-kd-ratio", type=float, default=0.0)
    parser.add_argument("--sequence-kd-confidence", type=float, default=0.60)
    parser.add_argument("--sequence-kd-batch-size", type=int, default=16)
    parser.add_argument("--expected-encoder-layers", type=int, default=6)
    parser.add_argument("--expected-decoder-layers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--require-mps", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.smoke:
        args.epochs = 1
        args.eval_every = 1
        args.checkpoint_every = 1
        args.minimum_epochs = 1
        args.early_stopping_patience = 1
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    source_dir = str(args.repo_dir.resolve() / "src")
    if source_dir not in sys.path:
        sys.path.insert(0, source_dir)

    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, set_seed

    set_seed(args.seed)
    random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / "COMPLETE").is_file():
        print(f"Already complete: {args.output_dir}")
        return
    device = choose_device(args.require_mps)
    print(f"Training device: {device}", flush=True)

    rolling = args.output_dir / "checkpoint-last"
    resume = (rolling / "trainer_state.json").is_file()
    tokenizer_source = rolling if resume else args.student_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    generation_config = download_generation_config(
        args.generation_config_url, args.generation_config_cache
    )
    teacher_path = download_teacher_artifact(
        args.teacher_artifact, args.teacher_cache_dir
    )
    teacher = AutoModelForSeq2SeqLM.from_pretrained(teacher_path)
    teacher.resize_token_embeddings(len(tokenizer))
    apply_safety_config(teacher, generation_config)
    teacher.to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    raw_train = read_rows(args.dataset_dir / "train.csv")
    raw_dev = read_rows(args.dataset_dir / "dev.csv")
    raw_test = read_rows(args.dataset_dir / "test.csv")
    if args.smoke:
        smoke_rng = random.Random(args.seed)
        raw_train = smoke_rng.sample(raw_train, min(64, len(raw_train)))
        raw_dev = smoke_rng.sample(raw_dev, min(32, len(raw_dev)))
        raw_test = smoke_rng.sample(raw_test, min(32, len(raw_test)))
    train_rows, dropped_train = deployable_rows(raw_train, tokenizer, args.max_target_length)
    dev_rows, dropped_dev = deployable_rows(raw_dev, tokenizer, args.max_target_length)
    test_rows, dropped_test = deployable_rows(raw_test, tokenizer, args.max_target_length)
    pseudo_rows, sequence_summary = sequence_kd_rows(
        train_rows,
        teacher=teacher,
        tokenizer=tokenizer,
        device=device,
        cache_path=args.output_dir / "prepared" / "teacher_sequence.csv",
        ratio=args.sequence_kd_ratio,
        confidence_threshold=args.sequence_kd_confidence,
        batch_size=args.sequence_kd_batch_size,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
        seed=args.seed,
    )
    for row in train_rows:
        row["target_source"] = "hard"
    train_rows = train_rows + pseudo_rows

    student = AutoModelForSeq2SeqLM.from_pretrained(rolling if resume else args.student_model)
    student.resize_token_embeddings(len(tokenizer))
    apply_safety_config(student, generation_config)
    student.config.use_cache = False
    if args.expected_encoder_layers is not None and student.config.num_layers != args.expected_encoder_layers:
        raise ValueError(f"expected {args.expected_encoder_layers} encoder layers, got {student.config.num_layers}")
    if args.expected_decoder_layers is not None and student.config.num_decoder_layers != args.expected_decoder_layers:
        raise ValueError(f"expected {args.expected_decoder_layers} decoder layers, got {student.config.num_decoder_layers}")
    student.to(device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    start_epoch = 1
    global_step = 0
    best_score = float("-inf")
    best_epoch = 0
    stale_evals = 0
    if resume:
        state = json.loads((rolling / "trainer_state.json").read_text())
        optimizer.load_state_dict(torch.load(rolling / "optimizer.pt", map_location="cpu", weights_only=True))
        for optimizer_state in optimizer.state.values():
            for key, value in optimizer_state.items():
                if key != "step" and torch.is_tensor(value):
                    optimizer_state[key] = value.to(device)
        start_epoch = int(state["epoch"]) + 1
        global_step = int(state["global_step"])
        best_score = float(state.get("best_score", best_score))
        best_epoch = int(state.get("best_epoch", 0))
        stale_evals = int(state.get("stale_evals", 0))
        print(f"Resuming epoch={start_epoch} global_step={global_step}", flush=True)

    manifest = {
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "device": str(device),
        "teacher_parameters": sum(parameter.numel() for parameter in teacher.parameters()),
        "student_parameters": sum(parameter.numel() for parameter in student.parameters()),
        "student_encoder_layers": student.config.num_layers,
        "student_decoder_layers": student.config.num_decoder_layers,
        "student_d_model": student.config.d_model,
        "student_d_ff": student.config.d_ff,
        "blocked_output_sequences": len(student.generation_config.bad_words_ids),
        "data": {
            "hard_train": len(train_rows) - len(pseudo_rows),
            "sequence_teacher_train": len(pseudo_rows),
            "development": len(dev_rows),
            "test": len(test_rows),
            "dropped_over_token_limit": {
                "train": len(dropped_train), "dev": len(dropped_dev), "test": len(dropped_test)
            },
        },
        "sequence_kd": sequence_summary,
        "selection": "development only; golden and test are never used for checkpoint selection",
    }
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)

    dataset = TopicDataset(train_rows, tokenizer, args.max_source_length, args.max_target_length)
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=student, label_pad_token_id=-100, padding=True)
    metrics_path = args.output_dir / "epoch_metrics.jsonl"
    stopped_early = False
    for epoch in range(start_epoch, args.epochs + 1):
        generator = torch.Generator().manual_seed(args.seed + epoch)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, generator=generator, collate_fn=collator)
        student.train()
        totals = Counter()
        started = time.perf_counter()
        for batch_index, batch in enumerate(loader, start=1):
            optimizer.zero_grad(set_to_none=True)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            decoder_input_ids = student._shift_right(labels)
            with torch.no_grad():
                teacher_logits = teacher(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids,
                    use_cache=False,
                ).logits
            student_logits = student(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                use_cache=False,
            ).logits
            loss, kl_loss, ce_loss = distillation_loss(
                student_logits, teacher_logits, labels, alpha=args.alpha, temperature=args.temperature
            )
            loss.backward()
            if args.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(), args.gradient_clip)
            optimizer.step()
            global_step += 1
            totals.update(loss=loss.item(), kl=kl_loss.item(), ce=ce_loss.item())
            if batch_index % 25 == 0 or batch_index == len(loader):
                print(
                    f"epoch={epoch}/{args.epochs} batch={batch_index}/{len(loader)} "
                    f"loss={totals['loss']/batch_index:.4f} kl={totals['kl']/batch_index:.4f} "
                    f"ce={totals['ce']/batch_index:.4f}", flush=True
                )
        record: dict[str, Any] = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": totals["loss"] / len(loader),
            "kl_loss": totals["kl"] / len(loader),
            "ce_loss": totals["ce"] / len(loader),
            "wall_seconds": time.perf_counter() - started,
        }
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            dev_metrics, dev_predictions = evaluate(
                dev_rows,
                model=student,
                tokenizer=tokenizer,
                device=device,
                calculate_metrics=calculate_metrics,
                batch_size=args.batch_size,
                max_source_length=args.max_source_length,
                max_target_length=args.max_target_length,
            )
            score = selection_score(dev_metrics)
            record["development"] = dev_metrics
            record["selection_score"] = score
            save_predictions(args.output_dir / "predictions" / f"development_epoch_{epoch:03d}.csv", dev_rows, dev_predictions)
            if score > best_score + 1e-8:
                best_score, best_epoch, stale_evals = score, epoch, 0
                save_model(student, tokenizer, args.output_dir / "model-best")
                print(f"New best checkpoint: epoch={epoch} score={score:.6f}", flush=True)
            else:
                stale_evals += 1
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        print(json.dumps(record, indent=2, sort_keys=True), flush=True)

        save_model(student, tokenizer, rolling)
        torch.save(optimizer.state_dict(), rolling / "optimizer.pt")
        state = {
            "epoch": epoch,
            "global_step": global_step,
            "best_score": best_score,
            "best_epoch": best_epoch,
            "stale_evals": stale_evals,
        }
        (rolling / "trainer_state.json").write_text(json.dumps(state, indent=2) + "\n")
        if epoch % args.checkpoint_every == 0 or epoch == args.epochs:
            save_model(student, tokenizer, args.output_dir / "checkpoints" / f"epoch-{epoch:03d}")
        if device.type == "mps":
            torch.mps.empty_cache()
        if epoch >= args.minimum_epochs and stale_evals >= args.early_stopping_patience:
            print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}", flush=True)
            stopped_early = True
            break

    del student
    if device.type == "mps":
        torch.mps.empty_cache()
    best_model = AutoModelForSeq2SeqLM.from_pretrained(args.output_dir / "model-best").to(device)
    best_model.eval()
    test_metrics, test_predictions = evaluate(
        test_rows,
        model=best_model,
        tokenizer=tokenizer,
        device=device,
        calculate_metrics=calculate_metrics,
        batch_size=args.batch_size,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
    )
    save_predictions(args.output_dir / "test_predictions.csv", test_rows, test_predictions)
    shutil.copytree(args.output_dir / "model-best", args.output_dir / "model", dirs_exist_ok=True)
    final_report = {
        "best_epoch": best_epoch,
        "best_development_selection_score": best_score,
        "stopped_early": stopped_early,
        "test": test_metrics,
        "test_used_for_selection": False,
        "golden_used_for_selection": False,
    }
    (args.output_dir / "final_report.json").write_text(json.dumps(final_report, indent=2, sort_keys=True) + "\n")
    (args.output_dir / "COMPLETE").write_text("complete\n")
    print(json.dumps(final_report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
