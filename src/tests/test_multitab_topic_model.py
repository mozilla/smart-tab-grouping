from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch

from jobs import BuildMultiTabTopicDataset as builder
from jobs import TuneGenTopicModelMultiTab as trainer


def test_masked_distillation_ignores_padded_positions():
    torch.manual_seed(4)
    student = torch.randn(1, 3, 13)
    teacher = torch.randn(1, 3, 13)
    labels = torch.tensor([[4, 1, -100]])
    first = trainer.distillation_loss(
        student, teacher, labels, alpha=0.9, temperature=2.0
    )[0]
    changed_student = student.clone()
    changed_teacher = teacher.clone()
    changed_student[:, 2] = torch.randn(13) * 100
    changed_teacher[:, 2] = torch.randn(13) * 100
    second = trainer.distillation_loss(
        changed_student, changed_teacher, labels, alpha=0.9, temperature=2.0
    )[0]
    assert torch.allclose(first, second)


def test_exact_title_owners_prefer_test_then_dev():
    rows = [
        {"split": "train", "input_titles": "Same\nTrain"},
        {"split": "dev", "input_titles": "Same\nDev"},
        {"split": "test", "input_titles": "Same\nTest"},
    ]
    owners = builder.assign_exact_title_owners(rows)
    assert owners["same"] == "test"
    assert owners["train"] == "train"


def test_selection_score_penalizes_unsafe_adult_outputs():
    safe = {
        "overall": {
            "token_f1": 0.5,
            "exact_match": 0.4,
            "adult_reference_examples": 10,
            "adult_abstention_rate": 1.0,
            "false_abstention_rate": 0.0,
        }
    }
    unsafe = {"overall": {**safe["overall"], "adult_abstention_rate": 0.5}}
    assert trainer.selection_score(safe) > trainer.selection_score(unsafe)


def test_default_recipe_is_full_small6_online_kd():
    args = trainer.parse_args(
        [
            "--dataset-dir",
            "dataset",
            "--output-dir",
            "output",
        ]
    )
    assert args.teacher_artifact == "moso/tab_grouping/model-v40xoz3q:v0"
    assert args.generation_config_url.endswith("v0.7.4/generation_config.json")
    assert args.student_model == "google/t5-efficient-small"
    assert args.expected_encoder_layers == 6
    assert args.expected_decoder_layers == 6
    assert args.epochs == 40
    assert args.sequence_kd_ratio == 0.0
    assert args.max_target_length == 6


def test_metrics_treat_safety_sentinels_as_firefox_abstentions():
    metrics = trainer.calculate_metrics(
        ["Adult Content", "None", "Travel"],
        ["Adult Content", "None", "Travel"],
    )
    assert metrics["adult_abstention_rate"] == 1.0
    assert metrics["false_abstention_rate"] == 0.0
    assert metrics["firefox_visible_exact_match"] == 1.0


def test_gcs_inputs_download_to_output_and_are_reused(tmp_path: Path):
    calls = []

    def download(bucket: str, blob: str, destination: str) -> None:
        calls.append((bucket, blob))
        Path(destination).write_text(blob, encoding="utf-8")

    paths = builder.download_gcs_inputs(tmp_path, download)
    assert set(paths) == set(builder.GCS_TOPIC_FILES)
    assert all(path.is_file() for path in paths.values())
    assert all(bucket == builder.GCS_BUCKET for bucket, _ in calls)

    builder.download_gcs_inputs(tmp_path, download)
    assert len(calls) == len(builder.GCS_TOPIC_FILES)


def test_teacher_artifact_downloads_once(tmp_path: Path):
    calls = []

    class Artifact:
        def download(self, root: str) -> str:
            destination = Path(root)
            destination.mkdir(parents=True)
            (destination / "config.json").write_text(json.dumps({"model": "teacher"}))
            return root

    class Api:
        def artifact(self, name: str, type: str) -> Artifact:
            calls.append((name, type))
            return Artifact()

    destination = tmp_path / "teacher"
    assert trainer.download_teacher_artifact("teacher:v0", destination, Api) == destination
    assert trainer.download_teacher_artifact("teacher:v0", destination, Api) == destination
    assert calls == [("teacher:v0", "model")]


def test_generation_config_downloads_once_and_requires_blocked_words(tmp_path: Path):
    calls = []
    payload = {"bad_words_ids": [[1], [2, 3]], "eos_token_id": 1}

    def get(url: str, timeout: int):
        calls.append((url, timeout))
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: payload,
        )

    destination = tmp_path / "generation_config.json"
    assert trainer.download_generation_config("https://example.test/config", destination, get) == payload
    assert trainer.download_generation_config("https://example.test/config", destination, get) == payload
    assert calls == [("https://example.test/config", 60)]
