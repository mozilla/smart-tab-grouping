#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$REPO_DIR/.venv/bin/python}"
INPUT_DIR="${INPUT_DIR:-$REPO_DIR/output/multitab-training-inputs}"
DATASET_DIR="${DATASET_DIR:-$REPO_DIR/output/multitab-topic-dataset}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/output/multitab-small6-online}"
TEACHER_ARTIFACT="${TEACHER_ARTIFACT:-moso/tab_grouping/model-v40xoz3q:v0}"
GENERATION_CONFIG_URL="${GENERATION_CONFIG_URL:-https://model-hub.mozilla.org/mozilla/smart-tab-topic/v0.7.4/generation_config.json}"
STUDENT_MODEL="${STUDENT_MODEL:-google/t5-efficient-small}"
EPOCHS="${EPOCHS:-40}"
BATCH_SIZE="${BATCH_SIZE:-8}"
EXPECTED_LAYERS="${EXPECTED_LAYERS:-6}"
SEQUENCE_KD_RATIO="${SEQUENCE_KD_RATIO:-0.0}"
REQUIRE_MPS="${REQUIRE_MPS:-1}"
REFRESH_DOWNLOADS="${REFRESH_DOWNLOADS:-0}"

export PYTHONPATH="$REPO_DIR/src"
export PYTORCH_ENABLE_MPS_FALLBACK=1
export TOKENIZERS_PARALLELISM=false

if [[ "$REFRESH_DOWNLOADS" == "1" || ! -f "$DATASET_DIR/audit.json" ]]; then
  dataset_args=(
    --download-dir "$INPUT_DIR/gcs"
    --output-dir "$DATASET_DIR"
    --seed 42
  )
  if [[ "$REFRESH_DOWNLOADS" == "1" ]]; then
    dataset_args+=(--refresh-downloads)
  fi
  "$PYTHON" "$REPO_DIR/src/jobs/BuildMultiTabTopicDataset.py" "${dataset_args[@]}"
fi

training_args=(
  --dataset-dir "$DATASET_DIR"
  --teacher-artifact "$TEACHER_ARTIFACT"
  --teacher-cache-dir "$INPUT_DIR/teacher"
  --generation-config-url "$GENERATION_CONFIG_URL"
  --generation-config-cache "$INPUT_DIR/generation_config.json"
  --student-model "$STUDENT_MODEL"
  --output-dir "$OUTPUT_DIR"
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --learning-rate 1e-4
  --alpha 0.9
  --temperature 2.0
  --max-source-length 512
  --max-target-length 6
  --eval-every 5
  --checkpoint-every 5
  --early-stopping-patience 3
  --minimum-epochs 20
  --sequence-kd-ratio "$SEQUENCE_KD_RATIO"
  --sequence-kd-confidence 0.60
  --expected-encoder-layers "$EXPECTED_LAYERS"
  --expected-decoder-layers "$EXPECTED_LAYERS"
  --seed 42
)
if [[ "$REQUIRE_MPS" == "1" ]]; then
  training_args+=(--require-mps)
fi

"$PYTHON" "$REPO_DIR/src/jobs/TuneGenTopicModelMultiTab.py" "${training_args[@]}"
