# Multi-tab topic-model distillation

`TuneGenTopicModelMultiTab.py` is the reproducible local trainer for the
multi-tab naming experiments. It is separate from the existing Metaflow
`TuneGenTopicModel.py` job. The default recipe is the selected Small6 online-KD
model: full `google/t5-efficient-small` with six encoder and six decoder layers,
masked token KL against the existing teacher, and hard-label cross entropy.

The builder creates only 2-10-tab examples using the Firefox-compatible prompt
and c-TF-IDF implementation. It assigns related variants to one split, removes
exact and character-ngram title leakage, and targets a 60/20/15/5 mix of clean,
noisy, `None`, and `Adult Content` groups. The two sentinel labels become empty
strings in Firefox. Checkpoint selection uses the development split only; the
test split is never used for selection.

## Inputs

- The builder downloads the four required training and validation CSVs from
  `gs://stage-fx-tab-grouping/topic/` with the repository's existing GCS helper.
- Historical inputs come only from the tracked `data/external`, `data/extract`,
  and `data/synthetic_datasets` directories.
- The held-out golden naming set is stored at
  `gs://stage-fx-tab-grouping/topic/eval/golden_tab_grouping.json`.
- The trainer downloads `moso/tab_grouping/model-v40xoz3q:v0` from W&B.
- The exact deployed v0.7.4 generation config, including all 1,466 blocked
  output sequences, is downloaded from Mozilla's public model hub.

Authenticate once before running:

```bash
gcloud auth application-default login
wandb login
```

All downloads, datasets, checkpoints, and models are written below the ignored
`output/` directory. No untracked local data is consumed.

## Run Small6 online KD

```bash
./experiments/run_multitab_distillation.sh
```

The run is resumable from `checkpoint-last`, evaluates every five epochs,
saves five-epoch checkpoints, starts early-stopping checks after epoch 20, and
stops after three development evaluations without improvement. The selected
Hugging Face model is written to `output/multitab-small6-online/model`.

Useful variants:

```bash
# Confidence-filtered teacher sequence targets plus online token KL.
SEQUENCE_KD_RATIO=0.5 OUTPUT_DIR=output/multitab-small6-sequence-kd \
  ./experiments/run_multitab_distillation.sh

# Full 8+8 tiny model.
STUDENT_MODEL=google/t5-efficient-tiny-nl8 EXPECTED_LAYERS=8 BATCH_SIZE=16 \
  OUTPUT_DIR=output/multitab-tiny-nl8-online \
  ./experiments/run_multitab_distillation.sh

# CUDA or CPU host.
REQUIRE_MPS=0 ./experiments/run_multitab_distillation.sh

# Redownload the canonical GCS inputs before rebuilding the dataset.
REFRESH_DOWNLOADS=1 ./experiments/run_multitab_distillation.sh
```

The selected model directory includes its tokenizer and generated
`bad_words_ids`, and is ready for the repository's existing ONNX/Q8 export
workflow.

## Golden-set evaluation

Download the canonical golden set to the ignored output directory:

```bash
mkdir -p output/eval
gcloud storage cp \
  gs://stage-fx-tab-grouping/topic/eval/golden_tab_grouping.json \
  output/eval/golden_tab_grouping.json
```

Use this set only for the final comparison after selecting a checkpoint from
the development split. Do not use golden-set results for training, early
stopping, hyperparameter tuning, or checkpoint selection.
