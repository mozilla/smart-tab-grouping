from metaflow import (
    FlowSpec,
    step,
    card,
    current, resources, kubernetes)
from metaflow.cards import Table

import pandas as pd

from distill_t5 import DistillTopicT5
from tune_bart import TuneTopicBart
from tune_t5 import TuneTopicT5
from tune_gpt2 import TuneTopicGPT2

from util.secrets import load_env
from util.storage import download_bucket_to_file, download_bucket_to_csv
from util.shorten_topic_length import ShortenTopicLength


def cleanup_wandb_args(config):
    def cleanup_bool(v):
        if v == "true":
            return True
        if v == "false":
            return False
        return v

    return {key: cleanup_bool(value) for (key, value) in config.items()}


TAB_GROUPING_BUCKET_NAME = "stage-fx-tab-grouping"
TUNING_DATA_PATHS = ["topic/topic_topic_fine_tuning_data__common_crawl_2025-02-23_08-18__filtered.csv",
                     "topic/topic_topic_fine_tuning_data__2025-02-21_16-50__filtered.csv",
                     "topic/search_simplified.csv"
                     ]  # "topic/topic_fine_tuning_data_extractive_2_15.csv"  # "topic/topic_fine_tuning_data_guided__02_11_processed.csv"

TUNING_DATA_PATHS_WITH_NONE = ["topic/fine_tuning_data__with_none__common_crawl_2025-02-23_08-18.csv",
                               "topic/common_corpus_noise_none_3_12.csv",
                               "topic/search_simplified.csv"
                               ]

NOISE_TRAINING_DATA_SET_INDEX = 1

SINGLE_TAB_VALIDATION_PATH = "topic/single_tab_validation.csv"


def create_trainer_for_config(config: dict[str, any]):
    if "t5" in config["model_name"]:
        if config.get("teacher_model_artifact", None) is None:
            return TuneTopicT5(**config)
        else:
            return DistillTopicT5(**config)
    if "gpt" in config["model_name"]:
        return TuneTopicGPT2(**config)
    if "bart" in config["model_name"]:
        return TuneTopicBart(**config)
    return None


class TuneGenTopicModel(FlowSpec):
    """
    A flow to tune the Generative Topic Model based on flan-t5-small
    This model is used for the Smart Tab Grouping project
    """

    @step
    def start(self):
        self.configs = [
            {
                "learning_rate": 1e-4,
                "batch_size": 8,
                "model_name": "google/t5-efficient-tiny",
                "label_column": "output",
                "use_keywords": True,
                "single_tab_handling": False,
                "learning_rate_decay": False,
                "shorten_training_label_boost": 0.05,
                "teacher_model_artifact": "moso/tab_grouping/model-uxfe87sy:v0" # noble-yogurt-330
            }
            ]
        self.configs = [
            {
                    "learning_rate": 3e-4,
                    "batch_size": 2,
                    "model_name": "google/flan-t5-base",
                    "label_column": "output",
                    "use_keywords": True,
                    "single_tab_handling": False,
                    "learning_rate_decay": False,
                    "shorten_training_label_boost": 0.05
            }
        ]

        self.next(self.train, foreach='configs')

    @kubernetes(image="us-docker.pkg.dev/moz-fx-mozsoc-ml-nonprod/metaflow-dockers/metaflow_gpu:onnx2",
                gpu_vendor="nvidia",
                gpu=1,
                memory=10240,
                disk=20240,
                cpu=2,
                )
    @card
    @step
    def train(self):
        """Extract feedback from prospecting given by curators"""
        train_config = self.input
        LABEL_MAX_LENGTH = 35

        load_env()
        print("Training Config: ")
        print(train_config)
        trainer = create_trainer_for_config(train_config)
        local_filename = "tuning_data.csv"
        training_files = TUNING_DATA_PATHS_WITH_NONE
        print(f"Using training files {training_files}")
        datasets = []
        for training_file in training_files:
            download_bucket_to_file(TAB_GROUPING_BUCKET_NAME, training_file, local_filename)
            datasets.append(pd.read_csv(local_filename, keep_default_na=False).fillna(""))
        datasets[NOISE_TRAINING_DATA_SET_INDEX] = datasets[NOISE_TRAINING_DATA_SET_INDEX].sample(n=500).reset_index(drop=True)  # reduce number a bit of None set

        topic_data = pd.concat(datasets, ignore_index=True).fillna("")
        topic_data = topic_data.drop_duplicates(subset=["input_titles"])
        shorten_boost = train_config.get("shorten_training_label_boost", None)
        if shorten_boost is not None:
            print(f"Shortening labels with setting {shorten_boost}")
            stl = ShortenTopicLength(shorten_boost)
            topic_data = stl.shorten_topics(topic_data)

        topic_data = topic_data[topic_data["output"].str.len() <= LABEL_MAX_LENGTH]
        self.topic_data = topic_data
        current.card.append(
            Table.from_dataframe(
                topic_data
            )
        )
        validation_data = download_bucket_to_csv(TAB_GROUPING_BUCKET_NAME, SINGLE_TAB_VALIDATION_PATH)
        trainer.setup_data(topic_data, validation_data)

        trainer.train()
        self.next(self.join)

    @step
    def join(self, inputs):
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == '__main__':
    TuneGenTopicModel()
