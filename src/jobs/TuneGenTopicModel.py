from metaflow import (
    FlowSpec,
    step,
    card,
    current, resources, kubernetes)
from metaflow.cards import Table

import pandas as pd

from util.secrets import load_env
from util.storage import download_bucket_to_file
from tune_code import GenTrainer


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
                     "topic/topic_topic_fine_tuning_data__2025-02-21_16-50__filtered.csv"]# "topic/topic_fine_tuning_data_extractive_2_15.csv"  # "topic/topic_fine_tuning_data_guided__02_11_processed.csv"

class TuneGenTopicModel(FlowSpec):
    """
    A flow to tune the Generative Topic Model based on flan-t5-small
    This model is used for the Smart Tab Grouping project
    """

    @step
    def start(self):
        self.configs = [
            {
                "learning_rate": 8e-4,
                "batch_size": 2,
                "model_name": "google/t5-efficient-tiny",
                "label_column": "output",
                "use_keywords": True,
                "single_tab_handling": False,
                "learning_rate_decay": True
            },
            {
                "learning_rate": 6e-4,
                "batch_size": 2,
                "model_name": "google/t5-efficient-tiny",
                "label_column": "output",
                "use_keywords": True,
                "single_tab_handling": False,
                "learning_rate_decay": True
            },
            {
                "learning_rate": 4e-4,
                "batch_size": 2,
                "model_name": "google/t5-efficient-tiny",
                "label_column": "output",
                "use_keywords": True,
                "single_tab_handling": False,
                "learning_rate_decay": True
            },
            {
                "learning_rate": 3e-4,
                "batch_size": 2,
                "model_name": "google/t5-efficient-tiny",
                "label_column": "output",
                "use_keywords": True,
                "single_tab_handling": False,
                "learning_rate_decay": True
            }
        ]

        self.next(self.train, foreach='configs')

    @resources(
        memory=10240,
        disk=20240,
        cpu=2,
    )
    @kubernetes(image="us-docker.pkg.dev/moz-fx-mozsoc-ml-nonprod/metaflow-dockers/metaflow_gpu:onnx2",
                gpu_vendor="nvidia",
                gpu=1
                )
    @card
    @step
    def train(self):
        """Extract feedback from prospecting given by curators"""
        train_config = self.input

        load_env()
        print("Training Config: ")
        print(train_config)
        trainer = GenTrainer(**train_config)
        local_filename = "tuning_data.csv"
        print(f"Using training files {TUNING_DATA_PATHS}")
        datasets = []
        for training_file in TUNING_DATA_PATHS:
            download_bucket_to_file(TAB_GROUPING_BUCKET_NAME, training_file, local_filename)
            datasets.append(pd.read_csv(local_filename).fillna(""))
        topic_data = pd.concat(datasets, ignore_index=True)

        current.card.append(
            Table.from_dataframe(
                topic_data
            )
        )
        trainer.setup_data(topic_data)
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
