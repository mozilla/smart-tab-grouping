from metaflow import (
    FlowSpec,
    step,
    card,
    current, resources, kubernetes)
from metaflow.cards import Table

import pandas as pd

from tune_bart import TuneTopicBart
from tune_t5 import TuneTopicT5
from tune_gpt2 import TuneTopicGPT2

from util.secrets import load_env
from util.storage import download_bucket_to_file, download_bucket_to_csv


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

SINGLE_TAB_VALIDATION_PATH = "topic/single_tab_validation.csv"

def create_trainer_for_config(config:dict[str, any]):
    if "t5" in config["model_name"]:
        return TuneTopicT5(**config)
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
                "learning_rate": 3e-4,
                "batch_size": 2,
                "model_name": "google/flan-t5-small",
                "label_column": "output",
                "use_keywords": True,
                "single_tab_handling": False,
                "learning_rate_decay": False,
                "shrink_decoder_index_remove": "6,5,4,3,2,0", # elated-lake
            },
            {
                "learning_rate": 3e-4,
                "batch_size": 2,
                "model_name": "google/flan-t5-small",
                "label_column": "output",
                "use_keywords": True,
                "single_tab_handling": False,
                "learning_rate_decay": False,
                "shrink_decoder_index_remove": "6,5,4,3,2,1", # new-test
            },
            {
                "learning_rate": 3e-4,
                "batch_size": 2,
                "model_name": "google/flan-t5-small",
                "label_column": "output",
                "use_keywords": True,
                "single_tab_handling": False,
                "learning_rate_decay": False,
                "shrink_encoder_index_remove": "1,3",
                "shrink_decoder_index_remove": "6,5,4,3,2,0",  # elated-lake
            },
            {
                "learning_rate": 3e-4,
                "batch_size": 2,
                "model_name": "google/flan-t5-small",
                "label_column": "output",
                "use_keywords": True,
                "single_tab_handling": False,
                "learning_rate_decay": False,
                "shrink_encoder_index_remove": "2,3,5",
                "shrink_decoder_index_remove": "6,5,4,3,2,1",  # new-test
            },
            {
                "learning_rate": 3e-4,
                "batch_size": 2,
                "model_name": "google/flan-t5-small",
                "label_column": "output",
                "use_keywords": True,
                "single_tab_handling": False,
                "learning_rate_decay": False,
                "shrink_encoder_index_remove": "3,7",
                "shrink_decoder_index_remove": "6,5,4,3,2,0",  # elated-lake
            },
            {
                "learning_rate": 3e-4,
                "batch_size": 2,
                "model_name": "google/flan-t5-small",
                "label_column": "output",
                "use_keywords": True,
                "single_tab_handling": False,
                "learning_rate_decay": False,
                "shrink_encoder_index_remove": "1,3,6",
                "shrink_decoder_index_remove": "6,5,4,3,2,1",  # new-test
            },
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
        LABEL_MAX_LENGTH = 35

        load_env()
        print("Training Config: ")
        print(train_config)
        trainer = create_trainer_for_config(train_config)
        local_filename = "tuning_data.csv"
        print(f"Using training files {TUNING_DATA_PATHS}")
        datasets = []
        for training_file in TUNING_DATA_PATHS:
            download_bucket_to_file(TAB_GROUPING_BUCKET_NAME, training_file, local_filename)
            datasets.append(pd.read_csv(local_filename).fillna(""))
        topic_data = pd.concat(datasets, ignore_index=True)
        topic_data = topic_data.drop_duplicates(subset=["input_titles"])
        topic_data = topic_data[topic_data["output"].str.len() <= LABEL_MAX_LENGTH]

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
