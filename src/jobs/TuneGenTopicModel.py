from metaflow import (
    FlowSpec,
    step,
    card,
    current)
from metaflow.cards import Table

import pandas as pd
from common.secrets import load_env
from common.deploy import kresources
from common.storage import download_bucket_to_file
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
TUNING_DATA_PATH = "topic/topic_fine_tuning_data_extractive_2_15.csv" # "topic/topic_fine_tuning_data_guided__02_11_processed.csv"
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
                "single_tab_handling": True
            },
            {
                "learning_rate": 3e-4,
                "batch_size": 2,
                "model_name": "google/flan-t5-small",
                "label_column": "output",
                "use_keywords": True,
                "single_tab_handling": False
            },
            {
                "learning_rate": 2e-4,
                "batch_size": 2,
                "model_name": "google/flan-t5-small",
                "label_column": "output",
                "use_keywords": True,
                "single_tab_handling": False
            },
            {
                "learning_rate": 4e-4,
                "batch_size": 2,
                "model_name": "google/flan-t5-small",
                "label_column": "output",
                "use_keywords": True,
                "single_tab_handling": False
            },
            {
                "learning_rate": 4e-4,
                "batch_size": 4,
                "model_name": "google/flan-t5-small",
                "label_column": "output",
                "use_keywords": True,
                "single_tab_handling": False
            }
        ]
        # self.configs = self.configs[:1]
        self.next(self.train, foreach='configs')

    @kresources(
        gpu="1",
        memory=10240,
        disk=20240,
        cpu=2,
        gpu_vendor="nvidia",
        image="us-docker.pkg.dev/moz-fx-mozsoc-ml-nonprod/metaflow-dockers/metaflow_gpu:onnx2",
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
        print(f"Using training file {TUNING_DATA_PATH}")
        download_bucket_to_file(TAB_GROUPING_BUCKET_NAME, TUNING_DATA_PATH, local_filename)
        topic_data = pd.read_csv(local_filename).fillna("")
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
