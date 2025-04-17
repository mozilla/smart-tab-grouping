from metaflow import (
    FlowSpec,
    step,
    card,
    current, resources, kubernetes, nvidia, conda)
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

UNLABELED_DATA_PATHS = ["topic/common_crawl_unlabeled_00000.csv", "topic/common_crawl_unlabeled_00002.csv", "topic/common_crawl_unlableled_00003.csv",
                        "topic/common_crawl_unlabeled_00004.csv", "topic/common_crawl_unlabeled_00005.csv"]

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

    @nvidia()
    @conda(python='3.11.9',
           libraries={
               'pandas': '1.5.3',
               'numpy': '1.26.4',
               'nltk': '3.9.1',
               'transformers': '4.40.2',
               'tqdm': '4.66.5',
               "pytorch::pytorch-cuda": "12.4",
               "pytorch::pytorch": "2.4.0",
               'scikit-learn': '1.5.1',
               'datasets': '2.19.2',
               'wandb': '0.16.6',
               'pydantic': '2.8.2',
               'conda-forge::sentencepiece': '0.2.0',
               'conda-forge::google-cloud-storage': '3.1.0',
               'conda-forge::pyspellchecker': '0.8.0',
               'conda-forge::google-cloud-secret-manager': '2.23.2',
               'conda-forge::rouge-score': '0.1.2',
               'conda-forge::python-dotenv': '1.1.0'
           })
    @step
    def start(self):
        self.configs = [
            {
                "learning_rate": 4e-4,
                "batch_size": 32,
                "model_name": "google/t5-efficient-tiny",
                "label_column": "output",
                "use_keywords": True,
                "single_tab_handling": False,
                "learning_rate_decay": False,
                "shorten_training_label_boost": 0.05,
                "teacher_model_artifact": "moso/tab_grouping/model-9lc3togr:v0", # azure-frost-334 ___ "moso/tab_grouping/model-uxfe87sy:v0" # noble-yogurt-330
                "shrink_encoder_index_remove": "1,3",
                "shrink_decoder_index_remove": "1,3",
            }
            ]
        self._skip_configs = [
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

    @nvidia()
    @conda(python='3.11.9',
           libraries={
               'pandas': '1.5.3',
               'numpy': '1.26.4',
               'nltk': '3.9.1',
               'transformers': '4.40.2',
               'tqdm': '4.66.5',
               "pytorch::pytorch-cuda": "12.4",
               "pytorch::pytorch": "2.4.0",
               'scikit-learn': '1.5.1',
               'datasets': '2.19.2',
               'wandb': '0.16.6',
               'pydantic': '2.8.2',
               'conda-forge::sentencepiece': '0.2.0',
               'conda-forge::google-cloud-storage': '3.1.0',
               'conda-forge::pyspellchecker': '0.8.0',
               'conda-forge::google-cloud-secret-manager': '2.23.2',
               'conda-forge::rouge-score': '0.1.2',
               'conda-forge::python-dotenv': '1.1.0'
           })
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
        def get_datasets(files):
            datasets = []
            print(f"Loading files files {files}")
            for training_file in files:
                download_bucket_to_file(TAB_GROUPING_BUCKET_NAME, training_file, local_filename)
                datasets.append(pd.read_csv(local_filename, keep_default_na=False).fillna(""))
            df = pd.concat(datasets, ignore_index=True).fillna("")
            return df

        #datasets[NOISE_TRAINING_DATA_SET_INDEX] = datasets[NOISE_TRAINING_DATA_SET_INDEX].sample(n=500).reset_index(drop=True)  # reduce number a bit of None set

        topic_data = get_datasets(TUNING_DATA_PATHS)
        topic_data = topic_data.drop_duplicates(subset=["input_titles"])

        unlabeled_data = get_datasets(UNLABELED_DATA_PATHS)
        unlabeled_data.loc[:, "input_keywords"] = ""
        unlabeled_data["input_titles"] = unlabeled_data["title"]
        unlabeled_data = unlabeled_data.drop_duplicates(subset=["input_titles"]).reset_index(drop=True)

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
        trainer.setup_data(topic_data,
                           validation=validation_data,
                           unlabeled=unlabeled_data)

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
