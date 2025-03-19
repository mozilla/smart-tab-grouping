import random
from datetime import datetime
from typing import List

import pandas as pd
from dotenv import load_dotenv
import os

from util.tab_titles import OpenAITopicGenerator, keyword_prompts
from util.key_document_finder import KeyDocumentFinder

LOW_COUNTS_FRACTION = 0.33

SYNTHETIC_DATA_PATH = "./data/synthetic_datasets"

include_additional_articles = False  # Include other non synthetic datasources


class TabTitleTrainingData:
    num_representative_docs = 3

    def get_meta_info_for_task(self, one_test, task_id, max_docs):
        docs_df = one_test[one_test.task_id == task_id].reset_index(drop=True)
        docs = docs_df["title"].to_list()
        random.shuffle(docs)
        docs = docs[:max_docs]
        descriptions = None
        # keyword generator
        if len(docs) > 1:
            key_doc_finder_ai = KeyDocumentFinder(docs_df, "task_id", "title")
            key_doc_finder_ai.compute_all(include_embeddings=False)
            keywords = key_doc_finder_ai.get_keywords_for_group(task_id)[:3]
        else:
            keywords = []
        if len(docs) == 1 and "description" in docs_df.columns:
            descriptions = docs_df["description"].to_list()[:1]
        return {"documents": docs, "keywords": keywords, "descriptions": descriptions}

    def compute_training_data_for_tests(self, ai_dataset, test_ids, max_docs):
        topic_gen = OpenAITopicGenerator(support_keywords=True)

        hint_db = pd.read_csv("./data/topic_model_fine_tune/topic_fine_tuning_data__01_05__grouped_with_hints.csv")
        topic_gen.prepare_hint_data(hint_db)

        results = []
        for ai_test_id in test_ids:
            one_test = ai_dataset[ai_dataset.test_set_id == ai_test_id].reset_index(drop=True)
            for task_id in one_test["task_id"].unique().tolist():
                cluster_data = self.get_meta_info_for_task(one_test, task_id, max_docs)
                if cluster_data is None:
                    print("Skipping invalid / missing metadata for item")
                    continue
                print(cluster_data)

                topic = topic_gen.get_topic(cluster_data)
                print(f"AI topic is: {topic}")

                cluster_data["keywords"] = list(filter(lambda a: a != "2023", cluster_data["keywords"]))

                input_for_fine_tuning_keywords = ",".join(cluster_data["keywords"][:3])
                input_for_fine_tuning_titles = "\n".join(cluster_data["documents"][:3])
                input_for_fine_tuning_description = "\n".join(cluster_data["descriptions"][:3])

                results.append({
                    "input_titles": input_for_fine_tuning_titles,
                    "input_keywords": input_for_fine_tuning_keywords,
                    "input_description": input_for_fine_tuning_description,
                    "output": topic,
                })
        return results

    def gen_data_single_document(self, dataset: pd.DataFrame, limit=0):
        for col in ['id', 'task', 'task_id', 'test_set_id']:
            if col not in dataset.columns:
                dataset[col] = dataset.index
        results = []
        test_ids = dataset["test_set_id"].unique().tolist()
        if limit > 0:
            test_ids = test_ids[:limit]
        results.extend(self.compute_training_data_for_tests(dataset, test_ids, 1))
        return pd.DataFrame(results)

    def gen_data_multi_document(self, dataset: pd.DataFrame, limit=0, low_counts_fraction=LOW_COUNTS_FRACTION):
        def split_array(arr: List[any], split_index):
            one = arr[:split_index]
            two = arr[split_index:]
            return one, two

        test_ids = dataset["test_set_id"].unique().tolist()
        num_total_items = len(test_ids)
        if limit > 0:
            num_total_items = min(num_total_items, limit)
            test_ids = test_ids[:num_total_items]

        random.shuffle(test_ids)
        num_one_and_two_article = int(len(test_ids) * low_counts_fraction)

        one_article_test_ids, test_ids = split_array(test_ids, num_one_and_two_article)
        two_article_test_ids, test_ids = split_array(test_ids, num_one_and_two_article)

        print(one_article_test_ids)
        print(two_article_test_ids)
        print(test_ids)

        results = []
        print("*** Setting up 1 articles in cluster ")
        results.extend(self.compute_training_data_for_tests(dataset, one_article_test_ids, 1))

        print("*** Setting up 2 articles in cluster ")
        results.extend(self.compute_training_data_for_tests(dataset, two_article_test_ids, 2))

        print(f"*** {self.num_representative_docs} articles in cluster ")
        results.extend(
            self.compute_training_data_for_tests(dataset, test_ids, self.num_representative_docs))
        return pd.DataFrame(results)


def create_tuning_data_for_synthetic_dataset():
    multi_doc_dataset = pd.concat([
        pd.read_csv(os.path.join(SYNTHETIC_DATA_PATH, "gen_test_set_2__12_9.csv")),
        pd.read_csv(os.path.join(SYNTHETIC_DATA_PATH, "gen_test_set_3_12_9_user_locs.csv"))
    ], axis=0).reset_index(drop=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    gen = TabTitleTrainingData()
    result = gen.gen_data(multi_doc_dataset)
    result.to_csv(f"./data/topic_model_fine_tune/topic_fine_tuning_data__{timestamp}.csv")


def create_tuning_data_for_single_item_common_crawl():
    #  "pocket_no_article_parsed__10_50_v_2024_processed.csv")
    df = pd.read_csv("./data/external/common_crawl.csv")
    df["description"] = df["description"].fillna("")
    df = df.dropna(subset=["title"]).reset_index(drop=True)
    df["description"] = df["description"].apply(lambda a: "" if a == "No Description" else a)
    gen = TabTitleTrainingData()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    result = gen.gen_data_single_document(df)
    result.to_csv(f"./data/topic_model_fine_tune/topic_fine_tuning_data__common_crawl_{timestamp}.csv")


if __name__ == "__main__":
    load_dotenv()
    create_tuning_data_for_single_item_common_crawl()
