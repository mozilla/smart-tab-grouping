import random
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv
import os

from util.tab_titles import OpenAITopicGenerator, keyword_prompts
from util.key_document_finder import KeyDocumentFinder

LOW_COUNTS_FRACTION = 0.1

SYNTHETIC_DATA_PATH = "./data/synthetic_datasets"

include_additional_articles = False  # Include other non synthetic datasources

class TabTitleTrainingData:
    num_representative_docs = 3

    def get_meta_info_for_task(self, one_test, task_id, max_docs):
        docs_df = one_test[one_test.task_id == task_id].reset_index(drop=True)
        docs = docs_df["title"].to_list()
        random.shuffle(docs)
        docs = docs[:max_docs]
        # keyword generator
        if len(docs) > 1:
            key_doc_finder_ai = KeyDocumentFinder(docs_df, "task_id", "title")
            key_doc_finder_ai.compute_all(include_embeddings=False)
            keywords = key_doc_finder_ai.get_keywords_for_group(task_id)[:3]
        else:
            keywords = []
        return {"documents": docs, "keywords": keywords}

    def compute_training_data_for_tests(self, ai_dataset, test_ids, max_docs):
        topic_gen = OpenAITopicGenerator(support_keywords=True)
        results = []
        for ai_test_id in test_ids:
            one_test = ai_dataset[ai_dataset.test_set_id == ai_test_id].reset_index(drop=True)
            for task_id in one_test["task_id"].unique().tolist():
                cluster_data = self.get_meta_info_for_task(one_test, task_id, max_docs)
                print(cluster_data)

                topic = topic_gen.get_topic(cluster_data)
                print(f"AI topic is: {topic}")

                input_for_fine_tuning_keywords = ",".join(cluster_data["keywords"][:3])
                input_for_fine_tuning_titles = "\n".join(cluster_data["documents"][:3])

                results.append({
                    "input_titles": input_for_fine_tuning_titles,
                    "input_keywords": input_for_fine_tuning_keywords,
                    "output": topic,
                })
        return results

    def gen_data(self, limit=0):
        ai_dataset = pd.concat([
            pd.read_csv(os.path.join(SYNTHETIC_DATA_PATH, "gen_test_set_2__12_9.csv")),
            pd.read_csv(os.path.join(SYNTHETIC_DATA_PATH, "gen_test_set_3_12_9_user_locs.csv"))
        ], axis=0).reset_index(drop=True)
        test_ids = ai_dataset["test_set_id"].unique().tolist()
        test_ids_first_pass = test_ids
        if limit > 0:
            test_ids_first_pass = test_ids_first_pass[:limit]
        results = []
        print(f"*** {self.num_representative_docs} articles in cluster ")
        results.extend(self.compute_training_data_for_tests(ai_dataset, test_ids_first_pass, self.num_representative_docs))

        # 2 articles
        print("*** 2 articles in cluster ")
        random.shuffle(test_ids)
        two_articles_tests = test_ids[:int(len(test_ids) * LOW_COUNTS_FRACTION)]
        if limit > 0:
            two_articles_tests = two_articles_tests[:limit]
        results.extend(self.compute_training_data_for_tests(ai_dataset, two_articles_tests, 2))

        print("*** 1 articles in cluster ")
        # one article
        random.shuffle(test_ids)
        one_article_tests = test_ids[:int(len(test_ids) * LOW_COUNTS_FRACTION)]
        if limit > 0:
            one_article_tests = one_article_tests[:limit]
        results.extend(self.compute_training_data_for_tests(ai_dataset, one_article_tests, 1))

        if include_additional_articles:
            print("*** special cluster ")
            # one article only dataset (no clusters)
            single_article_dataset = pd.read_csv(os.path.join(SYNTHETIC_DATA_PATH, "pocket_no_article_parsed__10_50_v_2024_processed.csv"))
            single_article_test_ids = single_article_dataset["test_set_id"].unique().tolist()
            if limit > 0:
                single_article_test_ids = single_article_test_ids[:limit]
            results.extend(self.compute_training_data_for_tests(single_article_dataset, single_article_test_ids, 1))

        return pd.DataFrame(results)

if __name__ == "__main__":
    load_dotenv()
    gen = TabTitleTrainingData()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    result = gen.gen_data()
    result.to_csv(f"./test_data/topic_fine_tuning_data__{timestamp}.csv")

