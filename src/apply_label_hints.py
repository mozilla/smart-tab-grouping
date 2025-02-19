import os
import pandas as pd
from dotenv import load_dotenv

from tab_grouping.tab_titles import OpenAITopicGenerator

TEST_PATH = "./test_data"


def apply_hints():
    hinted_dataset = pd.read_csv(
        os.path.join(TEST_PATH, "topic_fine_tuning_data__01_05__grouped.csv"))
    topic_gen = OpenAITopicGenerator(support_keywords=True, support_hints=True)
    topic_gen.prepare_hint_data(hinted_dataset)
    hinted_dataset = hinted_dataset.fillna("")
    subset_to_test = hinted_dataset
    subset_to_test["hint_guided_output"] = subset_to_test.apply(lambda a:
                                                                topic_gen.get_topic({
                                                                    "documents": a.input_titles,
                                                                    "keywords": a.input_keywords
                                                                }), axis=1)
    subset_to_test.to_csv("guided_output_2_10.csv")


if __name__ == "__main__":
    load_dotenv()
    apply_hints()
