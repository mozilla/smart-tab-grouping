"""
This file exports training data for the topic model from a set of user labeled groups.
"""
import pandas as pd

from labeled_data_utils import get_labeled_dataset, user_test_list
from tab_grouping.tab_titles import T5TopicGenerator
from topic_utils import create_topic_training_dataset

topic_generator = T5TopicGenerator()

result_dfs = []
for user_dataset_name in user_test_list:
    datasets, labeled_topics = get_labeled_dataset(user_dataset_name)
    for i in range(len(datasets)):
        dataset = datasets[i]
        cur_run_labeled_topics = labeled_topics[i]
        training_dataset = create_topic_training_dataset(dataset, "smart_group_label", topic_generator, predicted_id_topics=cur_run_labeled_topics)
        result_dfs.append(training_dataset)

all_users = pd.concat(result_dfs)
all_users.to_json("./output/label_training/all_users2.json", orient="records")
all_users.to_csv("./output/label_training/all_users2.csv")
