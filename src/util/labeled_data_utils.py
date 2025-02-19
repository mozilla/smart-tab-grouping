import json
from os import path
from typing import Tuple, List, Dict
import pandas as pd

user_test_list = ["jp", "a", "e", "g", "l", "p", "j"]

BATCH_ONE = "AI Gen Batch Set 1"
BATCH_TWO = "AI Gen Batch Set 2"
BATCH_ALL = "AI Gen Batch All"
BATCH_USER_TESTS = "All User Tests"


def find_domain(article_url, top_domain=True):
    """
    Returns a domain name from the article
    """
    https_url_start = "https://"
    http_url_start = "http://"

    if article_url.startswith(https_url_start):
        article_url = article_url.replace(https_url_start, "")
    elif article_url.startswith(http_url_start):
        article_url = article_url.replace(http_url_start, "")
    domain_items = article_url.split("/")[0].split(".")
    if domain_items[0] == "www":
        domain_items = domain_items[1:]
    if not top_domain:
        return ".".join(domain_items)
    if (
            len(domain_items[-1]) == 2 or len(domain_items) <= 2
    ):  # 2 char item likely a country domain -- return whole domain
        return ".".join(domain_items[-3:])
    else:
        return ".".join(domain_items[-2:])


def get_doc_name(s: str):
    """
    Page title processor that extracts the title of a document from the page title
    Only works with certain title types like google docs.

    Args:
        s: Page title string
    Returns: Document name
    """
    if s.find(" | ") > 0:
        s = s.split(" | ")[0]
    if s.find(" - ") > 0:
        s = s.split(" - ")[0]
    if s.find(" – ") > 0:
        s = s.split(" – ")[0]
    if s.find(" · ") > 0:
        s = s.split(" · ")[0]
    return s


def get_browse_group_from_history(history_df: pd.DataFrame) -> dict[str, int]:
    """
    This is a utility function that takes in browsing history as a dataframe
    and creates a 'browse_group' column based on the provenance of the browsing.

    Use the id column to represent the id of a page view, and from_visit to indicate
    the id that we came from to get to that id.
    """
    cur_assignments = {}
    # create list of mappings
    cur_index = 0
    all_ids = set(history_df["id"].to_list())
    for index, row in history_df.iterrows():
        if row.from_visit == 0 or row.from_visit not in all_ids:
            cur_assignments[row.id] = cur_index
            cur_index += 1
    num_changes = 1
    while num_changes > 0:
        num_changes = 0
        for index, row in history_df.iterrows():
            if row.id not in cur_assignments and row.from_visit in cur_assignments:
                cur_assignments[row.id] = cur_assignments[row.from_visit]
                num_changes += 1
    history_df["browse_group"] = history_df["id"].apply(lambda x: cur_assignments.get(x))
    result_dict = dict(zip(history_df.url, history_df.browse_group))
    return result_dict


def get_labeled_dataset(dataset_name: str) -> Tuple[List[pd.DataFrame], List[Dict[str, str]]]:
    from functools import partial
    if dataset_name is None:
        return []
    final_datasets = []
    category_title_datasets = []
    if dataset_name == BATCH_ALL or dataset_name == BATCH_ONE or dataset_name == BATCH_TWO:
        ai_dataset = pd.read_csv("./data/synthetic_datasets/gen_test_set_1__10_4.csv")
        test_ids = ai_dataset["test_set_id"].unique().tolist()
        ai_dataset["smart_group_label"] = ai_dataset["task"]  # TODO - consider task_id
        for ai_test_id in test_ids:
            one_test = ai_dataset[ai_dataset.test_set_id == ai_test_id].reset_index(drop=True)
            final_datasets.append(one_test)
            category_title_datasets.append({})
    else:
        dataset_names = user_test_list if dataset_name == BATCH_USER_TESTS else [dataset_name]
        for d in dataset_names:
            filename = f"./data/individual_tests/{d}.json"
            if not path.isfile(filename):
                filename = f"./data/individual_tests/private/{d}.json"
            with open(filename) as f:
                raw_data = json.load(f)
                if "tab_list" in raw_data:
                    tabs = raw_data["tab_list"]
                    category_titles = raw_data["group_titles"]
                else:
                    tabs = raw_data
                    category_titles = {}
                dataset_df = pd.DataFrame(tabs)
                history_filename = f"./data/individual_tests/private/graph_analysis/{d}.json"
                if path.isfile(history_filename):
                    history = pd.read_json(history_filename, orient="records")
                    history["url"] = history["url"].str.slice(0, 200)
                    group_id_map = get_browse_group_from_history(history)
                    dataset_df["browse_group"] = dataset_df["url"].apply(lambda a: group_id_map.get(a, -1))
                    orphan_browse_groups = dataset_df[dataset_df["browse_group"] == -1].index
                    dataset_df.loc[orphan_browse_groups, "browse_group"] = range(1000, 1000 + len(orphan_browse_groups))
                    # Items with history have no labels
                    if "smart_group_label" not in dataset_df.columns and "windowId" not in dataset_df.columns:
                        dataset_df["smart_group_label"] = 0
                final_datasets.append(dataset_df)
                category_title_datasets.append(category_titles)

    if dataset_name == BATCH_ONE:
        final_datasets = final_datasets[0:1]

    if dataset_name == BATCH_TWO:
        final_datasets = final_datasets[5:6]

    results = []
    for df in final_datasets:
        df["top_domain"] = df["url"].apply(partial(find_domain, top_domain=True))
        df["domain"] = df["url"].apply(partial(find_domain, top_domain=False))
        df["doc_name"] = df["title"].apply(get_doc_name)
        if "smart_group_label" not in df.columns:
            df["smart_group_label"] = df["windowId"]
        if "browse_group" not in df.columns:
            df["browse_group"] = 0
        # df = add_domain_descriptions(df)  # Not being used right now
        df.drop_duplicates(subset=["url"], inplace=True)
        df.reset_index(drop=True, inplace=True)
        results.append(df)
    return results, category_title_datasets
