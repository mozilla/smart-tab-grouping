from functools import partial
from typing import Set, List, Tuple, Dict
import regex as re
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import nltk

# from compute_site_descriptions import DOMAIN_DESCRIPTIONS_FILE
from util.key_document_finder import KeyDocumentFinder
from util.labeled_data_utils import get_labeled_dataset, BATCH_USER_TESTS, BATCH_ONE, BATCH_TWO, BATCH_ALL, user_test_list
from util.grouping_pipeline import run_pipeline, ModelProvider, EMBEDDING_MODEL_LIST, TOPIC_GENERATOR_OPTIONS, \
    NUM_CLUSTER_METHODS, CLUSTER_METHODS, EMBEDDING_TEXT_COMBINATIONS, OPENAI_CLOUD_LABEL, T5_BASE_LOCAL_LABEL
from util.topic_utils import compute_topic_using_digest
from util.tab_titles import OpenAITopicGenerator, TopicGenerator, T5TopicGenerator
from sentence_transformers import SentenceTransformer

st.title("Smart Tab Grouping")

DO_EXPORT_EMBEDDING_DATA = False # Output embeddings tsv in output folder
DO_OUTPUT_HTML_RESULT = False # Outputs HTML in ./output folder

class ModelProviderStreamlit(ModelProvider):
    def get_model(self, name):
        return get_model_streamlit_cached(name)

@st.cache_resource
def get_topic_generator(type: str = OPENAI_CLOUD_LABEL) -> TopicGenerator:
    if type == T5_BASE_LOCAL_LABEL:
        return T5TopicGenerator()
    if type == OPENAI_CLOUD_LABEL:
        return OpenAITopicGenerator()
    raise Exception("Invalid Topic Generator Type Specified")


@st.cache_resource
def get_model_streamlit_cached(name):
    return SentenceTransformer(name)

@st.cache_resource
def get_domain_dataset():
    domain_info = pd.read_csv("ext_data/domain_category_info_reformatted.csv")
    return domain_info


COLOR_LIST = [
    "#0063EA",
    "#8C39BC",
    "#00878C",
    "#BB4B00",
    "#985A00",
    "#C6256B",
    "#287F00",
    "#CC272E",
    "#596571",
    "#FF5733",  # Orange
    "#9B59B6",  # Violet
    "#C70039",  # Red
    "#2ECC71",  # Light Green
    "#2980B9",  # Blue
    "#1ABC9C",  # Turquoise
    "#27AE60",  # Green
    "#F1C40F",  # Yellow
    "#F39C12",  # Dark Yellow
    "#581845",  # Purple
    "#000000",  # Black
    "#E74C3C",  # Bright Red
    "#34495E",  # Dark Blue
    "#95A5A6",  # Light Gray
    "#7F8C8D"  # Dark Gray
]

STYLE_CSS = '''
        <style>
        .report-header {
            font-family: Arial, sans-serif;
            font-size: 22px;
        }

        .tab-container {
            font-family: Arial, sans-serif;
            display: flex;
            justify-content: center;
            flex-direction: column;
        }
        .tab-container-title {
            font-family: Arial, sans-serif;
            font-size: 22px;
            font-weight: 600;
            margin-left: 20px;
        }
        .group-match-pair {
            width: 950px;
            background-color: white;
            margin: 20px;
            justify-content: center;
            display: flex;
            flex-direction: row;
        }
        .tab-group {
            width: 464px;
            flex-shrink: 0;
            border-radius: 12px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            background-color: white;
            margin: 0 10px;
        }
        .tab-group-legend {
            width: 464px;
            flex-shrink: 0;
            font-size: 18px;
            margin: 0 10px;
            font-weight: 600;
        }
        .tab-group-legend-subhead {
            font-size: 14px;
            font-weight: 400;
        }
        .tab-header {
            padding: 10px;
            border-radius: 12px 12px 0 0;
            font-weight: bold;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .tab-header-gray {
            color: #D3D3D3;
        } 
        .tab-item {
            display: flex;
            align-items: center;
            padding: 1px 4px;
        }
        .tab-item-box {
            width: 16px;
            height: 16px;
            flex-shrink: 0;
            border-radius: 4px;
            margin-right: 4px;
            margin-left: 4px;
        }
        .tab-item-box-big {
            width: 24px;
            height: 24px;
        }
        .tab-item-text {
            padding: 6px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .legend {
            display: flex;
            align-items: center;
            margin-top: 20px;
            margin-left: 20px;
            font-family: Arial, sans-serif;
        }

        .legend-item {
            display: flex;
            align-items: center;
            margin-right: 20px;
        }

        .legend-icon {
            width: 20px;
            height: 20px;
            border-radius: 4px;
            margin-right: 8px;
        }
        </style>
            '''


def get_box(color: str, big=False):
    return f"<div class=\"tab-item-box{' tab-item-box-big' if big else ''}\" style=\"background-color:{color};\"></div> "



def compute_aligned_topics(frame: pd.DataFrame, user_label_key: str, ai_label_key: str):
    user_topics = frame[user_label_key].unique().tolist()
    ai_topics = frame[ai_label_key].unique().tolist()
    results = []
    items = []
    for user_topic in user_topics:
        for ai_topic in ai_topics:
            total_match = len(frame[(frame[user_label_key] == user_topic) & (frame[ai_label_key] == ai_topic)])
            num_ai = len(frame[frame[ai_label_key] == ai_topic])
            num_user = len(frame[frame[user_label_key] == user_topic])
            percent = total_match / (num_ai + num_user)
            items.append({"user": user_topic, "ai": ai_topic, "percentage": percent})
    all_combos = pd.DataFrame.from_records(items).sort_values(by="percentage", ascending=False)
    picked_ai = set()
    picked_user = set()
    for index, row in all_combos.iterrows():
        if row["ai"] not in picked_ai and row["user"] not in picked_user:
            results.append([row["user"], row["ai"]])
            picked_ai.add(row["ai"])
            picked_user.add(row["user"])
    for index, row in all_combos.iterrows():
        if row["ai"] not in picked_ai:
            results.append([None, row["ai"]])
            picked_ai.add(row["ai"])
        if row["user"] not in picked_user:
            results.append([row["user"], None])
            picked_user.add(row["user"])
    return results


def print_results(df, user_label_key: str, ai_label_key: str, topic_generator: str = None, predicted_id_topics=None,
                  docname="test_output", scores=None):
    ai_label_for_user_grouped_topic = {}
    html_buffer = ""

    key_doc_finder_ai = KeyDocumentFinder(df, ai_label_key, "title")
    key_doc_finder_ai.compute_all()
    key_doc_finder_user = KeyDocumentFinder(df, user_label_key, "title")
    key_doc_finder_user.compute_all()

    def add_html(s: str):
        nonlocal html_buffer
        html_buffer += s

    if predicted_id_topics is None:
        predicted_id_topics = {}
    color_table = {}

    topic_generator = get_topic_generator(topic_generator)

    add_html('<div class="tab-container">')

    def get_color(label: str):
        if label in color_table:
            return color_table[label]
        else:
            color_table[label] = COLOR_LIST[len(color_table)]
            return color_table[label]

    def get_tab_group(group_key: str, group_item: str, label_key: str, is_ai_group: False):
        if group_item is None:
            add_html('<div class="tab-group"> </div>')
            return
        items = df[df[group_key] == group_item]
        topic = None
        picked_documents_set = set()
        keywords = ""
        if len(items) > 0:
            key_doc_finder = key_doc_finder_ai if is_ai_group else key_doc_finder_user
            topic, picked_documents, keywords = compute_topic_using_digest(topic_generator, key_doc_finder, items,
                                                                           group_item)
        add_html('<div class="tab-group">')
        extra_topic = ""
        if is_ai_group:
            header_title = topic or "Unknown"
        else:
            if isinstance(group_item, float):
                group_item = int(group_item)
            header_title = predicted_id_topics.get(str(group_item), group_item)
            extra_topic = topic or "Unknown"
            ai_label_for_user_grouped_topic[str(group_item)] = extra_topic
        add_html(f'<div class="tab-header">{header_title}</div>')
        add_html(f'<div class="tab-content">')
        labels = items[label_key].astype(str).to_list()
        bullet_points = items["title"].to_list()
        for k in range(0, len(bullet_points)):
            add_html(f''' 
                    <div class="tab-item">
                        <div class="tab-item-box" style="background-color:{get_color(labels[k])}"></div>
                        <div class="tab-item-text">{bullet_points[k]}</div>
                    </div>            ''')
        add_html('</div>')  # tab-content
        add_html('</div>')  # tab-group

    aligned_user_ai_topic_list = compute_aligned_topics(df, user_label_key, ai_label_key)
    add_html('<div class="group-match-pair">')
    add_html('<div class="tab-group-legend">')
    add_html(docname)
    add_html('<div class="tab-group-legend-subhead">Tab groups you named</div>')
    add_html('</div>')
    add_html('<div class="tab-group-legend">')
    add_html('AI Generated')
    add_html('<div class="tab-group-legend-subhead">Generated tab groups, tabs sorted by group match accuracy</div>')
    add_html('</div>')
    add_html('</div>')
    for topic_item in aligned_user_ai_topic_list:
        user_topic = topic_item[0]
        ai_topic = topic_item[1]
        add_html('<div class="group-match-pair">')
        get_tab_group(user_label_key, user_topic, user_label_key, is_ai_group=False)
        get_tab_group(ai_label_key, ai_topic, user_label_key, is_ai_group=True)
        add_html('</div>')

    if scores is not None:
        add_html(
            f'<div class="report-header">Avg. Rand Score: {scores[0]:.2f} Adjusted Rand Score: {scores[1]:.2f}</div>')

    # Show label pairs for user labeled groups
    add_html(
        '<p /><p /><p /><div class="tab-container-title">Labels for Your Original Groupings</div><div class="report-header">')
    for topic_item in aligned_user_ai_topic_list:
        user_topic = topic_item[0]
        if user_topic is None:
            break
        if isinstance(user_topic, float):
            user_topic = int(user_topic)
        user_label = predicted_id_topics.get(str(user_topic), user_topic)
        ai_label = ai_label_for_user_grouped_topic.get(str(user_topic), "--")
        add_html(f'<p><b>User Label:</b> {user_label} &nbsp;&nbsp;&nbsp; <b>AI Label:</b> {ai_label} </p>')
    add_html('</div>')

    add_html('</div>')  # tab-container
    label_dict = predicted_id_topics or {}
    legend = f'''
            <div class="tab-container-title">Tab Grouping Report</div>
            <div class="legend">
             {"".join([f'<div class="legend-item">{get_box(v, big=True)}<div>{label_dict.get(str(k), k)}</div></div>' for k, v in color_table.items()])}
             </div>
             '''

    all_html = STYLE_CSS + legend + html_buffer
    st.html(all_html)
    if DO_OUTPUT_HTML_RESULT:
        with open(f"./output/{docname}_out.html", "w") as file:
            file.write(f"<html><body>{all_html}</body></html>")


def decipher_categories_to_set(cat_split: str) -> Set[str]:
    individual_items = set()
    if not isinstance(cat_split, str):
        return individual_items
    cat_item_list = cat_split.split('/')
    for cat_item in cat_item_list:
        cat_item = cat_item.strip()
        if len(cat_item) > 0:
            individual_items.add(cat_item)
    return individual_items


def compute_domain_categories(top_domain_str: str) -> Set:
    domain_info = get_domain_dataset()
    domain_cat_df = domain_info[domain_info.url == top_domain_str]
    if len(domain_cat_df) > 0:
        return set(decipher_categories_to_set(domain_cat_df.iloc[0]["Categories"]))
    return set()


def add_domain_descriptions(base_data: pd.DataFrame):
    domain_info = pd.read_csv("data/private/domain_descriptions.csv")
    domain_info = domain_info.fillna("")
    base_data = base_data.merge(domain_info, on="domain", how="left")
    if "description" not in base_data.columns:
        base_data["description"] = ""
    return base_data


@st.cache_resource
def get_labeled_dataset_cached(dataset_name: str) -> Tuple[List[pd.DataFrame], List[Dict[str, str]]]:
    return get_labeled_dataset(dataset_name)

@st.cache_resource
def get_title_embedding_transformer(model_name: str):
    return get_title_embedding_transformer_base(model_name)

def basic_clean(a: str):
    a = re.sub("Log in", "", a, flags=re.IGNORECASE)
    a = re.sub("Sign in", "", a, flags=re.IGNORECASE)
    a = re.sub("Sign Up", "", a, flags=re.IGNORECASE)
    a = re.sub("Signin", "", a, flags=re.IGNORECASE)
    a = re.sub("Log on", "", a, flags=re.IGNORECASE)
    a = re.sub("Register", "", a, flags=re.IGNORECASE)
    a = re.sub("Checkout", "", a, flags=re.IGNORECASE)
    return a

nltk.download('punkt')

load_dotenv()
st.write("Item Vector Components")
domain_scale = st.slider("Web Domain ID", 0.0, 1.0, 0.0)
title_embedding_scale = st.slider("Text Embedding", 0.0, 1.0, 1.0)
tf_idf_scale = st.slider("Frequent Words / Phrases (TF-IDF)", 0.0, 1.0, 0.0)
history_scale = st.slider("History", 0.0, 1.0, 0.0)

embedding_model = st.selectbox(
    "Which embedding model to use",
    EMBEDDING_MODEL_LIST)

config = {
    "domain_scale": domain_scale,
    "history_scale": history_scale,
    "title_embedding_scale": title_embedding_scale,
    "tf_idf_scale": tf_idf_scale,
    "embedding_model": embedding_model
}

text_for_embedding = st.radio(
    "Text For Embedding",
    EMBEDDING_TEXT_COMBINATIONS,
    index=1
)

cluster_method = st.radio(
    "Cluster Method",
    CLUSTER_METHODS,
    index=0,
)


remap = st.radio(
    "Reduce Dimensions",
    [0, 5, 15],
    index=1,
)

topic_name_generation = st.radio(
    "Topic Name Generation",
    TOPIC_GENERATOR_OPTIONS,
    index=0
)

num_cluster_method = st.radio(
    "Number of Cluster Optimizer",
    NUM_CLUSTER_METHODS,
    index=0
)

st.markdown("""---""")

dataset_name = st.radio(
    "Datasets",
    ["pittsburgh_trip",
     BATCH_ONE, BATCH_TWO,
     BATCH_ALL, BATCH_USER_TESTS],
    index=1
)
st.markdown("""---""")

dbscan_eps = 0.4  # st.slider("DBScan max distance", 0.0, 1.0, 0.4)

datasets, labeled_topics = get_labeled_dataset(dataset_name)
cur_run_labeled_topics = {}
topic_training_datasets = []
model_provider = ModelProviderStreamlit()
has_warned_no_description = False

if len(datasets) > 0:
    cur_run_labeled_topics = labeled_topics[0]
    if st.button("Compute Clusters"):
        run_count = 4 if len(datasets) == 1 else 1
        rscore_total = 0
        adj_rscore_total = 0
        actual_runs = 0
        res = None
        for df in datasets:
            if actual_runs < 4:
                st.write(df)
            for i in range(run_count):
                try:
                    config["clustering_method"] = cluster_method
                    config["dbscan_eps"] = dbscan_eps
                    config["remap"] = remap
                    config["num_cluster_method"] = num_cluster_method
                    config["text_for_embedding"] = text_for_embedding
                    if len(datasets) == 1 and DO_EXPORT_EMBEDDING_DATA:
                        save_set_name = dataset_name
                    else:
                        save_set_name = None
                    if "description" not in df.columns:
                        if not has_warned_no_description:
                            st.write("Warning: This dataset has no description")
                            has_warned_no_description = True
                        df["description"] = ""
                    res, score, adj_rscore = run_pipeline(config, df, save_set_name, model_provider=model_provider)
                    rscore_total += score
                    adj_rscore_total += adj_rscore
                    actual_runs += 1
                except Exception as ex:
                    st.write(ex)
                    print(ex)
        st.write(f"{actual_runs} total runs from {len(datasets)} labeled tests:")
        if res is not None:
            print_results(res, "smart_group_label", "predicted_cluster", topic_generator=topic_name_generation,
                          predicted_id_topics=cur_run_labeled_topics, docname=dataset_name,
                          scores=[rscore_total / actual_runs,
                                  adj_rscore_total / actual_runs])

else:
    st.write("No dataset found.")
