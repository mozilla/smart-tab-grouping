import streamlit as st
import numpy as np
import pandas as pd
import os
import re

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from urllib.parse import urlparse

PREPROCESS = True

def preprocess_text(text: str) -> str:
    if not PREPROCESS:
        return text

    delimiters = r"(?<=\s)[|–-]+(?=\s)"
    split_text = re.split(delimiters, text)
    has_enough_info = len(split_text) > 0 and len(" ".join(split_text[:-1])) > 5
    is_potential_domain_info = len(split_text) > 1 and len(split_text[-1]) < 20

    if has_enough_info and is_potential_domain_info:
        processed = " ".join(chunk.strip() for chunk in split_text[:-1] if chunk.strip()).strip()
        return processed
    return text

COMMON_TLDS = {"com", "net", "org", "edu", "gov", "co", "io", "info"}

def preprocess_url(url: str) -> str:
    parsed = urlparse(url)
    domain = parsed.netloc.replace("www.", "")
    domain_tokens = [part for part in domain.split(".") if part.lower() not in COMMON_TLDS]
    path_tokens = re.split(r'[/\-_.]', parsed.path.strip("/"))
    tokens = [
        t.lower() for t in (domain_tokens + path_tokens)
        if len(t) >= 3 and not any(c.isdigit() for c in t)
    ]
    return " ".join(tokens)

def sigmoid(z):
    return 1. / (1 + np.exp(-z))

def prod(gp, title, url_gp, url_url, max_other_gp, coef, intercept):
    return float(sigmoid(
        coef[0] * gp +
        coef[1] * title +
        coef[2] * url_gp +
        coef[3] * url_url +
        coef[4] * max_other_gp +
        intercept
    ))

def get_classifications(window, anchors, embedding_model, classifier_params, threshold):
    anchors = [preprocess_text(a) for a in anchors]
    window_titles = [preprocess_text(t) for t in window['titles']]
    window_urls = [preprocess_url(u) for u in window['urls']]
    group_names = window['group_name']

    anchor_indices = [window_titles.index(a) for a in anchors]
    anchor_group_name = group_names[anchor_indices[0]]

    candidate_indices = [i for i in range(len(window_titles)) if i not in anchor_indices]
    candidate_titles = [window_titles[i] for i in candidate_indices]
    candidate_groups = [group_names[i] for i in candidate_indices]
    candidate_urls = [window_urls[i] for i in candidate_indices]

    ct = {}
    group_embedding = embedding_model.encode(anchor_group_name)
    anchor_title_embeddings = [t for i,t in enumerate(embedding_model.encode(window_titles)) if i in anchor_indices]
    anchor_url_embeddings = [t for i,t in enumerate(embedding_model.encode(window_urls)) if i in anchor_indices]

    other_group_names = list(set(group_names).difference(set(anchor_group_name)))
    other_group_embeddings = [t for i,t in enumerate(embedding_model.encode(other_group_names))]
    for cg, c_title, c_url, ci in zip(candidate_groups, candidate_titles, candidate_urls, candidate_indices):
        ct[c_title] = {}
        ct[c_title]['index'] = ci

        title_emb = embedding_model.encode(c_title)
        url_emb = embedding_model.encode(c_url)

        ct[c_title]['group_similarity'] = cosine_similarity(
            group_embedding.reshape(1, -1),
            title_emb.reshape(1, -1)
        )[0][0]

        ct[c_title]['title_similarity'] = np.mean([
            cosine_similarity(title_emb.reshape(1, -1), at_emb.reshape(1, -1))[0][0]
            for at_emb in anchor_title_embeddings
        ])

        ct[c_title]['group_url_similarity'] = cosine_similarity(
            group_embedding.reshape(1, -1),
            url_emb.reshape(1, -1)
        )[0][0]

        ct[c_title]['url_similarity'] = np.mean([
            cosine_similarity(url_emb.reshape(1, -1), au_emb.reshape(1, -1))[0][0]
            for au_emb in anchor_url_embeddings
        ])

        ct[c_title]['max_group_similarity'] = np.max([cosine_similarity(gp_emb.reshape(1, -1), title_emb.reshape(1, -1))[0][0] for gp_emb in other_group_embeddings])

        coef, intercept = classifier_params[0], classifier_params[1]
        print(c_title, ct[c_title])
        ct[c_title]['proba'] = prod(
            ct[c_title]['group_similarity'],
            ct[c_title]['title_similarity'],
            ct[c_title]['group_url_similarity'],
            ct[c_title]['url_similarity'],
            ct[c_title]['max_group_similarity'],
            coef,
            intercept
        )
        ct[c_title]['similar'] = ct[c_title]['proba'] > threshold
        ct[c_title]['group_name'] = cg

        if ct[c_title]['similar']:
            ct[c_title]['classification'] = 'tp' if cg == anchor_group_name else 'fp'
        else:
            ct[c_title]['classification'] = 'fn' if cg == anchor_group_name else 'tn'

    return sorted(ct.items(), key=lambda x: x[1]['proba'], reverse=True)

def generate_window_from_df(df):
    return {
        'titles': list(df['title']),
        'urls': list(df['url']),
        'group_name': list(df['task'])
    }

def load_all_windows():
    BASE_DIR = os.path.dirname(os.path.realpath(__file__))
    df_path_default = os.path.join(BASE_DIR, "..", "data", "suggest_datasets", "processed.csv")
    df_synthetic_path = os.path.join(BASE_DIR, "..", "data", "suggest_datasets", "synthetic_processed.csv")

    if os.path.exists(df_path_default):
        df = pd.read_csv(df_path_default)
    elif os.path.exists(df_synthetic_path):
        df = pd.read_csv(df_synthetic_path)
    else:
        return []

    windows = []
    for test_set_id in df['test_set_id'].unique():
        subset = df[df['test_set_id'] == test_set_id]
        windows.append(generate_window_from_df(subset))
    return windows

def main():
    st.title("Classification Visualization App")

    all_windows = load_all_windows()
    windows = [
        {
            "titles": ["Sausage", "Burger", "Salads", "Github", "Cat Animal", "Dogs"],
            "urls": [
                "https://food.com/sausage",
                "https://food.com/burger",
                "https://food.com/salads",
                "https://github.com",
                "https://animals.com/cat",
                "https://animals.com/dogs"
            ],
            "group_name": ["Food", "Food", "Food", "Ungrouped-4", "Animals", "Animals"]
        }
    ]
    if len(all_windows) > 0:
        windows = all_windows

    window_idx = st.selectbox("Select Window Index:", list(range(len(windows))))
    window = windows[window_idx]

    title_group_options = [
        f"{t} ({g})"
        for t, g in zip(window["titles"], window["group_name"])
    ]
    title_group_map = {
        f"{t} ({g})": t
        for t, g in zip(window["titles"], window["group_name"])
    }

    anchor_labels = st.multiselect(
        "Select Anchor Titles (Group appended)",
        options=title_group_options,
        default=[]
    )
    anchors = [title_group_map[label] for label in anchor_labels]

    coef_1 = st.number_input("Coefficient for group-title similarity", value=5.9958344)
    coef_2 = st.number_input("Coefficient for title similarity", value=5.1170115)
    coef_3 = st.number_input("Coefficient for group-URL similarity", value=0)
    coef_4 = st.number_input("Coefficient for anchor URL similarity", value=0)
    coef_5 = st.number_input("Coefficient for other group similarity", value=-8.779929)
    intercept_val = st.number_input("Intercept", value=0.8938948)
    classifier_params = [[coef_1, coef_2, coef_3, coef_4, coef_5], intercept_val]

    threshold = st.slider("Threshold", min_value=0.0, max_value=1.0, value=0.7)

    model_name = st.text_input(
        "SentenceTransformer Model Name",
        value="sentence-transformers/all-MiniLM-L6-v2"
    )

    if st.button("Run Classification"):
        if not anchors:
            st.error("No anchors selected. Please select at least one title as an anchor.")
            return

        try:
            embedding_model = SentenceTransformer(model_name)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return

        anchor_indices = [window["titles"].index(a) for a in anchors]
        anchor_group_name = window['group_name'][anchor_indices[0]]

        st.write(f"**Anchor Titles**: {anchors}")
        st.write(f"**Anchor Group Name**: {anchor_group_name}")

        results = get_classifications(
            window,
            anchors,
            embedding_model,
            classifier_params,
            threshold
        )

        df = pd.DataFrame(
            [
                {
                    "Candidate Title": r[0],
                    "Index": r[1]["index"],
                    "Group Similarity": r[1]["group_similarity"],
                    "Title Similarity": r[1]["title_similarity"],
                    "Group URL Similarity": r[1]["group_url_similarity"],
                    "URL Similarity": r[1]["url_similarity"],
                    "Max Other Group Similarity": r[1]["max_group_similarity"],
                    "Probability": r[1]["proba"],
                    "Similar?": r[1]["similar"],
                    "Group Name": r[1]["group_name"],
                    "Classification": r[1]["classification"],
                }
                for r in results
            ]
        )

        def classification_row_style(row):
            color_map = {
                'tp': 'green',
                'fp': 'red',
                'fn': 'lightcoral',
                'tn': 'lightgreen'
            }
            return [
                f'background-color: {color_map.get(row["Classification"], "white")}'
                for _ in row.index
            ]

        styled_df = df.style.apply(classification_row_style, axis=1)

        st.dataframe(styled_df, width=1500, height=500)

        classification_counts = df["Classification"].value_counts()
        st.write("### Classification Counts")
        st.bar_chart(classification_counts)
        st.write(classification_counts)

if __name__ == "__main__":
    main()
