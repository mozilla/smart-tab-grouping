import streamlit as st
import numpy as np
import pandas as pd
import os

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -------------
# The classification code
# -------------
def sigmoid(z):
    return 1. / (1 + np.exp(-z))

def prod(gp, title, coef, intercept):
    return float(sigmoid(coef[0] * gp + coef[1] * title + intercept))

def get_classifications(window, anchors, embedding_model, classifier_params, threshold):
    """
    window: dict with at least
      - 'titles': list of strings
      - 'group_name': list of strings (same length as titles)
    anchors: list of anchor titles
    embedding_model: a SentenceTransformer model
    classifier_params: list or tuple [[coef_1, coef_2], intercept]
    threshold: float in [0,1]
    """

    window_titles = window['titles']
    # Convert anchor titles to indices
    anchor_indices = [window_titles.index(a) for a in anchors]
    anchor_group_name = window['group_name'][anchor_indices[0]]

    candidate_indices = [i for i, c in enumerate(window_titles) if i not in anchor_indices]
    candidate_titles = [window_titles[i] for i in candidate_indices]
    candidate_groups = [window['group_name'][i] for i in candidate_indices]

    ct = {}

    # Group embedding (from first anchor's group name)
    group_embedding = embedding_model.encode(anchor_group_name)
    # Anchor title embeddings
    anchor_title_embeddings = [embedding_model.encode(a) for a in anchors]

    for cg, c, ci in zip(candidate_groups, candidate_titles, candidate_indices):
        ct[c] = {}
        ct[c]['index'] = ci
        candidate_title_embedding = embedding_model.encode(c)
        ct[c]['group_similarity'] = cosine_similarity(
            group_embedding.reshape(1, -1),
            candidate_title_embedding.reshape(1, -1)
        )[0][0]

        # Average similarity to all anchor titles
        average_anchor_similarity = 0
        for anchor_title_embedding in anchor_title_embeddings:
            average_anchor_similarity += cosine_similarity(
                candidate_title_embedding.reshape(1, -1),
                anchor_title_embedding.reshape(1, -1)
            )[0][0]
        ct[c]['title_similarity'] = average_anchor_similarity / len(anchor_title_embeddings)

        # Probability from logistic formula
        coef, intercept = classifier_params[0], classifier_params[1]
        ct[c]['proba'] = prod(
            ct[c]['group_similarity'],
            ct[c]['title_similarity'],
            coef,
            intercept
        )
        # Compare with threshold
        ct[c]['similar'] = ct[c]['proba'] > threshold

        # True label check (if same group = "true" label)
        ct[c]['group_name'] = cg
        if ct[c]['similar']:
            # check if true positive or false positive
            ct[c]['classification'] = 'tp' if cg == anchor_group_name else 'fp'
        else:
            # false negative or true negative
            ct[c]['classification'] = 'fn' if cg == anchor_group_name else 'tn'

    # Return sorted list (descending by probability)
    return sorted(list(ct.items()), key=lambda x: x[1]['proba'], reverse=True)

# -------------
# Utility to generate windows from a DataFrame
# -------------
def generate_window_from_df(df):
    # Return the tab titles, urls, and group name
    return {
        'titles': list(df['title']),
        'urls': list(df['url']),
        'group_name': list(df['task'])
    }

# -------------
# Load windows from default or fallback CSV
# -------------
def load_all_windows():
    BASE_DIR = os.path.dirname(os.path.realpath(__file__))
    df_path_default = os.path.join(BASE_DIR, "..", "data", "suggest_datasets", "processed.csv")
    df_synthetic_path = os.path.join(BASE_DIR, "..", "data", "suggest_datasets", "synthetic_processed.csv")

    if os.path.exists(df_path_default):
        df = pd.read_csv(df_path_default)
    elif os.path.exists(df_synthetic_path):
        df = pd.read_csv(df_synthetic_path)
    else:
        # If neither CSV is found, return an empty list or handle it differently
        return []

    windows = []
    for test_set_id in df['test_set_id'].unique():
        subset = df[df['test_set_id'] == test_set_id]
        windows.append(generate_window_from_df(subset))
    return windows

# -------------
# Streamlit UI
# -------------
def main():
    st.title("Classification Visualization App")

    # Attempt to load windows from your CSVs
    all_windows = load_all_windows()

    # If none found, use dummy data; otherwise override with loaded data.
    windows = [
        {
            "titles": ["Sausage", "Burger", "Salads", "Github", "Cat Animal", "Dogs"],
            "group_name": ["Food", "Food", "Food", "Ungrouped-4", "Animals", "Animals"]
        }
    ]
    if len(all_windows) > 0:
        windows = all_windows

    # UI Control: pick which window
    window_idx = st.selectbox("Select Window Index:", list(range(len(windows))))

    window = windows[window_idx]

    # Build a list of "Title (Group)" options for anchors
    title_group_options = [
        f"{t} ({g})"
        for t, g in zip(window["titles"], window["group_name"])
    ]
    title_group_map = {
        f"{t} ({g})": t
        for t, g in zip(window["titles"], window["group_name"])
    }

    # Multi-select anchors
    anchor_labels = st.multiselect(
        "Select Anchor Titles (Group appended)",
        options=title_group_options,
        default=[]
    )
    anchors = [title_group_map[label] for label in anchor_labels]

    # Classifier hyperparameters
    coef_1 = st.number_input("Coefficient for group similarity", value=28.488064)
    coef_2 = st.number_input("Coefficient for title similarity", value=17.99544)
    intercept_val = st.number_input("Intercept", value=-37.541557)
    classifier_params = [[coef_1, coef_2], intercept_val]

    # Threshold
    threshold = st.slider("Threshold", min_value=0.0, max_value=1.0, value=0.5)

    # Model name
    model_name = st.text_input(
        "SentenceTransformer Model Name",
        value="thenlper/gte-small"  # or "sentence-transformers/all-MiniLM-L6-v2"
    )

    # Run classification
    if st.button("Run Classification"):
        if not anchors:
            st.error("No anchors selected. Please select at least one title as an anchor.")
            return

        try:
            embedding_model = SentenceTransformer(model_name)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return

        # For classification, we just need at least one anchor to identify group
        anchor_indices = [window["titles"].index(a) for a in anchors]
        anchor_group_name = window['group_name'][anchor_indices[0]]

        st.write(f"**Anchor Titles**: {anchors}")
        st.write(f"**Anchor Group Name**: {anchor_group_name}")

        # Compute results
        results = get_classifications(
            window,
            anchors,
            embedding_model,
            classifier_params,
            threshold
        )

        # Build DataFrame
        df = pd.DataFrame(
            [
                {
                    "Candidate Title": r[0],
                    "Index": r[1]["index"],
                    "Group Similarity": r[1]["group_similarity"],
                    "Title Similarity": r[1]["title_similarity"],
                    "Probability": r[1]["proba"],
                    "Similar?": r[1]["similar"],
                    "Group Name": r[1]["group_name"],
                    "Classification": r[1]["classification"],
                }
                for r in results
            ]
        )

        # Classification color map
        def classification_row_style(row):
            color_map = {
                'tp': 'green',
                'fp': 'red',
                'fn': 'lightcoral',   # 'light red'
                'tn': 'lightgreen'
            }
            return [
                f'background-color: {color_map.get(row["Classification"], "white")}'
                for _ in row.index
            ]

        styled_df = df.style.apply(classification_row_style, axis=1)

        # Make the DataFrame bigger: set width & height explicitly
        st.dataframe(styled_df, width=1200, height=800)

        # Classification counts
        classification_counts = df["Classification"].value_counts()
        st.write("### Classification Counts")
        st.bar_chart(classification_counts)
        st.write(classification_counts)

if __name__ == "__main__":
    main()
