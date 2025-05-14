import streamlit as st
import numpy as np
import pandas as pd
import os
import re
from typing import List, Dict, Any

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from urllib.parse import urlparse

from sklearn.cluster import (
    KMeans,
    DBSCAN,
    AgglomerativeClustering,
    Birch,
    OPTICS,
    SpectralClustering,
)
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

try:
    import umap.umap_ as umap  # type: ignore
except ImportError:
    umap = None


PREPROCESS = True


def preprocess_text(text: str) -> str:
    if not PREPROCESS:
        return text
    delimiters = r"(?<=\s)[|–-]+(?=\s)"
    parts = re.split(delimiters, text)
    if len(parts) > 1 and len(parts[-1]) < 20 and len(" ".join(parts[:-1])) > 5:
        return " ".join(p.strip() for p in parts[:-1] if p.strip())
    return text


COMMON_TLDS = {"com", "net", "org", "edu", "gov", "co", "io", "info"}

def preprocess_url(url: str) -> str:
    parsed = urlparse(url)
    domain = parsed.netloc.replace("www.", "")
    domain_tokens = [p for p in domain.split(".") if p.lower() not in COMMON_TLDS]
    path_tokens = re.split(r"[/\\_.\-]", parsed.path.strip("/"))
    tokens = [t.lower() for t in (*domain_tokens, *path_tokens) if len(t) >= 3 and not any(c.isdigit() for c in t)]
    return " ".join(tokens)


def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))

def lr_proba(gp, title, url_gp, url_url, max_other_gp, coef, intercept):
    return float(sigmoid(coef[0] * gp + coef[1] * title + coef[2] * url_gp + coef[3] * url_url + coef[4] * max_other_gp + intercept))


def get_lr_classifications(window: Dict[str, List[str]], anchors: List[str], model: SentenceTransformer,
                            classifier_params, threshold: float):
    titles = [preprocess_text(t) for t in window["titles"]]
    urls = [preprocess_url(u) for u in window["urls"]]
    groups = window["group_name"]

    anchors_clean = [preprocess_text(a) for a in anchors]
    try:
        anchor_idx = [titles.index(a) for a in anchors_clean]
    except ValueError as e:
        raise ValueError("One or more selected anchors were not found after preprocessing. Re‑select anchors or disable preprocessing.") from e

    anchor_group = groups[anchor_idx[0]]
    candidate_idx = [i for i in range(len(titles)) if i not in anchor_idx]

    title_embs = model.encode(titles)
    url_embs = model.encode(urls)
    group_emb = model.encode(anchor_group)
    anchor_title_embs = [title_embs[i] for i in anchor_idx]
    anchor_url_embs = [url_embs[i] for i in anchor_idx]
    other_groups = list(set(groups) - {anchor_group})
    other_group_embs = model.encode(other_groups) if other_groups else np.zeros((1, title_embs.shape[1]))

    coef, intercept = classifier_params
    results = []
    for ci in candidate_idx:
        t_emb = title_embs[ci]
        u_emb = url_embs[ci]
        cg = groups[ci]

        g_sim = cosine_similarity(group_emb.reshape(1, -1), t_emb.reshape(1, -1))[0][0]
        t_sim = float(np.mean([cosine_similarity(t_emb.reshape(1, -1), at.reshape(1, -1))[0][0] for at in anchor_title_embs]))
        g_u_sim = cosine_similarity(group_emb.reshape(1, -1), u_emb.reshape(1, -1))[0][0]
        u_sim = float(np.mean([cosine_similarity(u_emb.reshape(1, -1), au.reshape(1, -1))[0][0] for au in anchor_url_embs]))
        max_other = float(np.max([cosine_similarity(og.reshape(1, -1), t_emb.reshape(1, -1))[0][0] for og in other_group_embs])) if other_groups else 0.0

        proba = lr_proba(g_sim, t_sim, g_u_sim, u_sim, max_other, coef, intercept)
        similar = proba > threshold
        cls = "tp" if (similar and cg == anchor_group) else "fp" if similar else "fn" if cg == anchor_group else "tn"

        results.append((titles[ci], {
            "index": ci,
            "group_similarity": g_sim,
            "title_similarity": t_sim,
            "group_url_similarity": g_u_sim,
            "url_similarity": u_sim,
            "max_group_similarity": max_other,
            "proba": proba,
            "similar": similar,
            "group_name": cg,
            "classification": cls,
        }))

    return sorted(results, key=lambda x: x[1]["proba"], reverse=True)


def reduce_dimensions(emb: np.ndarray, method: str, n_components: int):
    if method == "None":
        return emb
    if method == "PCA":
        return PCA(n_components=min(n_components, emb.shape[1]), random_state=42).fit_transform(emb)
    if method == "UMAP":
        if umap is None:
            raise ImportError("UMAP not installed – run `pip install umap-learn`." )
        return umap.UMAP(n_components=min(n_components, 50), random_state=42).fit_transform(emb)
    raise ValueError(f"Unknown DR method: {method}")

def auto_k(embeddings: np.ndarray, algorithm: str, max_k: int = 10):
    best_k, best_score = 2, -1.0
    max_k = min(max_k, len(embeddings) - 1)
    for k in range(2, max_k + 1):
        if algorithm == "KMeans":
            labels = KMeans(n_clusters=k, random_state=42, n_init="auto").fit_predict(embeddings)
        elif algorithm == "Agglomerative":
            labels = AgglomerativeClustering(n_clusters=k).fit_predict(embeddings)
        elif algorithm == "Spectral":
            labels = SpectralClustering(n_clusters=k, random_state=42, assign_labels="discretize").fit_predict(embeddings)
        elif algorithm == "Birch":
            labels = Birch(n_clusters=k, threshold=0.5).fit_predict(embeddings)
        else:
            break
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(embeddings, labels)
        if score > best_score:
            best_k, best_score = k, score
    return best_k

def cluster_predict(window: Dict[str, List[str]], anchors: List[str], model: SentenceTransformer,
                    algorithm: str, dr_method: str, params: Dict[str, Any]):
    titles = [preprocess_text(t) for t in window["titles"]]
    groups = window["group_name"]

    anchors_clean = [preprocess_text(a) for a in anchors]
    anchor_idx = [titles.index(a) for a in anchors_clean]

    texts_for_embedding = [f"{grp} {title}" if i in anchor_idx else title for i, (title, grp) in enumerate(zip(titles, groups))]

    embeddings = model.encode(texts_for_embedding)
    n_comp = params.get("n_components", 50)
    emb_red = reduce_dimensions(embeddings, dr_method, n_comp)

    n_clusters = params.get("n_clusters")
    if algorithm in {"KMeans", "Agglomerative", "Spectral", "Birch"} and n_clusters is None:
        n_clusters = auto_k(emb_red, algorithm)

    if algorithm == "KMeans":
        labels = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto").fit_predict(emb_red)
    elif algorithm == "Agglomerative":
        labels = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(emb_red)
    elif algorithm == "Spectral":
        labels = SpectralClustering(n_clusters=n_clusters, random_state=42, assign_labels="discretize").fit_predict(emb_red)
    elif algorithm == "Birch":
        thresh = params.get("threshold", 0.5)
        labels = Birch(n_clusters=n_clusters, threshold=thresh).fit_predict(emb_red)
    elif algorithm == "DBSCAN":
        eps = params.get("eps", 0.5)
        ms = params.get("min_samples", 2)
        labels = DBSCAN(eps=eps, min_samples=ms).fit_predict(emb_red)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    elif algorithm == "OPTICS":
        ms = params.get("min_samples", 2)
        xi = params.get("xi", 0.05)
        labels = OPTICS(min_samples=ms, xi=xi).fit_predict(emb_red)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    else:
        raise ValueError("Unsupported algorithm: " + algorithm)

    anchor_group = groups[anchor_idx[0]]
    anchor_labels = {labels[i] for i in anchor_idx}

    results: list[tuple[str, Dict[str, Any]]] = []
    for i, (title, grp) in enumerate(zip(titles, groups)):
        if i in anchor_idx:
            continue
        similar = labels[i] in anchor_labels and labels[i] != -1
        cls = "tp" if (similar and grp == anchor_group) else "fp" if similar else "fn" if grp == anchor_group else "tn"
        results.append((title, {
            "index": i,
            "cluster_label": int(labels[i]),
            "similar": similar,
            "group_name": grp,
            "classification": cls,
        }))

    return results, n_clusters

def generate_window_from_df(df: pd.DataFrame):
    return {
        "titles": df["title"].tolist(),
        "urls": df["url"].tolist(),
        "group_name": df["task"].tolist(),
    }

def load_all_windows():
    base = os.path.dirname(os.path.realpath(__file__))
    p1 = os.path.join(base, "..", "data", "suggest_datasets", "processed.csv")
    p2 = os.path.join(base, "..", "data", "suggest_datasets", "synthetic_processed.csv")

    if os.path.exists(p1):
        df = pd.read_csv(p1)
    elif os.path.exists(p2):
        df = pd.read_csv(p2)
    else:
        return []

    windows = []
    for tsid in df["test_set_id"].unique():
        windows.append(generate_window_from_df(df[df["test_set_id"] == tsid]))
    return windows

def main():
    st.title("Classification & Clustering Evaluation App")

    windows = load_all_windows()
    if not windows:
        windows = [{
            "titles": ["Sausage", "Burger", "Salads", "Github", "Cat Animal", "Dogs"],
            "urls": [
                "https://food.com/sausage",
                "https://food.com/burger",
                "https://food.com/salads",
                "https://github.com",
                "https://animals.com/cat",
                "https://animals.com/dogs",
            ],
            "group_name": ["Food", "Food", "Food", "Ungrouped-4", "Animals", "Animals"],
        }]

    w_idx = st.selectbox("Select window index", list(range(len(windows))))
    window = windows[w_idx]

    algorithm = st.selectbox("Algorithm", [
        "Logistic Regression",
        "KMeans",
        "Agglomerative",
        "Spectral",
        "DBSCAN",
        "OPTICS",
        "Birch",
    ])

    dr_method = st.selectbox("Dimensionality Reduction", ["None", "PCA", "UMAP"])
    n_components = 50
    if dr_method != "None":
        n_components = int(st.number_input("Number of components", 2, 512, 50, step=1))

    display_labels = [f"{t} ({g})" for t, g in zip(window["titles"], window["group_name"])]
    label2title = {lbl: lbl.rsplit(" (", 1)[0] for lbl in display_labels}
    anchor_labels = st.multiselect("Select anchor titles", display_labels)
    anchors = [label2title[lbl] for lbl in anchor_labels]

    model_name = st.text_input("SentenceTransformer model", "sentence-transformers/all-MiniLM-L6-v2")

    cluster_params: Dict[str, Any] = {"n_components": n_components}

    if algorithm == "Birch":
        cluster_params["threshold"] = float(st.number_input("Birch threshold", 0.5))
    if algorithm == "DBSCAN":
        cluster_params["eps"] = float(st.number_input("DBSCAN eps", 0.5))
        cluster_params["min_samples"] = int(st.number_input("min_samples", 2))
    if algorithm == "OPTICS":
        cluster_params["min_samples"] = int(st.number_input("min_samples", 2))
        cluster_params["xi"] = float(st.number_input("xi", 0.05))

    if algorithm == "Logistic Regression":
        coef_1 = st.number_input("Coeff: group-title", value=5.9958344)
        coef_2 = st.number_input("Coeff: title", value=5.1170115)
        coef_3 = st.number_input("Coeff: group-url", value=0.0)
        coef_4 = st.number_input("Coeff: anchor-url", value=0.0)
        coef_5 = st.number_input("Coeff: other-group", value=-8.779929)
        intercept = st.number_input("Intercept", value=0.8938948)
        threshold = st.slider("Threshold", 0.0, 1.0, 0.7)
        lr_params = ([coef_1, coef_2, coef_3, coef_4, coef_5], intercept)

    if st.button("Run evaluation"):
        if not anchors:
            st.error("Please select at least one anchor title.")
            st.stop()

        try:
            embedder = SentenceTransformer(model_name)
        except Exception as e:
            st.error(f"Error loading embedding model: {e}")
            st.stop()

        if algorithm == "Logistic Regression":
            try:
                lr_results = get_lr_classifications(window, anchors, embedder, lr_params, threshold)
            except ValueError as e:
                st.error(str(e))
                st.stop()

            df = pd.DataFrame([
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
                for r in lr_results
            ])
            clusters_info = None
        else:
            anchors_clean = [preprocess_text(a) for a in anchors]
            c_results, k_opt = cluster_predict(window, anchors_clean, embedder, algorithm, dr_method, cluster_params)
            df = pd.DataFrame([
                {
                    "Candidate Title": r[0],
                    "Index": r[1]["index"],
                    "Cluster Label": r[1]["cluster_label"],
                    "Similar?": r[1]["similar"],
                    "Group Name": r[1]["group_name"],
                    "Classification": r[1]["classification"],
                }
                for r in c_results
            ])
            clusters_info = k_opt

        def style_row(row):
            colors = {"tp": "#d4edda", "fp": "#f8d7da", "fn": "#fce5cd", "tn": "#e2f0d9"}
            return [f"background-color: {colors.get(row['Classification'], '#fff')}"] * len(row)

        st.dataframe(df.style.apply(style_row, axis=1), height=500, width=1500)

        st.subheader("Classification counts")
        counts = df["Classification"].value_counts()
        st.bar_chart(counts)
        st.write(counts)

        if clusters_info is not None:
            st.info(f"Clusters used/discovered: {clusters_info}")


if __name__ == "__main__":
    main()
