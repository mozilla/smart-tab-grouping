from functools import partial

import math

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction import text
from sklearn.preprocessing import Normalizer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import nltk
from sklearn.metrics.cluster import rand_score, adjusted_rand_score
from util.silhouette import silh_find_optimal_k
import umap
import hdbscan
from kneed import KneeLocator
from sklearn.cluster import KMeans
import warnings
from sentence_transformers import SentenceTransformer

from util.labeled_data_utils import get_labeled_dataset, user_test_list

EMBEDDING_MODEL_MINILM = "all-MiniLM-L6-v2"
EMBEDDING_MODEL_MPNET = "all-mpnet-base-v2"
EMBEDDING_NOMIC = "nomic-ai/nomic-embed-text-v1.5"

EMBEDDING_MODEL_LIST = [EMBEDDING_MODEL_MINILM, EMBEDDING_MODEL_MPNET]

T5_BASE_LOCAL_LABEL = "T5 Fine Tuned (Local)"
OPENAI_CLOUD_LABEL = "OpenAI (Cloud)"


EMBEDDING_TEXT_COLUMN = "emb_text"
CUSTOM_STOP_WORDS = ["google", "slides", "docs", "sheets", "search"]

CLUSTER_METHODS = ["kmeans", "dbscan"]
EMBEDDING_TEXT_COMBINATIONS = ["title", "title+description", "name+description", "title+domain"]
DIM_REDUCE_OPTIONS = [0, 5, 15]
NUM_CLUSTER_METHODS = ["knee", "silhouette"]
TOPIC_GENERATOR_OPTIONS = [T5_BASE_LOCAL_LABEL, OPENAI_CLOUD_LABEL]


class ModelProvider:
    def __init__(self):
        self.models = {}
    def get_model(self, name):
        if name not in self.models:
            self.models[name] =  SentenceTransformer(name, trust_remote_code=True)
        return self.models[name]
    def get_prefix(self, name):
        if name == EMBEDDING_NOMIC:
            return "clustering: "
        return None

def generate_embedding_features(data, model_name, model_provider: ModelProvider):
    data_list = data[EMBEDDING_TEXT_COLUMN].values.tolist()
    prefix = model_provider.get_prefix(model_name)
    if prefix:
        data_list = list(map(lambda a: f"{prefix}{a}", data_list))
    return model_provider.get_model(model_name).encode(data_list, show_progress_bar=False)

def get_title_embedding_transformer(model_name: str, model_provider: ModelProvider):
    return FunctionTransformer(partial(generate_embedding_features, model_name=model_name, model_provider=model_provider))

class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, column):
        self.column = column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.column]


class EmbeddingScaler(BaseEstimator, TransformerMixin):
    def __init__(self, scale_factor=1.0):
        self.scale_factor = scale_factor

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X * self.scale_factor



class MultiLabelBinarizerWrapper:
    def __init__(self):
        self.mlb = MultiLabelBinarizer()

    def fit(self, X, y=None):
        self.mlb.fit(X["domain_category_info"].to_list())
        return self

    def transform(self, X):
        aa = self.mlb.transform(X["domain_category_info"].to_list())
        return aa


def generate_pipeline(config, model_provider: ModelProvider):
    history_scale = config["history_scale"]
    domain_scale = config["domain_scale"]
    title_embedding_scale = config["title_embedding_scale"]
    tf_idf_scale = config["tf_idf_scale"]

    pipeline_domain = Pipeline(
        [
            ("selector", ItemSelector(column=["domain"])),
            ("domain_features", OneHotEncoder(handle_unknown="ignore")),
            ('scaler', EmbeddingScaler(scale_factor=domain_scale))
        ]
    )

    pipeline_history = Pipeline(
        [
            ("selector", ItemSelector(column=["browse_group"])),
            ("domain_features", OneHotEncoder(handle_unknown="ignore")),
            ('scaler', EmbeddingScaler(scale_factor=history_scale))
        ]
    )

    #    pipeline_domain_category = Pipeline(
    #        [
    #            ("selector", ItemSelector(column=["domain_category_info"])),
    #            ("domain_cat_features", MultiLabelBinarizerWrapper()),
    #            ('scaler', EmbeddingScaler(scale_factor=domain_category_scale))
    #        ]
    #    )
    title_embedding_transformer = get_title_embedding_transformer(config["embedding_model"], model_provider=model_provider)
    pipeline_title_embeddings = Pipeline([("title_embedding_features", title_embedding_transformer),
                                          ('scaler', EmbeddingScaler(scale_factor=title_embedding_scale))])

    stemmer = PorterStemmer()

    def stem_preprocess(text):
        tokens = word_tokenize(text)
        return ' '.join([stemmer.stem(token) for token in tokens])

    stop_words = list(text.ENGLISH_STOP_WORDS)
    # stop_words.extend(CUSTOM_STOP_WORDS)
    pipeline_tfidf = Pipeline(
        [
            ("selector", ItemSelector(column=EMBEDDING_TEXT_COLUMN)),
            (
                "tfidf_title",
                TfidfVectorizer(
#                    preprocessor=stemming_tokenizer,
                    ngram_range=(1, 2),
                    stop_words= list(stop_words) + ["google", "search", "sheets", "docs"],
                    max_df=0.95,
                    min_df=3,
                    max_features=1000,
                )
            ),
            ('scaler', EmbeddingScaler(scale_factor=tf_idf_scale))
        ]
    )
    combined_features = FeatureUnion([
        ("pipeline_title_embeddings", pipeline_title_embeddings),
        ("pipeline_tfidf", pipeline_tfidf),
        ("pipeline_domain", pipeline_domain),
        ("pipeline_history", pipeline_history)
    ])
    final_pipeline = Pipeline(
        [
            ("features", combined_features),
            ('normalizer', Normalizer()),
        ]
    )
    return final_pipeline



def add_text_for_embedding(df: pd.DataFrame, fields):
    has_set = False
    for col in fields.split("+"):
        if not has_set:
            df[EMBEDDING_TEXT_COLUMN] = df[col]
            has_set = True
        else:
            df[EMBEDDING_TEXT_COLUMN] = df[EMBEDDING_TEXT_COLUMN] + ". " + df[col].fillna("")
    return df



def generate_best_cluster_model(embeddings, cluster_space, verbose=False, use_dbscan=True, eps=0.3,
                                num_cluster_method="knee"):
    """
    takes embeddings and returns the best model
    using the elbow method and kmeans
    cluster_space is the range to search for the best cluster - e.g. range(1, 50)
    This can take a while to run for large datasets
    """
    from sklearn.cluster import DBSCAN

    # HDBSCAN or another clustering algorithm that has .fit and .predict functions and
    # the .labels_ variable to extract the labels
    #    self.hdbscan_model = hdbscan_model or hdbscan.HDBSCAN(
    #        min_cluster_size=self.min_topic_size,
    #        metric="euclidean",
    #        cluster_selection_method="eom",
    #        prediction_data=True,
    #    )

    if use_dbscan:
        db = hdbscan.HDBSCAN(
            min_cluster_size=2,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True
        )
        #        db = DBSCAN(eps=eps, min_samples=2, metric='cosine').fit_predict(embeddings)
        #        db = DBSCAN(eps=eps, min_samples=2, metric='euclidian').fit_predict(embeddings)
        return db

    if num_cluster_method == "knee":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            sum_of_squared_distances = []
            k_to_model = {}
            for k in cluster_space:
                if k > len(embeddings):
                    break
                model_k = KMeans(n_clusters=k).fit(embeddings)
                sum_of_squared_distances.append(model_k.inertia_)
                k_to_model[k] = model_k
                if verbose:
                    print(k, model_k.inertia_)

            kn = KneeLocator(
                cluster_space,
                sum_of_squared_distances,
                curve='convex',
                direction='decreasing',
                interp_method='interp1d',
            )
            if verbose:
                print('Best number of clusters: {}'.format(kn.knee))

            # kn.knee returns optimal cluster value
            if kn.knee is None:
                print("Warning -- knee not found -- defaulting to 4")
                return k_to_model[4]
            return k_to_model[kn.knee]
    else:
        k = silh_find_optimal_k(embeddings, cluster_space)
        return KMeans(n_clusters=k).fit(embeddings)


def run_pipeline(config, df, saved_set_name=None, model_provider = None):
    """
    Runs a tab grouping pipeline, grouping tabs into clusters and labeling each cluster
    Args:
        config: Configuration options
        df: Dataset
        saved_set_name: Name to save embeddings as a tsv for other use
        model_provider: Language model class
    Returns:
        dataset, rand score (based on labels), adjusted rand score
    """
    df = df.sample(frac=1)
    dbscan = config["clustering_method"] == "dbscan"
    pipeline = generate_pipeline(config, model_provider=model_provider)
    df = add_text_for_embedding(df, config["text_for_embedding"])
    model = pipeline.fit(df)
    pipeline_result = model.transform(df).toarray()
    if saved_set_name is not None:
        np.savetxt(f"output/{saved_set_name}.tsv", pipeline_result, delimiter="\t")
        df[["title", "smart_group_label"]].to_csv(f"output/{saved_set_name}_labels.tsv", sep="\t")
    embeddings_as_list = [pipeline_result.tolist() for _row in pipeline_result]
    if config["remap"] > 0:
        umap_model = umap.UMAP(
            n_neighbors=config["remap"],
            n_components=5,
            min_dist=0.0,
            metric="cosine"
        )
        pipeline_result = umap_model.fit_transform(pipeline_result)
    #       embeddings_as_list = [pipeline_result.tolist() for _row in pipeline_result]

    max_clusters = min(math.floor(math.log(len(embeddings_as_list)) * 2.0 + 1), len(embeddings_as_list))
    best_cluster = generate_best_cluster_model(pipeline_result, range(2, max_clusters),
                                               verbose=False, use_dbscan=dbscan, eps=config["dbscan_eps"],
                                               num_cluster_method=config["num_cluster_method"])
    if dbscan:
        clusters = best_cluster.fit_predict(pipeline_result)
    else:
        clusters = best_cluster.predict(pipeline_result)
    df["predicted_cluster"] = clusters
    df["embeddings"] = embeddings_as_list
    rscore = rand_score(df["smart_group_label"], df["predicted_cluster"])
    adj_rscore = adjusted_rand_score(df["smart_group_label"], df["predicted_cluster"])
    return df, rscore, adj_rscore


def get_default_config():
    config = {}
    config["history_scale"] = 0.0
    config["domain_scale"] = 0.0
    config["title_embedding_scale"] = 1.0
    config["tf_idf_scale"] = 0.0
    config["clustering_method"] = "kmeans"
    config["dbscan_eps"] = 0.4
    config["remap"] = 5
    config["num_cluster_method"] = "knee"
    config["text_for_embedding"] = "title"
    config["embedding_model"] = EMBEDDING_MODEL_LIST[0]
    return config



def sweep_params():
    all_results = []
    dataset_names = user_test_list
    for dataset_id in dataset_names:
        datasets, labeled_topics = get_labeled_dataset(dataset_id)
        model_provider = ModelProvider()
        for embedding_model in EMBEDDING_MODEL_LIST:
            for clustering_method in CLUSTER_METHODS:
                dbscan_eps_params = [0.4]
                if clustering_method == "kmeans":
                    num_cluster_methods = NUM_CLUSTER_METHODS
                else:
                    num_cluster_methods = ["knee"]
                if clustering_method == "dbscan":
                    dbscan_eps_params = [0.4] # add others here
                for num_cluster_method in num_cluster_methods:
                    for dbscan_eps in dbscan_eps_params:
                        for remap in DIM_REDUCE_OPTIONS:
                            for tf_idf_scale in [0.0]:
                                config = get_default_config()
                                config["embedding_model"] = embedding_model
                                config["remap"] = remap
                                config["dbscan_eps"] = dbscan_eps
                                config["tf_idf_scale"] = tf_idf_scale
                                config["clustering_method"] = clustering_method
                                config["num_cluster_method"] = num_cluster_method
                                res, score, adj_rscore = run_pipeline(config, datasets[0], model_provider=model_provider)
                                result_dict = {**config, "dataset": dataset_id, "rand": score, "adj_rand": adj_rscore}
                                all_results.append(result_dict)
                                # wandb.log(result_dict)
                                print("got result")
    return all_results

if __name__ == "__main__":
    nltk.download('punkt')
#    wandb.init(
#        set the wandb project where this run will be logged
#        project="smart-tab-cluster-eval")
    all_results = []
    for _k in range(4):
        all_results.extend(sweep_params())
        print("***next sweep")
    df = pd.DataFrame(all_results)
    df.to_csv("./output/all_pipeline_embedding_test.csv")


