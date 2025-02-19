from typing import List

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
from bertopic.vectorizers import ClassTfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

MAX_WORDS_PER_TOPIC = 3

class KeyDocumentFinder:
    def __init__(self, df: pd.DataFrame, cluster_key="predicted_cluster", text_key="text_for_embedding"):
        self.df = df
        self.topic_id_key = cluster_key
        self.text_key = text_key

    def get_important_keywords(self):
        self.documents_per_topic = self.df.groupby([self.topic_id_key], as_index=False).agg({self.text_key: " ".join})
        self.topic_id_list = self.documents_per_topic[self.topic_id_key].unique().tolist()
        self.vectorizer_model = CountVectorizer(stop_words="english")
        self.document_list = self.documents_per_topic[self.text_key].values
        self.vectorizer_model.fit(self.document_list)
        self.vectorizer_words = self.vectorizer_model.get_feature_names_out()
        self.important_word_matrix = self.vectorizer_model.transform(self.document_list)
        self.ctfidf_model = ClassTfidfTransformer()
        self.weighted_important_word_matrix = self.ctfidf_model.fit_transform(self.important_word_matrix).toarray()

    def get_keywords_and_topics(self):
        self.sorted_indices = np.argsort(-self.weighted_important_word_matrix, 1)
        num_words = self.weighted_important_word_matrix.shape[1]
        num_clusters = self.weighted_important_word_matrix.shape[0]
        self.sorted_scores = np.take_along_axis(self.weighted_important_word_matrix, self.sorted_indices, axis=1)
        self.topic_info_list = []
        self.keyword_list = []
        for cluster in range(num_clusters):
            topic_words = []
            for top_score_ref in range(num_words):
                if self.sorted_scores[cluster, top_score_ref] > 0.05:
                    topic_words.append(self.vectorizer_words[self.sorted_indices[cluster, top_score_ref]])
                if len(topic_words) > MAX_WORDS_PER_TOPIC:
                    break
            self.topic_info_list.append(", ".join(topic_words))
            self.keyword_list.append(topic_words)

    def get_category_embeddings(self):
        bert_model = SentenceTransformer('all-mpnet-base-v2')
        self.topic_embeddings = bert_model.encode(self.topic_info_list)

    def select_best_documents(self, include_embeddings=False):
        if not include_embeddings:
            return self.select_first_documents()
        bert_model = SentenceTransformer('all-mpnet-base-v2')
        self.document_embeddings = bert_model.encode(self.df[self.text_key].to_list())
        #        self.similarities = cosine_similarity(self.document_embeddings, self.topic_embeddings)
        rep_docs_for_topic = {}
        for i in range(len(self.topic_id_list)):
            cur_topic_id = self.topic_id_list[i]
            indices = self.df[self.df[self.topic_id_key] == cur_topic_id].index.to_list()
            topic_embedding = self.topic_embeddings[i, :]
            documents_for_topic = self.document_embeddings[indices, :]
            similarity = cosine_similarity(documents_for_topic, topic_embedding.reshape(1, -1))
            top_items = np.argsort(-similarity.reshape(-1))
            rep_docs_for_topic[cur_topic_id] = np.array(indices)[top_items][:3]
        return rep_docs_for_topic

    def select_first_documents(self):
        first_docs_by_topic = {}
        for i in range(len(self.topic_id_list)):
            cur_topic_id = self.topic_id_list[i]
            selected_items = self.df[self.df[self.topic_id_key] == cur_topic_id].index.to_list()[:3]
            first_docs_by_topic[cur_topic_id]  = [int(a) for a in selected_items]
        return first_docs_by_topic

    def get_keywords_for_group(self, topic_id) -> List[str]:
        index = self.topic_id_list.index(topic_id)
        return self.keyword_list[index]
    def compute_all(self, include_embeddings=False):
        self.get_important_keywords()
        self.get_keywords_and_topics()
        if not include_embeddings:
            self.get_category_embeddings()
        self.best_document_indices = self.select_best_documents(include_embeddings=include_embeddings)
        return self.best_document_indices

    def get_best_documents(self):
       return self.best_document_indices
