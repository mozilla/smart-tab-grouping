import pandas as pd
from pandas import DataFrame
from functools import partial
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import numpy as np
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

class NLPEvaluator:
    def __init__(self):
        from rouge_score import rouge_scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.embedder = pipeline("feature-extraction", model="sentence-transformers/all-MiniLM-L6-v2", device=-1)
        self.stemmer = PorterStemmer()

    def compute_scores(self, row, label_key="label", pred_key=None):
        def cos_sim(s1, s2):
            embeddings = [np.mean(self.embedder(s)[0], axis=0) for s in [s1, s2]]
            similarity = cosine_similarity(embeddings[0].reshape(1, -1), embeddings[1].reshape(1, -1)).squeeze()
            return similarity

        def count_repeat_words(prediction):
            words = prediction.split()
            word_set = set()
            num_duplicate_words = 0
            for word in words:
                stem = self.stemmer.stem(word)
                if stem in word_set:
                    num_duplicate_words += 1
                else:
                    word_set.add(stem)
            return num_duplicate_words

        scores = self.rouge_scorer.score(row[label_key], row[pred_key])
        return {
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure,
            'pred_len': len(row[pred_key]),
            'label_len': len(row[label_key]),
            'repeat_words': count_repeat_words(row[pred_key]),
            'cos_sim': cos_sim(row[label_key], row[pred_key])
        }

    def get_avg_scores(self, input_df: DataFrame, compare_column: str, label_key="label"):
        rouge_scores_df = input_df.apply(partial(self.compute_scores, label_key=label_key, pred_key=compare_column),
                                         axis=1,
                                         result_type='expand')
        average_scores = rouge_scores_df.mean().to_dict()
        return average_scores


if __name__ == "__main__":
    eval = NLPEvaluator()
    df = pd.DataFrame({"label": ["dog", "cat", "apple"], "compare": ["bunny", "cat", "orange"]})
    print(eval.get_avg_scores(df, label_key="label", compare_column="compare"))
