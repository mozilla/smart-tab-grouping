import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

hint_embedder = pipeline("feature-extraction", model="sentence-transformers/all-MiniLM-L6-v2", device=-1)

FIXED_TOPICS = {"Adult Content", "None"}
class ShortenTopicLength:
    def __init__(self, boost_threshold=0.1):
        self.boost_threshold = boost_threshold
        self.hint_embedder = pipeline("feature-extraction", model="sentence-transformers/all-MiniLM-L6-v2", device=-1)
    def get_embedding_for_document(self, row):
        text = row["input_titles"] + " " + row["input_keywords"]
        return np.mean(self.hint_embedder(text)[0], axis=0)
    def shorten_single_topic(self, row):
        initial = row.output
        if initial in FIXED_TOPICS:
            return initial
        words = initial.split()
        initial_embedding = self.hint_embedder(initial)[0][0]
        if len(words) == 1:
            return initial
        alt_embeddings = [initial_embedding]
        alt_phrases = [initial]
        for missing_word_index in range(len(words)):
            # remove 1 word
            shortened_phrase = " ".join(words[:missing_word_index] + words[missing_word_index + 1:])
            embed = self.hint_embedder(shortened_phrase)[0][0]
            alt_phrases.append(shortened_phrase)
            alt_embeddings.append(embed)
        if len(words) > 2:
            # remove 2 words
            for missing_word_index in range(len(words) - 1):
                shortened_phrase = " ".join(words[:missing_word_index] + words[missing_word_index + 2:])
                embed = self.hint_embedder(shortened_phrase)[0][0]
                alt_phrases.append(shortened_phrase)
                alt_embeddings.append(embed)
        document_embedding = self.get_embedding_for_document(row).reshape(1, -1)
        similarity = cosine_similarity(document_embedding, np.array(alt_embeddings)).squeeze()
        similarity[0] -= self.boost_threshold
        closest_indices = np.argsort(-similarity)
        best_phrases = [alt_phrases[i] for i in closest_indices.tolist()]
        print(f"{initial} -> {best_phrases[0]}")
        return best_phrases[0]

    def shorten_topics(self, df: pd.DataFrame):
        df["output"] = df.apply(self.shorten_single_topic, axis=1)
        return df

if __name__ == "__main__":
    test_df = pd.DataFrame({"output": ["Desk concert concert", "dogs dogs dogs"], "input_titles":["Desk concert", "dogs"], "input_keywords":[" desk concert" , "bla"]})
    stl = ShortenTopicLength()
    df = stl.shorten_topics(test_df)
    print(df)
