import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import string

hint_embedder = pipeline("feature-extraction", model="sentence-transformers/all-MiniLM-L6-v2", device=-1)

FIXED_TOPICS = {"Adult Content", "None"}
PRESERVE_WORDS = {"real", "estate"} # real estate

class ShortenTopicLength:
    def __init__(self, boost_threshold=0.1):
        self.spell = None
        self.boost_threshold = boost_threshold
        self.hint_embedder = pipeline("feature-extraction", model="sentence-transformers/all-MiniLM-L6-v2", device=-1)
        self.translator = str.maketrans('', '', string.punctuation)

    def get_document_for_row(self, row):
        return row["input_titles"] + " " + row["input_keywords"]

    def get_embedding_for_document(self, row):
        text = self.get_document_for_row(row)
        return np.mean(self.hint_embedder(text)[0], axis=0)

    def pluralize_single_word(self, phrase, document_words_lower):
        words = phrase.split()
        new_words = []
        if len(words) > 1: # for now only look at single word cases
            return phrase
        for word in words:
            if word[-1] != 's':
                if (word.lower() + 's') in document_words_lower:
                    word = word + 's'
            new_words.append(word)
        return " ".join(new_words)

    def shorten_single_topic(self, row):
        initial = row.output
        if initial in FIXED_TOPICS:
            return initial
        try:
            words = initial.split()
            initial_embedding = self.hint_embedder(initial)[0][0]
            if len(words) == 1:
                return initial
            alt_embeddings = [initial_embedding]
            alt_phrases = [initial]

            document_words = set(self.get_document_for_row(row).translate(self.translator).lower().split())

            for missing_word_index in range(len(words)):
                # remove 1 word
                word_getting_removed = words[missing_word_index]
                if word_getting_removed.lower() in PRESERVE_WORDS:
                    continue
                if len(self.spell.unknown([word_getting_removed])) == 1: # skip rare word
                    continue
                shortened_phrase = " ".join(words[:missing_word_index] + words[missing_word_index + 1:])
                embed = self.hint_embedder(shortened_phrase)[0][0]
                alt_phrases.append(shortened_phrase)
                alt_embeddings.append(embed)

            if len(words) > 2:
                # remove 2 words
                for missing_word_index in range(len(words) - 1):
                    words_to_remove = words[missing_word_index:missing_word_index+2]
                    for w in words_to_remove:
                        if w.lower() in PRESERVE_WORDS:
                            continue
                    if len(self.spell.unknown(words_to_remove)) > 0: # either word unknown
                        continue
                    shortened_phrase = " ".join(words[:missing_word_index] + words[missing_word_index + 2:])
                    embed = self.hint_embedder(shortened_phrase)[0][0]
                    alt_phrases.append(shortened_phrase)
                    alt_embeddings.append(embed)
            document_embedding = self.get_embedding_for_document(row).reshape(1, -1)
            similarity = cosine_similarity(document_embedding, np.array(alt_embeddings)).squeeze()
            similarity[0] -= self.boost_threshold
            closest_indices = np.argsort(-similarity)
            best_phrases = [alt_phrases[i] for i in closest_indices.tolist()]
            best_phrase = best_phrases[0]
            best_phrase = self.pluralize_single_word(best_phrase, document_words)
            print(f"{initial} -> {best_phrase}")
            return best_phrase
        except:
            return initial


    def shorten_topics(self, df: pd.DataFrame):
        from spellchecker import SpellChecker
        self.spell = SpellChecker()
        self.spell.word_frequency.load_words(['microsoft', 'apple', 'google', 'bing', 'search', 'duckduckgo', 'yahoo'])
        df["output"] = df.apply(self.shorten_single_topic, axis=1)
        return df

if __name__ == "__main__":
    test_df = pd.DataFrame({"output": ["Desk concert concert",
                                       "dogs dogs dogs",
                                       "Guitar Store",
                                       "Phibbity News"],
                            "input_titles":["Desk concert",
                                            "dogs",
                                            "Guitars galore",
                                            "news"],
                            "input_keywords":["desk concert",
                                              "bla",
                                              "guitars",
                                              "news"]})
    stl = ShortenTopicLength()
    df = stl.shorten_topics(test_df)
    print(df)
