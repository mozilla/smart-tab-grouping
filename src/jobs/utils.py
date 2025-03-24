import requests
from transformers import T5Tokenizer

# Function to load words from a URL
def load_words_from_url(url):
    response = requests.get(url)
    response.raise_for_status()
    words = {line.strip() for line in response.text.splitlines()}
    return words

def get_bad_word_ids(model_name: str = "Mozilla/smart-tab-topic"):
    tokenizer = T5Tokenizer.from_pretrained(model_name, add_prefix_space=True)
    bad_words_profanity = load_words_from_url(
        "https://raw.githubusercontent.com/mozilla/fx-ml-train/refs/heads/main/trust_and_safety/word_blocking/profanity.txt"
    )
    bad_words_stg = load_words_from_url(
        "https://raw.githubusercontent.com/mozilla/fx-ml-train/refs/heads/main/trust_and_safety/word_blocking/smart_tab_grouping_specific.txt"
    )
    bad_words = list(set(list(bad_words_stg) + list(bad_words_profanity)))
    bad_words = bad_words + [b.title() for b in bad_words] # previous bad words + title case

    bad_words_ids = tokenizer(bad_words, add_special_tokens=False).input_ids
    print({
        'bad_words': bad_words,
        'bad_word_ids': bad_words_ids
    })
    return bad_words_ids

if __name__ == "__main__":
    profanity = load_words_from_url(
        "https://raw.githubusercontent.com/mozilla/fx-ml-train/refs/heads/main/trust_and_safety/word_blocking/profanity.txt"
    )
    stg_specific_words = load_words_from_url(
        "https://raw.githubusercontent.com/mozilla/fx-ml-train/main/trust_and_safety/word_blocking/smart_tab_grouping_specific.txt"
    )
    bad_words = list({*profanity, *stg_specific_words})
    print(bad_words)
