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

    bad_words = load_words_from_url(
        "https://raw.githubusercontent.com/snguyenthanh/better_profanity/master/better_profanity/profanity_wordlist.txt"
    )

    vocab = tokenizer.get_vocab()
    vocab = [tokenizer.convert_tokens_to_string([token]).lower().strip() for token in vocab]

    vocab_bad_words = [word for word in vocab if word in bad_words]
    bad_words_ids = []
    for word in vocab_bad_words:
        tokenized_word = tokenizer([word], add_special_tokens=False).input_ids[0]
        bad_words_ids.append(tokenized_word)
    print({
        'bad_words': vocab_bad_words,
        'bad_word_ids': bad_words_ids
    })
    return bad_words_ids

