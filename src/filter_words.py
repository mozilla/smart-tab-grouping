import requests
from transformers import AutoTokenizer
import json

model_name = "Mozilla/smart-tab-topic"

# Function to load words from a URL
def load_words_from_url(url):
    response = requests.get(url)
    response.raise_for_status()
    words = {line.strip() for line in response.text.splitlines()}
    return words

tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)

bad_words = load_words_from_url(
    "https://raw.githubusercontent.com/snguyenthanh/better_profanity/master/better_profanity/profanity_wordlist.txt"
)

vocab = tokenizer.get_vocab()
vocab = [tokenizer.convert_tokens_to_string([token]).lower().strip() for token in vocab]

vocab_bad_words = [word for word in vocab if word in bad_words]
print(len(vocab_bad_words), vocab_bad_words)

def get_tokens_as_list(word_list):
    tokens_list = []
    for word in word_list:
        tokenized_word = tokenizer([word], add_special_tokens=False).input_ids[0]
        tokens_list.append(tokenized_word)
    return tokens_list


bad_words_ids = get_tokens_as_list(vocab_bad_words)
print({'bad_word_ids': bad_words_ids})

with open("bad_words_smart_topic.json", "w") as file:
    json.dump({
        'bad_words': vocab_bad_words,
        'bad_words_ids': bad_words_ids
    }, file, indent=4)
