import os

import numpy as np
import pandas as pd
import torch
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity

from util.language_model import MessageSequenceItem, LanguageModel
from typing import List, Dict
import json


class Prompts(BaseModel):
    document_key: str = "[DOCUMENTS]"
    keyword_key: str = "[KEYWORDS]"
    header: str
    next: str


doc_only_prompts = Prompts(header="You are a webpage tab group classifier. Provide a succinct group "
                                  "name that encompasses the pages with the "
                                  "following page titles [DOCUMENTS]",
                           next="[DOCUMENTS]")

keyword_prompts = Prompts(header="You are a webpage tab group classifier. Provide a"
                                 " succinct group name that encompasses the pages with "
                                 "common keywords: [KEYWORDS]. Some of the titles are: [DOCUMENTS]",
                          next="Keywords: [KEYWORDS], Documents: [DOCUMENTS]")

from transformers import T5ForConditionalGeneration, T5Tokenizer, pipeline


class TopicGenerator:

    def __init__(self, prompts: Prompts = doc_only_prompts):
        self.prompts = prompts
        self.title_delimiter = "\n "
        pass

    def get_topic(self, sample_data, max_tokens: int) -> str:
        return "Base class"

    def create_simple_prompt(self, test_data):
        simple_prompt_string = "Generate a topic from these web titles: \n "
        return simple_prompt_string + self.title_delimiter.join(test_data["documents"])

    def create_simple_prompt_with_keywords(self, test_data, legacy=False):
        if legacy:
            # This was an experiment for local inference. No model was trained with these words
            return f"Generate a topic from these keywords: {', '.join(test_data['keywords'])} " \
                   f"/n documents: {self.title_delimiter.join(test_data['documents'])}"
        else:
            return f"Topic from keywords: {', '.join(test_data['keywords'])}. titles: \n{self.title_delimiter.join(test_data['documents'])}"

    def create_prompt_for_llm(self, documents: str, keywords: str, is_initial=False):
        base_prompt = self.prompts.header if is_initial else self.prompts.next
        return base_prompt.replace(self.prompts.document_key, documents).replace(self.prompts.keyword_key, keywords)

    def create_prompt_sequence(self, test_data, sample_data, limit_samples=999) -> List[MessageSequenceItem]:
        def serialize_docs(docs: List[str]):
            return "\n".join(docs)

        def serialize_keywords(keywords: List[str]):
            return ", ".join(keywords)

        def get_prompt_msg(info, is_initial=False):
            return self.create_prompt_for_llm(
                serialize_docs(info["documents"][:3]),
                serialize_keywords(info["keywords"][:3]),
                is_initial
            )

        cur_sample_data = sample_data[:limit_samples]
        remaining_sample_data = cur_sample_data[1:]
        initial_item = cur_sample_data[0]
        prompt_list = []

        start_message = self.lm.user_message(get_prompt_msg(initial_item, is_initial=True))
        response_message = self.lm.assistant_message(initial_item["title"])
        prompt_list.append(start_message)
        prompt_list.append(response_message)

        for item in remaining_sample_data:
            query_message = self.lm.user_message(get_prompt_msg(item))
            response_message = self.lm.assistant_message(item["title"])
            prompt_list.append(query_message)
            prompt_list.append(response_message)

        final_query = self.lm.user_message(get_prompt_msg(test_data))
        prompt_list.append(final_query)
        return prompt_list


class T5TopicGenerator(TopicGenerator):
    def __init__(self, model_name="./models/smart-tab-topic", title_delimiter=None):
        super().__init__()
        print(os.getcwd())
        with open("./data/multi_shot_topic_examples.json") as f:
            self.sample_data = json.load(f)
        self.model_name = model_name
        if title_delimiter is not None:
            self.title_delimiter = title_delimiter
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)


    def generate_response(self, prompt, max_tokens=34, prob_limit=None):
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        if prob_limit is not None:
            outputs = self.model.generate(input_ids, output_scores=True, return_dict_in_generate=True)
            probs = torch.softmax(outputs.scores[0], dim=-1)
            max_prob, _ = torch.max(probs, dim=-1)
            if max_prob < prob_limit:
                return "None"
            return self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        else:
            outputs = self.model.generate(input_ids, max_length=max_tokens, num_return_sequences=1)
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    def generate_token_response(self, prompt, max_tokens=34):
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        outputs = self.model.generate(input_ids, max_length=max_tokens, num_return_sequences=1,
                                      return_dict_in_generate=True, output_scores=True)
        return outputs

    def get_topic(self, data, max_tokens: int = 12) -> str:
        seq = self.create_simple_prompt(data)
        return self.generate_response(seq, max_tokens)

    def get_topic_with_keywords(self, data, max_tokens: int = 12, legacy=False, prob_limit=None) -> str:
        seq = self.create_simple_prompt_with_keywords(data, legacy=legacy)
        return self.generate_response(seq, max_tokens, prob_limit=prob_limit)


class OpenAITopicGenerator(TopicGenerator):
    def __init__(self, support_keywords=False, support_hints=False):
        super().__init__(prompts=doc_only_prompts if not support_keywords else keyword_prompts)
        self.hint_db = None
        self.lm = LanguageModel()
        with open("data/multi_shot_topic_examples.json") as f:
            self.sample_data = json.load(f)
        self.hint_embeddings = None
        self.hint_embedder = None

    def get_embeddings_for_hint_matching(self, input_df):
        embed_input = pd.Series(input_df["input_titles"] + " " + input_df["input_keywords"]).to_list()
        return np.array([np.mean(self.hint_embedder(sentence)[0], axis=0) for sentence in embed_input])

    def prepare_hint_data(self, hint_db):
        """
        Ingest a dataframe with a 'Hint' column that has user labeled
        hints. We will use embedding distance to use these examples to
        inform the OpenAI multi-shot example.
        """
        self.hint_embedder = pipeline("feature-extraction", model="sentence-transformers/all-MiniLM-L6-v2", device=-1)
        self.hint_db = hint_db.dropna(subset=['Hint']).reset_index(drop=True)
        self.hint_db.fillna("", inplace=True)
        self.hint_embeddings = self.get_embeddings_for_hint_matching(self.hint_db)

    def get_closest_embeddings(self, item_embedding: np.ndarray, num_items: int):
        similarity = cosine_similarity(item_embedding.reshape(1,-1), self.hint_embeddings).squeeze()
        closest_indices = np.argsort(-similarity)[:num_items]
        return closest_indices.tolist()

    def get_topic(self, data: Dict[str, List[str]], max_tokens: int = 6, num_similar_items=4) -> str:
        hint_item_list = self.sample_data
        if self.hint_embeddings is not None:
            embed_input = AnnotationItem(**data).get_combined_string_for_embedding()
            item_embedding = np.mean(self.hint_embedder(embed_input)[0], axis=0)
            closest_ids = self.get_closest_embeddings(item_embedding, num_items=num_similar_items)
            closest_rows = self.hint_db.iloc[closest_ids]
            custom_items = [{"documents": r.input_titles.split("\n"),
                             "keywords": r.input_keywords.split(","),
                             "title": r.Hint} for _, r in closest_rows.iterrows()]
            hint_item_list = hint_item_list + custom_items[:4]
        seq = self.create_prompt_sequence(AnnotationItem(data["documents"], data["keywords"], data["descriptions"]).get_dict(), hint_item_list)
        print("final sequence")
        print(seq[-1])
        return self.lm.generic_query(seq, max_tokens=max_tokens)


class AnnotationItem():
    def __init__(self, documents, keywords, descriptions=None):
        self.documents = documents
        self.keywords = keywords
        self.descriptions = descriptions
        if isinstance(self.documents, str):
            self.documents = list(filter(lambda a: len(a) > 0, self.documents.split("\n")))
        if isinstance(self.keywords, str):
            self.keywords = list(filter(lambda a: len(a) > 0, self.keywords.split(",")))

    def get_document_string(self):
        if not self.documents or len(self.documents) == 0:
            return ""
        else:
            return "\n".join(self.documents)

    def get_keyword_string(self):
        if not self.keywords or self.keywords != self.keywords or len(self.keywords) == 0:
            return ""
        else:
            return ",".join(self.keywords)

    def get_combined_string_for_embedding(self):
        return self.get_document_string() + " " + self.get_keyword_string()

    def get_dict(self):
        return {"documents": self.documents, "keywords": self.keywords}
