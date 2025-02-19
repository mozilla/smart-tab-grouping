import concurrent
import json
from enum import Enum
from typing import List, TypedDict
import logging
import os
import time
import re

import pandas as pd

from openai import OpenAI



class ModelType(Enum):
    OpenAIChat = "gpt-3.5-turbo"
    OpenAIDaVinci = "text-davinci-003"
    OpenAICurie = "text-curie-001"
    OpenAIBabbage = "text-babbage-001"
    OpenAIAda = "text-ada-001"
    OpenAI40 = "gpt-4o"


class MessageSequenceItem(TypedDict):
    role: str
    content: str


log_items = []


def stats_log(log_item):
    log_items.append(log_item)


def dump_lang_logs(filename):
    with open(filename, "w") as ff:
        json.dump({"logs": log_items}, ff)


class LanguageModel:
    """
    Wrapper for OpenAI or similar language models
    """

    def __init__(self, engine: ModelType = ModelType.OpenAI40, temperature=0.1,
                 logger: logging.Logger = logging.getLogger()):
        self._engine = engine.value
        self._temperature = temperature
        self._logger = logger
        logger.setLevel(10)
        self._retry_delay_sec = 1
        self._num_retries = 3
        self.setup_openai()

    def setup_openai(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def cleanup_result(self, test: str) -> str:
        return test.lower().replace(".", "").strip()

    @classmethod
    def user_message(cls, text) -> MessageSequenceItem:
        return {"role": "user", "content": text}

    @classmethod
    def assistant_message(cls, text) -> MessageSequenceItem:
        return {"role": "assistant", "content": text}

    def generic_query(self, messages: List[MessageSequenceItem], retry_count=0, max_tokens=None, json_only=False) -> str:
        """
        Do a basic openai query
        :param text: Prompt
        :return: Response, or empty string if there is a failure
        """
        retry_allowance = retry_count
        response_format = {"type": "json_object"} if json_only else None
        while retry_allowance >= 0:
            retry_allowance -= 1
            try:
                messages = [dict(m) for m in messages]
                url_response = self.client.chat.completions.create(model=self._engine,
                temperature=self._temperature,
                messages=messages,
                max_tokens=max_tokens,
                response_format=response_format)
                response = url_response.choices[0].message.content
                stats_log(
                    {"message": messages[0]["content"][0:50], "response": response})
                if self._logger is not None:
                    self._logger.debug(f"OpenAI Response {response}")
                return response
            except Exception as ex:
                if self._logger is not None:
                    self._logger.error(f"Error calling openai {ex}")
                if retry_count == 0:
                    raise(ex)
                if self._retry_delay_sec > 0:
                    time.sleep(self._retry_delay_sec)
        self._logger.error("Exceeded retry count. Returning empty string result")
        return ""

    def text_query(self, text: str, retry_count=0) -> str:
        """
        Do a basic openai query
        :param text: Prompt
        :return: Response, or empty string if there is a failure
        """
        return self.generic_query([self.user_message(text)], retry_count=retry_count)

    def ask_list_boolean(self, item_list, prompt, if_error=False, max_error=10, num_threads=4):
        """
        Asks the same question prompt with a list of data objects
        :param item_list:
        :param prompt:
        :param if_error:
        :return:
        """
        error_count = 0

        def get_boolean_result(item):
            combined_query = f"{prompt} Answer 1 for yes and 0 for no on a single line without any additional comments. ###{item}###"
            response = self.text_query(combined_query, retry_count=5)
            res = self.cleanup_result(response)
            if res == "0":
                return False
            elif res == "1":
                return True
            else:
                self._logger.warning(f"Boolean request failed with response {response}")
                return False

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            results = list(executor.map(get_boolean_result, item_list))

        return results

    def process_list_response(self, response):
        pattern = re.compile(r'(?:\d.)?(.+?)(?:\n|$)', re.DOTALL)
        # Extract the results using the regular expression pattern
        results = pattern.findall(response)
        return list(map(lambda a: a.strip(), results))

    def list_query(self, query) -> List[str]:
        """
        Use to call language model with a query that has a list of results.
        Typically, the query string should include 'Respond with each topic alone on a separate line'
        :return: a list
        """
        openai_response = self.text_query(query)
        return self.process_list_response(openai_response)

    def ask_df(self, message: str, key_items: List[str]) -> pd.DataFrame:
        key_item_text = ""
        for key_item in key_items:
            key_item_text = key_item_text + f'"{key_item}": "<text-only>",\n'
        if len(key_item_text) > 0:
            key_item_text = key_item_text[:-1]
        json_pattern = "{ \"RESULT\": [ {" + key_item_text + "} , ...]"
        complete_message = f"{message} Respond in JSON list format to match the following exact pattern with exact keys: " \
                           f"{json_pattern}"
        response = self.generic_query([self.user_message(complete_message)], retry_count=2, json_only=True)
        print("Paring response")
        print(response)
        array = json.loads(response)["RESULT"]
        response = pd.DataFrame(data=array)
        return response

    def get_list(self, message_str):
        res_df = self.ask_df(message_str, ["list_item"])
        return res_df["list_item"].to_list()
