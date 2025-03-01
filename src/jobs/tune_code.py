import os
from datetime import datetime

import pandas as pd
from pydantic import BaseModel

import wandb
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split

from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    TrainingArguments, Trainer
)

from util.storage import upload_directory
from utils import get_bad_word_ids

INPUT_PROMPT_ID = "input_with_prompt"


class PromptGenerator(BaseModel):
    document_key: str = "[DOCUMENTS]"
    keyword_key: str = "[KEYWORDS]"
    prompt: str

    def generate_prompt(self, documents, keywords):
        res = self.prompt.replace(self.keyword_key, keywords)
        res = res.replace(self.document_key, documents)
        return res

class ConditionalPromptGenerator(PromptGenerator):
    multi_doc_generator: PromptGenerator
    single_doc_generator: PromptGenerator
    def generate_prompt(self, documents, keywords):
        if len(keywords) > 0:
            return self.multi_doc_generator.generate_prompt(documents, keywords)
        else:
            return self.single_doc_generator.generate_prompt(documents, keywords)

document_prompt = PromptGenerator(prompt="Generate a topic from these web titles: \n [DOCUMENTS]")
keyword_prompt = PromptGenerator(prompt="Topic from keywords: [KEYWORDS]. titles: \n[DOCUMENTS]")
keyword_prompt_one_tab = PromptGenerator(prompt="Topic from title: \n[DOCUMENTS]")
hybrid_prompt_gen = ConditionalPromptGenerator(prompt="", multi_doc_generator=keyword_prompt,
                                               single_doc_generator=keyword_prompt_one_tab)

class GenTrainer:

    def __init__(self, learning_rate: float = 1e-4, batch_size: int = 2, model_name: str = 'google/flan-t5-base',
                 label_column: str = "output", use_keywords: bool = True, single_tab_handling: bool = False,
                 learning_rate_decay: bool = True):
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.label_column = label_column
        self.learning_rate_decay = learning_rate_decay
        self.single_tab_handling = single_tab_handling
        self.use_keywords = use_keywords
        self.prompter = keyword_prompt if use_keywords else document_prompt
        if self.single_tab_handling:
            self.prompter = hybrid_prompt_gen
        self.model = None

    def compute_metrics(self, eval_pred):
        from rouge_score import rouge_scorer, scoring
        predictions, labels = eval_pred
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_labels = [label.split(self.tokenizer.eos_token, 1)[0] for label in decoded_labels]
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        aggregator = scoring.BootstrapAggregator()

        for pred, label in zip(decoded_preds, decoded_labels):
            print(f"Pred:{pred} Label:{label}")
            scores = scorer.score(target=label, prediction=pred)
            aggregator.add_scores(scores)
        result = aggregator.aggregate()
        final_result = {key: value.mid.fmeasure for key, value in result.items()}
        wandb.log(final_result)
        return final_result

    def compute_metrics_text(self, decoded_preds, decoded_labels):
        from rouge_score import rouge_scorer, scoring
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        aggregator = scoring.BootstrapAggregator()
        for pred, label in zip(decoded_preds, decoded_labels):
            scores = scorer.score(target=label, prediction=pred)
            print(f"comparing {pred} with label {label}")
            aggregator.add_scores(scores)
        result = aggregator.aggregate()
        final_result = {key: value.mid.fmeasure for key, value in result.items()}
        wandb.log(final_result)
        return final_result

    def preprocess_function(self, examples):
        inputs = examples["input_text"]
        targets = examples["target_text"]
        model_inputs = self.tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(targets, max_length=512, truncation=True, padding="max_length")
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def setup_data(self, topic_data: pd.DataFrame, filename: str = "unknown"):
        self.filename = filename
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
        topic_data.input_keywords = topic_data.input_keywords.fillna("")
        topic_data[INPUT_PROMPT_ID] = topic_data.apply(lambda row: self.prompter.generate_prompt(row.input_titles, row.input_keywords), axis=1)
        topic_data_training, topic_data_eval = train_test_split(topic_data, test_size=0.2)
        train_data_dict = {"input_text": topic_data_training[INPUT_PROMPT_ID].to_list(),
                           "target_text": topic_data_training[self.label_column].to_list()}
        eval_data_dict = {"input_text": topic_data_eval[INPUT_PROMPT_ID].to_list(),
                          "target_text": topic_data_eval[self.label_column].to_list()}
        self.train_dataset = Dataset.from_pandas(pd.DataFrame(train_data_dict))
        self.eval_dataset = Dataset.from_pandas(pd.DataFrame(eval_data_dict))

        print(f"Training Dataset size {len(self.train_dataset)}")
        print(f"Eval Dataset size {len(self.eval_dataset)}")

    def train(self):
        torch.cuda.empty_cache()

        os.environ["WANDB_LOG_MODEL"] = "end"  # save the model to WandB

        wandb.init(project="tab_grouping",
                   config={"learning_rate": self.learning_rate, "batch_size": self.batch_size,
                           "model_name": self.model_name,
                           "label_column": self.label_column,
                           "use_keywords": self.use_keywords,
                           "learning_rate_decay": self.learning_rate_decay,
                           "single_tab_handling": self.single_tab_handling,
                           "input_prompt_id": INPUT_PROMPT_ID, "filename": self.filename})
        print(f"W&B Run ID: {wandb.run.id}")
        print(f"W&B Run Name: {wandb.run.name}")

        tokenized_training_dataset = self.train_dataset.map(self.preprocess_function, batched=True)
        tokenized_eval_dataset = self.eval_dataset.map(self.preprocess_function, batched=True)

        if self.learning_rate_decay:
            training_args = TrainingArguments(
                output_dir="./results",
                evaluation_strategy="epoch",
                learning_rate=self.learning_rate,
                per_device_train_batch_size=self.batch_size,
                per_device_eval_batch_size=1,
                num_train_epochs=3,
                weight_decay=0.01,
                save_total_limit=1,
                save_strategy="epoch",
                lr_scheduler_type="cosine",
                warmup_ratio=0.1
            )
        else:
            training_args = TrainingArguments(
                output_dir="./results",
                evaluation_strategy="epoch",
                learning_rate=self.learning_rate,
                per_device_train_batch_size=self.batch_size,
                per_device_eval_batch_size=1,
                num_train_epochs=3,
                weight_decay=0.01,
                save_total_limit=1,
                save_strategy="epoch",
            )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_training_dataset,
            eval_dataset=tokenized_training_dataset
        )

        trainer.train()
        results_labels = []
        results_output = []

        for item in tokenized_eval_dataset:
            input_ids = self.tokenizer(item["input_text"], return_tensors="pt").input_ids.to("cuda:0")
            label = item['target_text']
            outputs = self.model.generate(input_ids, max_length=30, num_return_sequences=1)
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            results_output.append(response)
            results_labels.append(label)
        self.compute_metrics_text(results_output, results_labels)
        validation_table = wandb.Table(columns=["input", "label", "prediction"], data=
                            [list(map(lambda a: a["input_text"], tokenized_eval_dataset)),
                            results_labels,
                            results_output])
        wandb.log({"Validation Data": validation_table})

        self.model.generation_config.update(bad_words_ids=get_bad_word_ids())
        self.model.save_pretrained("./t5-finetuned-topic")
        self.tokenizer.save_pretrained("./t5-finetuned-topic")

        current_date = datetime.now()
        date_string = current_date.isoformat().replace(":", "_")
        upload_directory("./t5-finetuned-topic", "stage-fx-tab-grouping", f"topic/models/{date_string}/", depth=1)

        wandb.finish()
        self.model = None
        torch.cuda.empty_cache()


if __name__ == '__main__':
    print(keyword_prompt.generate_prompt("Doc 1\nDoc 2\n", "key1, key2"))
#    t = GenTrainer()
#    filename = "./data/topic_fine_tuning_data2.csv"
#    topic_data = pd.read_csv(filename)
#    t.setup_data(topic_data, filename)
#    t.train()
