import os
from datetime import datetime
import pandas as pd
import wandb

from tune_base import TuneTopicBase, INPUT_PROMPT_ID, keyword_prompt
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split

from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments

from util.storage import upload_directory
from utils import get_bad_word_ids

class TuneTopicGPT2(TuneTopicBase):
    def __init__(self, learning_rate: float = 1e-4, batch_size: int = 2, model_name: str = 'google/flan-t5-base',
                 label_column: str = "output", use_keywords: bool = True, single_tab_handling: bool = False,
                 learning_rate_decay: bool = True):
        super().__init__(learning_rate, batch_size, model_name, label_column, use_keywords, single_tab_handling,
                         learning_rate_decay)
        self.tokenizer = None

    def preprocess_function(self, examples):
        inputs = examples["input_text"]
        targets = examples["target_text"]
        # Concatenate input and target with a separator (e.g., "<|endoftext|>")
        combined_texts = [f"{inp} {self.tokenizer.eos_token} {tgt}" for inp, tgt in zip(inputs, targets)]
        model_inputs = self.tokenizer(combined_texts,
                                      max_length=512,
                                      truncation=True,
                                      padding="max_length",
                                      return_attention_mask=True)
        model_inputs["labels"] = model_inputs["input_ids"]
        return model_inputs

    def setup_data(self, topic_data: pd.DataFrame, filename: str = "unknown"):
        self.filename = filename

        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
        self.model = GPT2LMHeadModel.from_pretrained(self.model_name)

        # Ensure the tokenizer has a padding token (GPT-2 does not have one by default)
        self.tokenizer.pad_token = self.tokenizer.eos_token
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
            eval_dataset=tokenized_training_dataset,
            tokenizer=self.tokenizer
        )

        trainer.train()
        results_labels = []
        results_output = []

        for item in tokenized_eval_dataset:
            input_ids = self.tokenizer(item["input_text"], return_tensors="pt", max_length=512, return_attention_mask=True).input_ids.to("cuda:0")
            label = item['target_text']
            outputs = self.model.generate(input_ids, max_new_tokens=30, num_return_sequences=1)
            # Remove the input prompt to get only the generated text
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            prompt_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
            response = response[len(prompt_text):].strip()
            results_output.append(response)
            print(response)
            results_labels.append(label)
        self.compute_metrics_text(results_output, results_labels)
#        validation_table = wandb.Table(columns=["input", "label", "prediction"], data=
#                            [list(map(lambda a: a["input_text"], tokenized_eval_dataset)),
#                            results_labels,
#                            results_output])
#        wandb.log({"Validation Data": validation_table})
        self.model.generation_config.update(bad_words_ids=get_bad_word_ids())
        self.model.save_pretrained("./gpt2-finetuned-topic")
        self.tokenizer.save_pretrained("./gpt2-finetuned-topic")

        current_date = datetime.now()
        date_string = current_date.isoformat().replace(":", "_")
        upload_directory("./gpt2-finetuned-topic", "stage-fx-tab-grouping", f"topic/models/{date_string}/", depth=1)

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
