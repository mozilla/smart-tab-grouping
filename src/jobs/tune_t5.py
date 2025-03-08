import os
from datetime import datetime
import pandas as pd
import wandb

from tune_base import TuneTopicBase, INPUT_PROMPT_ID, keyword_prompt
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


class TuneTopicT5(TuneTopicBase):
    def preprocess_function(self, examples):
        inputs = examples["input_text"]
        targets = examples["target_text"]
        model_inputs = self.tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(targets, max_length=512, truncation=True, padding="max_length")
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def setup_data(self, topic_data: pd.DataFrame, validation_data: pd.DataFrame, filename: str = "unknown"):
        self.filename = filename
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)

        topic_data.input_keywords = topic_data.input_keywords.fillna("")
        topic_data[INPUT_PROMPT_ID] = topic_data.apply(
            lambda row: self.prompter.generate_prompt(row.input_titles, row.input_keywords), axis=1)

        validation_data.input_keywords = validation_data.input_keywords.fillna("")
        validation_data[INPUT_PROMPT_ID] = validation_data.apply(
            lambda row: self.prompter.generate_prompt(row.input_titles, row.input_keywords), axis=1)

        topic_data_training, topic_data_eval = train_test_split(topic_data, test_size=0.2)
        train_data_dict = {"input_text": topic_data_training[INPUT_PROMPT_ID].to_list(),
                           "target_text": topic_data_training[self.label_column].to_list()}
        eval_data_dict = {"input_text": topic_data_eval[INPUT_PROMPT_ID].to_list(),
                          "target_text": topic_data_eval[self.label_column].to_list()}
        validation_data_dict = {"input_text": validation_data[INPUT_PROMPT_ID].to_list(),
                                "target_text": validation_data[self.label_column].to_list()}

        self.train_dataset = Dataset.from_pandas(pd.DataFrame(train_data_dict))
        self.eval_dataset = Dataset.from_pandas(pd.DataFrame(eval_data_dict))
        self.validation_dataset = Dataset.from_pandas(pd.DataFrame(validation_data_dict))

        print(f"Training Dataset size {len(self.train_dataset)}")
        print(f"Eval Dataset size {len(self.eval_dataset)}")
        print(f"Validation Dataset size {len(self.validation_dataset)}")

    def shrink_remove_layers(self, model, layer_name, new_size=None, layers_to_remove=None):
        """
        Note that new_size is supposed indicate the new size but it isn't exactly accurate
        This function needs to be updated
        """
        if layer_name == "encoder":
            config_name = "num_layers"
        else:
            config_name = f"num_{layer_name}_layers"
        current_size = getattr(model.config, config_name)
        if layers_to_remove is None:
            if new_size is None:
                new_size = int(current_size / 2)
            if current_size != new_size:
                remove_list = [i for i in range(1, new_size * 2, 2)]
        else:
            remove_list = list(map(int, layers_to_remove.split(",")))
            remove_list.sort()
        for i in reversed(remove_list):
            print(f"removing layer {layer_name} {i}")
            del getattr(model, layer_name).block[i]
        new_layer_size = len(getattr(model, layer_name).block)
        setattr(model.config, config_name, new_layer_size)
        print(f"Changed number of layers from {layer_name} from {current_size} to {new_layer_size}")

    def train(self):
        torch.cuda.empty_cache()
        shrink_model = self.shrink_remove_encoder_layers > 0 or self.shrink_remove_decoder_layers > 0 or \
            self.shrink_decoder_index_remove or self.shrink_encoder_index_remove

        os.environ["WANDB_LOG_MODEL"] = "end"  # save the model to WandB
        wandb.init(project="tab_grouping",
                   config={"learning_rate": self.learning_rate, "batch_size": self.batch_size,
                           "model_name": self.model_name,
                           "label_column": self.label_column,
                           "use_keywords": self.use_keywords,
                           "learning_rate_decay": self.learning_rate_decay,
                           "single_tab_handling": self.single_tab_handling,
                           "shrink_encoder_index_remove": self.shrink_encoder_index_remove,
                           "shrink_decoder_index_remove": self.shrink_decoder_index_remove,
                           "shrink_remove_encoder_layers": self.shrink_remove_encoder_layers,
                           "shrink_remove_decoder_layers": self.shrink_remove_decoder_layers,
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
                num_train_epochs=2,  # consider moving back to 3
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
                num_train_epochs=2,  # consider moving back to 3
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
        if shrink_model:
            self.run_eval(tokenized_eval_dataset, name="Pre Shrink Eval", prefix="preshrink")
        else:
            self.run_eval(tokenized_eval_dataset)
        tokenized_validation_dataset = self.validation_dataset.map(self.preprocess_function, batched=True)
        self.run_eval(tokenized_validation_dataset, name="Single Tab Validation", prefix="single_tab_val")

        if shrink_model:
            self.shrink_remove_layers(self.model, "encoder", self.shrink_remove_encoder_layers, self.shrink_encoder_index_remove)
            self.shrink_remove_layers(self.model, "decoder", self.shrink_remove_decoder_layers, self.shrink_decoder_index_remove)

            training_args.num_train_epochs = 1
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=tokenized_training_dataset,
                eval_dataset=tokenized_training_dataset
            )
            trainer.train()

            self.run_eval(tokenized_eval_dataset, name="Post Remove Eval", prefix="post_remove_eval")
            tokenized_validation_dataset = self.validation_dataset.map(self.preprocess_function, batched=True)
            self.run_eval(tokenized_validation_dataset, name="Post Remove Single Tab Validation",
                          prefix="post_remove_single_tab_val")

        self.model.generation_config.update(bad_words_ids=get_bad_word_ids())
        local_save_name = "./t5-finetuned-topic"

        self.model.save_pretrained(local_save_name)
        self.tokenizer.save_pretrained(local_save_name)

        current_date = datetime.now()
        date_string = current_date.isoformat().replace(":", "_")
        upload_directory("./t5-finetuned-topic", "stage-fx-tab-grouping", f"topic/models/{date_string}/", depth=1)

        del self.model
        torch.cuda.empty_cache()

        self.tokenizer = T5Tokenizer.from_pretrained(local_save_name)
        self.model = T5ForConditionalGeneration.from_pretrained(local_save_name).to(self.device)
        self.run_eval(tokenized_validation_dataset, name="2-Post Remove Single Tab Validation",
                      prefix="2_post_remove_single_tab_val")

        wandb.finish()
        self.model = None
        torch.cuda.empty_cache()

    def run_eval(self, eval_dataset, name="Eval", prefix=None):
        results_labels = []
        results_output = []
        losses = []
        for item in eval_dataset:
            input = item["input_text"]
            label = item['target_text']

            input_encodings = self.tokenizer(input, return_tensors="pt", truncation=True,
                                             padding=True)
            label_encodings = self.tokenizer(label, return_tensors="pt", truncation=True,
                                             padding=True)

            input_ids = input_encodings["input_ids"].to(self.device)
            attention_mask = input_encodings["attention_mask"].to(self.device)
            label_ids = label_encodings["input_ids"].to(self.device)

            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=label_ids
                )
            loss = outputs.loss.item()
            outputs = self.model.generate(input_ids, max_length=30, num_return_sequences=1)
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            results_output.append(response)
            results_labels.append(label)
            losses.append(loss)

        self.compute_metrics_text(results_output, results_labels, prefix=prefix)
        table = wandb.Table(
            columns=["input", "label", "prediction", "loss"],
            data=list(
                zip(
                    [d["input_text"] for d in eval_dataset],
                    results_labels,
                    results_output,
                    losses
                )
            ),
        )
        wandb.log({f"{name} Set": table})


if __name__ == '__main__':
    print(keyword_prompt.generate_prompt("Doc 1\nDoc 2\n", "key1, key2"))
#    t = GenTrainer()
#    filename = "./data/topic_fine_tuning_data2.csv"
#    topic_data = pd.read_csv(filename)
#    t.setup_data(topic_data, filename)
#    t.train()
