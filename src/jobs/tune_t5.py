import json
import os
from datetime import datetime
import pandas as pd
import wandb

from tune_base import TuneTopicBase, INPUT_PROMPT_ID, keyword_prompt
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    TrainingArguments, Trainer
)

from util.storage import upload_directory
from utils import get_bad_word_ids


PROCESSED_COLUMN = "processed_label_column"

class TuneTopicT5(TuneTopicBase):
    def preprocess_function(self, examples, teacher_model=None):
        inputs = examples["input_text"]
        model_inputs = self.tokenizer(
            inputs,
            max_length=512,
            truncation=True,
            padding="max_length"
        )
        if teacher_model is None:
            targets = examples["target_text"]
            with self.tokenizer.as_target_tokenizer():
                targets_tokenized = self.tokenizer(
                    targets,
                    max_length=64,
                    truncation=True,
                    padding="max_length"
                )
            labels = targets_tokenized["input_ids"]
        else:
            generated_ids = teacher_model.generate(
                self.tokenizer(inputs, return_tensors="pt", padding=True, truncation=True).input_ids.to(self.device),
                max_length=64
            ).to("cpu")
            decoded_outputs = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            print(f"Sample of decoded outputs from Unlabeled Dataset {decoded_outputs[:20]}")
            with self.tokenizer.as_target_tokenizer():
                tokenized_labels = self.tokenizer(
                    decoded_outputs,
                    max_length=64,
                    truncation=True,
                    padding="max_length"
                )
            labels = tokenized_labels["input_ids"]
        # not sure if this is needed.
    #        labels = [
#            [(token if token != self.tokenizer.pad_token_id else -100) for token in label_seq]
#            for label_seq in labels
#        ]
        model_inputs["labels"] = labels

        return model_inputs
    def prep_dataframe(self, data_df):
        data_df.input_keywords = data_df.input_keywords.fillna("")
        data_df[INPUT_PROMPT_ID] = data_df.apply(
            lambda row: self.prompter.generate_prompt(row.input_titles, row.input_keywords), axis=1)
        if self.label_prefix is not None:
            data_df[PROCESSED_COLUMN] = self.label_prefix + data_df[self.label_column]
        return data_df.reset_index(drop=True)

    def setup_data(self, topic_data: pd.DataFrame, validation: pd.DataFrame, filename: str = "unknown"):
        self.filename = filename
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
        topic_data = self.prep_dataframe(topic_data)
        validation = self.prep_dataframe(validation)
        if self.label_prefix is not None:
            self.label_column = PROCESSED_COLUMN
        topic_data_training, topic_data_eval = train_test_split(topic_data, test_size=0.2)
        train_data_dict = {"input_text": topic_data_training[INPUT_PROMPT_ID].to_list(),
                           "target_text": topic_data_training[self.label_column].to_list()}
        eval_data_dict = {"input_text": topic_data_eval[INPUT_PROMPT_ID].to_list(),
                          "target_text": topic_data_eval[self.label_column].to_list()}
        validation_data_dict = {"input_text": validation[INPUT_PROMPT_ID].to_list(),
                                "target_text": validation[self.label_column].to_list()}

        self.train_dataset = Dataset.from_pandas(pd.DataFrame(train_data_dict))
        self.eval_dataset = Dataset.from_pandas(pd.DataFrame(eval_data_dict))
        self.validation_dataset = Dataset.from_pandas(pd.DataFrame(validation_data_dict))

        print(f"Training Dataset size {len(self.train_dataset)}")
        print(f"Eval Dataset size {len(self.eval_dataset)}")
        print(f"Validation Dataset size {len(self.validation_dataset)}")

    def shrink_remove_layers(self, model, layer_name, layers_to_remove=None):
        """
        Note that new_size is supposed indicate the new size but it isn't exactly accurate
        This function needs to be updated
        """
        if layer_name == "encoder":
            config_name = "num_layers"
        else:
            config_name = f"num_{layer_name}_layers"
        current_size = getattr(model.config, config_name)
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
                           "brevity_weight": self.brevity_weight,
                           "use_keywords": self.use_keywords,
                           "learning_rate_decay": self.learning_rate_decay,
                           "single_tab_handling": self.single_tab_handling,
                           "shrink_encoder_index_remove": self.shrink_encoder_index_remove,
                           "shrink_decoder_index_remove": self.shrink_decoder_index_remove,
                           "shrink_remove_encoder_layers": self.shrink_remove_encoder_layers,
                           "shrink_remove_decoder_layers": self.shrink_remove_decoder_layers,
                           "shorten_training_label_boost": self.shorten_training_label_boost,
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
                num_train_epochs=2,
                weight_decay=0.01,
                save_total_limit=1,
                save_strategy="epoch",
                lr_scheduler_type="cosine",
                warmup_ratio=0.05
            )
        else:
            training_args = TrainingArguments(
                output_dir="./results",
                evaluation_strategy="epoch",
                learning_rate=self.learning_rate,
                per_device_train_batch_size=self.batch_size,
                per_device_eval_batch_size=1,
                num_train_epochs=2,
                weight_decay=0.01,
                save_total_limit=1,
                save_strategy="epoch",
                warmup_ratio=0.05)


        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_training_dataset,
            eval_dataset=tokenized_training_dataset
        )

        do_first_run = True # We were testing skipping the first run but got bad results

        tokenized_validation_dataset = self.validation_dataset.map(self.preprocess_function, batched=True)

        if do_first_run:
            trainer.train()
            if shrink_model:
                self.run_eval(tokenized_eval_dataset, name="Pre Shrink Eval", prefix="preshrink")
            else:
                self.run_eval(tokenized_eval_dataset)
            self.run_eval(tokenized_validation_dataset, name="Single Tab Validation", prefix="single_tab_val")

        has_second_train_run = False
        if shrink_model:
            has_second_train_run = True
            self.shrink_remove_layers(self.model, "encoder", self.shrink_remove_encoder_layers,
                                      self.shrink_encoder_index_remove)
            self.shrink_remove_layers(self.model, "decoder", self.shrink_remove_decoder_layers,
                                      self.shrink_decoder_index_remove)
        if has_second_train_run:
            training_args.num_train_epochs = 1 if do_first_run else 3
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=tokenized_training_dataset,
                eval_dataset=tokenized_training_dataset
            )
            trainer.train()
            name_prefix = "Post Remove"
            dataset_prefix = "post_remove"
            self.run_eval(tokenized_eval_dataset, name=f"{name_prefix} Eval", prefix=f"{dataset_prefix}_eval")
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

    def run_eval(self, eval_dataset, name="Eval", prefix=None, log_wandb=True):
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

        metrics = self.compute_metrics_text(results_output, results_labels, prefix=prefix)
        if (log_wandb):
            wandb.log(metrics)
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
        else:
            print(f"{name} stats {json.dumps(metrics)}")


if __name__ == '__main__':
    print(keyword_prompt.generate_prompt("Doc 1\nDoc 2\n", "key1, key2"))
#    t = GenTrainer()
#    filename = "./data/topic_fine_tuning_data2.csv"
#    topic_data = pd.read_csv(filename)
#    t.setup_data(topic_data, filename)
#    t.train()
