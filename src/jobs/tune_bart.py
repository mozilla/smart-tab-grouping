import os
from datetime import datetime
import pandas as pd
import wandb

from tune_base import TuneTopicBase, INPUT_PROMPT_ID
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split

from transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
    TrainingArguments,
    Trainer
)

from util.storage import upload_directory
from utils import get_bad_word_ids


class TuneTopicBart(TuneTopicBase):
    def preprocess_function(self, examples):
        inputs = examples["input_text"]
        targets = examples["target_text"]

        model_inputs = self.tokenizer(
            inputs,
            max_length=512,
            truncation=True,
            padding="max_length"
        )
        labels = self.tokenizer(
            targets,
            max_length=512,
            truncation=True,
            padding="max_length"
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def setup_data(self, topic_data: pd.DataFrame, filename: str = "unknown"):
        self.filename = filename
        self.tokenizer = BartTokenizer.from_pretrained(self.model_name)
        self.model = BartForConditionalGeneration.from_pretrained(self.model_name)
        topic_data.input_keywords = topic_data.input_keywords.fillna("")
        topic_data[INPUT_PROMPT_ID] = topic_data.apply(
            lambda row: self.prompter.generate_prompt(row.input_titles, row.input_keywords),
            axis=1
        )

        topic_data_training, topic_data_eval = train_test_split(topic_data, test_size=0.2)

        # Prepare data dictionaries
        train_data_dict = {
            "input_text": topic_data_training[INPUT_PROMPT_ID].tolist(),
            "target_text": topic_data_training[self.label_column].tolist()
        }
        eval_data_dict = {
            "input_text": topic_data_eval[INPUT_PROMPT_ID].tolist(),
            "target_text": topic_data_eval[self.label_column].tolist()
        }

        self.train_dataset = Dataset.from_pandas(pd.DataFrame(train_data_dict))
        self.eval_dataset = Dataset.from_pandas(pd.DataFrame(eval_data_dict))

        print(f"Training Dataset size {len(self.train_dataset)}")
        print(f"Eval Dataset size {len(self.eval_dataset)}")

    def train(self):
        """
        Fine-tune the model using the HF Trainer API, log with W&B, and upload artifacts.
        """
        torch.cuda.empty_cache()

        # Ensure W&B logs the model at the end of training
        os.environ["WANDB_LOG_MODEL"] = "end"

        # Initialize W&B run
        wandb.init(
            project="tab_grouping",
            config={
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
                "model_name": self.model_name,
                "label_column": self.label_column,
                "use_keywords": self.use_keywords,
                "learning_rate_decay": self.learning_rate_decay,
                "single_tab_handling": self.single_tab_handling,
                "input_prompt_id": INPUT_PROMPT_ID,
                "filename": self.filename
            }
        )
        print(f"W&B Run ID: {wandb.run.id}")
        print(f"W&B Run Name: {wandb.run.name}")

        # Tokenize datasets
        tokenized_training_dataset = self.train_dataset.map(self.preprocess_function, batched=True)
        tokenized_eval_dataset = self.eval_dataset.map(self.preprocess_function, batched=True)

        # Set training arguments
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

        # Create Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_training_dataset,
            eval_dataset=tokenized_eval_dataset
        )

        # Train
        trainer.train()

        # Generate predictions on the evaluation set
        results_labels = []
        results_output = []
        for item in tokenized_eval_dataset:
            # Prepare input tensors
            input_ids = self.tokenizer(
                item["input_text"],
                return_tensors="pt"
            ).input_ids.to(self.model.device)

            label = item["target_text"]

            # Generate with BART
            outputs = self.model.generate(
                input_ids,
                max_length=30,
                num_return_sequences=1
            )
            # Decode
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            results_output.append(response)
            results_labels.append(label)

        # Compute metrics (custom method from your base class)
        self.compute_metrics_text(results_output, results_labels)

        # Log sample predictions in a W&B Table
        validation_table = wandb.Table(
            columns=["input", "label", "prediction"],
            data=list(
                zip(
                    [d["input_text"] for d in tokenized_eval_dataset],
                    results_labels,
                    results_output
                )
            ),
        )
        wandb.log({"Validation Data": validation_table})

        # Optionally, update generation_config for BART
        # e.g., if you need to exclude certain words:
        self.model.config.bad_words_ids = get_bad_word_ids()

        # Save final model locally
        output_dir = "./bart-finetuned-topic"
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        # Upload directory to storage
        current_date = datetime.now()
        date_string = current_date.isoformat().replace(":", "_")
        upload_directory(
            output_dir,
            "stage-fx-tab-grouping",
            f"topic/models/{date_string}/",
            depth=1
        )

        # Finish W&B
        wandb.finish()

        # Free up memory
        self.model = None
        torch.cuda.empty_cache()
