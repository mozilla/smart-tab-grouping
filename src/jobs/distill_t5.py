import os
from datetime import datetime
from functools import partial

import numpy as np
import pandas as pd
import wandb
from torch.utils.data import DataLoader
from tune_t5 import TuneTopicT5
from tune_base import TuneTopicBase, INPUT_PROMPT_ID, keyword_prompt
import torch
from datasets import concatenate_datasets, Dataset
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from torch.nn.functional import log_softmax, softmax

from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    TrainingArguments, Trainer
)

from util.storage import upload_directory
from utils import get_bad_word_ids
from tqdm import tqdm


class DistillTopicT5(TuneTopicT5):

    def calculate_loss(self, student_outputs, teacher_outputs, labels, alpha=0.9, temperature=2.0):
        """
        Computes the blended distillation + CE loss for seq2seq models.

        Args:
            student_outputs: ModelOutput from the student (includes logits)
            teacher_outputs: ModelOutput from the teacher (includes logits)
            labels: Ground-truth target token IDs (shape [batch, tgt_len])
            alpha: Weight for distillation loss (vs. CE loss)
            temperature: Softmax temperature

        Returns:
            total_loss: Weighted sum of distillation and CE losses
        """

        student_logits = student_outputs.logits  # [batch, seq_len, vocab_size]
        teacher_logits = teacher_outputs.logits

        ce_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            labels.view(-1),
            ignore_index=self.tokenizer.pad_token_id,
            reduction='mean'
        )

        student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)

        # Compute KL loss (batchmean over all tokens)
        kl_loss = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction="batchmean"
        ) * (temperature ** 2)  # scale loss back due to temperature

        total_loss = alpha * kl_loss + (1.0 - alpha) * ce_loss

        return total_loss

    def setup_data(self, topic_data: pd.DataFrame, validation: pd.DataFrame, unlabeled=None,
                   filename: str = "unknown"):
        if unlabeled is not None:
            unlabeled_data = self.prep_dataframe(unlabeled)
            unlabeled_data_dict = {"input_text": unlabeled_data[INPUT_PROMPT_ID].to_list()}
            self.unlabeled_dataset = Dataset.from_pandas(pd.DataFrame(unlabeled_data_dict))
            print(f"Unlabeled Dataset size {len(self.unlabeled_dataset)}")
        else:
            self.unlabeled_dataset = None
        super().setup_data(topic_data, validation, filename=filename)

    def remove_layers(self):
        if self.shrink_encoder_index_remove or self.shrink_decoder_index_remove:
            self.shrink_remove_layers(self.model, "encoder",
                                      self.shrink_encoder_index_remove)
            self.shrink_remove_layers(self.model, "decoder",
                                      self.shrink_decoder_index_remove)

    def train(self):
        torch.cuda.empty_cache()

        if self.teacher_model_artifact is None:
            raise Exception("Teacher model is missing")

        os.environ["WANDB_LOG_MODEL"] = "end"  # save the model to WandB
        run = wandb.init(project="tab_grouping_distillation",
                         config={"learning_rate": self.learning_rate, "batch_size": self.batch_size,
                                 "model_name": self.model_name,
                                 "teacher_model_artifact": self.teacher_model_artifact,
                                 "label_column": self.label_column,
                                 "use_keywords": self.use_keywords,
                                 "learning_rate_decay": self.learning_rate_decay,
                                 "single_tab_handling": self.single_tab_handling,
                                 "input_prompt_id": INPUT_PROMPT_ID, "filename": self.filename})
        print(f"W&B Run ID: {wandb.run.id}")
        print(f"W&B Run Name: {wandb.run.name}")

        artifact = run.use_artifact(self.teacher_model_artifact, type='model')
        artifact_dir = artifact.download()
        teacher_model = T5ForConditionalGeneration.from_pretrained(artifact_dir)
        teacher_model.resize_token_embeddings(len(self.tokenizer))
        teacher_model.to(self.device)

        # we generate targets here
        if self.unlabeled_dataset is not None:
            tokenized_unlabeled_dataset = self.unlabeled_dataset.map(partial(self.preprocess_function, teacher_model=teacher_model), batched=True, batch_size=128)
        tokenized_training_dataset = self.train_dataset.map(self.preprocess_function, batched=True)
        if self.unlabeled_dataset is not None:
            # combine
            tokenized_training_dataset = concatenate_datasets([tokenized_unlabeled_dataset, tokenized_training_dataset])
        tokenized_eval_dataset = self.eval_dataset.map(self.preprocess_function, batched=True)

        def add_decoder_ids(item_input_dict):
            decoder_input_ids = teacher_model._shift_right(torch.tensor(item_input_dict["labels"]))
            item_input_dict["decoder_input_ids"] = decoder_input_ids
            return item_input_dict

        tokenized_training_dataset = tokenized_training_dataset.map(add_decoder_ids, batched=True)
        tokenized_training_dataset.set_format(
            type="torch", columns=["input_ids", "attention_mask", "labels", "decoder_input_ids"]
        )

        num_epochs = 40

        self.model.generation_config.update(bad_words_ids=None)

        tokenized_validation_dataset = self.validation_dataset.map(self.preprocess_function, batched=True)

        train_loader = DataLoader(
            tokenized_training_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        self.model.to(self.device)

        # clamp embeddings to what the tokenizer actually supports
        self.model.resize_token_embeddings(len(self.tokenizer))
        # Training Loop
        for epoch in range(num_epochs):
            self.model.train()
            print(f"Epoch {epoch}")

            if epoch == 8:
                # Modify the model
                self.remove_layers()
                # Re-create the optimizer with the new model parameters
                optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)

            batch_index = 0
            running_loss = 0.0
            for batch in train_loader:
                optimizer.zero_grad()

                batch = dict([(k, v.to(self.device)) for k, v in batch.items()])

                # Forward pass through the teacher model
                with torch.no_grad():
                    teacher_outputs = teacher_model(**batch)

                # Forward pass through the student model
                student_outputs = self.model(**batch)
                # assert student_outputs.logits.size() == teacher_outputs.logits.size()
                loss = self.calculate_loss(student_outputs, teacher_outputs, batch["labels"])
                # Backpropagation
                loss.backward()
                optimizer.step()

                batch_index += 1
                loss_val = loss.item()
                running_loss += loss_val
                if batch_index % 50 == 0:
                    wandb.log({"cur_loss": loss_val})
#                    progress_bar.set_postfix(loss_value=loss_val)
            avg_loss = running_loss / batch_index
            wandb.log({"train_loss": avg_loss})

            print(f"Average Loss at epoch {epoch}:{avg_loss}")
           # self.run_eval(tokenized_eval_dataset, log_wandb=False)
           # self.run_eval(tokenized_validation_dataset, name="Single Tab Validation", prefix="single_tab_val",
           #               log_wandb=False)

        print(f"**** DISTILLATION COMPLETE")
        self.run_eval(tokenized_eval_dataset)
        self.run_eval(tokenized_validation_dataset, name="Single Tab Validation", prefix="single_tab_val")

        self.model.generation_config.update(bad_words_ids=get_bad_word_ids())
        local_save_name = "./t5-distilled-topic"

        self.model.save_pretrained(local_save_name)
        self.tokenizer.save_pretrained(local_save_name)

        current_date = datetime.now()
        date_string = current_date.isoformat().replace(":", "_")
        upload_directory("./t5-distilled-topic", "stage-fx-tab-grouping", f"topic/models/{date_string}/", depth=1)

        del self.model
        torch.cuda.empty_cache()

        self.tokenizer = T5Tokenizer.from_pretrained(local_save_name)
        self.model = T5ForConditionalGeneration.from_pretrained(local_save_name).to(self.device)
        self.run_eval(tokenized_validation_dataset, name="Recheck-Post Remove Single Tab Validation",
                      prefix="recheck_post_remove_single_tab_val")

        wandb.finish()
        self.model = None
        torch.cuda.empty_cache()
