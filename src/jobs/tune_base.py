import os
from pydantic import BaseModel

import wandb
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

class TuneTopicBase:

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


if __name__ == '__main__':
    print(keyword_prompt.generate_prompt("Doc 1\nDoc 2\n", "key1, key2"))
#    t = GenTrainer()
#    filename = "./data/topic_fine_tuning_data2.csv"
#    topic_data = pd.read_csv(filename)
#    t.setup_data(topic_data, filename)
#    t.train()
