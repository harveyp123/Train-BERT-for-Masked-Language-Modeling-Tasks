
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"



from datasets import load_dataset

# wiki dataset: https://huggingface.co/datasets/wikipedia
dataset = load_dataset("wikipedia", "20220301.simple")


dataset


from transformers import AutoModel, AutoTokenizer


sample_text = dataset["train"]['text'][39]
print(sample_text)


tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


tokens = tokenizer(sample_text).input_ids
print([tokenizer.decode(id) for id in tokens])


from transformers import BertConfig, BertForMaskedLM

config = BertConfig(
    hidden_size = 384,
    vocab_size= tokenizer.vocab_size,
    num_hidden_layers = 6,
    num_attention_heads = 6,
    intermediate_size = 1024,
    max_position_embeddings = 256
)

model = BertForMaskedLM(config=config)
print(model.num_parameters()) #10457864


from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)


import torch
from torch.utils.data import Dataset
from accelerate import Accelerator, DistributedType

class LineByLineTextDataset(Dataset):
    def __init__(self, tokenizer, raw_datasets, max_length: int):
        self.padding = "max_length"
        self.text_column_name = 'text'
        self.max_length = max_length
        self.accelerator = Accelerator(gradient_accumulation_steps=1)
        self.tokenizer = tokenizer

        with self.accelerator.main_process_first():
            self.tokenized_datasets = raw_datasets.map(
                self.tokenize_function,
                batched=True,
                num_proc=4,
                remove_columns=[self.text_column_name],
                desc="Running tokenizer on dataset line_by_line",
            )
            self.tokenized_datasets.set_format('torch',columns=['input_ids'],dtype=torch.long)

    def tokenize_function(self,examples):
        examples[self.text_column_name] = [
            line for line in examples[self.text_column_name] if len(line[0]) > 0 and not line[0].isspace()
        ]
        return self.tokenizer(
            examples[self.text_column_name],
            padding=self.padding,
            truncation=True,
            max_length=self.max_length,
            return_special_tokens_mask=True,
        )
    def __len__(self):
        return len(self.tokenized_datasets)

    def __getitem__(self, i):
        return self.tokenized_datasets[i]


tokenized_dataset_train = LineByLineTextDataset(
    tokenizer= tokenizer,
    raw_datasets = dataset,
    max_length=256, # adjust this based on your requrements
)


from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./model",
    overwrite_output_dir=True,
    push_to_hub=True,
    hub_model_id="Ransaka/sinhala-bert-yt",
    learning_rate=1e-5,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    max_steps=500,
    eval_steps=100,
    logging_steps=100,
    weight_decay=0.01,
    evaluation_strategy="steps",
    save_strategy="steps",
    report_to='none',
    hub_private_repo = True,
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset_train['train'],
    eval_dataset= tokenized_dataset_train['train'], # change to your actual evaluation dataset
    )


trainer.train()


results = trainer.evaluate()


import math

print(f">>> Perplexity: {math.exp(results['eval_loss']):.2f}")


