from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    DataCollatorForLanguageModeling,
)
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
from constants import INPUT_DIR, FINETUNED_DIR

# Load the dataset
dataset = load_dataset("text", data_files={"train": str(INPUT_DIR)})
train_dataset = dataset["train"]

# Split the dataset into training and validation sets
train_dataset, val_dataset = train_dataset.train_test_split(test_size=0.1).values()

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({"pad_token": "[PAD]"})


def tokenize_function(examples):
    return tokenizer(examples["text"], padding=True, return_special_tokens_mask=True)


tokenized_train_dataset = train_dataset.map(
    tokenize_function, batched=True, remove_columns=["text"]
)
tokenized_val_dataset = val_dataset.map(
    tokenize_function, batched=True, remove_columns=["text"]
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

model = GPT2LMHeadModel.from_pretrained("gpt2")
model.resize_token_embeddings(len(tokenizer))

# Set up training arguments
training_args = TrainingArguments(
    output_dir=FINETUNED_DIR,
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=100,
    learning_rate=5e-5,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    data_collator=data_collator,
)

trainer.train()
trainer.save_model(FINETUNED_DIR / "finetuned")
