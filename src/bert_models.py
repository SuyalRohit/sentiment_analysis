import os
import numpy as np
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer
)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def get_tokenizer(model_name="google-bert/bert-base-uncased"):
    return AutoTokenizer.from_pretrained(model_name)

def tokenize_dataset(dataset, tokenizer):
    def tokenizer_function(batch):
        return tokenizer(batch["review"], padding="max_length", truncation=True, max_length=512)
    tokenized = dataset.map(tokenizer_function, batched=True, remove_columns=["review"])
    tokenized.set_format("torch")
    return tokenized

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions),
        "recall": recall_score(labels, predictions),
        "f1": f1_score(labels, predictions),
    }

def get_training_args(output_dir, run_name, num_train_epochs=2):
    os.makedirs(output_dir, exist_ok=True)
    return TrainingArguments(
        output_dir=output_dir,
        run_name=run_name,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=num_train_epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        disable_tqdm=False,
        fp16=True,
        logging_steps=100,
        report_to="none"
    )

def get_trainer(model, args, train_dataset, eval_dataset):
    return Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics
    )