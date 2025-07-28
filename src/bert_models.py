import os
import numpy as np
from transformers import TrainingArguments, Trainer, PreTrainedModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datasets import Dataset
from typing import Optional, Callable


def get_training_args(output_dir: str, run_name: str, num_train_epochs: int, learning_rate: float,
    per_device_train_batch_size: int, per_device_eval_batch_size: int, weight_decay: float, fp16: bool,) -> TrainingArguments:
    """
    Create Hugging Face TrainingArguments with specified training configuration.
    """
    os.makedirs(output_dir, exist_ok=True)
    return TrainingArguments(
        output_dir=output_dir,
        run_name=run_name,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        disable_tqdm=False,
        fp16=fp16,
        logging_steps=100,
        report_to="none"
    )

def get_trainer(model: PreTrainedModel, args: TrainingArguments, train_dataset: Dataset, eval_dataset: Dataset, compute_metrics: Optional[Callable] = None) -> Trainer:
    """
    Instantiate and return a Hugging Face Trainer for model training and evaluation.
    """
    return Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics
    )

def compute_metrics(eval_pred: tuple) -> dict:
    """
    Compute evaluation metrics (accuracy, precision, recall, f1) given predictions.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions),
        "recall": recall_score(labels, predictions),
        "f1": f1_score(labels, predictions),
    }