import os
import argparse
import logging
import sys
import yaml
import random
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from transformers import AutoModelForSequenceClassification

from src.data_loader import load_data, split_data
from src.eda import run_eda
from src.preprocessing import preprocess_text, batch_lemmatize, prepare_hf_dataset, get_tokenizer, tokenize_dataset
from src.traditional_models import train_models
from src.evaluation import confusion_matrix_, classification_report_, metrics_summary
from src.bert_models import get_training_args, get_trainer, compute_metrics


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for configuration file and output directory.
    """
    parser = argparse.ArgumentParser(description="Sentiment Analysis Pipeline")
    parser.add_argument(
        "--config", "-c", type=str, default="config.yaml",
        help="Path to YAML configuration file"
    )
    return parser.parse_args()


def main() -> None:
    """
    Execute the entire sentiment-analysis pipeline from data loading to evaluation.
    """
    args = parse_args()

    # Load configuration
    try:
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Configuration file '{args.config}' not found.", file=sys.stderr)
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing config file: {e}", file=sys.stderr)
        sys.exit(1)

    logging_config = cfg.get("logging", None)
    if logging_config:
        logging.config.dictConfig(logging_config)
    else:
        # fallback to simple logging if config missing
        logging.basicConfig(level=logging.INFO)
    
    logger = logging.getLogger(__name__)

    logger.info("Starting sentiment analysis pipeline")

    # Data Loading
    try:
        df = load_data(cfg["data"]["input_path"])
        logger.info(f"Loaded data has {len(df)} entries.")
    except Exception as e:
        logger.error(f"Data loading failed: {e}")
        sys.exit(1)
        
    df_traditional = df.copy()
    df_bert = df.copy()
    
    # EDA before Cleaning
    try:
        run_eda(df_traditional, "before data cleaning", logger)
    except Exception as e:
        logger.error(f"EDA before cleaning failed: {e}")
        sys.exit(1)
    
    # Text Cleaning
    try:
        logger.info("Starting text cleaning...")

        # Apply all cleaning steps (excluding lemmatization)
        df_traditional["clean_text"] = df_traditional["review"].astype(str).apply(preprocess_text)

        # Batch lemmatization with stopword removal
        df_traditional["clean_text"] = batch_lemmatize(df_traditional["clean_text"].tolist())

        logger.info("Text cleaning complete. Example clean text:")
        logger.info(df["clean_text"].head(3).to_list())
    except Exception as e:
        logger.error(f"Text cleaning failed: {e}")
        sys.exit(1)
    
    # EDA After Cleaning
    try:
        run_eda(df_traditional, "after data cleaning", logger)
    except Exception as e:
        logger.error(f"EDA after cleaning failed: {e}")
        sys.exit(1)
    
    # Splitting
    try:
        X_train, X_test, y_train, y_test = split_data(
            df_traditional,
            test_size=cfg["data"].get("test_size", 0.2),
            random_state=cfg.get("seed", 42)
        )
    except Exception as e:
        logger.error(f"Data splitting failed: {e}")
        sys.exit(1)
        
    # Label Encoding
    try:
        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(y_train)
        y_test = label_encoder.transform(y_test)
        logger.info(f"Labels encoded. Classes: {list(label_encoder.classes_)}")
    except Exception as e:
        logger.error(f"Label encoding failed: {e}")
        sys.exit(1)

    # Train Traditional ML Model
    try:
        models_to_train = cfg["traditional"]["models"]
        cv_folds = cfg["traditional"]["cv_folds"]
        logger.info(f"Training traditional models: {models_to_train} with {cv_folds}-fold CV")
        trained_models = train_models(X_train, y_train, models=models_to_train, cv=cv_folds, cfg_traditional=cfg["traditional"])
        logger.info("Traditional models training complete.")
    except Exception as e:
        logger.error(f"Traditional model training failed: {e}")
        sys.exit(1)
    
    # Evaluation of Traditional Model
    try:
        logger.info("Starting evaluation of traditional models...")
        for model_name, model in trained_models.items():
            y_pred = model.predict(X_test)
            
            out_dir = cfg["evaluation"].get("output_dir", "outputs/evaluation")
            os.makedirs(out_dir, exist_ok=True)
            file_prefix = os.path.join(out_dir, model_name)
            
            # For possible label mapping (classes could be numbers or strings)
            if hasattr(model, "classes_"):
                labels = list(model.classes_)
            else:
                labels = list(set(y_test))
    
            # Confusion Matrix
            confusion_matrix_(
                y_test, 
                y_pred, 
                labels=labels, 
                path=f"{file_prefix}_confusion_matrix.png"
            )
            # Classification Report
            classification_report_(
                y_test, 
                y_pred, 
                path=f"{file_prefix}_classification_report.json", 
                target_names=[str(lbl) for lbl in labels]
            )
            # Metrics Summary
            metrics_summary(
                y_test, 
                y_pred, 
                path=f"{file_prefix}_metrics_summary.json"
            )
            logger.info(f"Evaluation complete for {model_name}")
    
        logger.info("All traditional models evaluated successfully.")
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)
    
    # Prepareing Hugging Face Dataset
    try:
        hf_dataset = prepare_hf_dataset(
            df_bert,
            test_size=cfg["data"]["test_size"],
            seed=cfg["seed"]
        )
        logger.info("Prepared and cleaned Hugging Face Dataset.")
    except Exception as e:
        logger.error(f"Preparing HuggingFace dataset failed: {e}")
        sys.exit(1)
    
    # Initialize Tokenier
    try:
        tokenizer = get_tokenizer(cfg["bert"]["tokenizer"]["model_name"])
        logger.info(f"Loaded tokenizer: {cfg['bert']['tokenizer']['model_name']}")
    except Exception as e:
        logger.error(f"Tokenizer loading failed: {e}")
        sys.exit(1)
    
    # Tokenize Dataset
    try:
        max_length = cfg["bert"]["tokenizer"].get("max_length", 128)
        tokenized_dataset = tokenize_dataset(hf_dataset, tokenizer, max_length=max_length)
        logger.info("Tokenized Hugging Face Dataset for BERT.")
        # Ready for Dataloader/DataCollator and model training
    except Exception as e:
        logger.error(f"Tokenization failed: {e}")
        sys.exit(1)
    
    # BERT Model Fine Tuning
    try:
        bert_model_name = cfg["bert"]["tokenizer"]["model_name"]
        model = AutoModelForSequenceClassification.from_pretrained(bert_model_name, num_labels=2)
        logger.info(f"Loaded BERT model: {bert_model_name}")
    
        # TrainingArguments
        bert_cfg = cfg["bert"]
        tp = bert_cfg["training_params"]
        
        training_args = get_training_args(
            output_dir=bert_cfg["output_dir"],
            run_name="bert-finetune",
            num_train_epochs=tp["epochs"],
            learning_rate=tp["learning_rate"],
            per_device_train_batch_size=tp["per_device_train_batch_size"],
            per_device_eval_batch_size=tp["per_device_eval_batch_size"],
            weight_decay=tp["weight_decay"],
            fp16=tp["fp16"],
        )
    
        # Trainer
        trainer = get_trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["test"],
            compute_metrics=compute_metrics
        )
    
        # Train
        logger.info("Starting BERT fine-tuning...")
        trainer.train()
        logger.info("BERT training complete. Best model saved to output_dir.")
    except Exception as e:
        logger.error(f"BERT model training failed: {e}")
        sys.exit(1)
    
    # BERT Evaluation
    try:
        logger.info("Evaluating BERT model...")
        output_dir = cfg["evaluation"].get("output_dir", "outputs/evaluation")
        os.makedirs(output_dir, exist_ok=True)
    
        # 1. Predict on the test set
        eval_output = trainer.predict(tokenized_dataset["test"])
        y_pred = np.argmax(eval_output.predictions, axis=1)
        y_true = eval_output.label_ids

        target_names = cfg["label_mapping"]["target_names"]
        labels = cfg["label_mapping"]["labels"]

    
        # 2. Confusion Matrix
        confusion_matrix_(
            y_true,
            y_pred,
            labels=labels,
            path=os.path.join(output_dir, "bert_confusion_matrix.png")
        )
    
        # 3. Classification Report
        classification_report_(
            y_true,
            y_pred,
            path=os.path.join(output_dir, "bert_classification_report.json"),
            target_names=target_names
        )
    
        # 4. Metrics Summary
        metrics_summary(
            y_true,
            y_pred,
            path=os.path.join(output_dir, "bert_metrics_summary.json")
        )
        logger.info("BERT evaluation complete. Metrics, report, and confusion matrix saved.")
    
    except Exception as e:
        logger.error(f"BERT evaluation failed: {e}")
        sys.exit(1)
        
if __name__ == "__main__":
    main()