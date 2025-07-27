import re, unicodedata
from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer
from typing import Any
replacement_map = {
    '⁄': '/', 'ø': 'o', 'æ': 'ae', 'ß': 'ss', 'ð': 'd', 'þ': 'th',
}

def clean_text(text: str) -> str:
    """
    Performs regex-based cleaning,
    Normalized accented characters,  
    Replace specific special characters with ASCII equivalents.
    """
    text = text.lower()
    text = re.sub(r"(?:<br\s*/?>|</?i>|[^\w\s'])", "", text)
    text = ''.join(c for c in unicodedata.normalize('NFKD', text) if not unicodedata.combining(c))
    for old, new in replacement_map.items():
        text = text.replace(old, new)
    return text

def clean_batch(batch):
    """
    Using clean_text function to batch-clean the dataset's 'review' column.
    """
    batch["review"] = [clean_text(x) for x in batch["review"]]
    return batch

def prepare_hf_dataset(df: pd.DataFrame, test_size: float=0.2, seed: int=42) -> Dataset:
    """
    Prepare Hugging Face Dataset by mapping sentiment to labels,
    splitting into train/test, and batch cleaning.
    """
    df["labels"] = df["sentiment"].map({"positive": 1, "negative": 0})
    dataset = Dataset.from_pandas(df[["review", "labels"]])
    dataset = dataset.train_test_split(test_size=test_size, seed=seed, shuffle=True)
    dataset = dataset.map(clean_batch, batched=True)
    return dataset

def get_tokenizer(model_name: str = "bert-base-uncased") -> AutoTokenizer:
    """
    Load and return a HuggingFace tokenizer given a pretrained model name.
    """
    return AutoTokenizer.from_pretrained(model_name)

def tokenize_dataset(dataset: Dataset, tokenizer: AutoTokenizer, max_length: int = 512) -> Dataset:
    """
    Tokenize the 'review' field of the HuggingFace Dataset using the provided tokenizer.
    """

    def tokenizer_function(batch: dict[str, list[str]]) -> dict[str, Any]:
        """
        Tokenizes a batch of examples.

        Args:
            batch (dict): Batch dict containing a list of text under 'review' key.

        Returns:
            dict: Dictionary containing tokenized inputs with keys like 'input_ids' and 'attention_mask'.
        """
        return tokenizer(
            batch["review"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_attention_mask=True,
            return_token_type_ids=True,
        )
    tokenized = dataset.map(tokenizer_function, batched=True, remove_columns=["review"])
    tokenized.set_format("torch")

    return tokenized
