import re
import unicodedata
import spacy
from nltk.corpus import stopwords
from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer
from typing import Any

stop_words = set(stopwords.words('english'))
nlp = spacy.load('en_core_web_sm')

replacement_map = {
    '⁄': '/', 'ø': 'o', 'æ': 'ae', 'ß': 'ss', 'ð': 'd', 'þ': 'th',
}

def strip_html_and_punct(text: str) -> str:
    """    Remove HTML tags and punctuation from a text string."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"(?:<br\s*/?>|</?i>|[^\w\s'])", '', text)
    return text

def remove_accents(text: str) -> str:
    """Convert accented characters in text to unaccented counterparts."""
    return ''.join(
        c for c in unicodedata.normalize('NFKD', text)
        if not unicodedata.combining(c)
    )

def replace_special_chars(text: str) -> str:
    """Replace specific special characters with ASCII equivalents."""
    for old, new in replacement_map.items():
        text = text.replace(old, new)
    return text

def remove_non_ascii(text: str) -> str:
    """Remove all non-ASCII characters."""
    return re.sub(r'[^\x00-\x7F]', '', text)

def remove_stopwords(text: str, stopword_list: set = stop_words) -> str:
    """Remove English stopwords."""
    return ' '.join([w for w in text.split() if w not in stopword_list])

def lemmatize(text: str) -> str:
    """Lemmatize words using spaCy."""
    return ' '.join([token.lemma_ for token in nlp(text)])

def preprocess_text(text: str) -> str:
    """Apply all cleaning steps before lemmatization."""
    text = strip_html_and_punct(text)
    text = remove_accents(text)
    text = replace_special_chars(text)
    text = remove_non_ascii(text)
    return text

def batch_lemmatize(texts: list[str], stopword_list: set=stop_words) -> list[str]:
    """Batch lemmatize texts with spaCy and remove stopwords from each lemmatized text."""
    lemmatized_texts = []
    for doc in nlp.pipe(texts, batch_size=1000, disable=['parser', 'ner']):
        lemmas = ' '.join([token.lemma_ for token in doc])
        lemmas = remove_stopwords(lemmas, stopword_list)
        lemmatized_texts.append(lemmas)
    return lemmatized_texts

def clean_batch(batch):
    """
    Using clean_text function to batch-clean the dataset's 'review' column.
    """
    batch["review"] = [preprocess_text(x) for x in batch["review"]]
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