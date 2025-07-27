import re, unicodedata
from datasets import Dataset

replacement_map = {
    '⁄': '/', 'ø': 'o', 'æ': 'ae', 'ß': 'ss', 'ð': 'd', 'þ': 'th',
}

def clean_text(text):
    text = text.lower()
    text = re.sub(r"(?:<br\s*/?>|</?i>|[^\w\s'])", "", text)
    text = ''.join(c for c in unicodedata.normalize('NFKD', text) if not unicodedata.combining(c))
    for old, new in replacement_map.items():
        text = text.replace(old, new)
    return text

def clean_batch(batch):
    batch["review"] = [clean_text(x) for x in batch["review"]]
    return batch

def prepare_hf_dataset(df, test_size=0.2, seed=42):
    df["labels"] = df["sentiment"].map({"positive": 1, "negative": 0})
    dataset = Dataset.from_pandas(df[["review", "labels"]])
    dataset = dataset.train_test_split(test_size=test_size, seed=seed)
    dataset = dataset.map(clean_batch, batched=True)
    return dataset