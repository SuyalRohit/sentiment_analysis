import re
import unicodedata
import spacy
import nltk
from nltk.corpus import stopwords
from typing import List

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
nlp = spacy.load('en_core_web_sm')

replacement_map = {
    '⁄': '/', 'ø': 'o', 'æ': 'ae', 'ß': 'ss', 'ð': 'd', 'þ': 'th',
}

def strip_html_and_punct(text: str) -> str:
    """Remove HTML tags and most punctuation."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"(?:<br\s*/?>|</?i>|[^\w\s'])", '', text)
    return text

def remove_accents(text: str) -> str:
    """Normalize accented characters to ASCII equivalents."""
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

def remove_stopwords(text: str) -> str:
    """Remove English stopwords."""
    return ' '.join([w for w in text.split() if w not in stop_words])

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

def batch_lemmatize(texts: List[str], stopword_list: set=stop_words) -> List[str]:
    """Batch lemmatize texts with spaCy and remove stopwords from each lemmatized text."""
    lemmatized_texts = []
    for doc in nlp.pipe(texts, batch_size=1000, disable=['parser', 'ner']):
        lemmas = ' '.join([token.lemma_ for token in doc])
        lemmas = remove_stopwords(lemmas, stopword_list)
        lemmatized_texts.append(lemmas)
    return lemmatized_texts