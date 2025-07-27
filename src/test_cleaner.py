import re
import unicodedata
import spacy
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
nlp = spacy.load('en_core_web_sm')

replacement_map = {
    '⁄': '/',
    'ø': 'o',
    'æ': 'ae',
    'ß': 'ss',
    'ð': 'd',
    'þ': 'th',
}

def strip_html_and_punct(text):
    text = text.lower()
    text = re.sub(r"(?:<br\s*/?>|</?i>|[^\w\s'])", '', text)
    return text

def remove_accents(text):
    return ''.join(
        c for c in unicodedata.normalize('NFKD', text)
        if not unicodedata.combining(c)
    )

def replace_special_chars(text):
    for old, new in replacement_map.items():
        text = text.replace(old, new)
    return text

def remove_non_ascii(text):
    return re.sub(r'[^\x00-\x7F]', '', text)

def remove_stopwords(text):
    return ' '.join([w for w in text.split() if w not in stop_words])

def lemmatize(text):
    return ' '.join([token.lemma_ for token in nlp(text)])

def clean_text_pipeline(text, do_lemmatize=False):
    text = strip_html_and_punct(text)
    text = remove_accents(text)
    text = replace_special_chars(text)
    text = remove_non_ascii(text)
    text = remove_stopwords(text)
    if do_lemmatize:
        text = lemmatize(text)
    return text