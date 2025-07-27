import logging
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from typing import List, Tuple, Dict, Any
import pandas as pd

logger = logging.getLogger(__name__)

def model_factory(name: str) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Return an sklearn estimator or pipeline by name.
    """
    if name == "logreg":
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer()),
            ("clf", LogisticRegression(solver="liblinear"))
        ])
        params = {
            "tfidf__max_df": [0.75, 1.0],
            "clf__C": [0.1, 1.0]
        }
    
    elif name == "lsvc":
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer()),
            ("clf", LinearSVC())
        ])
        params = {
            "tfidf__max_df": [0.75, 1.0],
            "clf__alpha": [0.5, 1.0]
        }    
    
    elif name == "nb":
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer()),
            ("clf", MultinomialNB())
        ])
        params = {
            "tfidf__max_df": [0.75, 1.0],
            "clf__alpha": [0.5, 1.0]
        }
    
    else:
        raise ValueError(f"Unknown model: {name}")
    return pipeline, params

def train_models(X_train: pd.Series, y_train: pd.Series, models: List[str], cv: int = 5) -> dict:
    """
    Train and tune each model, returning fitted GridSearchCV objects.
    """
    trained = {}
    for name in models:
        pipe, params = model_factory(name)
        gs = GridSearchCV(pipe, param_grid=params, cv=cv, n_jobs=-1)
        gs.fit(X_train, y_train)
        trained[name] = gs
        logger.info(f"{name} best score: {gs.best_score_:.3f}")
    return trained
