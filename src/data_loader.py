import logging
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load CSV into DataFrame and remove duplicates.
    """
    df = pd.read_csv(filepath)
    required = {"review", "sentiment"}
    missing = required - set(df.columns)
    if missing:
        logger.error(f"Missing Columns: {missing}")
        raise ValueError(f"Missing Columns: {missing}")
    
    # Drop duplicates and log how many were removed
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        before = len(df)
        df.drop_duplicates(inplace=True)
        after = len(df)
        duplicate_removed = before - after
        logger.info(f"Removed {duplicate_removed} duplicate records from data")
    else:
        logger.info("No duplicate records found.")
    
    logger.info(f"Loaded {len(df)} rows from {filepath}")    
    return df

def split_data(
    df: pd.DataFrame, test_size: float, random_state: int) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Split DataFrame into train/test sets.
    """
    X = df["review"]
    y = df["sentiment"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    return X_train, X_test, y_train, y_test