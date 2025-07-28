import logging
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

def basic_dim(df):
    """Log shape, column names, and datatypes."""
    logger.info(f"Shape: {df.shape}")
    logger.info(f"Columns: {df.columns.tolist()}")
    logger.info(f"Datatypes:\n{df.dtypes}")

def check_missing(df):
    """Log total missing values."""
    total_missing = df.isnull().sum().sum()
    logger.info(f'Total missing values: {total_missing}')

def check_duplicates(df):
    """Log number of duplicate rows."""
    num_dupes = df.duplicated().sum()
    logger.info(f'Duplicate rows: {num_dupes}')


def plot_basic_eda(df):
    """
    Plots basic EDA visualizations for review data.
    Shows distribution of review lengths and sentiment classes.
    """
    df_new = df.copy()
    df_new['words_per_review'] = df_new['review'].astype(str).str.split().apply(len)

    # Distribution
    sns.histplot(df_new['words_per_review'], bins=50, kde=True)
    plt.title("Distribution of Words Per Review")
    plt.xlabel("Words per Review")
    plt.tight_layout()
    plt.show()

    # Boxplot by sentiment
    sns.boxplot(x='sentiment', y='words_per_review', data=df_new)
    plt.title("Review Length vs Sentiment")
    plt.tight_layout()
    plt.show()

    # Sentiment counts
    df_new['sentiment'].value_counts().plot.barh()
    plt.title("Sentiment Count")
    plt.tight_layout()
    plt.show()
    
    del(df_new)
    
def run_eda(df, stage: str, logger, plot=True):
    logger.info(f"Starting EDA {stage}")
    basic_dim(df)
    check_missing(df)
    check_duplicates(df)
    if plot:
        plot_basic_eda(df)
    logger.info(f"EDA {stage} complete.")