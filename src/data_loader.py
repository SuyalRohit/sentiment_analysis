import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    print(f"Initial shape: {df.shape}")

    # Drop duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        df.drop_duplicates(inplace=True)
        print(f"Removed {duplicates} duplicate rows.")

    return df

def plot_basic_eda(df):
    df['words_per_review'] = df['review'].str.split().apply(len)

    # Distribution
    sns.histplot(df['words_per_review'], bins=50, kde=True)
    plt.title("Distribution of Words Per Review")
    plt.show()

    # Boxplot by sentiment
    sns.boxplot(x='sentiment', y='words_per_review', data=df)
    plt.title("Review Length vs Sentiment")
    plt.show()

    # Sentiment counts
    df['sentiment'].value_counts().plot.barh()
    plt.title("Sentiment Count")
    plt.show()

    df.drop('words_per_review', axis=1, inplace=True)