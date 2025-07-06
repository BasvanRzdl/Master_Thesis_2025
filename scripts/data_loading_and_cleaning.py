import pandas as pd
import re
import numpy as np
import spacy
from sklearn.model_selection import train_test_split
import os

def run(raw_data_paths, output_dir, val_size, seed):
    """
    Loads, cleans, harmonizes, and splits two climate sentiment datasets.

    Args:
        raw_data_paths (list): List containing paths to Dataset A and Dataset B.
        output_dir (str): Directory to save processed files.
        val_size (int): Number of samples for the validation set.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: Paths to the train, test, and validation CSV files.
    """
    # Unpack dataset paths
    dataset_a_path, dataset_b_path = raw_data_paths

    # === Load and preprocess Dataset A (Guzman2020 - sentiment via TextBlob) ===
    df_a = pd.read_csv(dataset_a_path)
    # Keep only relevant columns
    df_a = df_a[['text', 'polarity']]
    # Rename 'polarity' to 'raw_sentiment' for harmonization
    df_a = df_a.rename(columns={'polarity': 'raw_sentiment'})
    # Add source identifier
    df_a['source'] = 'Dataset A'

    # === Load and preprocess Dataset B (Qian2019 - 5-class manual sentiment labels) ===
    df_b = pd.read_csv(dataset_b_path)
    # Keep only relevant columns
    df_b = df_b[['message', 'sentiment']]
    # Rename columns for harmonization
    df_b = df_b.rename(columns={'message': 'text', 'sentiment': 'raw_sentiment'})
    # Add source identifier
    df_b['source'] = 'Dataset B'

    # === Harmonize Dataset A: Convert TextBlob polarity to 3-class sentiment ===
    def map_sentiment_a(p):
        if p > 0:
            return 'positive'
        elif p < 0:
            return 'negative'
        else:
            return 'neutral'
    df_a['sentiment'] = df_a['raw_sentiment'].apply(map_sentiment_a)

    # === Harmonize Dataset B: Remove 'news' class and map to 3-class sentiment ===
    # Remove rows labeled as 'news' (class 2)
    df_b = df_b[df_b['raw_sentiment'] != 2]

    def map_sentiment_b(score):
        if score == 1:
            return 'positive'
        elif score == -1:
            return 'negative'
        else:
            return 'neutral'
    df_b['sentiment'] = df_b['raw_sentiment'].apply(map_sentiment_b)

    # === Keep only cleaned columns for both datasets ===
    df_a_clean = df_a[['text', 'sentiment', 'source']]
    df_b_clean = df_b[['text', 'sentiment', 'source']]

    # === Combine both datasets into a single DataFrame ===
    df_merged = pd.concat([df_a_clean, df_b_clean], ignore_index=True)

    # === Remove duplicates, missing values, and retweets ===
    # Drop duplicate tweets based on text
    df_merged.drop_duplicates(subset='text', inplace=True)
    # Drop rows with missing text or sentiment
    df_merged.dropna(subset=['text', 'sentiment'], inplace=True)
    # Remove retweets (tweets starting with "rt ")
    df_merged = df_merged[~df_merged['text'].str.contains(r"^rt\s", case=False, na=False)]

    # === Save merged dataset for reference ===
    merged_path = os.path.join(output_dir, "merged_climate_sentiment_dataset.csv")
    df_merged.to_csv(merged_path, index=False)

    # === Split the merged dataset into train and test sets (stratified by sentiment) ===
    train_df, test_df = train_test_split(
        df_merged,
        test_size=0.2,
        stratify=df_merged['sentiment'],
        random_state=seed
    )

    # === Further split test set into validation and test sets (stratified by sentiment) ===
    val_df, test_df = train_test_split(
        test_df,
        train_size=val_size,
        stratify=test_df["sentiment"],
        random_state=seed
    )

    # === Load spaCy English model for lemmatization ===
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

    # === Define text cleaning function ===
    def clean_text(text):
        """
        Cleans tweet text by lowercasing, removing URLs, mentions, special characters, numbers, and extra spaces.
        """
        text = text.lower()
        text = re.sub(r"\$q\$", "'", text)
        text = re.sub(r"http\S+|www\S+", "", text, flags=re.IGNORECASE)
        text = re.sub(r"@\w+", "@user", text)
        text = re.sub(r"&\w+;", "", text)
        text = re.sub(r"[^\w\s#]", "", text)
        text = re.sub(r"\d+", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    # === Define lemmatization function ===
    def lemmatise(text):
        """
        Lemmatizes cleaned text using spaCy, keeping only alphabetic, non-stopword tokens.
        """
        doc = nlp(text)
        return " ".join([token.lemma_ for token in doc if token.is_alpha and not token.is_stop])

    # === Apply cleaning and lemmatization to all splits ===
    for df in [train_df, test_df, val_df]:
        # Clean text
        df['text_cleaned'] = df['text'].astype(str).apply(clean_text)
        # Lemmatize cleaned text
        df['text_lemmatised'] = df['text_cleaned'].apply(lemmatise)

    # === Save processed splits to CSV files ===
    train_path = os.path.join(output_dir, "train_preprocessed.csv")
    test_path = os.path.join(output_dir, "test_preprocessed.csv")
    val_path = os.path.join(output_dir, "val_preprocessed.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    val_df.to_csv(val_path, index=False)

    # === Print summary of saved files ===
    print("Merged dataset saved as merged_climate_sentiment_dataset.csv")
    print("Train set saved as train_preprocessed.csv")
    print("Test set saved as test_preprocessed.csv")
    print("Validation set saved as val_preprocessed.csv")

    # === Return paths to processed splits ===
    return train_path, test_path, val_path