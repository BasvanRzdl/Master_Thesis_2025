import ssl
import nltk
# Uncomment the following line if you need to download the VADER lexicon for the first time
# nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run(test_path, output_path):
    """
    Run VADER sentiment analysis on a test set and save predictions.

    Args:
        test_path (str): Path to the input CSV file containing the test set.
        output_path (str): Path to save the output CSV file with VADER predictions.

    Returns:
        str: The path to the output file with predictions.
    """
    # Load the test set from CSV
    test_df = pd.read_csv(test_path)

    # Initialize the VADER sentiment analyzer
    vader = SentimentIntensityAnalyzer()

    def classify_vader(text):
        """
        Classify sentiment of a given text using VADER.

        Args:
            text (str): The input text to classify.

        Returns:
            str: 'positive', 'negative', or 'neutral' based on VADER compound score.
        """
        try:
            text = str(text)
            # Compute the compound sentiment score
            score = vader.polarity_scores(text)['compound']
            # Assign sentiment label based on compound score thresholds
            if score >= 0.05:
                return 'positive'
            elif score <= -0.05:
                return 'negative'
            else:
                return 'neutral'
        except Exception as e:
            # Print error and return 'neutral' if classification fails
            print("Error on text:", text)
            print(e)
            return 'neutral'

    # Apply VADER sentiment classification to the cleaned text column
    test_df['vader_pred'] = test_df['text_cleaned'].apply(classify_vader)

    # Save the DataFrame with VADER predictions to the specified output path
    test_df.to_csv(output_path, index=False)

    print("VADER results saved as test_preprocessed_VADER.csv")
    return output_path