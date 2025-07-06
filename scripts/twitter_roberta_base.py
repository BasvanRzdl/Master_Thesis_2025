import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run(input_path, output_path):
    """
    Run sentiment classification on a test set using the pre-trained Twitter-RoBERTa-base model.

    Args:
        input_path (str): Path to the input CSV file containing the test set.
        output_path (str): Path to save the output CSV file with predictions.

    Returns:
        str: The path to the output file with predictions.
    """
    # Load the test set from CSV
    test_df = pd.read_csv(input_path)

    # Specify the pre-trained model name
    model_name = "cardiffnlp/twitter-roberta-base-sentiment"

    # Load the tokenizer and model from HuggingFace
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # Create a sentiment analysis pipeline using the loaded model and tokenizer
    classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

    # Prepare the texts for classification (ensure string type and handle NaNs)
    texts = test_df['text_cleaned'].astype(str).fillna('').tolist()

    # Run the sentiment classifier on the texts
    preds = classifier(texts, truncation=True)

    # Map model output labels to human-readable sentiment classes
    label_mapping = {
        'LABEL_0': 'negative',
        'LABEL_1': 'neutral',
        'LABEL_2': 'positive'
    }

    # Extract the predicted label for each text and map to sentiment
    test_df['bert_untrained_pred'] = [label_mapping[p['label']] for p in preds]

    # Save the DataFrame with predictions to the specified output path
    test_df.to_csv(output_path, index=False)

    print('Twitter-RoBERTa Base results saved to test_preprocessed_VADER_BERTBASE.csv')
    return output_path