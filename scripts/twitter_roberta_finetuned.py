from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import pandas as pd
import numpy as np
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run(model_dir, input_path, output_path):
    # Load the test set from a CSV file
    test_df = pd.read_csv(input_path)

    # Map sentiment string labels to numeric class ids for evaluation
    label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
    test_df['label'] = test_df['sentiment'].str.lower().map(label_map)
    # Ensure text is string type and handle missing values
    test_df['text'] = test_df['text_cleaned'].astype(str).fillna('')
    # Convert the DataFrame to a HuggingFace Dataset for use with Trainer
    test_dataset_bert = Dataset.from_pandas(test_df[['text', 'label']], preserve_index=False)

    # Load the fine-tuned model and tokenizer from the specified directory
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    # Define a function to tokenize the text data
    def tokenize_function(batch):
        return tokenizer(list(batch['text']), truncation=True)
    # Apply tokenization to the test dataset
    test_dataset_bert = test_dataset_bert.map(tokenize_function, batched=True)
    # Set the format of the dataset for PyTorch and specify required columns
    columns = ['input_ids', 'attention_mask', 'label']
    test_dataset_bert.set_format(type='torch', columns=columns)

    # Set up Trainer arguments for evaluation (no training)
    training_args = TrainingArguments(
        output_dir=model_dir,
        do_train=False,
        do_eval=True,
        per_device_eval_batch_size=16,
        report_to="none",
    )

    # Initialize the HuggingFace Trainer for evaluation
    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer
    )

    # Run prediction on the test set
    test_results_bert = trainer.predict(test_dataset_bert)
    logits = test_results_bert.predictions
    y_true_bert = test_results_bert.label_ids
    # Get predicted class ids by taking the argmax over logits
    y_pred_bert = np.argmax(logits, axis=1)

    # Map predicted class ids back to sentiment string labels
    id2label = {0: "negative", 1: "neutral", 2: "positive"}
    test_df = test_df.copy()
    test_df["bert_trained_pred"] = y_pred_bert
    test_df["bert_trained_pred"] = test_df["bert_trained_pred"].map(id2label)
    # Optionally drop the numeric label column (not saved to output)
    test_df.drop('label', axis=1)
    # Save the DataFrame with predictions to the specified output path
    test_df.to_csv(output_path, index=False)

    return output_path