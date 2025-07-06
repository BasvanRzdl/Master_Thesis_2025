import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
import evaluate
import os

def run(train_path, model_output_dir):
    # === Load the training data ===
    train_df = pd.read_csv(train_path)

    # === Map sentiment strings to numeric labels for classification ===
    label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
    train_df_bert = train_df.copy()
    train_df_bert['label'] = train_df_bert['sentiment'].str.lower().map(label_map)

    # === Split the data into training and validation sets (stratified by label) ===
    train_df_bert, val_df_bert = train_test_split(
        train_df_bert,
        test_size=0.1,
        stratify=train_df_bert['label'],
        random_state=42
    )

    # === Keep only relevant columns and rename for Hugging Face compatibility ===
    train_df_bert = train_df_bert[['text_cleaned', 'label']].rename(columns={'text_cleaned': 'text'})
    val_df_bert   = val_df_bert[['text_cleaned', 'label']].rename(columns={'text_cleaned': 'text'})

    # === Ensure the 'text' column is string type and handle potential NaNs ===
    train_df_bert['text'] = train_df_bert['text'].astype(str).fillna('')
    val_df_bert['text'] = val_df_bert['text'].astype(str).fillna('')

    # === Convert pandas DataFrames to Hugging Face Datasets ===
    train_dataset_bert = Dataset.from_pandas(train_df_bert, preserve_index=False)
    val_dataset_bert   = Dataset.from_pandas(val_df_bert, preserve_index=False)

    # === Load the tokenizer for the pre-trained Twitter-RoBERTa model ===
    model_name = 'cardiffnlp/twitter-roberta-base-sentiment'
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # === Define tokenization function for batched processing ===
    def tokenize_function(batch):
        return tokenizer(list(batch['text']), truncation=True)

    # === Apply tokenization to both train and validation datasets ===
    train_dataset_bert = train_dataset_bert.map(tokenize_function, batched=True)
    val_dataset_bert   = val_dataset_bert.map(tokenize_function, batched=True)

    # === Set dataset format to PyTorch tensors for model training ===
    columns = ['input_ids', 'attention_mask', 'label']
    train_dataset_bert.set_format(type='torch', columns=columns)
    val_dataset_bert.set_format(type='torch', columns=columns)

    # === Create a data collator for dynamic padding during training ===
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # === Load the pre-trained Twitter-RoBERTa model for sequence classification ===
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

    # === Load evaluation metrics (accuracy and weighted F1) ===
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    # === Define compute_metrics function for Trainer evaluation ===
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        accuracy = accuracy_metric.compute(predictions=preds, references=labels)['accuracy']
        f1 = f1_metric.compute(predictions=preds, references=labels, average='weighted')['f1']
        return {"accuracy": accuracy, "f1": f1}

    # === Set up training arguments for the Trainer ===
    training_args = TrainingArguments(
        output_dir=model_output_dir,
        evaluation_strategy="epoch",         # Evaluate at the end of each epoch
        save_strategy="epoch",               # Save checkpoint at the end of each epoch
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,
        load_best_model_at_end=True,         # Restore best model at end of training
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,
        report_to="none",                    # Disable reporting to external services
        logging_strategy="epoch",
        seed=42
    )

    # === Initialize the Hugging Face Trainer for fine-tuning ===
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset_bert,
        eval_dataset=val_dataset_bert,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]  # Early stopping if no improvement
    )

    # === Fine-tune the model ===
    trainer.train()

    # === Save the fine-tuned model and tokenizer to disk ===
    os.makedirs(model_output_dir, exist_ok=True)
    trainer.save_model(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)
    print("Model checkpoint written to:", model_output_dir)

    # === Return the output directory path for downstream use ===
    return model_output_dir