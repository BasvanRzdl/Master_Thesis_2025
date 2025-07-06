import os
from typing import Dict
from scripts import (
    data_loading_and_cleaning,
    descriptive_statistics,
    vader,
    twitter_roberta_base,
    fine_tune_twitter_roberta,
    twitter_roberta_finetuned,
    prompt_engineering,
    prompt_robustness,
    lda,
    results_evaluation
)

def main():
    # Flag to control whether to fine-tune Twitter-RoBERTa
    FINETUNE_ROBERTA = False

    # Paths to the raw datasets
    DATA_PATHS = ["data/Dataset A.csv", "data/Dataset B.csv"]

    # Number of samples to use for validation
    VALIDATION_SAMPLE_SIZE = 1000

    # Random seed for reproducibility
    RANDOM_SEED = 42

    # 1. Data Loading and Cleaning
    # This step loads the raw data, splits it into train/test/validation, and performs cleaning.
    train_path, test_path, val_path = data_loading_and_cleaning.run(
        raw_data_paths=DATA_PATHS,  # specify your raw data files
        output_dir="data/",
        val_size=VALIDATION_SAMPLE_SIZE,
        seed=RANDOM_SEED
    )

    # 2. Descriptive Statistics
    # Generate and save descriptive statistics for train, test, and validation sets.
    descriptive_statistics.run(
        train_path=train_path,
        test_path=test_path,
        val_path=val_path,
        output_dir="results/descriptive_statistics/"
    )

    # 3. VADER
    # Apply VADER sentiment analysis to the test set and save the results.
    vader_test_path = vader.run(
        test_path=test_path,
        output_path="data/test_preprocessed_VADER.csv"
    )
    
    # 4. Twitter-RoBERTa Base
    # Run the base Twitter-RoBERTa model on the test set.
    bertbase_path = twitter_roberta_base.run(
        input_path=vader_test_path,
        output_path="data/test_preprocessed_VADER_BERTBASE.csv"
    )

    # 5. Fine-Tune Twitter-RoBERTa (optional)
    # Fine-tune the Twitter-RoBERTa model on the training data if enabled.
    if FINETUNE_ROBERTA == True:
        fine_tune_twitter_roberta.run(
            train_path=train_path,
            model_output_dir="data/Twitter-RoBERTA_Fine_Tuning_Weights/"
        )

    # 6. Twitter-RoBERTa Fine-Tuned
    # Run the fine-tuned Twitter-RoBERTa model on the BERTBASE-processed test set.
    bertft_path = twitter_roberta_finetuned.run(
        model_dir="data/Twitter-RoBERTA_Fine_Tuning_Weights/",
        input_path=bertbase_path,
        output_path="data/test_preprocessed_VADER_BERTBASE_BERTFT.csv"
    )

    # Define the base prompt for LLM-based sentiment classification
    BASE_PROMPT = (
        "Classify the sentiment of the following tweet as 'positive', 'neutral', or 'negative'.\n"
        "Tweet: \"{tweet_text}\"\n"
        "Sentiment:"
    )

    # Components for prompt engineering, including role, domain, label explanations, few-shot examples, etc.
    PROMPT_COMPONENTS: Dict[str, str] = {
        "role": "You are an impartial social-media analyst.",
        "domain": "Tweets discuss climate change, climate action, or sustainability.",
        "label_explanation": (
            " - positive: if the tweet supports climate action and sustainability, expresses concern about climate change, or affirms its reality.\n"
            " - negative: if the tweet denies, mocks, criticizes, or downplays climate change, climate action, or sustainability.\n"
            " - neutral:  if the tweet does not clearly express a stance or is purely factual."
        ),
        "few_shot": (
            "Here are some examples:\n"
            "Tweet: \"Climate change is real and we need to act now to save the planet.\"\n"
            "Sentiment: positive\n\n"
            "Tweet: \"Global warming is a hoax created to control us.\"\n"
            "Sentiment: negative\n\n"
            "Tweet: \"The IPCC released its latest climate report today.\"\n"
            "Sentiment: neutral"
        ),
        "sarcasm": "Tweets may contain sarcasm.",
        "self_check": "Verify label before replying.",
    }

    # 7. Prompt Engineering
    # Use prompt engineering approach to classify sentiment on the validation set using an LLM.
    prompt_engineering.run(
        base_prompt=BASE_PROMPT,
        prompt_components=PROMPT_COMPONENTS,
        validation_path=val_path,
        output_dir="results/prompt_engineering"
    )

    # 8. Prompt Robustness
    # Evaluate the robustness of the prompt/LLM.
    llm_path = prompt_robustness.run(
        input_path=bertft_path,
        output_path="data/test_preprocessed_VADER_BERTBASE_BERTFT_LLM.csv",
        metrics_path="results/prompt_robustness/"
    )

    # 9. LDA
    # Apply Latent Dirichlet Allocation (LDA) topic modeling.
    lda_path = lda.run(
        input_path=llm_path,
        output_path="data/test_preprocessed_VADER_BERTBASE_BERTFT_LLM_LDA.csv"
    )

    # 10. Final Results Evaluation
    # Evaluate and summarize the final results after all steps.
    results_evaluation.run(
        input_path=lda_path,
        output_dir="results/final_results"
    )

if __name__ == "__main__":
    # Entry point for the script
    main()