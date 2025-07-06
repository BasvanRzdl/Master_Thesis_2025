from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd
import requests
from sklearn.metrics import accuracy_score, f1_score
from tenacity import retry, wait_random_exponential, stop_after_attempt

import openai
from dotenv import load_dotenv

###############################################################################
# CONFIG & ENV
###############################################################################

# Column names for tweet text and sentiment label
TEXT_COL = "text"
LABEL_COL = "sentiment"
# How often to checkpoint results (in number of runs)
CHECKPOINT_EVERY = 1

# Supported LLM models and their configuration
MODELS: Dict[str, Dict[str, str]] = {
    "gpt-4o": {"provider": "openai", "engine": "gpt-4o"},
    "deepseek": {"provider": "deepseek", "engine": "deepseek-chat"},
}

# Load API keys from .env file
load_dotenv("data/apikey.env")
openai.api_key = os.getenv("OPENAI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

###############################################################################
# PROMPT BUILDERS
###############################################################################

def baseline_gpt4o(tweet: str) -> str:
    """
    Build the baseline prompt for GPT-4o for sentiment classification.
    """
    return f"""You are an impartial social-media analyst.

Tweets discuss climate change, climate action, or sustainability.

 - positive: if the tweet supports climate action and sustainability, expresses concern about climate change, or affirms its reality.
 - negative: if the tweet denies, mocks, criticizes, or downplays climate change, climate action, or sustainability.
 - neutral:  if the tweet does not clearly express a stance or is purely factual.

Classify the sentiment of the following tweet as 'positive', 'neutral', or 'negative'.
Tweet: \"{tweet}\"
Sentiment:
Verify label before replying.
"""

def baseline_deepseek(tweet: str) -> str:
    """
    Build the baseline prompt for DeepSeek for sentiment classification.
    """
    return f"""You are an impartial social-media analyst.

 - positive: if the tweet supports climate action and sustainability, expresses concern about climate change, or affirms its reality.
 - negative: if the tweet denies, mocks, criticizes, or downplays climate change, climate action, or sustainability.
 - neutral:  if the tweet does not clearly express a stance or is purely factual.

Classify the sentiment of the following tweet as 'positive', 'neutral', or 'negative'.
Tweet: \"{tweet}\"
Sentiment:
"""

def no_punct(text: str) -> str:
    """
    Remove periods, commas, and colons from the text and lowercase it.
    """
    stripped = text.replace(".", "").replace(",", "").replace(":", "")
    lowered = stripped.lower()
    return " ".join(lowered.split())

def false_example_inject(prompt: str) -> str:
    """
    Inject a misleading example into the prompt to test model robustness.
    """
    wrong = """Tweet: \"Global warming is a hoax created to control us.\"\nSentiment: positive\n\n"""
    return prompt.replace("Classify the sentiment", wrong + "Classify the sentiment", 1)

###############################################################################
# API CALL & LABEL EXTRACTION
###############################################################################

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
def call_model(model: str, prompt: str) -> str:
    """
    Call the specified LLM model with the given prompt and return the response.

    Args:
        model (str): Model key from MODELS dict.
        prompt (str): Prompt to send to the model.

    Returns:
        str: The model's response as a string.
    """
    cfg = MODELS[model]
    if cfg["provider"] == "openai":
        # Call OpenAI chat completion endpoint
        r = openai.chat.completions.create(
            model=cfg["engine"],
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        return r.choices[0].message.content

    if cfg["provider"] == "deepseek":
        # Call DeepSeek chat completion endpoint
        if not DEEPSEEK_API_KEY:
            raise RuntimeError("DEEPSEEK_API_KEY not set")
        headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
        payload = {"model": cfg["engine"], "messages": [{"role": "user", "content": prompt}], "temperature": 0.0}
        r = requests.post("https://api.deepseek.com/v1/chat/completions", headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    raise NotImplementedError(cfg["provider"])

def extract_label(reply: str) -> str:
    """
    Extract the sentiment label from the model's reply.

    Args:
        reply (str): The model's response.

    Returns:
        str: One of 'positive', 'neutral', 'negative', or 'unknown'.
    """
    low = reply.lower()
    for lab in ("positive", "neutral", "negative"):
        if lab in low:
            return lab
    return "unknown"

###############################################################################
# MAIN FUNCTION
###############################################################################

def run(input_path: str, output_path: str, metrics_path: str) -> str:
    """
    Run prompt robustness experiments on LLMs for sentiment classification.

    Args:
        input_path (str): Path to input CSV with tweets and gold labels.
        output_path (str): Path to save the augmented CSV with predictions.
        metrics_path (str): Directory to save metrics CSV.

    Returns:
        str: Path to the output CSV file.
    """
    # Timestamp for unique metrics file
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_path = Path(metrics_path) / f"perturb_metrics_{ts}.csv"

    # Load input data
    df = pd.read_csv(input_path)
    metrics: List[Dict[str, object]] = []

    total_runs = 6  # 2 models × 3 prompt variants
    run_counter = 0

    def checkpoint():
        """
        Save current predictions and metrics to disk.
        """
        df.to_csv(output_path, index=False)
        pd.DataFrame(metrics).to_csv(metrics_path, index=False)

    overall_start = time.perf_counter()

    try:
        # Loop over each model (gpt-4o, deepseek)
        for model in MODELS:
            # Select the appropriate baseline prompt builder for the model
            base_fn = baseline_gpt4o if model == "gpt-4o" else baseline_deepseek

            # Build prompt variants for each tweet
            variant_prompts = {
                "baseline": [base_fn(tw) for tw in df[TEXT_COL]],
                "no_punctuation": [no_punct(base_fn(tw)) for tw in df[TEXT_COL]],
                "false_examples": [false_example_inject(base_fn(tw)) for tw in df[TEXT_COL]],
            }

            # Run each prompt variant for the current model
            for var, prompts in variant_prompts.items():
                run_counter += 1
                print(f"\n▶️  Run {run_counter}/{total_runs} — {var} — {model}")
                start = time.perf_counter()
                # Get predictions from the model for each prompt
                preds = [extract_label(call_model(model, p)) for p in prompts]
                elapsed = time.perf_counter() - start

                # Store predictions in the dataframe
                df[f"pred_{model}_{var}"] = preds

                # Compute accuracy and F1 metrics
                acc = accuracy_score(df[LABEL_COL], preds)
                f1w = f1_score(df[LABEL_COL], preds, average="weighted")
                f1m = f1_score(df[LABEL_COL], preds, average="macro")

                # Record metrics for this run
                metrics.append({
                    "model": model,
                    "variant": var,
                    "accuracy": acc,
                    "f1_weighted": f1w,
                    "f1_macro": f1m,
                    "elapsed_sec": round(elapsed, 2)
                })

                print(f"      ↳ {elapsed / 60:.1f} min  acc {acc:.3f}  macro-F1 {f1m:.3f}")

                # Checkpoint results after each run
                if run_counter % CHECKPOINT_EVERY == 0:
                    checkpoint()

    finally:
        # Always save results, even if interrupted
        checkpoint()

    tot = time.perf_counter() - overall_start
    print(f"\n✅ Finished in {tot / 60:.1f} min — augmented CSV: {Path(output_path).name}")

    print("Robustness test results have been saved to test_preprocessed_VADER_BERTBASE_BERTFT_LLM.csv")
    return str(output_path)
