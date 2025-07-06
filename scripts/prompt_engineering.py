from __future__ import annotations

import itertools
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import requests
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from tenacity import retry, wait_random_exponential, stop_after_attempt

import openai

###############################################################################
# CONFIGURATION
###############################################################################

# Column names for text and label in the validation set
TEXT_COL = "text"
LABEL_COL = "sentiment"

# How often to checkpoint results (in number of runs)
CHECKPOINT_EVERY = 1

# Supported models and their configuration
MODELS: Dict[str, Dict[str, str]] = {
    "gpt-4o": {"provider": "openai", "engine": "gpt-4o"},
    "deepseek": {"provider": "deepseek", "engine": "deepseek-chat"},
}

###############################################################################
# API KEYS
###############################################################################

# Set OpenAI API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")
# Set DeepSeek API key from environment variable
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

###############################################################################
# HELPER FUNCTIONS
###############################################################################

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def call_model(model_name: str, prompt: str) -> str:
    """
    Calls the specified LLM model with the given prompt and returns the response.

    Args:
        model_name (str): The key for the model in the MODELS dict.
        prompt (str): The prompt to send to the model.

    Returns:
        str: The model's response as a string.

    Raises:
        RuntimeError: If the required API key is not set.
        NotImplementedError: If the provider is not supported.
    """
    cfg = MODELS[model_name]
    if cfg["provider"] == "openai":
        # Call OpenAI chat completion endpoint
        resp = openai.chat.completions.create(
            model=cfg["engine"],
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return resp.choices[0].message.content
    if cfg["provider"] == "deepseek":
        # Call DeepSeek API
        if not DEEPSEEK_API_KEY:
            raise RuntimeError("DEEPSEEK_API_KEY not set")
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": cfg["engine"],
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
        }
        r = requests.post("https://api.deepseek.com/v1/chat/completions", headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    raise NotImplementedError(cfg["provider"])


def extract_label(text: str) -> str:
    """
    Extracts the sentiment label from the model's response text.

    Args:
        text (str): The model's response.

    Returns:
        str: One of "positive", "neutral", "negative", or "unknown".
    """
    low = text.lower()
    for lab in ("positive", "neutral", "negative"):
        if lab in low:
            return lab
    return "unknown"


def build_prompt(tweet_text: str, active_components: List[str], base_prompt: str) -> str:
    """
    Constructs the full prompt for the model by combining active prompt components and the base prompt.

    Args:
        tweet_text (str): The tweet text to classify.
        active_components (List[str]): List of prompt component strings to prepend.
        base_prompt (str): The base prompt template (should include {tweet_text}).

    Returns:
        str: The full prompt to send to the model.
    """
    components_text = "\n\n".join(active_components)
    return components_text + "\n\n" + base_prompt.format(tweet_text=tweet_text)


###############################################################################
# MAIN FUNCTION
###############################################################################

def run(base_prompt: str, prompt_components: Dict[str, str], validation_path: str, output_dir: str) -> None:
    """
    Runs prompt engineering experiments by evaluating all combinations of prompt components
    and models on a validation set, saving metrics and predictions.

    Args:
        base_prompt (str): The base prompt template (should include {tweet_text}).
        prompt_components (Dict[str, str]): Dict of named prompt components (flags).
        validation_path (str): Path to the validation CSV file.
        output_dir (str): Directory to save results.
    """
    # Timestamp for output file uniqueness
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Paths for saving metrics and predictions
    metrics_path = output_dir / f"prompt_engineering_metrics_{ts}.csv"
    preds_path = output_dir / f"prompt_engineering_preds_{ts}.csv"

    print(f"Loading validation data from: {validation_path}")
    df = pd.read_csv(validation_path)
    # Ensure required columns are present
    assert {TEXT_COL, LABEL_COL}.issubset(df.columns)

    # Load validation data and prepare predictions DataFrame
    val_df = pd.read_csv(validation_path)
    full_preds_df = val_df.copy().reset_index(drop=True)

    # Generate all combinations of prompt component flags (on/off for each)
    flag_keys = list(prompt_components.keys())
    flag_grid = [dict(zip(flag_keys, combo)) for combo in itertools.product([False, True], repeat=len(flag_keys))]

    total_runs = len(flag_grid) * len(MODELS)
    run_counter = 0
    metrics_rec: List[Dict[str, object]] = []

    def flush_results(final: bool = False):
        """
        Saves current metrics and predictions to disk.

        Args:
            final (bool): If True, prints a message indicating final save.
        """
        pd.DataFrame(metrics_rec).to_csv(metrics_path, index=False)
        full_preds_df.to_csv(preds_path, index=False)
        if final:
            print(f"Results saved → {metrics_path.name}, {preds_path.name}")

    overall_start = time.perf_counter()

    try:
        # Iterate over all prompt component flag combinations
        for flags in flag_grid:
            # Determine which components are active for this run
            active_keys = [key for key, use in flags.items() if use]
            active_components = [prompt_components[k] for k in active_keys]
            tag = "base" if not active_components else "_".join(active_keys)

            # Build prompts for all tweets in the validation set
            prompts = [build_prompt(text, active_components, base_prompt) for text in full_preds_df[TEXT_COL]]

            # Evaluate each model for this prompt variant
            for model in MODELS:
                run_counter += 1
                print(f"\n▶️  Run {run_counter}/{total_runs} — {tag} — {model}")
                start_run = time.perf_counter()

                # Get predictions for all prompts using the current model
                preds = [extract_label(call_model(model, prompt)) for prompt in prompts]
                elapsed = time.perf_counter() - start_run

                # Store predictions in the DataFrame
                full_preds_df[f"pred_{model}_{tag}"] = preds

                # Compute evaluation metrics
                acc = accuracy_score(full_preds_df[LABEL_COL], preds)
                f1w = f1_score(full_preds_df[LABEL_COL], preds, average="weighted")
                f1m = f1_score(full_preds_df[LABEL_COL], preds, average="macro")

                # Record metrics for this run
                metrics_rec.append({
                    "run_idx": run_counter,
                    "variant": tag,
                    "model": model,
                    "accuracy": acc,
                    "f1_weighted": f1w,
                    "f1_macro": f1m,
                    "elapsed_sec": round(elapsed, 2)
                })

                print(f"      ↳ {elapsed/60:.1f} min, acc {acc:.3f}")

                # Periodically save results to disk
                if run_counter % CHECKPOINT_EVERY == 0:
                    flush_results()

    finally:
        # Always save final results, even if interrupted
        flush_results(final=True)

    total = time.perf_counter() - overall_start
    print(f"\n✅ Completed {total_runs} runs in {total/60:.1f} min ({total/3600:.2f} h)")
