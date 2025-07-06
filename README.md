# Master Thesis: Prompt-Based Sentiment Analysis of Sustainability Discourse on Social Media Using Generative AI

This repository contains the complete code and analysis for a master thesis project focused on **climate change sentiment analysis using multiple NLP approaches**. The research compares traditional sentiment analysis methods, transformer models, and Large Language Models (LLMs) on climate change-related social media data.

# Project Structure

```
masterThesis536165/
├── data/                           # Data files and models
│   ├── Dataset A.csv              # Raw dataset A
│   ├── Dataset B.csv              # Raw dataset B
│   ├── merged_climate_sentiment_dataset.csv  # Combined dataset
│   ├── train_preprocessed.csv     # Training set
│   ├── test_preprocessed.csv      # Test set
│   ├── val_preprocessed.csv       # Validation set
│   ├── test_preprocessed_VADER.csv  # VADER results
│   ├── test_preprocessed_VADER_BERTBASE.csv  # BERT base results
│   ├── test_preprocessed_VADER_BERTBASE_BERTFT.csv  # BERT fine-tuned results
│   ├── test_preprocessed_VADER_BERTBASE_BERTFT_LLM.csv  # LLM results
│   ├── test_preprocessed_VADER_BERTBASE_BERTFT_LLM_LDA.csv  # Final results with topics
│   ├── Twitter-RoBERTa_Fine_Tuning_Weights/  # Fine-tuned model weights
│   └── apikey.env                 # API keys (not tracked in git)
├── scripts/                       # Analysis scripts
│   ├── data_loading_and_cleaning.py      # Data preprocessing
│   ├── descriptive_statistics.py         # Exploratory data analysis
│   ├── vader.py                          # VADER sentiment analysis
│   ├── twitter_roberta_base.py           # Base BERT model
│   ├── fine_tune_twitter_roberta.py      # BERT fine-tuning
│   ├── twitter_roberta_finetuned.py      # Fine-tuned BERT
│   ├── prompt_engineering.py             # LLM prompt engineering
│   ├── prompt_robustness.py              # Robustness testing
│   ├── lda.py                           # Topic modeling
│   ├── results_evaluation.py             # Final evaluation
│   └── utils/
│       └── component_analysis_utils.py   # Helper functions
├── results/                       # Analysis results and visualizations
│   ├── descriptive_statistics/    # EDA results
│   ├── final_results/             # Main results and plots
│   ├── lda/                       # Topic modeling results
│   ├── prompt_engineering/        # Prompt engineering results
│   └── prompt_robustness/         # Robustness test results
├── main.py                        # Main execution script
├── requirements.txt               # Python dependencies
└── Thesis Text/                   # Thesis documentation
```

## Quick Start

### Prerequisites
- Python 3.8+
- 8GB+ RAM (16GB+ recommended for BERT fine-tuning)
- OpenAI API key (for GPT-4o)
- DeepSeek API key (for DeepSeek-Chat)

### Download Fine-tuned Model (Optional)
To run the fine-tuned Twitter-RoBERTa model, download the model weights:

```bash
# Create the model directory
mkdir -p data/Twitter-RoBERTa_Fine_Tuning_Weights/

# Download the model file (476 MB)
# Option 1: Using curl (if you have a direct download link)
# curl -L "YOUR_DOWNLOAD_LINK" -o data/Twitter-RoBERTa_Fine_Tuning_Weights/model.safetensors

# Option 2: Manual download
# Download from: [Add your download link here - Google Drive, Dropbox, etc.]
# Place the file at: data/Twitter-RoBERTa_Fine_Tuning_Weights/model.safetensors
```

**Note:** The fine-tuned model is optional. The analysis can run without it, using only the base Twitter-RoBERTa model.

### Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd masterThesis536165
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables:**
```bash
# Create environment file
cp data/apikey.env.example data/apikey.env

# Edit the file and add your API keys:
# OPENAI_API_KEY=your_openai_key_here
# DEEPSEEK_API_KEY=your_deepseek_key_here
```

5. **Run the complete analysis:**
```bash
python main.py
```

## Step-by-Step Replication Guide

### Step 1: Data Preparation
The analysis starts with two raw datasets containing climate change-related tweets:

```bash
python scripts/data_loading_and_cleaning.py
```

**What this does:**
- Loads and merges Dataset A and B
- Removes duplicates, retweets, and missing values
- Splits data into train/test/validation sets (stratified by sentiment)
- Performs text preprocessing (lemmatization, cleaning)
- Saves processed datasets

**Expected output:**
- `data/train_preprocessed.csv`
- `data/test_preprocessed.csv` 
- `data/val_preprocessed.csv`

### Step 2: Exploratory Data Analysis
```bash
python scripts/descriptive_statistics.py
```

**What this does:**
- Generates descriptive statistics for all datasets
- Creates word clouds for positive/negative sentiments
- Saves visualizations to `results/descriptive_statistics/`

### Step 3: VADER Sentiment Analysis
```bash
python scripts/vader.py
```

**What this does:**
- Applies VADER sentiment analysis to test set
- Saves results with predictions to `data/test_preprocessed_VADER.csv`

### Step 4: BERT Base Model
```bash
python scripts/twitter_roberta_base.py
```

**What this does:**
- Loads pre-trained Twitter-RoBERTa model
- Classifies sentiment on test set
- Saves results to `data/test_preprocessed_VADER_BERTBASE.csv`

### Step 5: BERT Fine-tuning (Optional)
```bash
# Set FINETUNE_ROBERTA = True in main.py
python scripts/fine_tune_twitter_roberta.py
```

**What this does:**
- Fine-tunes Twitter-RoBERTa on training data
- Saves model weights to `data/Twitter-RoBERTa_Fine_Tuning_Weights/`
- **Note:** This step is computationally intensive and optional

### Step 6: Fine-tuned BERT
```bash
python scripts/twitter_roberta_finetuned.py
```

**What this does:**
- Loads fine-tuned model (if available)
- Makes predictions on test set
- Saves results to `data/test_preprocessed_VADER_BERTBASE_BERTFT.csv`

### Step 7: Prompt Engineering
```bash
python scripts/prompt_engineering.py
```

**What this does:**
- Tests all combinations of prompt components (64 variants)
- Uses GPT-4o and DeepSeek-Chat models
- Evaluates on validation set
- Saves results to `results/prompt_engineering/`

**Prompt Components Tested:**
- **Role**: "You are an impartial social-media analyst"
- **Domain**: "Tweets discuss climate change, climate action, or sustainability"
- **Label Explanation**: Detailed definitions of positive/negative/neutral
- **Few-shot**: Example tweets with labels
- **Sarcasm**: Warning about sarcastic content
- **Self-check**: Instruction to verify before responding

### Step 8: Robustness Testing
```bash
python scripts/prompt_robustness.py
```

**What this does:**
- Tests model performance under perturbations:
  - No punctuation
  - False examples in prompts
- Evaluates robustness across models
- Saves results to `results/prompt_robustness/`

### Step 9: Topic Modeling
```bash
python scripts/lda.py
```

**What this does:**
- Applies Latent Dirichlet Allocation (LDA)
- Identifies 8 climate change discourse topics
- Assigns topics to each tweet
- Saves results to `data/test_preprocessed_VADER_BERTBASE_BERTFT_LLM_LDA.csv`

**Identified Topics:**
1. Human Cause & Sea Level Rise
2. Urban & Environmental Impacts  
3. Global Warming Attribution
4. Political Denial & Polarisation
5. Science Denial & Public Beliefs

### Step 10: Final Evaluation
```bash
python scripts/results_evaluation.py
```

**What this does:**
- Compares all models across multiple metrics
- Creates confusion matrices and heatmaps
- Analyzes prompt component effects
- Generates topic-wise performance analysis
- Saves comprehensive results to `results/final_results/`

## Configuration Options

### Main Parameters (in `main.py`)
```python
FINETUNE_ROBERTA = False          # Enable BERT fine-tuning
VALIDATION_SAMPLE_SIZE = 1000     # Validation set size
RANDOM_SEED = 42                  # Reproducibility seed
```

### Prompt Engineering (in `scripts/prompt_engineering.py`)
```python
MODELS = {
    "gpt-4o": {"provider": "openai", "engine": "gpt-4o"},
    "deepseek": {"provider": "deepseek", "engine": "deepseek-chat"},
}
```

### LDA Parameters (in `scripts/lda.py`)
```python
k_values = list(range(2, 20, 2))  # Topic numbers to test
final_k = 8                       # Final number of topics
```

## Output Files

### Main Results
- `results/final_results/main_model_comparison.csv` - Model performance metrics
- `results/final_results/robustness_prompt_comparison.csv` - Robustness test results
- `results/final_results/lda_topic_distribution.csv` - Topic distribution
- `results/final_results/per_topic_model_comparison.csv` - Topic-wise performance

### Visualizations
- `results/final_results/confusion_matrices/` - Confusion matrices for all models
- `results/final_results/heatmap_topic_model_accuracy.png` - Topic-model accuracy heatmap
- `results/final_results/heatmap_prompt_variant_model_accuracy.png` - Prompt variant heatmap
- `results/final_results/plots/components/` - Prompt component effect plots

### Prompt Engineering Results
- `results/prompt_engineering/prompt_engineering_metrics_*.csv` - All prompt variant metrics
- `results/prompt_engineering/prompt_engineering_preds_*.csv` - All predictions

## Troubleshooting

### Common Issues

1. **API Key Errors**
   - Ensure API keys are set in `data/apikey.env`
   - Check API key validity and quota limits

2. **Memory Issues**
   - Reduce `VALIDATION_SAMPLE_SIZE` for lower memory usage
   - Use smaller batch sizes in BERT scripts

3. **Model Loading Errors**
   - Ensure internet connection for downloading models
   - Check available disk space for model weights

4. **Reproducibility**
   - Use the same `RANDOM_SEED` value
   - Ensure same Python environment and package versions

### Performance Optimization

1. **Faster Execution**
   - Set `FINETUNE_ROBERTA = False` to skip BERT fine-tuning
   - Reduce validation set size
   - Use fewer prompt variants

2. **Memory Optimization**
   - Process data in smaller batches
   - Use CPU-only mode for BERT (modify scripts)

## Dependencies

### Core Libraries
- `pandas>=1.5.0` - Data manipulation
- `numpy>=1.21.0` - Numerical computing
- `scikit-learn>=1.1.0` - Machine learning utilities

### NLP Libraries
- `transformers>=4.20.0` - Hugging Face transformers
- `torch>=1.12.0` - PyTorch
- `nltk>=3.7` - Natural language toolkit
- `vaderSentiment>=3.3.2` - VADER sentiment analysis
- `gensim>=4.2.0` - Topic modeling

### Visualization
- `matplotlib>=3.5.0` - Plotting
- `seaborn>=0.11.0` - Statistical visualization
- `wordcloud>=1.8.0` - Word cloud generation

### API Libraries
- `openai>=0.27.0` - OpenAI API
- `requests>=2.28.0` - HTTP requests
- `tenacity>=8.2.0` - Retry logic

## Citation

If you use this code in your research, please cite:

```bibtex
@thesis{climate_sentiment_analysis_2024,
  title={Prompt-Based Sentiment Analysis of Sustainability Discourse on Social Media Using Generative AI},
  author={Bas van Roozendaal},
  year={2025},
  institution={[Erasmus School of Economics]},
  type={Master's Thesis}
}
```

## Contributing

This is a master thesis project. For questions, issues, or suggestions, please contact the author.

## License

This project is for academic research purposes. Please respect the original data sources and API terms of service.

## Author

Bas van Roozendaal - Master Thesis  
Erasmus University Rotterdam, Erasmus School of Economics.

---

**Note:** This replication guide provides complete instructions for reproducing the research. All code, data processing steps, and analysis procedures are documented to ensure full reproducibility of the results. 