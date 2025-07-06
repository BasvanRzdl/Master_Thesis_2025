import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from scripts.utils.component_analysis_utils import (
    compute_groupwise_means,
    plot_component_effect,
    ttest_component_effect,
    component_delta_ranking,
    plot_all_components_grouped_bar
)

def run(input_path: str, output_dir: str) -> None:
    # Create output directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    plot_dir = os.path.join(output_dir, "plots/components")
    csv_dir = os.path.join(output_dir, "component_stats")
    cm_dir = os.path.join(output_dir, "confusion_matrices")

    for d in [plot_dir, csv_dir, cm_dir]:
        os.makedirs(d, exist_ok=True)

    # Load main results and prompt engineering results
    df = pd.read_csv(input_path)
    prompt_df = pd.read_csv("results/prompt_engineering/prompt_engineering_metrics_20250607_202310.csv")

    # Helper function to compute accuracy and macro-F1 for a set of model columns
    def evaluate_predictions(df, model_columns, label_column='sentiment'):
        metrics = []
        for model in model_columns:
            acc = accuracy_score(df[label_column], df[model])
            f1 = f1_score(df[label_column], df[model], average='macro')
            metrics.append({'Model': model, 'Accuracy': acc, 'Macro-F1': f1})
        return pd.DataFrame(metrics)

    # Helper function to save confusion matrices for each model
    def save_confusion_matrices(df, model_columns, label_column='sentiment', save_dir=cm_dir):
        for model in model_columns:
            cm = confusion_matrix(df[label_column], df[model], labels=sorted(df[label_column].unique()))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=sorted(df[label_column].unique()))
            fig, ax = plt.subplots(figsize=(6, 6))
            disp.plot(ax=ax, cmap='Blues', colorbar=False)
            plt.title(f'Confusion Matrix: {model}')
            plt.tight_layout()
            fig.savefig(f"{save_dir}/confusion_matrix_{model}.png")
            plt.close()

    # Define model columns for main and robustness evaluation
    main_models = ['vader_pred', 'bert_trained_pred', 'bert_untrained_pred', 'pred_gpt-4o_baseline', 'pred_deepseek_baseline']
    robust_models = main_models + [
        'pred_gpt-4o_no_punctuation', 'pred_gpt-4o_false_examples',
        'pred_deepseek_no_punctuation', 'pred_deepseek_false_examples'
    ]

    # Evaluate and save metrics for main models
    main_metrics_df = evaluate_predictions(df, main_models)
    main_metrics_df.to_csv(f"{output_dir}/main_model_comparison.csv", index=False)

    # Evaluate and save metrics for robustness models
    robust_metrics_df = evaluate_predictions(df, robust_models)
    robust_metrics_df.to_csv(f"{output_dir}/robustness_prompt_comparison.csv", index=False)

    # Save LDA topic distribution
    lda_stats = df['lda_topic'].value_counts().reset_index()
    lda_stats.columns = ['Topic', 'Tweet_Count']
    lda_stats.to_csv(f"{output_dir}/lda_topic_distribution.csv", index=False)

    # Compute and save per-topic, per-model metrics
    topic_metrics = []
    for topic in df['lda_topic'].unique():
        topic_df = df[df['lda_topic'] == topic]
        for model in main_models:
            acc = accuracy_score(topic_df['sentiment'], topic_df[model])
            f1 = f1_score(topic_df['sentiment'], topic_df[model], average='macro')
            topic_metrics.append({'Topic': topic, 'Model': model, 'Accuracy': acc, 'Macro-F1': f1})

    topic_metrics_df = pd.DataFrame(topic_metrics)
    topic_metrics_df.to_csv(f"{output_dir}/per_topic_model_comparison.csv", index=False)

    # Save confusion matrices for all robust models
    save_confusion_matrices(df, robust_models)

    # Create and save accuracy table for each prompt variant and model
    accuracy_table = prompt_df.pivot_table(index='variant', columns='model', values='accuracy')
    accuracy_table.to_csv(f"{output_dir}/prompt_variant_accuracy_table.csv")

    # Prepare component presence dataframe for regression and group analysis
    components = ['role', 'domain', 'label_explanation', 'few_shot', 'sarcasm', 'self_check']
    variants = prompt_df['variant'].unique().tolist()
    components_df = pd.DataFrame({'variant': variants})

    # Mark presence (1/0) of each component in each variant
    for comp in components:
        components_df[comp] = components_df['variant'].apply(lambda x: int(comp in x))

    # Merge component presence with accuracy table for further analysis
    accuracy_table_components = prompt_df.pivot(index='variant', columns='model', values='accuracy').reset_index()
    merged_df = pd.merge(components_df, accuracy_table_components, on='variant')

    # Compute groupwise means and t-tests for each component
    t_test_rows, all_group_means = [], []
    for comp in components:
        # Compute group means for component present/absent
        means = compute_groupwise_means(merged_df, comp).reset_index()
        means['component'] = comp
        means = means.rename(columns={comp: 'component_value'})
        all_group_means.append(means)

        # Plot effect of component
        plot_component_effect(merged_df, comp, save_path=f"{plot_dir}/{comp}_effect.png")
        # Run t-test for each model
        for model in ['gpt-4o', 'deepseek']:
            t, p = ttest_component_effect(merged_df, comp, model)
            t_test_rows.append({'component': comp, 'model': model, 't_statistic': t, 'p_value': p})

    # Save group means, t-test results, and component delta ranking
    pd.concat(all_group_means).to_csv(f"{csv_dir}/group_means_all_components.csv", index=False)
    pd.DataFrame(t_test_rows).to_csv(f"{csv_dir}/t_test_results.csv", index=False)
    component_delta_ranking(merged_df).to_csv(f"{csv_dir}/component_delta_ranking.csv", index=False)

    # Plot grouped bar chart for all components
    plot_all_components_grouped_bar(merged_df, save_path=f"{plot_dir}/all_components_grouped_bar.png")

    # Run OLS regression for each model to estimate effect of each component
    regression_results = {}
    for model in ['gpt-4o', 'deepseek']:
        reg_df = merged_df.copy().dropna(subset=[model])
        X = reg_df[components]
        X = sm.add_constant(X)
        y = reg_df[model]

        model_fit = sm.OLS(y, X).fit()
        regression_results[model] = model_fit

    # Summarize regression results and save to file
    summary_df = summary_col(list(regression_results.values()),
                             stars=True,
                             model_names=['GPT-4o', 'DeepSeek-Chat'],
                             info_dict={'R-squared': lambda x: f"{x.rsquared:.3f}", 'N': lambda x: f"{int(x.nobs)}"})

    with open(f"{csv_dir}/component_regression_summary.txt", "w") as f:
        f.write(summary_df.as_text())

    # --- Additional plots: Heatmaps for topic/model and prompt variant/model accuracy ---

    # Mapping for renaming prompt variants for better plot readability
    variant_rename_map = {
        "base": "Base",
        "domain": "Domain",
        "domain_few_shot": "Domain + Few-shot",
        "domain_few_shot_sarcasm": "Domain + Few-shot + Sarcasm",
        "domain_few_shot_sarcasm_self_check": "Domain + Few-shot + Sarcasm + Self-check",
        "domain_few_shot_self_check": "Domain + Few-shot + Self-check",
        "domain_label_explanation": "Domain + Label Explanation",
        "domain_label_explanation_few_shot": "Domain + Label Exp. + Few-shot",
        "domain_label_explanation_few_shot_sarcasm": "Domain + Label Exp. + Few-shot + Sarcasm",
        "domain_label_explanation_few_shot_sarcasm_self_check": "Domain + Label Exp. + Few-shot + Sarcasm + Self-check",
        "domain_label_explanation_few_shot_self_check": "Domain + Label Exp. + Few-shot + Self-check",
        "domain_label_explanation_sarcasm": "Domain + Label Exp. + Sarcasm",
        "domain_label_explanation_sarcasm_self_check": "Domain + Label Exp. + Sarcasm + Self-check",
        "domain_label_explanation_self_check": "Domain + Label Exp. + Self-check",
        "domain_sarcasm": "Domain + Sarcasm",
        "domain_sarcasm_self_check": "Domain + Sarcasm + Self-check",
        "domain_self_check": "Domain + Self-check",
        "few_shot": "Few-shot",
        "few_shot_sarcasm": "Few-shot + Sarcasm",
        "few_shot_sarcasm_self_check": "Few-shot + Sarcasm + Self-check",
        "few_shot_self_check": "Few-shot + Self-check",
        "label_explanation": "Label Explanation",
        "label_explanation_few_shot": "Label Exp. + Few-shot",
        "label_explanation_few_shot_sarcasm": "Label Exp. + Few-shot + Sarcasm",
        "label_explanation_few_shot_sarcasm_self_check": "Label Exp. + Few-shot + Sarcasm + Self-check",
        "label_explanation_few_shot_self_check": "Label Exp. + Few-shot + Self-check",
        "label_explanation_sarcasm": "Label Exp. + Sarcasm",
        "label_explanation_sarcasm_self_check": "Label Exp. + Sarcasm + Self-check",
        "label_explanation_self_check": "Label Exp. + Self-check",
        "role": "Role",
        "role_domain": "Role + Domain",
        "role_domain_few_shot": "Role + Domain + Few-shot",
        "role_domain_few_shot_sarcasm": "Role + Domain + Few-shot + Sarcasm",
        "role_domain_few_shot_sarcasm_self_check": "Role + Domain + Few-shot + Sarcasm + Self-check",
        "role_domain_few_shot_self_check": "Role + Domain + Few-shot + Self-check",
        "role_domain_label_explanation": "Role + Domain + Label Exp.",
        "role_domain_label_explanation_few_shot": "Role + Domain + Label Exp. + Few-shot",
        "role_domain_label_explanation_few_shot_sarcasm": "Role + Domain + Label Exp. + Few-shot + Sarcasm",
        "role_domain_label_explanation_few_shot_sarcasm_self_check": "Role + Domain + Label Exp. + Few-shot + Sarcasm + Self-check",
        "role_domain_label_explanation_few_shot_self_check": "Role + Domain + Label Exp. + Few-shot + Self-check",
        "role_domain_label_explanation_sarcasm": "Role + Domain + Label Exp. + Sarcasm",
        "role_domain_label_explanation_sarcasm_self_check": "Role + Domain + Label Exp. + Sarcasm + Self-check",
        "role_domain_label_explanation_self_check": "Role + Domain + Label Exp. + Self-check",
        "role_domain_sarcasm": "Role + Domain + Sarcasm",
        "role_domain_sarcasm_self_check": "Role + Domain + Sarcasm + Self-check",
        "role_domain_self_check": "Role + Domain + Self-check",
        "role_few_shot": "Role + Few-shot",
        "role_few_shot_sarcasm": "Role + Few-shot + Sarcasm",
        "role_few_shot_sarcasm_self_check": "Role + Few-shot + Sarcasm + Self-check",
        "role_few_shot_self_check": "Role + Few-shot + Self-check",
        "role_label_explanation": "Role + Label Exp.",
        "role_label_explanation_few_shot": "Role + Label Exp. + Few-shot",
        "role_label_explanation_few_shot_sarcasm": "Role + Label Exp. + Few-shot + Sarcasm",
        "role_label_explanation_few_shot_sarcasm_self_check": "Role + Label Exp. + Few-shot + Sarcasm + Self-check",
        "role_label_explanation_few_shot_self_check": "Role + Label Exp. + Few-shot + Self-check",
        "role_label_explanation_sarcasm": "Role + Label Exp. + Sarcasm",
        "role_label_explanation_sarcasm_self_check": "Role + Label Exp. + Sarcasm + Self-check",
        "role_label_explanation_self_check": "Role + Label Exp. + Self-check",
        "role_sarcasm": "Role + Sarcasm",
        "role_sarcasm_self_check": "Role + Sarcasm + Self-check",
        "role_self_check": "Role + Self-check",
        "sarcasm": "Sarcasm",
        "sarcasm_self_check": "Sarcasm + Self-check",
        "self_check": "Self-check"
    }
    # Mapping for renaming model columns for plots
    model_rename_map = {
        "deepseek": "DeepSeek-Chat",
        "gpt-4o": "GPT-4o"
    }
    # Mapping for renaming model names in topic metrics
    rename_map_models = {
        'vader_pred': 'VADER',
        'bert_trained_pred': 'BERT\n(Fine-Tuned)',
        'bert_untrained_pred': 'BERT\n(Base)',
        'pred_gpt-4o_baseline': 'GPT-4o',
        'pred_deepseek_baseline': 'DeepSeek-Chat',
    }
    # Mapping for renaming LDA topic numbers to descriptive names
    rename_map_topics = {
        0: 'Human Cause & Sea Level Rise',
        1: 'Urban & Environmental Impacts',
        2: 'Global Warming Attribution',
        3: 'Political Denial & Polarisation',
        5: 'Science Denial & Public Beliefs',
    }

    # Apply renaming for better plot readability
    topic_metrics_df['Model'] = topic_metrics_df['Model'].replace(rename_map_models)
    topic_metrics_df['Topic'] = topic_metrics_df['Topic'].replace(rename_map_topics)

    # Rename prompt variants and models for the accuracy table
    accuracy_table_renamed = accuracy_table.rename(index=variant_rename_map, columns=model_rename_map)

    # Plot heatmap: accuracy per LDA topic and model
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        topic_metrics_df.pivot(index='Topic', columns='Model', values='Accuracy'),
        cmap="Oranges",
        cbar_kws={'label': 'Accuracy'}
    )
    plt.title("Accuracy per LDA Topic and Model")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/heatmap_topic_model_accuracy.png")
    plt.close()

    # Plot heatmap: accuracy per prompt variant and model
    plt.figure(figsize=(12, 13))
    sns.heatmap(
        accuracy_table_renamed,
        cmap="Greens",
        cbar_kws={'label': 'Accuracy'}
    )
    plt.title("Accuracy per Prompt Variant and Model")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/heatmap_prompt_variant_model_accuracy.png")
    plt.close()
