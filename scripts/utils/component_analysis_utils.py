
# === COMPONENT EFFECT ANALYSIS FUNCTIONS ===

import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import pandas as pd

# 1. Groupwise mean accuracy by component
def compute_groupwise_means(df, component, models=['gpt-4o', 'deepseek']):
    """
    Computes the mean accuracy for each model, grouped by the presence/absence of a given prompt component.

    Args:
        df (pd.DataFrame): DataFrame containing results and component columns.
        component (str): The component column to group by (e.g., 'role', 'domain').
        models (list): List of model column names to compute means for.

    Returns:
        pd.DataFrame: Grouped mean accuracies for each model by component presence (0/1).
    """
    return df.groupby(component)[models].mean()

# 2. Plot mean accuracy by component
def plot_component_effect(df, component, save_path=None):
    """
    Plots a bar chart of mean accuracy for each model, grouped by the presence/absence of a given component.

    Args:
        df (pd.DataFrame): DataFrame containing results and component columns.
        component (str): The component column to group by.
        save_path (str, optional): If provided, saves the plot to this path; otherwise, displays the plot.
    """
    # Compute groupwise means and reshape for plotting
    avg_df = df.groupby(component)[['gpt-4o', 'deepseek']].mean().reset_index()
    avg_df = pd.melt(avg_df, id_vars=[component], var_name='model', value_name='accuracy')

    sns.set(style="whitegrid")
    plt.figure(figsize=(6, 4))
    sns.barplot(data=avg_df, x=component, y='accuracy', hue='model')
    plt.title(f'Accuracy by Component Presence: {component}')
    plt.ylabel('Mean Accuracy')
    plt.xlabel(f'{component} Present (0/1)')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_all_components_grouped_bar(df, save_path=None):
    """
    Plots a grouped bar chart showing mean accuracy for each model, for all prompt components,
    split by component presence (0/1).

    Args:
        df (pd.DataFrame): DataFrame containing results and component columns.
        save_path (str, optional): If provided, saves the plot to this path; otherwise, displays the plot.
    """
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Map internal component names to display names for plot labels
    component_name_map = {
        'role': 'Role Instruction',
        'domain': 'Domain Context',
        'label_explanation': 'Label Explanation',
        'few_shot': 'Few-Shot Examples',
        'sarcasm': 'Sarcasm Trigger',
        'self_check': 'Self-Check Step'
    }

    components = list(component_name_map.keys())

    # Build a long-format DataFrame for plotting
    rows = []
    for comp in components:
        # Compute groupwise means for each component
        group_means = df.groupby(comp)[['gpt-4o', 'deepseek']].mean().reset_index()
        for _, row in group_means.iterrows():
            rows.append({
                'component': component_name_map[comp],  # Human-readable name
                'presence': int(row[comp]),             # 0 (absent) or 1 (present)
                'gpt-4o': row['gpt-4o'],
                'deepseek': row['deepseek']
            })

    long_df = pd.DataFrame(rows)
    melted_df = pd.melt(long_df, id_vars=['component', 'presence'],
                        value_vars=['gpt-4o', 'deepseek'],
                        var_name='model', value_name='accuracy')

    # Create grouped bar plot
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=melted_df,
        x='component',
        y='accuracy',
        hue='presence',
        errorbar=None,
        palette='pastel',
        edgecolor='black',
        dodge=True
    )
    plt.title('Mean Accuracy by Prompt Component Presence')
    plt.ylabel('Mean Accuracy')
    plt.xlabel('Prompt Component')
    plt.legend(title='Component Present')
    plt.xticks(rotation=30)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

# 3. T-test for accuracy difference based on component
def ttest_component_effect(df, component, model):
    """
    Performs an independent t-test to compare model accuracy between groups
    with and without a given prompt component.

    Args:
        df (pd.DataFrame): DataFrame containing results and component columns.
        component (str): The component column to compare (0 vs 1).
        model (str): The model column to test (e.g., 'gpt-4o').

    Returns:
        tuple: (t_statistic, p_value)
    """
    group_1 = df[df[component] == 1][model]
    group_0 = df[df[component] == 0][model]
    t_stat, p_val = ttest_ind(group_1, group_0, equal_var=False)
    return t_stat, p_val

# 4. Compute accuracy deltas (effect size) for all components
def component_delta_ranking(df, models=['gpt-4o', 'deepseek']):
    """
    Computes the difference in mean accuracy (effect size) for each model,
    for all prompt components (present vs absent).

    Args:
        df (pd.DataFrame): DataFrame containing results and component columns.
        models (list): List of model column names to compute deltas for.

    Returns:
        pd.DataFrame: DataFrame with effect size (mean difference) for each component and model.
    """
    effect_dict = {}
    for comp in ['role', 'domain', 'label_explanation', 'few_shot', 'sarcasm', 'self_check']:
        # Compute the difference in mean accuracy between presence (1) and absence (0) of the component
        mean_diff = df.groupby(comp)[models].mean().diff().iloc[-1]
        effect_dict[comp] = mean_diff
    # Rename columns to indicate they are differences
    return pd.DataFrame(effect_dict).T.rename(columns={m: f"{m}_diff" for m in models})
