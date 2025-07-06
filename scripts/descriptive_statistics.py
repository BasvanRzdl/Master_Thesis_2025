import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import os

def run(train_path, test_path, val_path, output_dir):
    # === Load the train, test, and validation datasets ===
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    val_df = pd.read_csv(val_path)
    # Concatenate all splits for overall statistics
    df_merged = pd.concat([train_df, test_df, val_df], ignore_index=True)

    # Ensure the output directory for statistics exists
    os.makedirs("results/descriptive_statistics", exist_ok=True)

    def generate_descriptive_stats(df, name):
        """
        Generate a string summary of descriptive statistics for a given DataFrame.

        Args:
            df (pd.DataFrame): The dataset to summarize.
            name (str): Name of the dataset (e.g., 'TRAIN', 'TEST', etc.).

        Returns:
            str: Formatted statistics summary.
        """
        stats = []
        stats.append(f"[{name.upper()}]")
        stats.append(f"Tweet count: {df.shape[0]}")
        # Sentiment label distribution
        stats.append(f"Sentiment distribution:\n{df['sentiment'].value_counts()}")
        # Average tweet length in characters
        stats.append(f"Average tweet length (chars): {df['text'].str.len().mean():.2f}")
        # Average tweet length in tokens (words)
        stats.append(f"Average tweet length (tokens): {df['text'].str.split().apply(len).mean():.2f}")
        # Percentage of tweets containing a hashtag
        stats.append(f"Contains hashtag (%): {(df['text'].str.contains('#')).mean() * 100:.2f}")
        # Percentage of tweets containing an emoji (basic unicode range)
        emoji_tag = '[\U0001F600-\U0001F64F]'
        stats.append(
            f"Contains emoji (%): {(df['text'].str.contains(emoji_tag, regex=True)).mean() * 100:.2f}")
        stats.append("\n")
        return "\n".join(stats)

    # === Collect descriptive statistics for all splits and the full dataset ===
    all_stats = [
        generate_descriptive_stats(df_merged, "FULL"),
        generate_descriptive_stats(train_df, "TRAIN"),
        generate_descriptive_stats(test_df, "TEST"),
        generate_descriptive_stats(val_df, "VAL")
    ]

    # Combine all statistics into a single string
    final_stats = "\n".join(all_stats)

    # === Write the descriptive statistics to a text file ===
    with open("results/descriptive_statistics/descriptive_statistics.txt", "w", encoding="utf-8") as f:
        f.write(final_stats)

    print("Descriptive statistics saved to descriptive_statistics.txt")

    # === Set up stopwords for word clouds (add domain-specific stopwords) ===
    custom_stopwords = set(STOPWORDS)
    custom_stopwords.update(["climate", "change", "rt", "amp"])

    def generate_wordcloud(df, sentiment, color_map, output_name):
        """
        Generate and save a word cloud image for tweets of a given sentiment.

        Args:
            df (pd.DataFrame): The dataset containing tweets.
            sentiment (str): The sentiment label to filter by ('positive', 'negative', etc.).
            color_map (str): Matplotlib colormap for the word cloud.
            output_name (str): Filename for saving the word cloud image.
        """
        # Concatenate all lemmatized text for the given sentiment
        text = " ".join(df[df['sentiment'] == sentiment]['text_lemmatised'].dropna().tolist())
        # Generate the word cloud
        wordcloud = WordCloud(
            width=1000,
            height=500,
            background_color='white',
            stopwords=custom_stopwords,
            colormap=color_map,
            max_words=50
        ).generate(text)
        # Plot and save the word cloud
        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f"Word Cloud of {sentiment.capitalize()} Tweets", fontsize=14)
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, output_name))
        plt.close()

    # === Generate and save word clouds for positive and negative tweets ===
    generate_wordcloud(df_merged, 'positive', 'Greens_r', 'wordcloud_positive.png')
    generate_wordcloud(df_merged, 'negative', 'Reds_r', 'wordcloud_negative.png')

    print("Wordclouds saved to wordcloud_positive.png and wordcloud_negative.png")