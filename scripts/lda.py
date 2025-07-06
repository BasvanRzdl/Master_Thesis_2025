import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from gensim import corpora, models
from gensim.models.coherencemodel import CoherenceModel
from gensim.parsing.preprocessing import STOPWORDS

def run(input_path: str, output_path: str) -> str:
    # === Set up output directories for results and plots ===
    results_dir = Path("results/lda")
    plots_dir = results_dir / "plots"
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    txt_output = results_dir / "lda_results.txt"

    # === Load lemmatized text data ===
    full_df = pd.read_csv(input_path)
    # Split lemmatized text into tokens
    texts = full_df["text_lemmatised"].fillna("").astype(str).apply(str.split)

    # === Remove stopwords (including domain-specific ones) ===
    custom_stopwords = STOPWORDS.union({"climate", "change"})
    texts = texts.apply(lambda tokens: [w for w in tokens if w not in custom_stopwords])

    # === Create Gensim dictionary and corpus ===
    dictionary = corpora.Dictionary(texts)
    # Filter out rare and overly common words
    dictionary.filter_extremes(no_below=5, no_above=0.4)
    # Convert texts to bag-of-words representation
    corpus = [dictionary.doc2bow(text) for text in texts]

    # === Helper function: Train LDA and compute coherence for a given k ===
    def compute_coherence(k):
        lda_model = models.LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=k,
            random_state=42,
            passes=10,
            iterations=100,
            eval_every=0  # disables intermediate evaluation
        )
        coherence_model = CoherenceModel(
            model=lda_model,
            texts=texts,
            dictionary=dictionary,
            coherence='c_v'
        )
        return lda_model, coherence_model.get_coherence()

    # === Range of topic numbers to evaluate ===
    k_values = list(range(2, 20, 2))
    coherence_scores = []
    diversity_scores = []
    lda_models = []

    results = []

    # === Train LDA models for each k and compute metrics ===
    for k in k_values:
        lda_k, coh = compute_coherence(k)
        lda_models.append(lda_k)
        coherence_scores.append(coh)

        # Compute topic diversity: proportion of unique words in top N words per topic
        topn = 8
        topic_words = [
            [word for word, _ in lda_k.show_topic(i, topn=topn)]
            for i in range(k)
        ]
        unique_words = set(word for topic in topic_words for word in topic)
        diversity = len(unique_words) / (k * topn)
        diversity_scores.append(diversity)

        results.append(f"k={k} → Coherence: {coh:.4f} | Diversity: {diversity:.4f}")

    # === Select best model by coherence and also pick a fixed k for reporting ===
    best_k_index = int(np.argmax(coherence_scores))
    best_k = k_values[best_k_index]
    best_model = lda_models[best_k_index]

    final_k = 8  # fixed number of topics for reporting
    final_k_index = k_values.index(final_k)
    final_model = lda_models[final_k_index]

    # === Plot coherence scores vs number of topics ===
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, coherence_scores, marker="o")
    plt.title("Topic Coherence vs Number of Topics")
    plt.xlabel("Number of Topics (k)")
    plt.ylabel("Coherence Score (c_v)")
    plt.grid(True)
    plt.savefig(plots_dir / "coherence_plot.png")
    plt.close()

    # === Plot topic diversity vs number of topics ===
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, diversity_scores, marker="o", color="green")
    plt.title("Topic Diversity vs Number of Topics")
    plt.xlabel("Number of Topics (k)")
    plt.ylabel("Topic Diversity (top 8 words)")
    plt.grid(True)
    plt.savefig(plots_dir / "diversity_plot.png")
    plt.close()

    # === Compute per-topic coherence for the final model ===
    coherence_model = CoherenceModel(
        model=final_model,
        texts=texts,
        dictionary=dictionary,
        coherence='c_v'
    )
    per_topic_scores = coherence_model.get_coherence_per_topic()

    results.append(f"\nTop terms per topic for k={final_k}:")

    # === List top terms and coherence for each topic ===
    for i in range(final_k):
        top_terms = ', '.join([t for t, _ in final_model.show_topic(i, topn=8)])
        score = per_topic_scores[i]
        results.append(f"Topic {i} (coherence = {score:.4f}): {top_terms}")

    # === Compute and report topic diversity for the final model ===
    topn = 8
    topic_words = [
        [word for word, _ in final_model.show_topic(i, topn=topn)]
        for i in range(final_k)
    ]
    unique_words = set(word for topic in topic_words for word in topic)
    topic_diversity = len(unique_words) / (final_k * topn)

    results.append(f"\nTopic Diversity (top {topn} words): {topic_diversity:.4f}")

    # === Assign topics to each tweet in the input file ===
    allowed_topics = {0, 1, 2, 3, 5}  # Only allow assignment to these topics
    final_test_df = pd.read_csv(input_path)
    test_texts = final_test_df["text_lemmatised"].fillna("").astype(str).apply(str.split)
    test_texts = test_texts.apply(lambda tokens: [w for w in tokens if w not in custom_stopwords])
    test_corpus = [dictionary.doc2bow(text) for text in test_texts]

    assigned_topics = []
    for dist in final_model.get_document_topics(test_corpus, minimum_probability=0.0):
        # Only consider allowed topics for assignment
        allowed_dist = [t for t in dist if t[0] in allowed_topics]
        best_topic = max(allowed_dist, key=lambda x: x[1])[0] if allowed_dist else -1
        assigned_topics.append(best_topic)

    # Add topic assignments to DataFrame and save
    final_test_df["lda_topic"] = assigned_topics
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    final_test_df.to_csv(output_path, index=False)

    results.append(f"\nEnriched file with topic assignments saved to: {output_path}")

    # === Write all results and summaries to a text file ===
    with open(txt_output, "w", encoding="utf-8") as f:
        f.write("\n".join(results))

    return str(output_path)
