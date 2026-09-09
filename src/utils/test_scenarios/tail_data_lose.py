import os
from sentence_transformers import util
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from utils.db import select
from utils.test_scenarios import model_collapse
import re

K = 3                 # upper cap on HDBSCAN min_cluster_size / min_samples
MIN_CLUSTER_SIZE_FLOOR = 2   # smallest usable cluster size on very small runs

def embed_sentences(sentences):
    return np.asarray(model_collapse.embedding(sentences))

def adaptive_min_cluster_size(total_points, k=K):
    return max(MIN_CLUSTER_SIZE_FLOOR, min(k, total_points // 10))

def run_hdbscan(embeddings, min_cluster_size):
    db = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_cluster_size, metric="cosine")
    return db.fit_predict(embeddings)

def plot_clusters_2d(embeddings, labels, save_path):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(embeddings)

    plt.figure()
    noise_mask = labels == -1
    plt.scatter(
        reduced[~noise_mask, 0], reduced[~noise_mask, 1],
        c=labels[~noise_mask], cmap="tab10", marker="o", label="clustered"
    )
    plt.scatter(
        reduced[noise_mask, 0], reduced[noise_mask, 1],
        c="black", marker="x", label="noise/outlier"
    )
    plt.xlabel("PCA component 1")
    plt.ylabel("PCA component 2")
    plt.title("HDBSCAN clusters (PCA projection)")
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def read_file(filename):
    try:
        with open(filename, "r") as file:
            text = file.read()
    except Exception as e:
        print(f"Failed to Read file {filename}, {e}")
        return []

    sentences = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        for sentence in re.split(r'(?<=[.!?])\s+', line):
            sentence = sentence.strip()
            if sentence:
                sentences.append(sentence)
    return sentences

def _mean_pairwise_distance(embeddings):
    n = len(embeddings)
    if n < 2:
        return 0.0
    sims = util.cos_sim(embeddings, embeddings)
    mean_sim = (sims.sum().item() - n) / (n * (n - 1))  # exclude self-similarity diagonal
    return 1 - mean_sim

def diversity_loss_pct(reference_embeddings, compare_embeddings):
    reference_spread = _mean_pairwise_distance(reference_embeddings)
    compare_spread = _mean_pairwise_distance(compare_embeddings)
    if reference_spread <= 0:
        return 0.0
    return max(0.0, (reference_spread - compare_spread) / reference_spread) * 100

def run(gen_id, engine, output):

    # Connect with DB and get the iteration details
    SessionLocal = select.initiate_sessionlocal(engine)
    iteration_details = select.fetch_iteration_id_using_gen_id(
        SessionLocal,
        gen_id
    ).all()

    iteration_id = []
    for iteration in iteration_details:
        iteration_id.append(iteration.iteration_id)

    iteration_id.sort()
    file_path = []

    for i in range(len(iteration_id)):
        file_path.append(select.fetch_documents_per_iteration(
            SessionLocal,
            iteration_id[i]
        ).all()[0].file_path)


    per_iteration_embeddings = []

    for file in file_path:
        sentences = read_file(file)
        embeddings = embed_sentences(sentences) if sentences else np.empty((0, 384))
        per_iteration_embeddings.append(embeddings)

    all_embeddings = np.vstack(per_iteration_embeddings) if per_iteration_embeddings else np.empty((0, 384))
    all_iteration_index = np.concatenate([
        np.full(len(emb), idx) for idx, emb in enumerate(per_iteration_embeddings)
    ]) if per_iteration_embeddings else np.empty((0,), dtype=int)

    min_cluster_size = adaptive_min_cluster_size(len(all_embeddings))
    labels = run_hdbscan(all_embeddings, min_cluster_size=min_cluster_size)

    if len(all_embeddings) > 0:
        save_path = output + "gen" + str(gen_id) + "_clusters.png"
        plot_clusters_2d(all_embeddings, labels, save_path)

    
    baseline_embeddings = per_iteration_embeddings[0] if per_iteration_embeddings else np.empty((0, 384))

    data_lose = []
    for idx in range(1, len(iteration_id)):
        results = []
        this_embeddings = per_iteration_embeddings[idx]
        previous_embeddings = per_iteration_embeddings[idx - 1]

        this_mask = all_iteration_index == idx
        this_labels = labels[this_mask]
        outlier_ratio_pct = (
            (this_labels == -1).sum() / len(this_labels) * 100
            if len(this_labels) > 0 else 0.0
        )

        diversity_loss_from_baseline_pct = diversity_loss_pct(baseline_embeddings, this_embeddings)
        diversity_loss_from_previous_pct = diversity_loss_pct(previous_embeddings, this_embeddings)

        cumulative_loss_pct = (outlier_ratio_pct + diversity_loss_from_baseline_pct) / 2
        step_loss_pct = diversity_loss_from_previous_pct

        results.append({
            "iteration_id": iteration_id[idx],
            "cumulative_loss_pct": cumulative_loss_pct, # Average of outlier_ration and diversity_loss_from_baseline
            "step_loss_pct": step_loss_pct, # Same as diversity loss from previous
            "outlier_ratio_pct": outlier_ratio_pct, # How unlike it is from other iterations, based on the result from HDBSCAN
            "diversity_loss_from_baseline_pct": diversity_loss_from_baseline_pct, # How much data lost or narrowed from baseline
            "diversity_loss_from_previous_pct": diversity_loss_from_previous_pct, # How much data lost or narrowed from prev run
        })
        data_lose.append(results)

    return data_lose
