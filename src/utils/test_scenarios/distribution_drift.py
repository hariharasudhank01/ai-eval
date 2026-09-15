import numpy as np
from sentence_transformers import util
from utils.db import select, insert
from utils.test_scenarios.case_split import get_source_cases, embed, read_iteration_items
from utils.test_scenarios.tail_data_lose import adaptive_min_cluster_size, run_hdbscan
from utils.helpers.logger import get_logger, timed

logger = get_logger("test_scenarios.distribution_drift")

ASSIGNMENT_SIM_THRESHOLD = 0.6

def category_label_for(label):
    return "unassigned" if label == -1 else f"cluster_{label}"

def cluster_baseline(source_embeddings):
    if len(source_embeddings) == 0:
        return np.array([])
    min_cluster_size = adaptive_min_cluster_size(len(source_embeddings))
    return run_hdbscan(source_embeddings, min_cluster_size=min_cluster_size)

def compute_centroids(source_embeddings, labels):
    centroids = {}
    for label in set(labels):
        if label == -1:
            continue
        mask = labels == label
        centroids[category_label_for(label)] = source_embeddings[mask].mean(axis=0)
    return centroids

def proportions_from_labels(category_labels, total):
    if total == 0:
        return {}
    counts = {}
    for cat in category_labels:
        counts[cat] = counts.get(cat, 0) + 1
    return {cat: (count / total * 100) for cat, count in counts.items()}

def assign_items_to_categories(item_embeddings, centroids):
    if len(item_embeddings) == 0:
        return []
    if not centroids:
        return ["unassigned"] * len(item_embeddings)

    centroid_labels = list(centroids.keys())
    centroid_matrix = np.vstack([centroids[c] for c in centroid_labels])
    sims = util.cos_sim(item_embeddings, centroid_matrix).numpy()  # (n_items, n_centroids)
    best_idx = sims.argmax(axis=1)
    best_sim = sims.max(axis=1)

    assigned = []
    for idx, sim in zip(best_idx, best_sim):
        assigned.append(centroid_labels[idx] if sim >= ASSIGNMENT_SIM_THRESHOLD else "unassigned")
    return assigned

def total_variation_distance_pct(baseline_props, current_props):
    # baseline_props/current_props are already in percentage points (0-100, summing to ~100),
    # so 0.5 * sum(|diff|) here yields TVD directly as a 0-100 percentage - no extra scaling needed.
    all_categories = set(baseline_props) | set(current_props)
    return 0.5 * sum(abs(baseline_props.get(c, 0.0) - current_props.get(c, 0.0)) for c in all_categories)

def run(gen_id, engine, filename):
    logger.info(f"Running distribution_drift for gen_id {gen_id}")
    source_cases = get_source_cases(filename)
    source_embeddings = embed(source_cases)
    n_cases = len(source_cases)

    labels = cluster_baseline(source_embeddings)
    baseline_categories = [category_label_for(l) for l in labels]
    baseline_proportions = proportions_from_labels(baseline_categories, n_cases)
    centroids = compute_centroids(source_embeddings, labels)

    SessionLocal = select.initiate_sessionlocal(engine)
    iteration_details = select.fetch_iteration_id_using_gen_id(
        SessionLocal,
        gen_id
    ).all()

    iteration_id = sorted(iteration.iteration_id for iteration in iteration_details)
    file_path = [
        select.fetch_documents_per_iteration(SessionLocal, iid).all()[0].file_path
        for iid in iteration_id
    ]

    results = []

    for iid, fpath in zip(iteration_id, file_path):
        items = read_iteration_items(fpath)
        item_embeddings = embed(items)
        item_categories = assign_items_to_categories(item_embeddings, centroids)
        current_proportions = proportions_from_labels(item_categories, len(items))

        all_categories = set(baseline_proportions) | set(current_proportions)
        for category in all_categories:
            insert.run(engine, "distribution_drift", {
                "category_label": category,
                "baseline_proportion_pct": baseline_proportions.get(category, 0.0),
                "current_proportion_pct": current_proportions.get(category, 0.0),
                "iteration_id": iid,
            })

        drift_pct = total_variation_distance_pct(baseline_proportions, current_proportions)

        largest_shift_category = "none"
        largest_shift_pct = 0.0
        for category in all_categories:
            shift = abs(baseline_proportions.get(category, 0.0) - current_proportions.get(category, 0.0))
            if shift > largest_shift_pct:
                largest_shift_pct = shift
                largest_shift_category = category

        minority_candidates = {c: p for c, p in baseline_proportions.items() if c != "unassigned" and p > 0}
        if minority_candidates:
            minority_category = min(minority_candidates, key=minority_candidates.get)
            baseline_minority_pct = minority_candidates[minority_category]
            current_minority_pct = current_proportions.get(minority_category, 0.0)
            minority_retention_pct = (current_minority_pct / baseline_minority_pct * 100) if baseline_minority_pct > 0 else 0.0
        else:
            minority_category = "none"
            minority_retention_pct = 0.0

        summary = {
            "iteration_id": iid,
            "distribution_drift_pct": drift_pct, # Total Variation Distance between baseline and current category proportions
            "largest_shift_category": largest_shift_category,
            "largest_shift_pct": largest_shift_pct,
            "minority_category": minority_category, # smallest non-"unassigned" baseline category
            "minority_retention_pct": minority_retention_pct, # % of that category's original share that survives this iteration
        }
        results.append(summary)

        insert.run(engine, "distribution_drift_summary", summary)

    logger.debug(f"distribution_drift results for gen_id {gen_id}: {results}")
    return results
