from sentence_transformers import util
import numpy as np
from utils.text_split import read_file as read_source_file
from utils.db import select
from utils.test_scenarios import model_collapse

COVERAGE_SIM_THRESHOLD = 0.6

def get_source_cases(filename):
    content = read_source_file(filename)
    return [line.rstrip("\n").strip() for line in content if line.strip()]

def read_iteration_items(filename):
    try:
        with open(filename, "r") as file:
            text = file.read()
    except Exception as e:
        print(f"Failed to Read file {filename}, {e}")
        return []
    return [line.strip() for line in text.split("\n") if line.strip()]

def embed(texts):
    return np.asarray(model_collapse.embedding(texts)) if texts else np.empty((0, 384))

def run(gen_id, engine, filename):
    source_cases = get_source_cases(filename)
    source_embeddings = embed(source_cases)
    n_cases = len(source_cases)

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

    case_labels = [f"case_{i+1}" for i in range(n_cases)]
    case_weightage_totals = [0.0] * n_cases
    per_iteration_results = []

    for iid, fpath in zip(iteration_id, file_path):
        items = read_iteration_items(fpath)
        item_embeddings = embed(items)
        total_items = len(items)

        case_counts = [0] * n_cases
        unassigned_count = 0

        if total_items > 0 and n_cases > 0:
            sims = util.cos_sim(item_embeddings, source_embeddings).numpy()  # (n_items, n_cases)
            best_case_idx = sims.argmax(axis=1)
            best_sim = sims.max(axis=1)
            for idx, sim in zip(best_case_idx, best_sim):
                if sim >= COVERAGE_SIM_THRESHOLD:
                    case_counts[idx] += 1
                else:
                    unassigned_count += 1
        else:
            unassigned_count = total_items

        weightage_pct = {}
        for i in range(n_cases):
            pct = (case_counts[i] / total_items * 100) if total_items > 0 else 0.0
            weightage_pct[case_labels[i]] = pct
            case_weightage_totals[i] += pct

        unassigned_pct = (unassigned_count / total_items * 100) if total_items > 0 else 0.0

        per_iteration_results.append({
            "iteration_id": iid,
            "weightage_pct": weightage_pct, # % of this iteration's generated items assigned to each source case
            "unassigned_pct": unassigned_pct, # % of items with no good match to any source case
        })

    total_iterations = len(iteration_id)
    per_case_avg_weightage_pct = {
        case_labels[i]: (
            case_weightage_totals[i] / total_iterations
            if total_iterations > 0 else 0.0
        ) # average share of generated volume this case received, across the whole run
        for i in range(n_cases)
    }

    result = {
        "cases": dict(zip(case_labels, source_cases)), # legend: case label -> full source text, listed once
        "per_case_avg_weightage_pct": per_case_avg_weightage_pct,
        "per_iteration": per_iteration_results,
    }
    return result
