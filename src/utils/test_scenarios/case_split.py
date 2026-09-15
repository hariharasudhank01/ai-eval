from sentence_transformers import util
import numpy as np
from utils.helpers.text_split import read_file as read_source_file
from utils.db import select, insert
from utils.test_scenarios import model_collapse
from utils.helpers.logger import get_logger, timed

logger = get_logger("test_scenarios.case_split")

COVERAGE_SIM_THRESHOLD = 0.6

def get_source_cases(filename):
    content = read_source_file(filename)
    return [line.rstrip("\n").strip() for line in content if line.strip()]

def read_iteration_items(filename):
    try:
        with open(filename, "r") as file:
            text = file.read()
    except Exception:
        logger.exception(f"Failed to Read file {filename}")
        return []
    return [line.strip() for line in text.split("\n") if line.strip()]

def embed(texts):
    return np.asarray(model_collapse.embedding(texts)) if texts else np.empty((0, 384))

def run(gen_id, engine, filename):
    logger.info(f"Running case_split for gen_id {gen_id}")
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
    case_weightage_totals = [0.0] * n_cases # Creates list of size of identified cases
    per_iteration_results = []

    for iid, fpath in zip(iteration_id, file_path):
        items = read_iteration_items(fpath)
        item_embeddings = embed(items)
        total_items = len(items)

        case_counts = [0] * n_cases # Creates list of size of identified cases
        unassigned_count = 0

        if total_items > 0 and n_cases > 0:
            sims = util.cos_sim(item_embeddings, source_embeddings).numpy()  # (n_items, n_cases)
            best_case_idx = sims.argmax(axis=1) # Returns the Index of Max Values
            best_sim = sims.max(axis=1) # Returns the Max Value
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

        for case_label, pct in weightage_pct.items():
            insert.run(engine, "case_coverage", {
                "case_label": case_label,
                "weightage_pct": pct,
                "iteration_id": iid,
            })
        insert.run(engine, "case_coverage", {
            "case_label": "unassigned",
            "weightage_pct": unassigned_pct,
            "iteration_id": iid,
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
        "per_case_avg_weightage_pct": per_case_avg_weightage_pct,
        "per_iteration": per_iteration_results,
    }
    logger.debug(f"case_split results for gen_id {gen_id}: {result}")
    return result
