TAIL_DATA_LOSE_METRICS = [
    "cumulative_loss_pct",
    "step_loss_pct",
    "outlier_ratio_pct",
    "diversity_loss_from_baseline_pct",
    "diversity_loss_from_previous_pct",
]

def unwrap(entry):
    return entry[0] if isinstance(entry, list) else entry # Mixed data type - dict - list of dict - so normalize

def flatten_model_collapse(gen_id, result):
    rows = []
    for entry in result:
        r = unwrap(entry)
        for metric_name in ("step_similarity", "baseline_drift_pct"):
            rows.append({
                "gen_id": gen_id,
                "iteration_id": r["iteration_id"],
                "test_type": "model_collapse",
                "metric_name": metric_name,
                "value": r[metric_name],
            })
    return rows

def flatten_tail_data_lose(gen_id, result):
    rows = []
    for entry in result:
        r = unwrap(entry)
        for metric_name in TAIL_DATA_LOSE_METRICS:
            rows.append({
                "gen_id": gen_id,
                "iteration_id": r["iteration_id"],
                "test_type": "tail_data_lose",
                "metric_name": metric_name,
                "value": r[metric_name],
            })
    return rows

def flatten_pii(gen_id, result):
    rows = []
    for r in result:
        rows.append({
            "gen_id": gen_id,
            "iteration_id": r["iteration_id"],
            "test_type": "pii",
            "metric_name": "total_score",
            "value": r["total_score"],
        })
        rows.append({
            "gen_id": gen_id,
            "iteration_id": r["iteration_id"],
            "test_type": "pii",
            "metric_name": "entity_count",
            "value": len(r["entities"]),
        })
    return rows

def flatten_case_split(gen_id, result):
    rows = []
    for r in result["per_iteration"]:
        iteration_id = r["iteration_id"]
        for case_label, pct in r["weightage_pct"].items():
            rows.append({
                "gen_id": gen_id,
                "iteration_id": iteration_id,
                "test_type": "case_split",
                "metric_name": f"weightage_{case_label}",
                "value": pct,
            })
        rows.append({
            "gen_id": gen_id,
            "iteration_id": iteration_id,
            "test_type": "case_split",
            "metric_name": "unassigned_pct",
            "value": r["unassigned_pct"],
        })
    return rows

def unify_results(gen_id, model_collapse_result, tail_data_lose_result, pii_result, case_split_result):
    rows = []
    rows.extend(flatten_model_collapse(gen_id, model_collapse_result))
    rows.extend(flatten_tail_data_lose(gen_id, tail_data_lose_result))
    rows.extend(flatten_pii(gen_id, pii_result))
    rows.extend(flatten_case_split(gen_id, case_split_result))
    for row in rows:
        row["value"] = float(row["value"])
    return rows

