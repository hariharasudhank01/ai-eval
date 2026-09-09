from sentence_transformers import SentenceTransformer, util
from utils.db import select
import dotenv

dotenv.load_dotenv()

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2') # SBERT Model

def embedding(value):
    return model.encode(value, convert_to_numpy=True)

def similarity(value1, value2):
    sim = util.cos_sim(value1, value2)
    return sim.item()

def drift_pct(reference_vector, compare_vector):
    return max(0.0, (1 - similarity(reference_vector, compare_vector))) * 100

def run(gen_id, base_sim, engine):
    SessionLocal = select.initiate_sessionlocal(engine)
    iteration_details = select.fetch_iteration_id_using_gen_id(
        SessionLocal,
        gen_id
    ).all()

    iteration_id = sorted(iteration.iteration_id for iteration in iteration_details)

    vector_list = [
        select.fetch_documents_per_iteration(SessionLocal, iid).all()[0].vector_value
        for iid in iteration_id
    ]

    baseline_vector = vector_list[0] if vector_list else None

    collapse = []
    for i in range(1, len(vector_list)):
        results = []
        step_similarity = similarity(vector_list[i], vector_list[i - 1])
        baseline_drift_pct = drift_pct(baseline_vector, vector_list[i])

        results.append({
            "iteration_id": iteration_id[i],
            "step_similarity": step_similarity, # How similar this iteration's output is to the immediately preceding one - trending toward 1.0 signals collapse
            "baseline_drift_pct": baseline_drift_pct, # How far this iteration has drifted from iteration 1's output, regardless of direction
        })
        collapse.append(results)

    #print({"source_to_first_gen_similarity": base_sim, "iterations": results})
    return collapse
