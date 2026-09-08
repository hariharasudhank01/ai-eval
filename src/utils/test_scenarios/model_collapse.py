from sentence_transformers import SentenceTransformer, util
from utils.db import select

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2') # SBERT Model

def embedding(value):
    return model.encode(value, convert_to_numpy=True)

def similarity(value1, value2):
    sim = util.cos_sim(value1, value2)
    return sim.item()

def run(gen_id, base_sim, engine):
    SessionLocal = select.initiate_sessionlocal(engine)
    iteration_details = select.fetch_iteration_id_using_gen_id(
        SessionLocal,
        gen_id
    ).all()

    iteration_id = []
    for iteration in iteration_details:
        iteration_id.append(iteration.iteration_id)

    iteration_id.sort()
    vector = {}
    vector[iteration_id[0]] = select.fetch_documents_per_iteration(
            SessionLocal,
            iteration_id[0]
        ).all()[0].vector_value
    
    for i in range(1, len(iteration_id)):
        vector[iteration_id[i]] = select.fetch_documents_per_iteration(
            SessionLocal,
            iteration_id[i]
        ).all()[0].vector_value

    vector_list = list(vector.values())

    relative_change = []
    sim = []
    sim.append(base_sim)
    for i in range(1, len(vector_list)):
        current_vector = vector_list[i]
        previous_vector = vector_list[i - 1]
        
        similarity_score = similarity(current_vector, previous_vector)
        sim.append(similarity_score)

        #relative_change.append(((similarity_score - base_sim) / base_sim))

    print(sim)

