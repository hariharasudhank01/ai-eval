"""
Script to handle the logic flow
"""
from utils.llm import olama
from datetime import datetime
from utils.db import schema, psqlcon, insert, update, select
from utils.text_split import read_file
from utils.test_scenarios import pii, model_collapse, tail_data_lose, case_split
from utils.test_scenarios.test_results import unify_results

engine = psqlcon.db_connect()

def update_table(table_name, PK, values):
    update.run(
        engine,
        table_name,
        PK,
        values
    )

def write_file(new_file, content):
    with open(new_file, "w", encoding="utf-8") as w:
        w.write(content)

def generate_content(filename):
    content = read_file(filename)
    merge_content = "\n".join(line.rstrip("\n") for line in content)
    case_count = len(content)
    return merge_content, case_count

def generate_synthetic_data(prompt, content, case_count, iteration, model, output):

    gen_values = {
        "start_time": datetime.now().strftime("%Y-%m-%d %H:%M")
    }
    gen_id = insert.run(engine, "generation", gen_values)

    step_count = 1
    for i in range(0, iteration):
        model_name = model

        p = prompt + "\n" + content
        resp = olama.run(p, model, case_count)

        iteration_values = {
            "step_count": step_count,
            "model_name": model_name,
            "prompt": p,
            "gen_id": gen_id
        }

        iteration_id = insert.run(engine, "iteration", iteration_values)

        new_file_path = output + ("gen"+str(gen_id)+"doc"+str(step_count)) + ".txt"
        write_file(new_file_path, resp.content)

        if step_count == 1:
            dep_doc_id = iteration_id
        else:
            dep_doc_id = iteration_id - 1

        doc_values = {
            "doc_id": iteration_id,
            "step_count": step_count,
            "file_path": new_file_path,
            "vector_value": model_collapse.embedding(resp.content),
            "dep_doc_id": dep_doc_id,
            "iteration_id": iteration_id
        }

        doc_id = insert.run(engine, "document", doc_values)
        step_count += 1

    update_table(
        "generation",
        gen_id,
        {"end_time": datetime.now().strftime("%Y-%m-%d %H:%M")}
    )
    return gen_id

def run(user_input):
    schema.create_table(engine)

    # Parent
    content, case_count = generate_content(user_input.filename)
    prompt = user_input.prompt + " " + content
    resp = olama.run(prompt, user_input.model, case_count)
    parent_data_embed = model_collapse.embedding(content)
    base_sim = model_collapse.similarity(parent_data_embed, model_collapse.embedding(resp.content))
    print(base_sim)

    gen_id = generate_synthetic_data(
        user_input.prompt,
        resp.content,
        case_count,
        user_input.iteration,
        user_input.model,
        user_input.output
    )

    collapse_result = model_collapse.run(gen_id, base_sim, engine)
    data_lose_result = tail_data_lose.run(gen_id, engine, user_input.output)
    pii_result = pii.run(gen_id, engine)
    case_split_result = case_split.run(gen_id, engine, user_input.filename)


    #print(f"AI Eval Results for model {user_input.model} is as follows,\n\n Generation ID {gen_id} \n\n Model Collapse {collapse_result} \n\n Data Lose {data_lose_result} \n\n PII {pii_result} \n\n Case Split {case_split_result}")

    results = unify_results(gen_id, collapse_result, data_lose_result, pii_result, case_split_result)
    id = []
    for result in results:
        values = {
            "test_type": result["test_type"],
            "metric_name": result["metric_name"],
            "value": result["value"],
            "iteration_id": result["iteration_id"],
            "gen_id": result["gen_id"]
        }
        result_id = insert.run(
            engine,
            "result",
            values
        )
        id.append(result_id)


            
