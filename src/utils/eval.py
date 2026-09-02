"""
Script to handle the logic flow
"""
from utils.llm import olama
from pypdf import PdfReader
from datetime import datetime
from utils.db import schema, psqlcon, insert, update, select
from utils.test_scenarios import pii, model_collapse

engine = psqlcon.db_connect()

def update_table(table_name, PK, values):
    update.run(
        engine,
        table_name,
        PK,
        values
    )

def read_file(filename):
    if filename.endswith(".txt"):
        with open(filename, "r") as r:
            return r.readlines()
    else:
        reader = PdfReader(filename)
        for page in reader.pages:
            page_lines = page.extract_text().split("\n")
            return [line.strip() for line in page_lines if line.strip()]

def write_file(new_file, content):
    with open(new_file, "w", encoding="utf-8") as w:
        w.write(content)

def generate_content(filename):
    content = read_file(filename)
    merge_content = ""
    for con in content:
        merge_content = merge_content + con
    return merge_content

def generate_synthetic_data(prompt, content, iteration, model, output):

    gen_values = {
        "start_time": datetime.now().strftime("%Y-%m-%d %H:%M")
    }
    gen_id = insert.run(engine, "generation", gen_values)

    step_count = 1
    for i in range(0, iteration):
        model_name = model

        p = prompt + "\n" + content
        resp = olama.run(p, model)

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
    content = generate_content(user_input.filename)
    prompt = user_input.prompt + " " + content
    resp = olama.run(prompt, user_input.model)
    parent_data_embed = model_collapse.embedding(content)
    base_sim = model_collapse.similarity(parent_data_embed, model_collapse.embedding(resp.content))
    print(base_sim)

    gen_id = generate_synthetic_data(
        user_input.prompt,
        resp.content,
        user_input.iteration,
        user_input.model,
        user_input.output
    )

    model_collapse.check_model_collapse(gen_id, base_sim, engine)
            
