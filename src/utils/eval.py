"""
Script to handle the logic flow
"""
from utils.llm import llm
from datetime import datetime
from utils.db import schema, psqlcon, insert, update, select
from utils.helpers.text_split import read_file
from utils.test_scenarios import pii, model_collapse, tail_data_lose, case_split, toxicity, distribution_drift
from utils.helpers.logger import get_logger, timed, RunLogCapture

logger = get_logger("eval")

engine = psqlcon.db_connect()

def update_table(table_name, PK, values):
    update.run(
        engine,
        table_name,
        PK,
        values
    )

def write_file(new_file, content):
    try:
        with open(new_file, "w", encoding="utf-8") as w:
            w.write(content)
    except Exception:
        logger.exception(f"Failed to write file '{new_file}'")
        raise

def generate_content(filename):
    content = read_file(filename)
    merge_content = "\n".join(line.rstrip("\n") for line in content)
    case_count = len(content)
    logger.debug(f"Loaded {case_count} case(s) from '{filename}'")
    return merge_content, case_count

def generate_synthetic_data(prompt, content, case_count, iteration, model, output, provider):

    gen_values = {
        "start_time": datetime.now().strftime("%Y-%m-%d %H:%M")
    }
    gen_id = insert.run(engine, "generation", gen_values)
    logger.info(f"Generation {gen_id} started - {iteration} iteration(s) with model '{provider}/{model}'")

    step_count = 1
    with timed(logger, f"Generation {gen_id} - all iterations"):
        for i in range(0, iteration):
            model_name = model

            p = prompt + "\n" + content

            with timed(logger, f"Iteration {step_count}/{iteration}"):
                resp = llm.run(p, model, case_count, provider)

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
    logger.info(f"Generation {gen_id} complete")
    return gen_id

def _run_test_scenario(name, func, *args):
    try:
        with timed(logger, f"Test scenario '{name}'"):
            result = func(*args)
        logger.info(f"Test scenario '{name}' completed")
        return result
    except Exception:
        logger.exception(f"Test scenario '{name}' failed - continuing with remaining scenarios")
        return None

def run(user_input):
    run_log = RunLogCapture()
    try:
        schema.create_table(engine)

        # Parent
        content, case_count = generate_content(user_input.filename)
        prompt = user_input.prompt + " " + content
        resp = llm.run(prompt, user_input.model, case_count, user_input.provider)
        parent_data_embed = model_collapse.embedding(content)
        base_sim = model_collapse.similarity(parent_data_embed, model_collapse.embedding(resp.content))
        logger.debug(f"Baseline source-to-first-generation similarity: {base_sim}")

        # gen_id only exists once generation has actually succeeded - nothing
        # catches exceptions here, so a failure propagates to the caller (the
        # future API layer) instead of ever reaching this point.
        gen_id = generate_synthetic_data(
            user_input.prompt,
            resp.content,
            case_count,
            user_input.iteration,
            user_input.model,
            user_input.output,
            user_input.provider
        )
        run_log.attach_gen_id(gen_id)
    except Exception:
        logger.exception("AI Eval run failed before generation completed")
        run_log.close()
        raise

    try:
        # Each test scenario writes its own results directly to its dedicated table.
        # A failure in one scenario is logged and does not stop the others from
        # running, and does not affect the gen_id already returned to the caller.
        collapse_result = _run_test_scenario("model_collapse", model_collapse.run, gen_id, base_sim, engine)
        data_lose_result = _run_test_scenario("tail_data_lose", tail_data_lose.run, gen_id, engine, user_input.output)
        pii_result = _run_test_scenario("pii", pii.run, gen_id, engine)
        case_split_result = _run_test_scenario("case_split", case_split.run, gen_id, engine, user_input.filename)
        toxicity_result = _run_test_scenario("toxicity", toxicity.run, gen_id, engine)
        distribution_drift_result = _run_test_scenario("distribution_drift", distribution_drift.run, gen_id, engine, user_input.filename)

        logger.info(
            f"AI Eval Results for model {user_input.provider}/{user_input.model} is as follows,\n\n"
            f" Generation ID {gen_id} \n\n"
            f" Model Collapse {collapse_result} \n\n"
            f" Data Lose {data_lose_result} \n\n"
            f" PII {pii_result} \n\n"
            f" Case Split {case_split_result} \n\n"
            f" Toxicity {toxicity_result} \n\n"
            f" Distribution Drift {distribution_drift_result}"
        )
    finally:
        run_log.close()

    return gen_id
