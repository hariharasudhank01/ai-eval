from sqlalchemy.orm import sessionmaker
from .schema import GenerationModel, IterationModel, DocumentModel, TestResultModel
import sys

def insert_gen_table(SessionLocal, values):
    with SessionLocal() as session:
        new_gen = GenerationModel(
            start_time=values["start_time"]
        )
        session.add(new_gen)
        session.flush()
        id = new_gen.gen_id
        print(f"Generation {id} created in db")
        session.commit()
    return id

def insert_iteration_table(SessionLocal, values):
    with SessionLocal() as session:
        new_iteration = IterationModel(
            step_count=values["step_count"],
            model_name=values["model_name"],
            prompt=values["prompt"],
            gen_id=values["gen_id"]
        )
        session.add(new_iteration)
        session.flush()
        id = new_iteration.iteration_id
        print(f"Iteration {id} created in db")
        session.commit()
    return id 
     
def insert_document_table(SessionLocal, values):
    with SessionLocal() as session:
        new_document = DocumentModel(
            doc_id=values["doc_id"],
            step_count=values["step_count"],
            file_path=values["file_path"],
            vector_value=values["vector_value"],
            dep_doc_id=values["dep_doc_id"],
            iteration_id=values["iteration_id"]
        )
        session.add(new_document)
        session.flush()
        id = new_document.doc_id
        print(f"Document {id} created in db")
        session.commit()
    return id

def insert_results_table(SessionLocal, values):
    with SessionLocal() as session:
        new_result = TestResultModel(
            test_type=values["test_type"],
            metric_name=values["metric_name"],
            value=values["value"],
            iteration_id=values["iteration_id"],
            gen_id=values["gen_id"]
        )
        session.add(new_result)
        session.flush()
        id = new_result.result_id
        print(f"Result {id} stored in db")
        session.commit()
    return id 

def run(engine, table_name, values):
    SessionLocal = sessionmaker(
        bind=engine,
        autoflush=False,
        expire_on_commit=True
    )
    if table_name == "generation":
        id = insert_gen_table(SessionLocal, values)
        print(values)
    elif table_name == "iteration":
        id = insert_iteration_table(SessionLocal, values)
    elif table_name == "document":
        id = insert_document_table(SessionLocal, values)
    elif table_name == "result":
        id = insert_results_table(SessionLocal, values)
    else:
        print(f"{table_name} not found")
        sys.exit()
    return id