from sqlalchemy.orm import sessionmaker
from .schema import GenerationModel, IterationModel, DocumentModel
import sys

def update_gen_table(SessionLocal, pk, values):
    with SessionLocal() as session:
        gen_detail = session.get(GenerationModel, pk)
        if gen_detail:
            gen_detail.end_time = values["end_time"]
            print(f"Gen ID {pk} updated")
            session.commit()
        else:
            print(f"Gen ID {pk} not found")

def run(engine, table_name, pk, values):
    SessionLocal = sessionmaker(
        bind=engine,
        autoflush=False,
        expire_on_commit=True
    )
    if table_name == "generation":
        update_gen_table(SessionLocal, pk, values)