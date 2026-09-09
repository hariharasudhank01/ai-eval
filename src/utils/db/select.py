from sqlalchemy.orm import sessionmaker
from .schema import GenerationModel, IterationModel, DocumentModel, TestResultModel
from sqlalchemy import select
import sys

def fetch_iteration_id_using_gen_id(SessionLocal, gen_id):
    with SessionLocal() as session:
        stmt = (
            select(IterationModel).where(
                IterationModel.gen_id == gen_id
            )
        )
        if not session.scalars(stmt):
            print(f"{gen_id} not Found")
            sys.exit()
    return session.scalars(stmt)

def fetch_documents_per_iteration(SessionLocal, iteration_id):
    with SessionLocal() as session:
        stmt = (
            select(DocumentModel).where(
                DocumentModel.iteration_id == iteration_id
            )
        )
        if not session.scalars(stmt):
            print(f"{iteration_id} not Found")
            sys.exit()
    return session.scalars(stmt)

def fetch_test_results(SessionLocal, gen_id):
    with SessionLocal() as session:
        stmt = (
            select(TestResultModel).where(
                TestResultModel.gen_id == gen_id
            )
        )
        if not session.scalars(stmt):
            print(f"{gen_id} not Found")
            sys.exit()
    return session.scalars(stmt)       

def initiate_sessionlocal(engine):
    return sessionmaker(
        bind=engine,
        autoflush=False,
        expire_on_commit=True
    )