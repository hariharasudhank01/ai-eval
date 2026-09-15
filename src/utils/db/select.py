from sqlalchemy.orm import sessionmaker
from .schema import GenerationModel, IterationModel, DocumentModel
from sqlalchemy import select
import sys
from utils.helpers.logger import get_logger, timed

logger = get_logger("db.select")

def fetch_iteration_id_using_gen_id(SessionLocal, gen_id):
    try:
        with SessionLocal() as session, timed(logger, f"fetch_iteration_id_using_gen_id({gen_id})"):
            stmt = (
                select(IterationModel).where(
                    IterationModel.gen_id == gen_id
                )
            )
            if not session.scalars(stmt):
                logger.error(f"{gen_id} not Found")
                sys.exit()
        return session.scalars(stmt)
    except Exception:
        logger.exception(f"Failed to fetch iterations for gen_id {gen_id}")
        raise

def fetch_documents_per_iteration(SessionLocal, iteration_id):
    try:
        with SessionLocal() as session, timed(logger, f"fetch_documents_per_iteration({iteration_id})"):
            stmt = (
                select(DocumentModel).where(
                    DocumentModel.iteration_id == iteration_id
                )
            )
            if not session.scalars(stmt):
                logger.error(f"{iteration_id} not Found")
                sys.exit()
        return session.scalars(stmt)
    except Exception:
        logger.exception(f"Failed to fetch documents for iteration_id {iteration_id}")
        raise

def initiate_sessionlocal(engine):
    return sessionmaker(
        bind=engine,
        autoflush=False,
        expire_on_commit=True
    )
