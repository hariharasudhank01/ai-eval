from sqlalchemy.orm import sessionmaker
from .schema import GenerationModel, IterationModel, DocumentModel
import sys
from utils.helpers.logger import get_logger, timed

logger = get_logger("db.update")

def update_gen_table(SessionLocal, pk, values):
    with SessionLocal() as session:
        gen_detail = session.get(GenerationModel, pk)
        if gen_detail:
            gen_detail.end_time = values["end_time"]
            logger.debug(f"Gen ID {pk} updated")
            session.commit()
        else:
            logger.error(f"Gen ID {pk} not found")

def run(engine, table_name, pk, values):
    SessionLocal = sessionmaker(
        bind=engine,
        autoflush=False,
        expire_on_commit=True
    )
    try:
        with timed(logger, f"Update '{table_name}' pk={pk}"):
            if table_name == "generation":
                update_gen_table(SessionLocal, pk, values)
            else:
                logger.error(f"{table_name} not found")
                sys.exit()
    except Exception:
        logger.exception(f"Failed to update '{table_name}' pk={pk} with values {values}")
        raise
