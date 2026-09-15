"""
This Script handles the psql connection and creates pgvector
"""
from dotenv import load_dotenv
import os
import time
import sqlalchemy
import psycopg
from utils.helpers.logger import get_logger, timed

load_dotenv()

logger = get_logger("db.psqlcon")

def db_connect():
    engine = sqlalchemy.create_engine(os.getenv("POSTGRES_URL"), pool_pre_ping=True)
    with timed(logger, "Database connection"):
        for attempt in range(1, 30 + 1):
            try:
                with engine.connect() as conn:
                    conn.execute(sqlalchemy.text("CREATE EXTENSION IF NOT EXISTS vector;"))
                    conn.execute(sqlalchemy.text("SELECT 1"))
                    conn.commit()
                    logger.info("Database engine is ready.")
                break
            except sqlalchemy.exc.OperationalError as e:
                logger.debug(f"DB not ready yet (attempt {attempt}/30): {e}")
                time.sleep(5)
        else:
            logger.error("Database engine was not ready after maximum retries.")
            raise RuntimeError("Database engine was not ready after maximum retries.")
    return engine
