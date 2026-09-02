"""
This Script handles the psql connection and creates pgvector
"""
from dotenv import load_dotenv
import os
import time
import sqlalchemy
import psycopg

load_dotenv()

def db_connect():
    engine = sqlalchemy.create_engine(os.getenv("POSTGRES_URL"), pool_pre_ping=True)
    for _ in range(1, 30 + 1):
        try:
            with engine.connect() as conn:
                conn.execute(sqlalchemy.text("CREATE EXTENSION IF NOT EXISTS vector;"))
                conn.execute(sqlalchemy.text("SELECT 1"))
                conn.commit()
                print("Database engine is ready.")
            break
        except sqlalchemy.exc.OperationalError as e:
            print(f"DB not ready yet: {e}")
            time.sleep(5)
    else:
        raise RuntimeError("Database engine was not ready after maximum retries.")
    return engine