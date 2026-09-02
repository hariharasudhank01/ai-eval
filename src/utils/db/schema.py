"""
This Script creates the table
"""
from datetime import datetime
from typing import Optional
from sqlalchemy import String, DateTime, Boolean, ForeignKey, Integer, Float, inspect
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from pgvector.sqlalchemy import Vector
import sys


class Base(DeclarativeBase):
    pass

class GenerationModel(Base):
    __tablename__ = "generations"

    gen_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True) #PK
    start_time: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    end_time: Mapped[datetime] = mapped_column(DateTime, nullable=True)

class IterationModel(Base):
    __tablename__ = "iterations"

    iteration_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True) #PK
    step_count: Mapped[int] = mapped_column(Integer, nullable=False)
    model_name: Mapped[str] = mapped_column(String(36), nullable=True)
    prompt: Mapped[str] = mapped_column(String, nullable=False)

    #FK
    gen_id: Mapped[int] = mapped_column(Integer, ForeignKey("generations.gen_id", ondelete="CASCADE"))

class DocumentModel(Base):
    __tablename__ = "documents"

    doc_id: Mapped[int] = mapped_column(Integer, primary_key=True) # PK
    step_count: Mapped[int] = mapped_column(Integer, nullable=False)
    file_path: Mapped[str] = mapped_column(String, nullable=True)
    vector_value: Mapped[Optional[list]] = mapped_column(Vector(384), nullable=False)

    #FK
    dep_doc_id: Mapped[int] = mapped_column(Integer, ForeignKey("documents.doc_id", ondelete="CASCADE"), nullable=True)
    iteration_id: Mapped[int] = mapped_column(Integer, ForeignKey("iterations.iteration_id", ondelete="CASCADE"))

class TestResultModel(Base):
    __tablename__ = "test_results"

    result_id: Mapped[int] = mapped_column(Integer, primary_key=True) #PK
    test_type: Mapped[str] = mapped_column(String, nullable=False)
    value: Mapped[float] = mapped_column(Float, nullable=False)

    # FK
    doc_id: Mapped[int] = mapped_column(Integer, ForeignKey("documents.doc_id", ondelete="CASCADE"))
    iteration_id: Mapped[int] = mapped_column(Integer, ForeignKey("iterations.iteration_id", ondelete="CASCADE"))

def verify_database(engine, tables):
    inspector = inspect(engine)
    existing_tables = inspector.get_table_names()
    all_present = True

    for table in tables:
        if table in existing_tables:
            print(f"{table} created successfully.")
        else:
            all_present = False
            print(f"{table} not found")

    if not all_present:
        sys.exit(0)

def create_table(engine):
    tables = ["generations", "iterations", "documents", "test_results"]
    Base.metadata.create_all(bind=engine)
    verify_database(engine, tables)