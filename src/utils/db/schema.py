"""
This Script creates the table
"""
from datetime import datetime
from typing import Optional
from sqlalchemy import String, DateTime, Boolean, ForeignKey, Integer, Float, inspect
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from pgvector.sqlalchemy import Vector
import sys
from utils.helpers.logger import get_logger, timed

logger = get_logger("db.schema")


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

class ModelCollapseModel(Base):
    __tablename__ = "model_collapse"

    collapse_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True) # PK
    step_similarity: Mapped[float] = mapped_column(Float)
    baseline_drift: Mapped[float] = mapped_column(Float)

    # FK
    iteration_id: Mapped[int] = mapped_column(Integer, ForeignKey("iterations.iteration_id", ondelete="CASCADE"))   

class ModelDataLose(Base):
    __tablename__ = "datalose"

    lose_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True) # PK

    cumulative_loss: Mapped[float] = mapped_column(Float)
    outlier_ratio: Mapped[float] = mapped_column(Float)
    diversity_loss_from_baseline: Mapped[float] = mapped_column(Float)
    diversity_loss_from_previous_iteration: Mapped[float] = mapped_column(Float)

    # FK
    iteration_id: Mapped[int] = mapped_column(Integer, ForeignKey("iterations.iteration_id", ondelete="CASCADE"))

class ModelPIIEntities(Base):
    __tablename__ = "pii_entities"

    entity_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True) # PK
    entity_type: Mapped[str] = mapped_column(String, unique=True, nullable=False) # lookup vocabulary, get-or-create by this name

class ModelPII(Base):
    __tablename__ = "pii"

    pii_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True) # PK
    total_score: Mapped[float] = mapped_column(Float)

    # FK
    iteration_id: Mapped[int] = mapped_column(Integer, ForeignKey("iterations.iteration_id", ondelete="CASCADE"))

class ModelPIIEntityFinding(Base):
    __tablename__ = "pii_entity_findings"

    finding_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True) # PK
    text: Mapped[str] = mapped_column(String, nullable=False) # matched text span for this finding
    score: Mapped[float] = mapped_column(Float, nullable=False) # ensemble-averaged confidence for this finding
    source: Mapped[str] = mapped_column(String, nullable=False) # comma-joined detectors that agreed on this finding, e.g. "presidio,llm"

    # FK - junction table: one PII check (pii_id) can link to many entity types
    pii_id: Mapped[int] = mapped_column(Integer, ForeignKey("pii.pii_id", ondelete="CASCADE"))
    entity_id: Mapped[int] = mapped_column(Integer, ForeignKey("pii_entities.entity_id", ondelete="CASCADE"))

class ModelCaseCoverage(Base):
    __tablename__ = "case_coverage"

    coverage_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True) # PK
    case_label: Mapped[str] = mapped_column(String, nullable=False) # opaque label e.g. "case_1", or "unassigned" - never raw case text
    weightage_pct: Mapped[float] = mapped_column(Float, nullable=False)

    # FK - one row per (iteration_id, case_label), including a case_label="unassigned" row
    iteration_id: Mapped[int] = mapped_column(Integer, ForeignKey("iterations.iteration_id", ondelete="CASCADE"))

class ModelToxicityCategories(Base):
    __tablename__ = "toxicity_categories"

    category_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True) # PK
    category_name: Mapped[str] = mapped_column(String, unique=True, nullable=False) # lookup vocabulary, get-or-create by this name

class ModelToxicity(Base):
    __tablename__ = "toxicity"

    toxicity_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True) # PK
    total_score: Mapped[float] = mapped_column(Float, nullable=False)

    # FK
    iteration_id: Mapped[int] = mapped_column(Integer, ForeignKey("iterations.iteration_id", ondelete="CASCADE"))

class ModelToxicityFinding(Base):
    __tablename__ = "toxicity_findings"

    finding_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True) # PK
    text: Mapped[str] = mapped_column(String, nullable=False) # flagged sentence
    score: Mapped[float] = mapped_column(Float, nullable=False) # this detector's confidence for this finding
    source: Mapped[str] = mapped_column(String, nullable=False) # "detoxify" or "llm" - not required to agree with each other

    # FK - junction table: one toxicity check (toxicity_id) can link to many categories
    toxicity_id: Mapped[int] = mapped_column(Integer, ForeignKey("toxicity.toxicity_id", ondelete="CASCADE"))
    category_id: Mapped[int] = mapped_column(Integer, ForeignKey("toxicity_categories.category_id", ondelete="CASCADE"))

class ModelDistributionDrift(Base):
    __tablename__ = "distribution_drift"

    drift_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True) # PK
    category_label: Mapped[str] = mapped_column(String, nullable=False) # opaque, scoped to this run only e.g. "cluster_1" or "unassigned"
    baseline_proportion_pct: Mapped[float] = mapped_column(Float, nullable=False)
    current_proportion_pct: Mapped[float] = mapped_column(Float, nullable=False)

    # FK - one row per (iteration_id, category_label)
    iteration_id: Mapped[int] = mapped_column(Integer, ForeignKey("iterations.iteration_id", ondelete="CASCADE"))

class ModelDistributionDriftSummary(Base):
    __tablename__ = "distribution_drift_summary"

    summary_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True) # PK
    distribution_drift_pct: Mapped[float] = mapped_column(Float, nullable=False) # Total Variation Distance between baseline and current category proportions
    largest_shift_category: Mapped[str] = mapped_column(String, nullable=False)
    largest_shift_pct: Mapped[float] = mapped_column(Float, nullable=False)
    minority_category: Mapped[str] = mapped_column(String, nullable=False)
    minority_retention_pct: Mapped[float] = mapped_column(Float, nullable=False)

    # FK
    iteration_id: Mapped[int] = mapped_column(Integer, ForeignKey("iterations.iteration_id", ondelete="CASCADE"))



"""
This table will be dropped

class TestResultModel(Base):
    __tablename__ = "test_results"

    result_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True) #PK
    test_type: Mapped[str] = mapped_column(String, nullable=False)
    metric_name: Mapped[str] = mapped_column(String, nullable=False)
    value: Mapped[float] = mapped_column(Float, nullable=False)

    # FK
    iteration_id: Mapped[int] = mapped_column(Integer, ForeignKey("iterations.iteration_id", ondelete="CASCADE"))
    gen_id: Mapped[int] = mapped_column(Integer, ForeignKey("generations.gen_id", ondelete="CASCADE"))
"""

def verify_database(engine, tables):
    inspector = inspect(engine)
    existing_tables = inspector.get_table_names()
    all_present = True

    for table in tables:
        if table in existing_tables:
            logger.debug(f"{table} created successfully.")
        else:
            all_present = False
            logger.error(f"{table} not found")

    if not all_present:
        logger.error("Not all required tables are present - aborting.")
        sys.exit(0)

def create_table(engine):
    tables = ["generations", 
              "iterations", 
              "documents", 
             # "test_results", 
              "model_collapse",
              "datalose",
              "pii_entities",
              "pii",
              "pii_entity_findings",
              "case_coverage",
              "toxicity_categories",
              "toxicity",
              "toxicity_findings",
              "distribution_drift",
              "distribution_drift_summary"
              ]
    try:
        with timed(logger, "Table creation"):
            Base.metadata.create_all(bind=engine)
    except Exception:
        logger.exception("Failed to create database tables")
        raise
    verify_database(engine, tables)