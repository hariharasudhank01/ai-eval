from sqlalchemy.orm import sessionmaker
from sqlalchemy import select
from .schema import (
    GenerationModel,
    IterationModel,
    DocumentModel,
    ModelCollapseModel,
    ModelDataLose,
    ModelPII,
    ModelPIIEntities,
    ModelPIIEntityFinding,
    ModelCaseCoverage,
    ModelToxicityCategories,
    ModelToxicity,
    ModelToxicityFinding,
    ModelDistributionDrift,
    ModelDistributionDriftSummary,
)
import sys
from utils.helpers.logger import get_logger, timed

logger = get_logger("db.insert")

def insert_gen_table(SessionLocal, values):
    with SessionLocal() as session:
        new_gen = GenerationModel(
            start_time=values["start_time"]
        )
        session.add(new_gen)
        session.flush()
        id = new_gen.gen_id
        logger.debug(f"Generation {id} created in db")
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
        logger.debug(f"Iteration {id} created in db")
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
        logger.debug(f"Document {id} created in db")
        session.commit()
    return id

def insert_model_collapse_table(SessionLocal, values):
    with SessionLocal() as session:
        new_collapse = ModelCollapseModel(
            step_similarity=values["step_similarity"],
            baseline_drift=values["baseline_drift"],
            iteration_id=values["iteration_id"]
        )
        session.add(new_collapse)
        session.flush()
        id = new_collapse.collapse_id
        logger.debug(f"Model collapse result {id} stored in db")
        session.commit()
    return id

def insert_datalose_table(SessionLocal, values):
    with SessionLocal() as session:
        new_lose = ModelDataLose(
            cumulative_loss=values["cumulative_loss"],
            outlier_ratio=values["outlier_ratio"],
            diversity_loss_from_baseline=values["diversity_loss_from_baseline"],
            diversity_loss_from_previous_iteration=values["diversity_loss_from_previous_iteration"],
            iteration_id=values["iteration_id"]
        )
        session.add(new_lose)
        session.flush()
        id = new_lose.lose_id
        logger.debug(f"Data lose result {id} stored in db")
        session.commit()
    return id

def insert_pii_table(SessionLocal, values):
    with SessionLocal() as session:
        new_pii = ModelPII(
            total_score=values["total_score"],
            iteration_id=values["iteration_id"]
        )
        session.add(new_pii)
        session.flush()
        id = new_pii.pii_id
        logger.debug(f"PII result {id} stored in db")
        session.commit()
    return id

def get_or_create_pii_entity(SessionLocal, values):
    entity_type = values["entity_type"]
    with SessionLocal() as session:
        stmt = select(ModelPIIEntities).where(ModelPIIEntities.entity_type == entity_type)
        existing = session.scalars(stmt).first()
        if existing:
            return existing.entity_id

        new_entity = ModelPIIEntities(entity_type=entity_type)
        session.add(new_entity)
        session.flush()
        id = new_entity.entity_id
        logger.debug(f"PII entity type '{entity_type}' created in db")
        session.commit()
    return id

def insert_pii_entity_finding_table(SessionLocal, values):
    with SessionLocal() as session:
        new_finding = ModelPIIEntityFinding(
            pii_id=values["pii_id"],
            entity_id=values["entity_id"],
            text=values["text"],
            score=values["score"],
            source=values["source"]
        )
        session.add(new_finding)
        session.flush()
        id = new_finding.finding_id
        logger.debug(f"PII entity finding {id} stored in db")
        session.commit()
    return id

def insert_case_coverage_table(SessionLocal, values):
    with SessionLocal() as session:
        new_coverage = ModelCaseCoverage(
            case_label=values["case_label"],
            weightage_pct=values["weightage_pct"],
            iteration_id=values["iteration_id"]
        )
        session.add(new_coverage)
        session.flush()
        id = new_coverage.coverage_id
        logger.debug(f"Case coverage {id} stored in db")
        session.commit()
    return id

def insert_toxicity_table(SessionLocal, values):
    with SessionLocal() as session:
        new_toxicity = ModelToxicity(
            total_score=values["total_score"],
            iteration_id=values["iteration_id"]
        )
        session.add(new_toxicity)
        session.flush()
        id = new_toxicity.toxicity_id
        logger.debug(f"Toxicity result {id} stored in db")
        session.commit()
    return id

def get_or_create_toxicity_category(SessionLocal, values):
    category_name = values["category_name"]
    with SessionLocal() as session:
        stmt = select(ModelToxicityCategories).where(ModelToxicityCategories.category_name == category_name)
        existing = session.scalars(stmt).first()
        if existing:
            return existing.category_id

        new_category = ModelToxicityCategories(category_name=category_name)
        session.add(new_category)
        session.flush()
        id = new_category.category_id
        logger.debug(f"Toxicity category '{category_name}' created in db")
        session.commit()
    return id

def insert_toxicity_finding_table(SessionLocal, values):
    with SessionLocal() as session:
        new_finding = ModelToxicityFinding(
            toxicity_id=values["toxicity_id"],
            category_id=values["category_id"],
            text=values["text"],
            score=values["score"],
            source=values["source"]
        )
        session.add(new_finding)
        session.flush()
        id = new_finding.finding_id
        logger.debug(f"Toxicity finding {id} stored in db")
        session.commit()
    return id

def insert_distribution_drift_table(SessionLocal, values):
    with SessionLocal() as session:
        new_drift = ModelDistributionDrift(
            category_label=values["category_label"],
            baseline_proportion_pct=values["baseline_proportion_pct"],
            current_proportion_pct=values["current_proportion_pct"],
            iteration_id=values["iteration_id"]
        )
        session.add(new_drift)
        session.flush()
        id = new_drift.drift_id
        logger.debug(f"Distribution drift {id} stored in db")
        session.commit()
    return id

def insert_distribution_drift_summary_table(SessionLocal, values):
    with SessionLocal() as session:
        new_summary = ModelDistributionDriftSummary(
            distribution_drift_pct=values["distribution_drift_pct"],
            largest_shift_category=values["largest_shift_category"],
            largest_shift_pct=values["largest_shift_pct"],
            minority_category=values["minority_category"],
            minority_retention_pct=values["minority_retention_pct"],
            iteration_id=values["iteration_id"]
        )
        session.add(new_summary)
        session.flush()
        id = new_summary.summary_id
        logger.debug(f"Distribution drift summary {id} stored in db")
        session.commit()
    return id

_TABLE_HANDLERS = {
    "generation": insert_gen_table,
    "iteration": insert_iteration_table,
    "document": insert_document_table,
    "model_collapse": insert_model_collapse_table,
    "datalose": insert_datalose_table,
    "pii": insert_pii_table,
    "pii_entity": get_or_create_pii_entity,
    "pii_entity_finding": insert_pii_entity_finding_table,
    "case_coverage": insert_case_coverage_table,
    "toxicity": insert_toxicity_table,
    "toxicity_category": get_or_create_toxicity_category,
    "toxicity_finding": insert_toxicity_finding_table,
    "distribution_drift": insert_distribution_drift_table,
    "distribution_drift_summary": insert_distribution_drift_summary_table,
}

def run(engine, table_name, values):
    handler = _TABLE_HANDLERS.get(table_name)
    if handler is None:
        logger.error(f"{table_name} not found")
        sys.exit()

    SessionLocal = sessionmaker(
        bind=engine,
        autoflush=False,
        expire_on_commit=True
    )

    try:
        with timed(logger, f"Insert into '{table_name}'"):
            id = handler(SessionLocal, values)
    except Exception:
        logger.exception(f"Failed to insert into '{table_name}' with values {values}")
        raise

    return id
