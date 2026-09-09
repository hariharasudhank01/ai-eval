import re
import ollama
from pydantic import BaseModel
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from utils.db import select

configuration = {
    "nlp_engine_name": "spacy",
    "models": [
        {"lang_code": "en", "model_name": "en_core_web_lg"}
    ],
}

provider = NlpEngineProvider(nlp_configuration=configuration)
nlp_engine = provider.create_engine()

analyzer = AnalyzerEngine(
    nlp_engine=nlp_engine,
    supported_languages=["en"]
)

PII_RISK_CAP = 5

LOCATION_DENYLIST = {
    "city", "town", "office", "downtown", "area", "region", "place",
    "location", "neighborhood", "district", "building", "street", "road",
    "state", "country"
}

RELATIVE_DATE_DENYLIST = {
    "soon", "recently", "last week", "next week", "yesterday", "today",
    "tomorrow", "last month", "next month", "later", "now"
}

MONTHS = (
    "january|february|march|april|may|june|july|august|september|"
    "october|november|december"
)

ABSOLUTE_DATE_PATTERNS = [
    re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"),
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
    re.compile(rf"\b(?:{MONTHS})\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,?\s+\d{{4}})?\b", re.IGNORECASE),
    re.compile(rf"\b\d{{1,2}}\s+(?:{MONTHS})(?:\s+\d{{4}})?\b", re.IGNORECASE),
]

REGEX_PATTERNS = {
    "EMAIL_ADDRESS": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "PHONE_NUMBER": re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b"),
    "US_SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "CREDIT_CARD": re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
    "IP_ADDRESS": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
}

PII_LLM_SYSTEM_PROMPT = """
Role:
- PII detection reviewer.
Task:
- Identify any personally identifiable information in the text provided, including
  information that is only indirectly or contextually identifying (e.g. a person
  described by a unique combination of role, employer, and location, even without
  a name).
Action:
- Return each PII instance as an entity_type (e.g. PERSON, LOCATION, EMAIL_ADDRESS,
  PHONE_NUMBER, DATE_TIME, ORGANIZATION, INDIRECT_IDENTIFIER, OTHER) and the exact
  text span from the input that triggered it.
- Do not flag generic nouns, generic place references (e.g. "the office", "downtown"),
  or vague/relative time references (e.g. "last week", "soon") as PII.
- Only flag information that could reasonably identify a specific real individual or
  their private details.
- If no PII is found, return an empty entities list.
"""

class PiiEntity(BaseModel):
    entity_type: str
    text: str

class PiiLlmResponse(BaseModel):
    entities: list[PiiEntity]

def read_file(filename):
    try:
        with open(filename, "r") as file:
            return file.read()
    except Exception as e:
        print(f"Failed to Read file {filename}, {e}")
    return ""

def _is_valid_location(matched_text):
    normalized = matched_text.strip().lower()
    if normalized in LOCATION_DENYLIST:
        return False
    for article in ("the ", "a ", "an "):
        if normalized.startswith(article) and normalized[len(article):] in LOCATION_DENYLIST:
            return False
    return True

def _is_valid_date_time(matched_text):
    normalized = matched_text.strip().lower()
    if normalized in RELATIVE_DATE_DENYLIST:
        return False
    return any(pattern.search(matched_text) for pattern in ABSOLUTE_DATE_PATTERNS)

def _is_valid_entity(entity_type, matched_text):
    if entity_type == "LOCATION":
        return _is_valid_location(matched_text)
    if entity_type == "DATE_TIME":
        return _is_valid_date_time(matched_text)
    return True

def check_pii(text):
    results = analyzer.analyze(text=text, language="en")
    entities = []
    for res in results:
        if res.score <= 0.0:
            continue
        matched_text = text[res.start:res.end]
        if not _is_valid_entity(res.entity_type, matched_text):
            continue
        entities.append({
            "entity_type": res.entity_type,
            "text": matched_text,
            "score": res.score,
            "source": "presidio"
        })
    return entities

def check_pii_regex(text):
    entities = []
    for entity_type, pattern in REGEX_PATTERNS.items():
        for match in pattern.finditer(text):
            entities.append({
                "entity_type": entity_type,
                "text": match.group(),
                "score": 1.0,
                "source": "regex"
            })
    return entities

def check_pii_llm(text, model):
    try:
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": PII_LLM_SYSTEM_PROMPT},
                {"role": "user", "content": text}
            ],
            format=PiiLlmResponse.model_json_schema(),
            options={
                'think': False,
                'num_predict': -1
            }
        )
        parsed = PiiLlmResponse.model_validate_json(response.message.content)
    except Exception as e:
        print(f"PII LLM judge failed for model {model}, {e}")
        return []

    return [
        {
            "entity_type": entity.entity_type,
            "text": entity.text,
            "score": 1.0,
            "source": "llm"
        }
        for entity in parsed.entities
    ]

def _normalize(text):
    return " ".join(text.strip().lower().split())

def run_ensemble(text, model):
    all_entities = check_pii(text) + check_pii_regex(text) + check_pii_llm(text, model)

    groups = []
    for ent in all_entities:
        norm = _normalize(ent["text"])
        if not norm:
            continue
        matched_group = None
        for g in groups:
            if norm == g["norm"] or norm in g["norm"] or g["norm"] in norm:
                matched_group = g
                break
        if matched_group is None:
            matched_group = {"norm": norm, "entity_types": set(), "sources": set(), "scores": []}
            groups.append(matched_group)
        matched_group["entity_types"].add(ent["entity_type"])
        matched_group["sources"].add(ent["source"])
        matched_group["scores"].append(ent["score"])

    weighted_severity = 0.0
    entities = []
    for g in groups:
        if len(g["sources"]) < 2:
            continue
        avg_score = sum(g["scores"]) / len(g["scores"])
        weighted_severity += avg_score
        for entity_type in g["entity_types"]:
            if entity_type not in entities:
                entities.append(entity_type)

    total_score = min(1.0, weighted_severity / PII_RISK_CAP)
    return total_score, entities

def run(gen_id, engine):
    SessionLocal = select.initiate_sessionlocal(engine)
    iteration_details = select.fetch_iteration_id_using_gen_id(
        SessionLocal,
        gen_id
    ).all()

    iterations = []
    for iteration in iteration_details:
        iterations.append({
            "iteration_id": iteration.iteration_id,
            "model_name": iteration.model_name
        })

    iterations.sort(key=lambda it: it["iteration_id"])
    file_path = []

    for it in iterations:
        file_path.append(select.fetch_documents_per_iteration(
            SessionLocal,
            it["iteration_id"]
        ).all()[0].file_path)

    pii = []

    for file_id in range(len(file_path)):
        result = {
            "iteration_id": None,
            "total_score": None,
            "entities": []
        }
        sentences = read_file(file_path[file_id])
        result["iteration_id"] = iterations[file_id]["iteration_id"]
        total_score, entities = run_ensemble(sentences, iterations[file_id]["model_name"])
        result["total_score"] = total_score
        result["entities"] = entities
        pii.append(result)

    return pii
