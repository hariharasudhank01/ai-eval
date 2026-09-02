from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider

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

def run(text):
    results = analyzer.analyze(text=text, language="en")
    total_score = 0.0
    entities = []
    for res in results:
        temp = res.to_dict()
        if res.score > 0.0:
            if temp["entity_type"] not in entities:
                entities.append(temp["entity_type"])
            total_score = total_score + res.score
    return total_score, entities
