import ollama
from .base import LLMProvider
from utils.helpers.logger import get_logger, timed

logger = get_logger("llm.providers.ollama")

class OllamaProvider(LLMProvider):
    def generate(self, system_prompt, user_prompt, model, schema):
        try:
            with timed(logger, f"Ollama generate call to '{model}'"):
                response = ollama.chat(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    format=schema.model_json_schema(),
                    options={
                        'think': False,
                        'num_predict': -1
                    }
                )
                return schema.model_validate_json(response.message.content)
        except Exception:
            logger.exception(f"Ollama generate call to '{model}' failed")
            raise
