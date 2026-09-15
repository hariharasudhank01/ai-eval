import os
from openai import OpenAI
from .base import LLMProvider
from utils.helpers.logger import get_logger, timed

logger = get_logger("llm.providers.openai")

class OpenAIProvider(LLMProvider):
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def generate(self, system_prompt, user_prompt, model, schema):
        try:
            with timed(logger, f"OpenAI generate call to '{model}'"):
                completion = self.client.chat.completions.parse(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    response_format=schema
                )
                parsed = completion.choices[0].message.parsed
                if parsed is None:
                    raise ValueError("OpenAI response did not contain parsed structured output")
                return parsed
        except Exception:
            logger.exception(f"OpenAI generate call to '{model}' failed")
            raise
