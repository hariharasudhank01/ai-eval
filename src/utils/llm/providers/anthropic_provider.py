import os
from anthropic import Anthropic
from .base import LLMProvider
from utils.helpers.logger import get_logger, timed

logger = get_logger("llm.providers.anthropic")

MAX_TOKENS = 4096

class AnthropicProvider(LLMProvider):
    def __init__(self):
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def generate(self, system_prompt, user_prompt, model, schema):
        tool_name = schema.__name__
        try:
            with timed(logger, f"Anthropic generate call to '{model}'"):
                response = self.client.messages.create(
                    model=model,
                    max_tokens=MAX_TOKENS,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                    tools=[{
                        "name": tool_name,
                        "description": f"Return the result as {tool_name}.",
                        "input_schema": schema.model_json_schema(),
                    }],
                    tool_choice={"type": "tool", "name": tool_name},
                )
                for block in response.content:
                    if block.type == "tool_use":
                        return schema.model_validate(block.input)
                raise ValueError("Anthropic response did not contain a tool_use block")
        except Exception:
            logger.exception(f"Anthropic generate call to '{model}' failed")
            raise
