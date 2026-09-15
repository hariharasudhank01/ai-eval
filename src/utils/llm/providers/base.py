"""
Common interface every model provider implements.
"""
from abc import ABC, abstractmethod

class LLMProvider(ABC):
    @abstractmethod
    def generate(self, system_prompt, user_prompt, model, schema):
        """
        Call the provider's chat API and return a validated instance of `schema`
        (a pydantic BaseModel subclass), using whichever structured-output mechanism
        that provider natively supports.
        """
        raise NotImplementedError
