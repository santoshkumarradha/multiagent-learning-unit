from typing import Type

import instructor
from openai import OpenAI
from pydantic import BaseModel

from .base import BaseLLM


class OpenAIChatGPT(BaseLLM):
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-3.5-turbo",
        temperature: float = None,
        top_p: float = None,
        max_tokens: int = 1500,
    ):
        super().__init__(temperature, top_p, max_tokens)
        self.client = instructor.patch(OpenAI(api_key=api_key))
        self.model = model

    def generate(self, prompt: str, schema: Type[BaseModel]) -> str:
        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant. Please provide your response in the exact JSON format specified.",
                },
                {"role": "user", "content": prompt},
            ]

            response = self.client.chat.completions.create(
                model=self.model,
                response_model=schema,  # Ensure this is a class type, not an instance
                messages=messages,
                # temperature=self.temperature,
                # top_p=self.top_p,
                max_tokens=self.max_tokens,
            )

            # Instructor returns the parsed object, so we need to convert it back to JSON
            return response.model_dump_json()

        except Exception as e:
            # Handle any API errors or other exceptions
            print(f"Error in OpenAI API call: {str(e)}")
            raise
