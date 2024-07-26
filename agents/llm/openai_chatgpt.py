from typing import Type

import instructor
from openai import OpenAI
from pydantic import BaseModel, ValidationError

from .base import BaseLLM


class OpenAIChatGPT(BaseLLM):
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        temperature: float = None,
        top_p: float = None,
        max_tokens: int = 1500,
    ):
        super().__init__(temperature, top_p, max_tokens)
        self.client = self.client = instructor.patch(OpenAI(api_key=api_key))
        self.model = model

    def generate(
        self, system_prompt: str, context: str, task: str, schema: Type[BaseModel]
    ) -> str:
        try:
            messages = [
                {
                    "role": "system",
                    "content": system_prompt
                    + f". Return maximum {self.max_tokens} tokens",
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nTask: {task}",
                },
            ]

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                response_model=schema,
            )
            try:
                return response.model_dump_json()
            except:
                # Get the raw text response
                raw_response = response.choices[0].message.content

                try:
                    # Attempt to parse the response into the schema
                    parsed_response = schema.model_validate_json(raw_response)
                    return parsed_response.model_dump_json()
                except ValidationError:
                    # If parsing fails, return the raw response
                    return raw_response

        except Exception as e:
            print(f"Error in OpenAI API call: {str(e)}")
            # Return any partial response if available
            if (
                "response" in locals()
                and hasattr(response, "choices")
                and len(response.choices) > 0
            ):
                return response.choices[0].message.content
            raise
