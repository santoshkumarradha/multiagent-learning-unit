import json
import random
import string
from typing import Any, Dict

from .base import BaseLLM


class MockLLM(BaseLLM):
    def generate(self, prompt: str, schema: Dict[str, Any]) -> str:
        random_string = "".join(
            random.choices(string.ascii_letters + string.digits, k=self.max_tokens)
        )

        response = {}
        for key, value in schema.get("properties", {}).items():
            if value.get("type") == "string":
                response[key] = f"Mock {key}: {random_string[:20]}"
            elif value.get("type") == "number":
                response[key] = random.uniform(0, 1)
            elif value.get("type") == "integer":
                response[key] = random.randint(0, 100)
            elif value.get("type") == "boolean":
                response[key] = random.choice([True, False])
            elif value.get("type") == "array":
                response[key] = [f"Item {i}" for i in range(3)]
            else:
                response[key] = None

        return json.dumps(response)
