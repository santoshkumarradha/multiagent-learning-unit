import inspect
import uuid
from typing import Any, Optional, Type, Union

from pydantic import BaseModel, Field, create_model

from .agent_status import AgentStatus
from .context import GlobalContext


class StandardOutputModel(BaseModel):
    response: str = Field(..., title="Response from the agent")


import json


class StandardOutputModel(BaseModel):
    response: str = Field(..., title="Response from the agent")


class BaseAgent:
    def __init__(
        self,
        name: str,
        role: str,
        function: str,
        output_model: Union[Type[BaseModel], Type] = StandardOutputModel,
    ):
        self.name = name
        self.role = role
        self.function = function
        self.original_output_model = output_model

        if inspect.isclass(output_model) and issubclass(output_model, BaseModel):
            self.output_model = output_model
        else:
            model_name = f"DynamicOutputModel_{uuid.uuid4().hex[:8]}"
            self.output_model = create_model(
                model_name,
                output=(output_model, ...),
            )

        self.status = AgentStatus.PENDING

    def __call__(self, task: str, context: GlobalContext = None) -> Any:
        if context is None:
            context = GlobalContext()
        raise NotImplementedError("Subclasses should implement this method.")

    def _create_system_prompt(self) -> str:
        return f"""You are: {self.name}
Your role: {self.role}
Your function: {self.function}
Based on your role and function, do the task you are given."""

    def _execute_task(self, system_prompt: str, context: str, task: str) -> str:
        raise NotImplementedError("Subclasses should implement this method.")

    def _to_pydantic_model(self, output: Any) -> BaseModel:
        if isinstance(output, BaseModel):
            return output
        if inspect.isclass(self.original_output_model) and issubclass(
            self.original_output_model, BaseModel
        ):
            return self.output_model.parse_obj(output)
        return self.output_model(output=output)

    def _from_pydantic_model(self, pydantic_output: BaseModel) -> Any:
        if inspect.isclass(self.original_output_model) and issubclass(
            self.original_output_model, BaseModel
        ):
            return pydantic_output
        return pydantic_output.output
