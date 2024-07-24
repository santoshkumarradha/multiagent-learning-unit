import inspect
from datetime import datetime
from typing import Any, Optional, Type, Union

from pydantic import BaseModel, Field

from ..llm.base import BaseLLM
from ..utils.print_utils import print_agent_details
from .base_agent import AgentStatus, BaseAgent
from .context import GlobalContext
from .llm_manager import LLMManager


class StandardOutputModel(BaseModel):
    response: str = Field(..., title="Response from the agent")


class Agent(BaseAgent):
    def __init__(
        self,
        name: str,
        role: str,
        function: str,
        output_model: Union[Type[BaseModel], Type] = StandardOutputModel,
        llm: Optional[BaseLLM] = None,
        max_retries: int = 3,
        disable_logging: bool = False,
        retry_sleep_time=1,
    ):
        super().__init__(name, role, function, output_model)
        self.llm = llm or LLMManager.get_global_llm()
        self.max_retries = max_retries
        self.disable_logging = disable_logging
        self.retry_sleep_time = retry_sleep_time
        if self.llm is None:
            raise ValueError(
                "No LLM provided and no global LLM set. Use LLMManager.set_global_llm() or provide an LLM instance."
            )

    def __call__(self, task: str, context: GlobalContext = None) -> Any:
        if context is None:
            context = GlobalContext()

        self.status = AgentStatus.RUNNING
        start_time = datetime.now().isoformat()
        system_prompt = self._create_system_prompt()
        context_str = context.get_context_string()

        attempts = 0
        max_attempts = self.max_retries
        while attempts < max_attempts:
            attempts += 1
            try:
                result = self._execute_task(system_prompt, context_str, task)
                if inspect.isclass(self.output_model) and issubclass(
                    self.output_model, BaseModel
                ):
                    pydantic_result = self.output_model.parse_raw(result)
                else:
                    pydantic_result = self._to_pydantic_model(result)
                end_time = datetime.now().isoformat()

                context.add(
                    self.name,
                    task,
                    pydantic_result.model_dump(),
                    status=AgentStatus.COMPLETED.value,
                    start_time=start_time,
                    end_time=end_time,
                )

                print_agent_details(
                    self.name,
                    task,
                    pydantic_result.model_dump(),
                    self.llm.__class__.__name__,
                    attempts,
                )
                self.status = AgentStatus.COMPLETED
                return self._from_pydantic_model(pydantic_result)
            except Exception as e:
                import time

                time.sleep(self.retry_sleep_time)
                if attempts == max_attempts:
                    end_time = datetime.now().isoformat()
                    context.add(
                        self.name,
                        task,
                        str(e),
                        status=AgentStatus.FAILED.value,
                        start_time=start_time,
                        end_time=end_time,
                    )

                    self.status = AgentStatus.FAILED
                    raise e

    def _execute_task(self, system_prompt: str, context: str, task: str) -> Any:
        return self.llm.generate(
            system_prompt=system_prompt,
            context=context,
            task=task,
            schema=self.output_model,
        )
