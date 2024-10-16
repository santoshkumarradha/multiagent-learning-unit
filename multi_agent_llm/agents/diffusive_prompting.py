import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field
from rich.color import Color
from rich.console import Console
from rich.panel import Panel
from rich.style import Style
from rich.table import Table
from rich.theme import Theme

# Set up rich console for logging
my_theme = Theme(
    {
        "info": Style(color=Color.from_rgb(50, 150, 200)),
        "warning": Style(color=Color.from_rgb(200, 150, 50)),
        "error": Style(color=Color.from_rgb(200, 50, 50)),
        "highlight": Style(color=Color.from_rgb(100, 200, 100)),
    }
)
console = Console(theme=my_theme)


# Define pydantic models
class TaskRefinementResponse(BaseModel):
    refined_tasks: List[str] = Field(
        ...,
        description="A list of refined and improved versions of the task, exploring different approaches.",
    )
    reasoning: str = Field(
        ...,
        description="Explanation of how the tasks were refined and different approaches considered.",
    )


class AgentFinalResponse(BaseModel):
    response: str = Field(..., description="The agent's detailed answer to the task.")


# Define the CompositeMetaAgentSystem class
class CompositeMetaAgentSystem:
    def __init__(
        self,
        llm,
        main_goal: str,
        name: Optional[str] = None,
    ):
        self.llm = llm
        self.main_goal = main_goal
        self.name = name or f"CompositeMetaAgentSystem_{str(uuid.uuid4())}"

    def _log_step(self, agent_name: str, input_data: str, output_data: str):
        table = Table(title=f"[bold]{agent_name}[/bold]", show_header=False, box=None)
        table.add_row("[cyan]Input:[/cyan]", str(input_data))
        table.add_row("[green]Output:[/green]", str(output_data))
        console.print(Panel(table, expand=False))

    def _prompt_task_refinement_agent(
        self,
        agent_name: str,
        task: str,
        previous_refinements: List[str],
        verbose: bool = False,
    ) -> TaskRefinementResponse:
        # Construct the system prompt
        system_prompt = f"""You are {agent_name}.

Role:
Your role is to analyze and refine the given task to make it clearer, more detailed, and easier to solve by exploring multiple approaches.

Function:
Your function is to carefully consider the task, identify any ambiguities or missing information, and improve it by adding necessary details.
You should explore different ways to approach the task, providing multiple refined tasks that consider various solutions or methods.
If previous refinements did not lead toward a solution, consider alternative approaches.
You should provide refined versions of the task that are more specific and thorough, along with your reasoning.
"""

        # Include previous refinements if any
        previous = ""
        if previous_refinements:
            previous = "\nPrevious Refinements:\n" + "\n".join(previous_refinements)

        # User prompt for task refinement
        user_prompt = f"""Task:
{task}
{previous}

Please refine the task according to your function and provide your reasoning in the following JSON format:

{{
    "refined_tasks": [
        "Your first refined and improved version of the task with all details",
        "Your second refined task",
        "..."
    ],
    "reasoning": "Your reasoning behind the refinements made and different approaches considered."
}}

Ensure that you explore multiple ways to approach the task, providing different refined tasks that consider various solutions or methods. If previous refinements did not lead toward a solution, consider alternative approaches. Make sure each refined task is more detailed, clearer, and helps in solving the problem better.
"""

        # Generate the response using the LLM
        query = self.llm.format_prompt(
            system_prompt=system_prompt, user_prompt=user_prompt
        )
        result = self.llm.generate(query, schema=TaskRefinementResponse)

        if verbose:
            self._log_step(agent_name, query, result.model_dump_json(indent=2))

        return result

    def _prompt_final_agent(
        self,
        agent_name: str,
        task: str,
        schema: BaseModel,
        verbose: bool = False,
    ) -> Any:
        # Construct the system prompt
        system_prompt = f"""You are {agent_name}.

Role:
Your role is to provide detailed answers to the task.

Function:
Using all the refinements made, provide comprehensive solutions to the task, considering different approaches.
"""

        # User prompt for the final agent
        user_prompt = f"""Task:
{task}

Please provide detailed answers to the task, considering different approaches if applicable.
"""

        # Generate the response using the LLM
        query = self.llm.format_prompt(
            system_prompt=system_prompt, user_prompt=user_prompt
        )
        result = self.llm.generate(query, schema=schema)

        if verbose:
            self._log_step(agent_name, query, result.model_dump_json(indent=2))

        return result

    def run(
        self, n: int, initial_task: str, schema: BaseModel, verbose: bool = False
    ) -> Any:
        task = initial_task
        reasonings = []
        previous_refinements = []

        for level in range(n):
            agent_name = f"Agent Level {level+1}"
            result = self._prompt_task_refinement_agent(
                agent_name, task, previous_refinements, verbose
            )
            reasonings.append(result.reasoning)
            previous_refinements.extend(result.refined_tasks)

            # For the next agent, combine the refined tasks
            task = "\n".join(result.refined_tasks)

        # Run the final agent to solve the refined task(s)
        final_agent_name = "Final Agent"

        class FinalAgentAnswer(BaseModel):
            answer: schema = Field(
                ..., description="The final agent's detailed answer to the task."
            )
            reasoning: str = Field(
                ..., description="The reasoning behind the final agent's answer."
            )

        final_result = self._prompt_final_agent(
            final_agent_name, task, FinalAgentAnswer, verbose
        )

        # Print the final agent's response
        console.print(
            f"[bold]{final_agent_name}'s Response:[/bold]\n{final_result.answer}"
        )
        console.print(f"[bold]Reasoning:[/bold]\n{final_result.reasoning}")

        return final_result


# Example usage
def example_usage():
    # Define a mock LLM class for demonstration
    class MockLLM:
        def generate(self, prompt, schema):
            # This is a simplistic mock. In reality, you'd use a real LLM here.
            return schema(
                **{
                    field: f"Mocked {field} based on: {prompt}"
                    for field in schema.__fields__
                }
            )

        def format_prompt(self, system_prompt, user_prompt):
            return f"{system_prompt}\n{user_prompt}"

    # Initialize the LLM
    llm = MockLLM()

    # Initialize the CompositeMetaAgentSystem
    meta_agent_system = CompositeMetaAgentSystem(
        llm=llm,
        main_goal="Solve complex tasks by refining them and exploring multiple approaches",
    )

    # Define the initial task
    initial_task = """
Given:
"Neural networks transform data efficiently" → "eeraf"
"Artificial intelligence automates decisions" → "rnue"
"Amazing Large language models" → "maao"
Query:
What is "Gradient descent optimizes loss functions" → ?
"""

    # Set the number of meta-agent levels
    n = 5  # For example, 5 levels of meta-agents

    # Run the meta-agent system
    final_result = meta_agent_system.run(
        n=n, initial_task=initial_task, schema=AgentFinalResponse, verbose=True
    )

    # Print the final result
    console.print(f"[bold green]Final Result:[/bold green]\n{final_result.response}")


if __name__ == "__main__":
    example_usage()
