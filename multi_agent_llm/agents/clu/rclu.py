import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field
from rich.color import Color
from rich.console import Console
from rich.panel import Panel
from rich.style import Style
from rich.theme import Theme

from .clu import CompositeLearnUnit
from .kmu import KnowledgeManagementUnit

my_theme = Theme(
    {
        "info": Style(color=Color.from_rgb(50, 150, 200)),
        "warning": Style(color=Color.from_rgb(200, 150, 50)),
        "error": Style(color=Color.from_rgb(200, 50, 50)),
        "highlight": Style(color=Color.from_rgb(100, 200, 100)),
    }
)

console = Console(theme=my_theme)


class ReinforcedCompositeLearnUnit(CompositeLearnUnit):
    def __init__(
        self,
        llm,
        main_goal: str,
        name: Optional[str] = None,
        general_main_goal: Optional[str] = None,
        prompt_main_goal: Optional[str] = None,
        storage_goal: Optional[str] = None,
        retrieval_goal: Optional[str] = None,
        compress_knowledge: bool = True,
        retrieval_memory_count: int = 50,
    ):
        self.llm = llm
        self.main_goal = main_goal
        self.name = name or str(uuid.uuid4())
        self.compress = compress_knowledge
        self.retrieval_memory_count = retrieval_memory_count

        # Set default goals if not provided
        self.storage_goal = (
            storage_goal
            or f"Directly store all the successful and unsuccessful strategies and knowledge that led to the correct and incorrect outcomes, respectively, with the main goal: {main_goal} in mind. do not lose any information."
        )
        self.retrieval_goal = (
            retrieval_goal
            or f"Retrieve relevant strategies and knowledge that can guide reasoning across various types of tasks with the main goal: {main_goal} in mind."
        )
        self.general_main_goal = (
            general_main_goal
            or f"Develop and maintain a versatile knowledge base of abstract concepts, patterns, and problem-solving approaches with the main goal: {main_goal} in mind."
        )
        self.prompt_main_goal = (
            prompt_main_goal
            or f"Refine strategies for generating effective prompts that can guide reasoning across various types of tasks. Extract relevant information needed to construct high-quality prompts based on feedback from answering specific task-related queries with given old prompts. Main goal: {main_goal}"
        )

        # Initialize four KMUs for positive and negative feedback
        self.positive_general_kmu = KnowledgeManagementUnit(
            llm,
            main_goal=self.general_main_goal,
            storage_goal=self.storage_goal
            + "Store all the successful strategies and knowledge that led to the correct outcome.",
            retrieval_goal=self.retrieval_goal,
            name=f"PositiveGeneralKMU_{self.name}",
        )
        self.negative_general_kmu = KnowledgeManagementUnit(
            llm,
            main_goal=self.general_main_goal,
            storage_goal=self.storage_goal
            + "Store all the unsuccessful strategies and knowledge that led to the incorrect outcome.",
            retrieval_goal=self.retrieval_goal,
            name=f"NegativeGeneralKMU_{self.name}",
        )
        self.positive_prompt_kmu = KnowledgeManagementUnit(
            llm,
            main_goal=self.prompt_main_goal,
            storage_goal=self.storage_goal,
            retrieval_goal=self.retrieval_goal,
            name=f"PositivePromptKMU_{self.name}",
        )
        self.negative_prompt_kmu = KnowledgeManagementUnit(
            llm,
            main_goal=self.prompt_main_goal,
            storage_goal=self.storage_goal,
            retrieval_goal=self.retrieval_goal,
            name=f"NegativePromptKMU_{self.name}",
        )

    def parallel_kmu_retrieval(self, query: str):
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(
                    self.positive_general_kmu.retrieve,
                    query,
                    n=self.retrieval_memory_count,
                ),
                executor.submit(
                    self.negative_general_kmu.retrieve,
                    query,
                    n=self.retrieval_memory_count,
                ),
                executor.submit(
                    self.positive_prompt_kmu.retrieve,
                    query,
                    n=self.retrieval_memory_count,
                ),
                executor.submit(
                    self.negative_prompt_kmu.retrieve,
                    query,
                    n=self.retrieval_memory_count,
                ),
            ]
            results = [future.result() for future in futures]
        return results

    def reason(
        self,
        query: str,
        schema: BaseModel,
        verbose: bool = False,
        _capture_knowledge: bool = False,
    ) -> Any:
        if verbose:
            console.print("[info]Starting Reasoning Process[/info]")

        # Retrieve positive and negative knowledge in parallel
        (
            positive_general_knowledge,
            negative_general_knowledge,
            positive_prompt_knowledge,
            negative_prompt_knowledge,
        ) = self.parallel_kmu_retrieval(query)

        if verbose:
            self._log_step(
                "Positive General KMU Retrieval", query, positive_general_knowledge
            )
            self._log_step(
                "Negative General KMU Retrieval", query, negative_general_knowledge
            )
            self._log_step(
                "Positive Prompt KMU Retrieval", query, positive_prompt_knowledge
            )
            self._log_step(
                "Negative Prompt KMU Retrieval", query, negative_prompt_knowledge
            )

        # Generate task-specific prompt
        task_prompt = self._prompt_meta_prompt_agent(
            query,
            positive_general_knowledge,
            negative_general_knowledge,
            positive_prompt_knowledge,
            negative_prompt_knowledge,
            verbose,
        )

        # Combine general knowledge
        general_knowledge = positive_general_knowledge + negative_general_knowledge

        # Execute task using Operational Agent
        result = self._prompt_operational_agent(
            query, general_knowledge, task_prompt, schema, verbose
        )

        if verbose:
            console.print("[info]Reasoning Process Completed[/info]")

        if _capture_knowledge:
            return result, general_knowledge, task_prompt
        else:
            return result

    def train(
        self,
        x: Any,
        y: Optional[Any] = None,
        schema: BaseModel = None,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        if verbose:
            console.print("[info]Starting Training Process[/info]")

        # Perform reasoning and capture knowledge used
        generated_output, general_knowledge_used, task_prompt = self.reason(
            x, schema, verbose, _capture_knowledge=True
        )

        if y is None:
            if verbose:
                console.print(
                    "[warning]No expected output provided. Skipping comparison and feedback.[/warning]"
                )
            return {"generated_output": generated_output}

        # Compare generated output with expected output
        is_equivalent, comparison_explanation = self._prompt_comparison_agent(
            x, self.main_goal, generated_output.answer, y, verbose
        )

        # Generate feedback based on comparison results and knowledge used
        feedback = self._prompt_feedback_agent(
            is_equivalent,
            x,
            self.main_goal,
            f"Answer:{str(generated_output.answer)}"
            + f"Reasoning :{generated_output.explanation}",
            y,
            comparison_explanation,
            general_knowledge_used,
            task_prompt,
            verbose,
        )

        # Save feedback to the appropriate KMUs
        if is_equivalent:
            # Positive feedback
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [
                    executor.submit(
                        self._save_feedback_knowledge,
                        feedback["general_knowledge_feedback"],
                        self.positive_general_kmu,
                        verbose,
                    ),
                    executor.submit(
                        self._save_feedback_knowledge,
                        feedback["prompt_knowledge_feedback"],
                        self.positive_prompt_kmu,
                        verbose,
                    ),
                ]
        else:
            # Negative feedback
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [
                    executor.submit(
                        self._save_feedback_knowledge,
                        feedback["general_knowledge_feedback"],
                        self.negative_general_kmu,
                        verbose,
                    ),
                    executor.submit(
                        self._save_feedback_knowledge,
                        feedback["prompt_knowledge_feedback"],
                        self.negative_prompt_kmu,
                        verbose,
                    ),
                ]

        # Wait for all futures to complete
        for future in futures:
            future.result()

        if verbose:
            console.print("[info]Training Process Completed[/info]")

        return {
            "generated_output": generated_output.answer,
            "explanation": generated_output.explanation,
            "is_equivalent": is_equivalent,
            "comparison_explanation": comparison_explanation,
            "feedback": feedback,
        }

    def _save_feedback_knowledge(
        self,
        feedback_entries: List[str],
        kmu: KnowledgeManagementUnit,
        verbose: bool = False,
        align_knowledge=False,
    ):
        # for now combine all feedback entries into a single entry so that we can retrieve larger feedback entries
        feedback_entries = [
            (
                "|".join(feedback_entries)
                if len(feedback_entries) > 1
                else feedback_entries[0] if len(feedback_entries) == 1 else ""
            )
        ]
        # Add feedback entries to the KMU
        for entry in feedback_entries:
            if entry.strip():
                kmu.save(entry, compress=self.compress, align_knowledge=align_knowledge)
                if verbose:
                    console.print(f"Saved feedback to KMU {kmu.name}: {entry}")

    def _prompt_operational_agent(
        self,
        task: str,
        general_knowledge: List[str],
        task_prompt: str,
        schema: BaseModel,
        verbose: bool = False,
    ) -> Any:
        system_prompt = f"""You are the Operational Agent in the Composite Learning Unit.
Main Goal: {self.main_goal}
Your task is to process the given task using the provided general knowledge and following the detailed task-specific prompt.

Consider the following:
- Previous successful approaches:
{[k for k in general_knowledge if k in self.positive_general_kmu.collection.get()['documents']]}

- Previous unsuccessful approaches and mistakes to avoid:
{[k for k in general_knowledge if k in self.negative_general_kmu.collection.get()['documents']]}

Begin by enclosing all thoughts within <thinking> tags, exploring multiple angles and approaches.
Break down the solution into clear steps within <step> tags. Start with a 20-step budget, requesting more for complex problems if needed.
Use <count> tags after each step to show the remaining budget. Stop when reaching 0.
Continuously adjust your reasoning based on intermediate results and reflections, adapting your strategy as you progress.
Regularly evaluate progress using <reflection> tags. Be critical and honest about your reasoning process.
Assign a quality score between 0.0 and 1.0 using <reward> tags after each reflection. Use this to guide your approach:

0.8+: Continue current approach
0.5-0.7: Consider minor adjustments
Below 0.5: Seriously consider backtracking and trying a different approach


If unsure or if reward score is low, backtrack and try a different approach, explaining your decision within <thinking> tags.
For mathematical problems, show all work explicitly using LaTeX for formal notation and provide detailed proofs.
Explore multiple solutions individually if possible, comparing approaches in reflections.
Use thoughts as a scratchpad, writing out all calculations and reasoning explicitly.
Synthesize the final answer within <answer> tags, providing a clear, concise summary.
Conclude with a final reflection on the overall solution, discussing effectiveness, challenges, and solutions. Assign a final reward score.

Before answering, think step by step and explain the reasoning behind the answer, including how various parts of the general knowledge and prompt knowledge were used to arrive at the answer. Provide detailed reasoning and explanation and then finally provide the answer in answer json field."""

        user_prompt = f"Task: {task}\nDetailed Task-Specific Prompt: {task_prompt}\nExecute the task:"

        class OperationalAgentOutput(BaseModel):
            explanation: str = Field(
                ..., description="Detailed explanation of the response"
            )
            answer: schema = Field(
                ...,
                description="The response generated by the Operational Agent",
            )

        query = self.llm.format_prompt(
            system_prompt=system_prompt, user_prompt=user_prompt
        )
        result = self.llm.generate(query, schema=OperationalAgentOutput)

        if verbose:
            self._log_step("Operational Agent", query, str(result))

        return result

    def _prompt_meta_prompt_agent(
        self,
        task: str,
        positive_general_knowledge: List[str],
        negative_general_knowledge: List[str],
        positive_prompt_knowledge: List[str],
        negative_prompt_knowledge: List[str],
        verbose: bool = False,
    ) -> str:
        system_prompt = f"""You are the Meta-Prompt Agent in the Composite Learning Unit.
Main Goal: {self.prompt_main_goal}

Your task is to generate a detailed, task-specific prompt for the Operational Agent based on the given task and the retrieved prompt knowledge.

You have access to the memory of all past mistakes (unsuccessful strategies) and all past successes (successful strategies). Use this information to guide your prompt creation.

Consider the following:

- Previous successful strategies:
{positive_prompt_knowledge if positive_prompt_knowledge else 'No positive strategies recorded yet.'}
{positive_general_knowledge if positive_general_knowledge else 'No positive strategies recorded yet.'}

- Previous unsuccessful strategies (mistakes to avoid):
{negative_prompt_knowledge if negative_prompt_knowledge else 'No negative strategies recorded yet.'}
{negative_general_knowledge if negative_general_knowledge else 'No negative strategies recorded yet.'}

Your goal is to learn from past mistakes by avoiding strategies that did not work, and to learn from past successes by leveraging strategies that have worked before.

When creating the prompt:

- Build upon previous successful strategies to enhance the likelihood of success.
- Avoid repeating past mistakes by not using unsuccessful strategies.
- If no relevant past successes or failures are available, explore new approaches.

Important: Come up with unique and diverse strategies that have not been tried yet based on this guidance for the Operational Agent to follow.

Ensure that the generated prompt effectively guides the Operational Agent to accomplish the task, learning from the history of what worked and what did not."""

        user_prompt = f"Task: {task}\nGenerate a detailed task-specific prompt that leverages previous successes and avoids past mistakes."

        class MetaPromptAgentOutput(BaseModel):
            prompt: str = Field(
                ...,
                description="The detailed task-specific prompt generated by the Meta-Prompt Agent",
            )

        query = self.llm.format_prompt(
            system_prompt=system_prompt, user_prompt=user_prompt
        )
        result = self.llm.generate(query, schema=MetaPromptAgentOutput)

        if verbose:
            self._log_step("Meta-Prompt Agent", query, result.prompt)

        return result.prompt

    def _prompt_feedback_agent(
        self,
        is_positive: bool,
        query: str,
        main_goal: str,
        generated_output: Any,
        expected_output: Any,
        comparison_explanation: str,
        general_knowledge_used: List[str],
        task_prompt: str,
        verbose: bool = False,
        max_feedback: int = 5,
    ) -> Dict[str, List[str]]:
        agent_type = "Positive" if is_positive else "Negative"

        if is_positive:
            # Positive Feedback Agent
            system_prompt = f"""You are the Positive Feedback Agent in the Composite Learning Unit.
Main Goal: {main_goal}

Your task is to act as a memory of what worked. Record what was tried, that it worked, and potential reasons why it was successful. Focus on documenting the strategies and knowledge that led to the correct outcome, so they can be replicated in the future.

Please provide:

- A summary of the strategies and knowledge that worked.
- Potential reasons why they were successful.

Remember to maintain a history of what worked.

Ensure that your feedback is clear and concise, accurately reflecting the successful attempts to help future reasoning processes replicate these successes.

Provide detailed feedback for both the general knowledge and prompt knowledge, with no more than {max_feedback} feedback items for each. If we tried 'n' strategies and 'm' of them worked, provide feedback for 'm' strategies that worked and 'n-m' strategies that did not work, condense the feedback such that we have only one feedback for each strategy that worked and one feedback for each strategy that did not work"""
        else:
            # Negative Feedback Agent
            system_prompt = f"""You are the Negative Feedback Agent in the Composite Learning Unit.
Main Goal: {main_goal}

Your task is to act as a memory of past mistakes. Record what was tried, that it did not work, and potential reasons why it failed. Focus on documenting the strategies and knowledge that led to the incorrect outcome, so they can be avoided in the future.

Please provide:

- A summary of the strategies and knowledge that did not work.
- Potential reasons why they failed. (if you are not sure, just provide a summary of the unsuccessful strategies)

Do not provide suggestions for improvement or how to avoid these issues. Your role is to maintain a history of what did not work.

Ensure that your feedback is clear and concise, accurately reflecting the unsuccessful attempts to help future reasoning processes avoid repeating these mistakes.
Note if the strategy is correct but the execution is wrong, provide feedback on the execution.

Provide detailed feedback for both the general knowledge and prompt knowledge, with no more than {max_feedback} feedback items for each.
If we tried 'n' strategies and 'm' of them worked, provide feedback for 'm' strategies that worked and 'n-m' strategies that did not work, condense the feedback such that we have only one feedback for each strategy that worked and one feedback for each strategy that did not work."""

        user_prompt = f"""Query: {query}
Generated Output: {generated_output}
Expected Output: {expected_output}
Comparison Explanation: {comparison_explanation}
General Knowledge Used: {general_knowledge_used if general_knowledge_used else 'None'}
Prompt Knowledge Used: {task_prompt if task_prompt else 'None'}

Always prefer to condense multiple feedback items into a single feedback item if they are related and can be summarized effectively.
Provide the requested feedback:"""

        class FeedbackAgentOutput(BaseModel):
            general_knowledge_feedback: List[str] = Field(
                ..., description="Feedback for the general knowledge base"
            )
            prompt_knowledge_feedback: List[str] = Field(
                ..., description="Feedback for the prompt knowledge base"
            )

        query = self.llm.format_prompt(
            system_prompt=system_prompt, user_prompt=user_prompt
        )
        result = self.llm.generate(query, schema=FeedbackAgentOutput)

        if verbose:
            self._log_step(
                f"{agent_type} Feedback Agent",
                query,
                f"General Knowledge Feedback: {result.general_knowledge_feedback}\nPrompt Knowledge Feedback: {result.prompt_knowledge_feedback}",
            )

        return {
            "general_knowledge_feedback": result.general_knowledge_feedback,
            "prompt_knowledge_feedback": result.prompt_knowledge_feedback,
        }

    def _prompt_comparison_agent(
        self,
        query: str,
        main_goal: str,
        generated_output: Any,
        expected_output: Any,
        verbose: bool = False,
    ) -> Tuple[bool, str]:
        system_prompt = f"""You are the Comparison Agent in the Composite Learning Unit.
Main Goal: {main_goal}
Your task is to compare the generated output with the expected output, considering the query and main goal.
Determine if they are equivalent, either verbatim or semantically, based on the context of the task.

Provide:
- A boolean indicating whether the outputs are equivalent.
- A detailed explanation of your comparison, highlighting similarities and differences."""

        user_prompt = f"""Query: {query}
Generated Output: {generated_output}
Expected Output: {expected_output}
Are these outputs equivalent? Provide a boolean result and a detailed explanation."""

        class ComparisonAgentOutput(BaseModel):
            is_equivalent: bool = Field(
                ..., description="Boolean indicating if the outputs are equivalent"
            )
            explanation: str = Field(
                ..., description="Detailed explanation of the equivalence decision"
            )

        query = self.llm.format_prompt(
            system_prompt=system_prompt, user_prompt=user_prompt
        )
        result = self.llm.generate(query, schema=ComparisonAgentOutput)

        if verbose:
            self._log_step(
                "Comparison Agent",
                query,
                f"Equivalent: {result.is_equivalent}\nExplanation: {result.explanation}",
            )

        return result.is_equivalent, result.explanation

    def print_knowledge(self, verbose: bool = False):
        console.print("[bold]Positive General Knowledge:[/bold]")
        self.positive_general_kmu.print_knowledge(verbose)

        console.print("\n[bold]Negative General Knowledge:[/bold]")
        self.negative_general_kmu.print_knowledge(verbose)

        console.print("\n[bold]Positive Prompt Knowledge:[/bold]")
        self.positive_prompt_kmu.print_knowledge(verbose)

        console.print("\n[bold]Negative Prompt Knowledge:[/bold]")
        self.negative_prompt_kmu.print_knowledge(verbose)
