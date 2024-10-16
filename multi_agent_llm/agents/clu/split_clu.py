import random
import uuid
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel, Field
from rich import print
from rich.console import Console
from rich.panel import Panel
from rich.theme import Theme

from .kmu import KMU

custom_theme = Theme(
    {
        "agent_name": "bold cyan",
        "task": "green",
        "response": "yellow",
    }
)

console = Console(
    theme=custom_theme, width=80
)  # Set a fixed width for consistent formatting


def format_agent_output(name: str, task: str, response: str):
    agent_info = f"[agent_name]{name}[/agent_name]\n"
    task_section = "[task]Task:[/task]\n" + task + "\n"
    response_section = "[response]Response:[/response]\n" + response

    full_content = agent_info + task_section + response_section
    return full_content


verbosity = False
disable_logging = True


## -- utility functions ---
def print_cond(text, verbose=verbosity):
    if verbose:
        print(text)


def generate_short_uuid(length=8):
    short_uuid = str(uuid.uuid4()).replace("-", "")[:length]
    return short_uuid


# pydantic models ---
class FeedbackOutput(BaseModel):
    feedback: str = Field(..., description="Detailed feedback on the response")
    improvement_suggestions: List[str] = Field(
        ..., description="Suggestions for improvement"
    )
    knowledge_gaps: List[str] = Field(..., description="Identified knowledge gaps")


class OperationalAgentOutput(BaseModel):
    reasoning: str = Field(..., description="Detailed reasoning behind the response")
    response: str = Field(
        ..., description="Concise response from the Operational Agent"
    )
    confidence: float = Field(
        ...,
        description="Confidence in the response between 0 and 1, with 1 being highest confidence",
    )


class CLU:
    def __init__(
        self,
        main_role,
        pruning_queue_size=1,
        compress_knowledge=True,
        collection_name=None,
        retrival_limit=10,
        exploration_rate=0.5,
        llm=None,
        verbose=False,
    ):
        self.main_role = main_role
        self.clu_id = generate_short_uuid(5)
        self.pruning_queue_size = pruning_queue_size
        self.retrival_limit = retrival_limit
        self.verbose = verbose
        self.llm = llm

        learning_collection_name = (
            f"{collection_name}_learning" if collection_name else None
        )
        mistake_collection_name = (
            f"{collection_name}_mistake" if collection_name else None
        )

        self.learning_kmu = KMU(
            main_goal=f"Extract relevant positive knowledge from the feedback that indicates what worked well in accomplishing the main goal - Main goal: {main_role}",
            storage_goal=f"Store knowledge about strategies, reasoning, and approaches that have been effective in accomplishing the main goal - Main goal: {main_role}. Encourage storing knowledge that can be used in future tasks and not just for the current task.",
            retrieval_goal=f"Retrieve relevant positive knowledge about what has worked well in the past to help solve the current task based on the main role - Main goal: {main_role}",
            persist_directory="./.db",
            compress_knowledge=compress_knowledge,
            collection_name=learning_collection_name,
            llm=llm,
        )

        self.mistake_kmu = KMU(
            main_goal=f"Extract relevant negative knowledge from the feedback that indicates what did not work in accomplishing the main goal - Main goal: {main_role}",
            storage_goal=f"Store knowledge about mistakes, pitfalls, and ineffective strategies to avoid in order to accomplish the main goal - Main goal: {main_role}. Encourage storing knowledge that can be used in future tasks and not just for the current task.",
            retrieval_goal=f"Retrieve relevant negative knowledge about what has not worked in the past to help avoid similar mistakes in solving the current task based on the main role - Main goal: {main_role}",
            persist_directory="./.db",
            compress_knowledge=compress_knowledge,
            collection_name=mistake_collection_name,
            llm=llm,
        )

        self.operational_agent = None
        self.pruning_queue = []
        self.exploration_rate = exploration_rate
        self.training_iterations = 0
        self.custom_analysis_agent = None

        # Initialize agents
        self.initialize_agents()

    def _generate_response(
        self,
        name: str,
        role: str,
        function: str,
        user_prompt: str,
        response_model: Type[BaseModel],
    ):
        system_prompt = f"""
        You are: {name}
        Your role: {role}
        Your function: {function}
        Based on your role and function, do the task you are given.
        """

        # Generate response using LLM
        result = self.llm.generate(
            self.llm.format_prompt(
                system_prompt=system_prompt, user_prompt=user_prompt
            ),
            schema=response_model,
        )

        if self.verbose:
            formatted_output = format_agent_output(name, user_prompt, str(result))
            console.print(
                Panel(
                    formatted_output,
                    title=name,
                    border_style="cyan",
                    expand=False,
                )
            )

        return result

    def create_operational_agent(self, prompt, response_schema=None):
        if self.custom_analysis_agent is None:
            self.operational_agent = lambda input: self._generate_response(
                name="OperationalAgent",
                role="Execute tasks based on the provided prompt",
                function=prompt,
                user_prompt=input,
                response_model=response_schema or OperationalAgentOutput,
            )
        else:
            self.operational_agent = lambda input: self.custom_analysis_agent(
                prompt, input
            )

    def initialize_agents(self):
        class PromptGenerationOutput(BaseModel):
            prompt: str = Field(..., description="Generated prompt")

        self.meta_prompt_agent = lambda input: self._generate_response(
            name="MetaPromptAgent",
            role="Generate and optimize prompts for the Operational Agent",
            function=f"""
            Main Goal: {self.main_role}

            Your task is to generate and optimize prompts for the Operational Agent.
            Consider the main goal, available positive and negative knowledge, and task at hand.
            Combine goal-oriented and task-specific elements to create effective prompts that will help solve the given task.
            Utilize positive knowledge to reinforce effective strategies and negative knowledge to avoid past mistakes.

            Output a prompt that includes:
            1. A clear definition of the agent's role
            2. A detailed description of the agent's function and responsibilities based on relevant knowledge and task at hand
            3. Specific instructions to help solve the current task or work on it better with the main goal in mind based on relevant knowledge.
            4. Relevant knowledge to consider (if available), both positive and negative
            5. Use latest prompting techniques and strategies to generate the best prompt like Chain of thought prompts (asking to think step by step) etc., or more novel and innovative strategies based on the task at hand
            """,
            user_prompt=input,
            response_model=PromptGenerationOutput,
        )

        class ResponseComparisonOutput(BaseModel):
            is_correct: bool = Field(
                ..., description="Whether the response matches the expected output"
            )
            explanation: str = Field(
                ..., description="Explanation of the comparison result"
            )

        self.response_comparison_agent = lambda input: self._generate_response(
            name="ResponseComparison",
            role="Compare the agent's response to the expected output",
            function=f"""
            Your task is to compare the agent's response to the expected output, considering the main role: {self.main_role}.

            Based on the main role, determine the appropriate criteria for comparison for the given task.

            - First, understand the main role and how it affects the expectations for the response.
            - Review the task, the agent's response, and the expected output in the context of {self.main_role}.
            - Decide whether an exact match is required, or if acceptable variations are permitted based on meaning, intent, or style.
            - Determine if the response fulfills the requirements of the task and aligns with the main role.
            - Provide a detailed explanation of your comparison, highlighting similarities and differences, and how they relate to the main role.
            - Return a boolean indicating whether the response is correct and an explanation.

            **Guidelines:**

            - If {self.main_role} requires strict adherence to specific information, focus on exact matches.
            - If {self.main_role} involves flexibility or creativity, allow for variations in wording or approach, as long as the core meaning aligns.
            - Always consider how the response supports the objectives of {self.main_role}.
            """,
            user_prompt=input,
            response_model=ResponseComparisonOutput,
        )

        class FeedbackOutput(BaseModel):
            feedback: str = Field(..., description="Detailed feedback on the response")
            improvement_suggestions: List[str] = Field(
                ..., description="Suggestions for improvement"
            )
            knowledge_gaps: List[str] = Field(
                ..., description="Identified knowledge gaps"
            )

        self.general_feedback_agent = lambda input: self._generate_response(
            name="GeneralFeedbackAnalysis",
            role="Provide general feedback on the agent's performance",
            function="""
            You are responsible for providing general feedback on the Operational Agent's performance when no expected output is provided.
            Your tasks include:
            - Evaluate the relevance and quality of the response in relation to the given task
            - Assess the reasoning process and its effectiveness
            - Suggest potential improvements or areas for further exploration
            - Identify any apparent knowledge gaps
            - Provide a performance score and detailed feedback

            Consider the overall goal of the CLU while providing feedback.
            - Question and evaluate things you are not sure about, and provide feedback on the reasoning and the response.
            - Ask about knowledge gaps and provide feedback on the reasoning and the response or lack of such knowledge.
            Important: Prefer giving detailed and comprehensive yet not verbose feedback that is not already there in the knowledge, but will help and is correct.
            """,
            user_prompt=input,
            response_model=FeedbackOutput,
        )

        class PositiveFeedbackOutput(BaseModel):
            feedback: str = Field(
                ..., description="Detailed feedback on what worked well"
            )
            success_factors: List[str] = Field(
                ..., description="Factors that contributed to the success"
            )

        self.positive_feedback_agent = lambda input: self._generate_response(
            name="PositiveFeedbackAnalysis",
            role="Analyze successful responses and extract insights",
            function="""
            You are responsible for analyzing successful responses from the Operational Agent.
            Your tasks include:
            - Evaluate why the response correctly matches the expected output
            - Identify key elements in the reasoning that led to the correct answer
            - Suggest how this success can be replicated in future tasks
            - Highlight any particularly effective strategies used
            - Provide a performance score and detailed feedback
            Make sure to align your feedback with the main goal of the CLU in mind.
            Main Goal: {self.main_role}
            """,
            user_prompt=input,
            response_model=PositiveFeedbackOutput,
        )

        class NegativeFeedbackOutput(BaseModel):
            feedback: str = Field(
                ..., description="Detailed feedback on what went wrong"
            )
            improvement_suggestions: List[str] = Field(
                ..., description="Suggestions for improvement"
            )

        self.negative_feedback_agent = lambda input: self._generate_response(
            name="NegativeFeedbackAnalysis",
            role="Analyze incorrect responses and provide improvement suggestions",
            function="""
            You are responsible for analyzing incorrect responses from the Operational Agent.
            Your tasks include:
            - Evaluate why the response does not match the expected output
            - Identify gaps in knowledge or reasoning that led to the incorrect answer
            - Suggest specific improvements to the knowledge base or agent's function
            - Provide strategies to avoid similar mistakes in the future
            - Suggest potential improvements or areas for further exploration
            - Make sure in your suggestions you are not repeating the same thing that is already there in the knowledge base but provide new insights as to how to improve the response and what worked and what did not work.
            - Provide feedback of new knowledge that can be added to the knowledge base to try diverse solutions.
            Make sure to align your feedback with the main goal of the CLU in mind.
            Main Goal: {self.main_role}
            """,
            user_prompt=input,
            response_model=NegativeFeedbackOutput,
        )

        class KnowledgeInsightOutput(BaseModel):
            positive_knowledge: List[str] = Field(
                ...,
                description="List of positive knowledge entries about what worked well",
            )
            negative_knowledge: List[str] = Field(
                ...,
                description="List of negative knowledge entries about what did not work",
            )

        self.knowledge_insight_agent = lambda input: self._generate_response(
            name="KnowledgeInsight",
            role="Extract and generalize key knowledge from the feedback",
            function="""
            You are responsible for extracting and generalizing key knowledge from the feedback.
            Your tasks include:
            - Analyzing feedback, tasks, responses, and context to understand the main goal.
            - Extract positive knowledge about what worked well.
            - Extract negative knowledge about what did not work.
            - Formulate general, reusable knowledge entries that can be applied to future tasks.
            - Provide clear lists of positive and negative knowledge to be added to the respective databases.
            - The lists should be diverse and each element should be unique and contain new knowledge to be added.
            - These lists may include concepts, ideas, principles, and guidelines for optimal performance or avoidance of mistakes.
            
            Make sure you do all of this with the main goal in mind and align your knowledge gathering with the main goal.
            Main Goal : {self.main_role}
            """,
            user_prompt=input,
            response_model=KnowledgeInsightOutput,
        )

    def _get_feedback(
        self,
        oa_response: Any,
        expected_output: Optional[str],
        oa_prompt: Optional[str],
        knowledge: Optional[str],
        task: str,
    ) -> FeedbackOutput:
        oa_response_str = str(oa_response)
        if expected_output is not None and expected_output != "":
            comparison_result = self.response_comparison_agent(
                f"Compare Response: <{oa_response_str}>\n to Expected Output: <{expected_output}>\n for the Task: {task}",
            )
            feedback_input = f"""
            Overall Goal: {self.main_role}
            Task: {task}
            Agent Response: {oa_response_str}
            Agent Confidence: {getattr(oa_response, 'confidence', 'N/A')}
            Expected Output: {expected_output}
            Is Correct: {comparison_result.is_correct}
            Knowledge: {knowledge}
            Prompt for Operational Agent generated based on Knowledge: {oa_prompt}
            Comparison Explanation: {comparison_result.explanation}
            """
            feedback = (
                self.positive_feedback_agent(
                    feedback_input,
                )
                if comparison_result.is_correct
                else self.negative_feedback_agent(
                    feedback_input,
                )
            )
        else:
            feedback_input = f"""
            Overall Goal: {self.main_role}
            Task: {task}
            Agent Response: {oa_response_str}
            Agent Confidence: {getattr(oa_response, 'confidence', 'N/A')}
            """
            feedback = self.general_feedback_agent(
                feedback_input,
            )
            comparison_result = None
        return feedback, comparison_result.is_correct if comparison_result else None

    def prune_knowledge_bases(self):
        for entry in self.pruning_queue:

            def prune_learning_kmu():
                if entry["learning_ids"]:
                    with Lock():
                        self.learning_kmu.prune(
                            entry["feedback"], entry["learning_ids"]
                        )

            def prune_mistake_kmu():
                if entry["mistake_ids"]:
                    with Lock():
                        self.mistake_kmu.prune(entry["feedback"], entry["mistake_ids"])

            with ThreadPoolExecutor(max_workers=2) as executor:
                executor.submit(prune_learning_kmu)
                executor.submit(prune_mistake_kmu)
        self.pruning_queue = []

    def inference(
        self, task: str, response_schema: Optional[Type[BaseModel]] = None, **kwargs
    ):
        # Retrieve knowledge in parallel
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_learning = executor.submit(
                self.learning_kmu.retrieve_knowledge,
                task,
                n_results=self.retrival_limit,
            )
            future_mistake = executor.submit(
                self.mistake_kmu.retrieve_knowledge, task, n_results=self.retrival_limit
            )
            learning_knowledge = future_learning.result()
            mistake_knowledge = future_mistake.result()

        # Process learning knowledge
        learning_knowledge_str = (
            " ".join(
                [
                    item
                    for sublist in learning_knowledge["documents"]
                    for item in (sublist if isinstance(sublist, list) else [sublist])
                ]
            )
            if learning_knowledge["documents"]
            else "No relevant positive knowledge found."
        )

        # Process mistake knowledge
        mistake_knowledge_str = (
            " ".join(
                [
                    item
                    for sublist in mistake_knowledge["documents"]
                    for item in (sublist if isinstance(sublist, list) else [sublist])
                ]
            )
            if mistake_knowledge["documents"]
            else "No relevant negative knowledge found."
        )
        explore = random.random() < self.exploration_rate and kwargs.get(
            "training", False
        )
        if explore and self.verbose:
            print(
                "Exploration phase: Ignoring knowledge base for prompt generation and task execution."
            )

        # Generate prompt using Meta-Prompt Agent
        prompt_input = f"""
        Main Goal: {self.main_role}
        Task: {task}
        Relevant Positive Knowledge that we have learnt and works (Positive knowledge), the success we had: {learning_knowledge_str}
        Relevant Negative Knowledge we know that has failed (Negative knowledge), the mistakes we made: {mistake_knowledge_str}
        
        { "IMPORTANT: You are now in exploration phase, so ignore above instructions. The prompt you generate should try out a completely new approach that is not present in our knowledge base at all." if explore else "" }
        """
        generated_prompt = self.meta_prompt_agent(prompt_input)

        # Create or update Operational Agent with the generated prompt
        self.create_operational_agent(generated_prompt.prompt, response_schema)

        # Prepare knowledge input for operational agent
        oa_input = f"""
        Task: {task}
        Relevant Positive Knowledge that we have learnt and works (Positive knowledge), the success we had: {learning_knowledge_str}
        Relevant Negative Knowledge we know that has failed (Negative knowledge), the mistakes we made: {mistake_knowledge_str}

        Use the above relevant knowledge to inform your response. If no relevant knowledge is provided, rely on your general understanding.
        """
        # Execute task with relevant knowledge
        oa_response = self.operational_agent(oa_input)

        return {
            "response": oa_response,
            "generated_prompt": generated_prompt.prompt,
            "learning_knowledge_str": learning_knowledge_str,
            "mistake_knowledge_str": mistake_knowledge_str,
            "learning_knowledge": learning_knowledge,
            "mistake_knowledge": mistake_knowledge,
        }

    def train(
        self,
        task: str,
        expected_output: Optional[str] = None,
        prune_after=1,
        response_schema: Optional[Type[BaseModel]] = None,
        **kwargs,
    ):
        # First perform inference
        inference_result = self.inference(
            task, response_schema=response_schema, training=True, **kwargs
        )
        oa_response = inference_result["response"]

        right_answer = None
        feedback, right_answer = self._get_feedback(
            oa_response=oa_response,
            expected_output=expected_output,
            oa_prompt=inference_result["generated_prompt"],
            knowledge=f"{inference_result['learning_knowledge_str']} {inference_result['mistake_knowledge_str']}",
            task=task,
        )

        # Extract knowledge insights
        knowledge_update = self.knowledge_insight_agent(
            f"Overall Goal: {self.main_role}\nAnalyze feedback: {feedback}\nTask: {task}\nResponse: {str(oa_response)}\nPrompt generated using Knowledge: {inference_result['generated_prompt']}",
        )

        # Save knowledge updates in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            if knowledge_update.positive_knowledge:
                executor.map(
                    self.learning_kmu.save_knowledge,
                    knowledge_update.positive_knowledge,
                )
            if knowledge_update.negative_knowledge:
                executor.map(
                    self.mistake_kmu.save_knowledge, knowledge_update.negative_knowledge
                )

        # Flatten and check IDs
        learning_ids = inference_result["learning_knowledge"].get("ids", [])
        mistake_ids = inference_result["mistake_knowledge"].get("ids", [])

        flattened_learning_ids = [item for sublist in learning_ids for item in sublist]
        flattened_mistake_ids = [item for sublist in mistake_ids for item in sublist]

        if self.training_iterations % prune_after == 0:  # Control the pruning frequency
            if flattened_learning_ids or flattened_mistake_ids:
                full_feedback_input = f"""
                Overall Goal: {self.main_role}
                Task: {task}
                Agent Response: {str(oa_response)}
                Expected Output: {expected_output}
                Feedback for the agent's response: {feedback.feedback}
                """
                self.pruning_queue.append(
                    {
                        "feedback": full_feedback_input,
                        "learning_ids": flattened_learning_ids,
                        "mistake_ids": flattened_mistake_ids,
                    }
                )

        # Prune if necessary
        if len(self.pruning_queue) >= self.pruning_queue_size:
            self.prune_knowledge_bases()

        self.training_iterations += 1
        return {
            "response": oa_response,
            "feedback": feedback.dict(),
            "knowledge_update": {
                "positive": knowledge_update.positive_knowledge,
                "negative": knowledge_update.negative_knowledge,
            },
            "right_answer": right_answer,
        }

    def print_knowledge_base(self):
        console = Console()
        console.print("Learning Knowledge Base (Positive Knowledge):")
        learning_entries = self.learning_kmu.collection.get()
        if not learning_entries["ids"]:
            console.print("[bold red]Knowledge base is empty.[/bold red]")
        else:
            for idx, doc in enumerate(learning_entries["documents"]):
                console.print(f"Entry {idx + 1}: {doc}")

        console.print("\nMistake Knowledge Base (Negative Knowledge):")
        mistake_entries = self.mistake_kmu.collection.get()
        if not mistake_entries["ids"]:
            console.print("[bold red]Knowledge base is empty.[/bold red]")
        else:
            for idx, doc in enumerate(mistake_entries["documents"]):
                console.print(f"Entry {idx + 1}: {doc}")
