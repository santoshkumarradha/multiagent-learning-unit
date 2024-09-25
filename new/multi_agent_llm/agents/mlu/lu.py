import asyncio
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field
from rich import box
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import Progress
from rich.table import Table
from rich.text import Text

# Setup rich logging
console = Console()
# logging.basicConfig(level="NOTSET", format="%(message)s", handlers=[RichHandler()])
# log = logging.getLogger("rich")

from ...agent_class import Agent
from ...llm import LLMBase
from .kmu import KnowledgeManagementUnit


def set_verbose(verbose: bool = False):
    global verbosity
    verbosity = verbose


set_verbose(False)


def generate_short_uuid(length=8):
    short_uuid = str(uuid.uuid4()).replace("-", "")[:length]
    return short_uuid


def print_cond_rich(panel_text, title=None, subtitle=None, style="blue"):
    if verbosity:
        panel = Panel(
            panel_text,
            title=title,
            subtitle=subtitle,
            border_style=style,
            padding=(1, 2),
        )
        console.print(panel)


def format_result_table(results: dict, title: str = "Results", style: str = "green"):
    """Helper function to create and print a rich table for results"""
    table = Table(title=title, box=box.ROUNDED, border_style=style)
    table.add_column("Field", justify="right", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")

    for key, value in results.items():
        table.add_row(key, str(value))

    console.print(table)


class FeedbackOutput(BaseModel):
    performance_score: float = Field(
        ..., description="Performance score of the Operational Agent between 0 and 1"
    )
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
        ..., description="Confidence in the response between 0 and 1"
    )


class InferenceOutput(BaseModel):
    operational_agent_response: OperationalAgentOutput = Field(
        ..., description="Operational Agent response"
    )
    knowledge_used: str = Field(
        ..., description="Knowledge used by the Operational Agent"
    )


class TrainingOutput(BaseModel):
    inference_output: InferenceOutput = Field(
        ...,
        description="The result of the inference step, containing operational agent response and used knowledge.",
    )
    feedback: FeedbackOutput = Field(
        ..., description="The feedback provided during training."
    )
    knowledge_update: List[str] = Field(
        ..., description="The updated knowledge entries."
    )
    right_answer: Optional[bool] = Field(
        None, description="Whether the Operational Agent's response was correct."
    )


class PromptGenerationOutput(BaseModel):
    prompt: str = Field(..., description="Generated prompt")


class ResponseComparisonOutput(BaseModel):
    is_correct: bool = Field(
        ..., description="Whether the response matches the expected output"
    )
    explanation: str = Field(..., description="Explanation of the comparison result")


class KnowledgeInsightOutput(BaseModel):
    new_knowledge: List[str] = Field(..., description="List of new knowledge entries")


class CLU:
    def __init__(
        self,
        llm: LLMBase,
        main_role,
        pruning_queue_size=1,
        compress_knowledge=True,
        collection_name=None,
        retrieval_limit=10,
    ):
        self.llm = llm
        self.main_role = main_role
        self.clu_id = generate_short_uuid(5)
        self.pruning_queue_size = pruning_queue_size
        self.retrieval_limit = retrieval_limit

        self.general_kmu = KnowledgeManagementUnit(
            llm=self.llm,
            main_goal=f"Extract relevant information that can be inferred from the task-specific feedback from the feedback agent to do the main goal better- Main goal: {main_role}",
            storage_goal=f"Store general knowledge needed by agents to do the main role - Main goal: {main_role} in the best way possible. Encourage storing knowledge that can be used in future tasks and not just for the current task.",
            retrieval_goal=f"Retrieve relevant general knowledge needed to solve the task based on the main role - Main goal: {main_role}",
            chroma_db="./.db",
            compress_knowledge=compress_knowledge,
            collection_name=collection_name,
        )
        self.prompt_kmu = KnowledgeManagementUnit(
            llm=self.llm,
            main_goal=f"Extract relevant information needed to construct high-quality prompts based on the feedback from answering specific task-related query with a given old prompt. - Main goal:  {main_role} ",
            storage_goal=f"Store prompt/role/function of agent-related knowledge and prompt's effectiveness insights like strengths and weaknesses needed to perform the main role - Main goal: {main_role} better on given tasks. Encourage storing knowledge that can be used in future tasks and not just for the current task.",
            retrieval_goal=f"Retrieve relevant prompt insights needed for constructing better prompts to achieve the - Main goal: {main_role} for given tasks",
            chroma_db="./.db",
            compress_knowledge=compress_knowledge,
            collection_name=collection_name,
        )

        self.operational_agent = None
        self.pruning_queue = []
        self.training_iterations = 0
        self.custom_analysis_agent = None

        # Initialize agents
        self.initialize_agents()

        self._loop = asyncio.get_event_loop()
        self._executor = ThreadPoolExecutor(max_workers=1)

    def set_analysis_agent(self, custom_analysis_agent):
        self.custom_analysis_agent = custom_analysis_agent

    async def create_operational_agent(self, prompt):
        print_cond_rich(
            f"Creating Operational Agent with the prompt:\n{prompt}",
            title="Agent Creation",
        )
        if self.custom_analysis_agent is None:
            self.operational_agent = Agent(
                name="OperationalAgent",
                role="Execute tasks based on the provided prompt",
                function=prompt,
            )
        else:
            self.operational_agent = lambda input: self.custom_analysis_agent(
                prompt, input
            )

    def initialize_agents(self):
        self.meta_prompt_agent = Agent(
            name="MetaPromptAgent",
            role="Generate and optimize prompts for the Operational Agent",
            function=f"""
            Main Goal: {self.main_role}
            
            Your task is to generate and optimize prompts for the Operational Agent.
            Consider the main goal, available knowledge, and task at hand.
            Combine goal-oriented and task-specific elements to create effective prompts that will help solve the given task.
            
            Output a prompt that includes:
            1. A clear definition of the agent's role
            2. A detailed description of the agent's function and responsibilities based on relevant knowledge and task at hand
            3. Specific instructions to help solve the current task or work on it better with the main goal in mind based on relevant knowledge.
            4. Relevant knowledge to consider (if available)
            """,
        )

        self.response_comparison_agent = Agent(
            name="ResponseComparison",
            role="Compare the agent's response to the expected output",
            function="""
            Your task is to compare the agent's response to the expected output.
            check the task, and the expected output and then compare the response to the expected output for the given task.
            - Determine if the response matches the expected output exactly
            - If it does not match exactly see if it matches in meaning and intent clearly and unambiguously.
            - Provide an explanation of your comparison, highlighting similarities and differences
            - Return a boolean indicating whether the response is correct and an explanation
            - if the response says the same thing as expected output (first check verbatim, if not in intent), then it is correct, if not, then it is wrong. First compare them verbatim, if it does not match, then compare them in terms of meaning and intent.
            """,
        )

        self.general_feedback_agent = Agent(
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
             - question and evaluate things you are not sure about, and provide feedback on the reasoning and the response.
            - ask about knowledge gaps and provide feedback on the reasoning and the response or lack of such knowledge.
            Consider the overall goal of the CLU while providing feedback.
            If no expected output is provided, evaluate the response based on its relevance and quality for the given task and overall goal.
            check if the reasoning and the reply makes sense given the main goal of what we are trying to achieve.
            Important: Prefer giving detailed and comprehensive yet not verbose feedback that is not already there in the knowledge, but will help and is correct.
            """,
        )

        self.positive_feedback_agent = Agent(
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
            Consider the overall goal of the CLU while providing feedback.
             - question and evaluate things you are not sure about, and provide feedback on the reasoning and the response.
            - ask about knowledge gaps and provide feedback on the reasoning and the response or lack of such knowledge.
            Consider the overall goal of the CLU while providing feedback.
            If no expected output is provided, evaluate the response based on its relevance and quality for the given task.
            
            Make sure to compare the output to that of expected output, if it is the same give reasons as to what made the main agent reply this way and extract insights, if its wrong, extract and explain what caused it wrong warning signs and how to avoid it in future to make sure we use this for future new tasks.
            Do not give positive feedback for wrong expected output, make sure to give feedback on what went wrong and how to avoid it in future.
            Important: Prefer giving detailed and comprehensive yet not verbose feedback that is not already there in the knowledge, but will help and is correct.
            """,
        )

        self.negative_feedback_agent = Agent(
            name="NegativeFeedbackAnalysis",
            role="Analyze incorrect responses and provide improvement suggestions",
            function="""
            You are responsible for analyzing incorrect responses from the Operational Agent.
            Your tasks include:
            - Evaluate why the response does not match the expected output
            - Identify gaps in knowledge or reasoning that led to the incorrect answer
            - Suggest specific improvements to the knowledge base or agent's function
            - Provide strategies to avoid similar mistakes in the future
            - provide negative feedback as to what not to do in future.
            - Provide a performance score and detailed feedback
            Consider the overall goal of the CLU while providing feedback.
             - question and evaluate things you are not sure about, and provide feedback on the reasoning and the response.
            - ask about knowledge gaps and provide feedback on the reasoning and the response or lack of such knowledge.
            Consider the overall goal of the CLU while providing feedback.
            If no expected output is provided, evaluate the response based on its relevance and quality for the given task.
            
            Make sure to compare the output to that of expected output, if it is the same give reasons as to what made the main agent reply this way and extract insights, if its wrong, extract and explain what caused it wrong warning signs and how to avoid it in future to make sure we use this for future new tasks.
            Do not give positive feedback for wrong expected output, make sure to give feedback on what went wrong and how to avoid it in future.
            Important: Prefer giving detailed and comprehensive yet not verbose feedback that is not already there in the knowledge, but will help and is correct.
            """,
        )

        self.knowledge_insight_agent = Agent(
            name="KnowledgeInsight",
            role="""
            You are tasked with extracting and generalizing key knowledge from various inputs to support the main goal. 
            Your primary responsibility is to distill generalizable insights that can be applied to future tasks with similar main goals. 
            This involves identifying patterns, formulating reusable knowledge entries, and ensuring the knowledge extracted is broadly applicable and not overly specific to individual tasks.
            """,
            function="""
            You are responsible for extracting and generalizing key knowledge. 
            Your tasks include:

            - Analyzing feedback, tasks, responses, and context to understand the main goal.
            - Provide positive feedback on what worked.
            - Detecting patterns and formulating general rules based on the analysis.
            - Extracting key elements that influence the quality of task answers.
            - Formulating general, reusable knowledge entries that can be applied to future tasks.
            - Provide a clear list of knowledge to be added to the database, the list should be diverse and each element should be unique and contain new knowledge to be added.
            - This list may include the following in a combined format as well: Concepts: Fundamental principles that underpin the task structure to achieve the main goal. Ideas: Innovative concepts that can improve task performance. Principles: Core values or standards to uphold. Rules: Guidelines or protocols that should be followed for optimal task performance.
            - include things in format that will help us do the main goal better based on the task, feedback and reasoning.
            - Include the applicability: Contexts or scenarios where each piece of knowledge is relevant.
            
            derive general rules and principles from the feedback and reasoning that can be applied to future tasks.
            
            you are responsible for adding knowledge in a database that will help us do this the main goal better based on the task, feedback and reasoning.
            IMPORTANT: 
            1.Please add diverse knowledge that is not already there in the context given.
            **2.Make sure you extract and add knowledge that is aligned and geared towards achieving the overall goal.**
            3. do not memorize and store exact input-output pairs but rather the reasoning and the knowledge that can be used to solve the task.
            """,
        )

    async def _get_feedback(
        self,
        oa_response: OperationalAgentOutput,
        expected_output: Optional[str],
        knowledge: Optional[str],
        task: str,
    ) -> FeedbackOutput:
        if expected_output is not None and expected_output != "":
            system_prompt, user_prompt = self.response_comparison_agent.prompt(
                f"Compare Response: <{oa_response.response}>\n to Expected Output: <{expected_output}>\n for the Task: {task}"
            )
            formatted_prompt = self.llm.format_prompt(system_prompt, user_prompt)
            comparison_result = await self.llm.generate_async(
                formatted_prompt, ResponseComparisonOutput
            )

            feedback_input = f"""
            Overall Goal: {self.main_role}
            Task: {task}
            Agent Response: {oa_response.response}
            Agent Reasoning: {oa_response.reasoning}
            Agent Confidence: {oa_response.confidence}
            Expected Output: {expected_output}
            Is Correct: {comparison_result.is_correct}
            Knowledge: {knowledge}
            Comparison Explanation: {comparison_result.explanation}
            """
            system_prompt, user_prompt = (
                self.positive_feedback_agent
                if comparison_result.is_correct
                else self.negative_feedback_agent
            ).prompt(feedback_input)
            formatted_prompt = self.llm.format_prompt(system_prompt, user_prompt)
            feedback = await self.llm.generate_async(formatted_prompt, FeedbackOutput)

        else:
            feedback_input = f"""
            Overall Goal: {self.main_role}
            Task: {task}
            Agent Response: {oa_response.response}
            Agent Reasoning: {oa_response.reasoning}
            Agent Confidence: {oa_response.confidence}
            """
            system_prompt, user_prompt = self.general_feedback_agent.prompt(
                feedback_input
            )
            formatted_prompt = self.llm.format_prompt(system_prompt, user_prompt)
            feedback = await self.llm.generate_async(formatted_prompt, FeedbackOutput)
        return feedback, (
            comparison_result.is_correct if "comparison_result" in locals() else None
        )

    async def prune_knowledge_bases(self):
        for entry in self.pruning_queue:
            if entry["prompt_ids"]:
                await self.prompt_kmu.prune(entry["feedback"], entry["prompt_ids"])
            if entry["general_ids"]:
                await self.general_kmu.prune(entry["feedback"], entry["general_ids"])
        self.pruning_queue = []

    def inference(self, task: str):
        """
        Synchronous wrapper for inference_async method.
        Can be called from Jupyter notebooks or any synchronous context.
        """

        def run_async_in_new_loop():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            return new_loop.run_until_complete(self.inference_async(task))

        return self._executor.submit(run_async_in_new_loop).result()

    def training(self, task: str, expected_output: Optional[str] = None, prune_after=1):
        """
        Synchronous wrapper for training_async method.
        Can be called from Jupyter notebooks or any synchronous context.
        """

        def run_async_in_new_loop():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            return new_loop.run_until_complete(
                self.training_async(task, expected_output, prune_after)
            )

        return self._executor.submit(run_async_in_new_loop).result()

    async def inference_async(self, task: str):
        start_time = time.time()

        # Step 1: Retrieve prompt knowledge
        print_cond_rich(
            f"Retrieving relevant prompt knowledge for task: [italic]{task}[/italic]",
            title="Knowledge Retrieval 📚",
            style="cyan",
        )
        prompt_knowledge = await self.prompt_kmu.retrieve_knowledge(
            task, n_results=self.retrieval_limit
        )
        prompt_knowledge_str = (
            " ".join(
                [item for sublist in prompt_knowledge["documents"] for item in sublist]
            )
            if prompt_knowledge["documents"]
            else "No relevant prompt knowledge found."
        )

        # Step 2: Generate prompt using Meta-Prompt Agent
        print_cond_rich(
            f"Generating prompt for task: [italic]{task}[/italic] using MetaPromptAgent",
            title="Prompt Generation 📝",
            style="magenta",
        )
        prompt_input = f"Main Goal: {self.main_role}\nTask: {task}\nRelevant Prompt Knowledge: {prompt_knowledge_str}"
        system_prompt, user_prompt = self.meta_prompt_agent.prompt(prompt_input)
        formatted_prompt = self.llm.format_prompt(system_prompt, user_prompt)
        generated_prompt = await self.llm.generate_async(
            formatted_prompt, PromptGenerationOutput
        )

        # Step 3: Create operational agent
        await self.create_operational_agent(generated_prompt.prompt)

        # Step 4: Retrieve general knowledge
        print_cond_rich(
            f"Retrieving general knowledge for task: [italic]{task}[/italic]",
            title="General Knowledge Retrieval 📖",
            style="cyan",
        )
        general_knowledge_input = f"Main Goal: {self.main_role}\nTask: {task}\nTask Solving Prompt: {generated_prompt.prompt}"
        general_knowledge = await self.general_kmu.retrieve_knowledge(
            general_knowledge_input, n_results=self.retrieval_limit
        )
        general_knowledge_str = (
            " ".join(
                [item for sublist in general_knowledge["documents"] for item in sublist]
            )
            if general_knowledge["documents"]
            else "No relevant general knowledge found."
        )

        # Step 5: Execute operational agent with retrieved knowledge
        print_cond_rich(
            f"Executing Operational Agent for task: [italic]{task}[/italic] with the following knowledge:\n[dim]{general_knowledge_str}[/dim]",
            title="Agent Execution ⚙️",
            style="bold blue",
        )
        oa_input = f"Task: {task}\nRelevant Knowledge: {general_knowledge_str}"
        system_prompt, user_prompt = self.operational_agent.prompt(oa_input)
        formatted_prompt = self.llm.format_prompt(system_prompt, user_prompt)
        oa_response = await self.llm.generate_async(
            formatted_prompt, OperationalAgentOutput
        )

        # Step 6: Show results
        end_time = time.time()
        execution_time = end_time - start_time

        print_cond_rich(
            f"Inference Completed for task: [italic]{task}[/italic] in [bold]{execution_time:.2f}[/bold] seconds",
            title="Inference Complete 🎉",
            style="green",
        )

        # Displaying inference output
        format_result_table(
            {
                "Response": oa_response.response,
                "Reasoning": oa_response.reasoning,
                "Confidence": oa_response.confidence,
                "Knowledge Used": general_knowledge_str,
            },
            title="Inference Result",
            style="green",
        )

        return InferenceOutput(
            operational_agent_response=oa_response, knowledge_used=general_knowledge_str
        )

    async def training_async(
        self, task: str, expected_output: Optional[str] = None, prune_after=1
    ) -> TrainingOutput:
        start_time = time.time()

        print_cond_rich(
            f"Starting [bold]Training[/bold] for task: [italic]{task}[/italic]",
            title="Training Start 🎯",
            style="yellow",
        )

        # Step 1: Run inference first
        inference_result = await self.inference_async(task)

        # Step 2: Generate feedback
        print_cond_rich(
            f"Generating feedback for task: [italic]{task}[/italic]",
            title="Feedback Generation 📝",
            style="magenta",
        )
        feedback, right_answer = await self._get_feedback(
            oa_response=inference_result.operational_agent_response,
            expected_output=expected_output,
            knowledge=inference_result.knowledge_used,
            task=task,
        )

        # Step 3: Store feedback and knowledge updates
        print_cond_rich(
            "Storing prompt feedback and updating knowledge entries...",
            title="Knowledge Management 💾",
            style="cyan",
        )
        await self.store_prompt_feedback(
            inference_result.operational_agent_response.response,
            task,
            inference_result.operational_agent_response.confidence,
            feedback.feedback,
        )

        # Step 4: Knowledge update and prune
        system_prompt, user_prompt = self.knowledge_insight_agent.prompt(
            f"Overall Goal: {self.main_role}\nAnalyze feedback: {feedback}\nTask: {task}\nResponse: {inference_result.operational_agent_response.response}\nReasoning: {inference_result.operational_agent_response.reasoning}"
        )
        formatted_prompt = self.llm.format_prompt(system_prompt, user_prompt)
        knowledge_update = await self.llm.generate_async(
            formatted_prompt, KnowledgeInsightOutput
        )

        if knowledge_update.new_knowledge:
            tasks = [
                self.general_kmu.save_knowledge(knowledge)
                for knowledge in knowledge_update.new_knowledge
            ]
            await asyncio.gather(*tasks)

        if len(self.pruning_queue) >= self.pruning_queue_size:
            print_cond_rich(
                "Pruning knowledge base...",
                title="Pruning Knowledge 🌳",
                style="bold cyan",
            )
            await self.prune_knowledge_bases()

        # Show results
        end_time = time.time()
        execution_time = end_time - start_time

        print_cond_rich(
            f"Training Completed for task: [italic]{task}[/italic] in [bold]{execution_time:.2f}[/bold] seconds",
            title="Training Complete 🏁",
            style="green",
        )

        # Display training output neatly
        format_result_table(
            {
                "Feedback": feedback.feedback,
                "Right Answer": right_answer,
                "New Knowledge": knowledge_update.new_knowledge,
            },
            title="Training Result",
            style="green",
        )

        return TrainingOutput(
            inference_output=inference_result,
            feedback=feedback,
            knowledge_update=knowledge_update.new_knowledge,
            right_answer=right_answer,
        )

    async def store_prompt_feedback(self, prompt, task, performance, feedback):
        insight = f"Prompt: {prompt}\nTask: {task}\nPerformance: {performance}\nFeedback: {feedback}"
        await self.prompt_kmu.save_knowledge(insight)

    def print_knowledge_base(self):
        console = Console()
        console.print("General Knowledge Base:")
        general_entries = self.general_kmu.collection.get()
        if not general_entries["ids"]:
            console.print("[bold red]Knowledge base is empty.[/bold red]")
        else:
            for idx, doc in enumerate(general_entries["documents"]):
                console.print(f"Entry {idx + 1}: {doc}")

        console.print("\nPrompt Knowledge Base:")
        prompt_entries = self.prompt_kmu.collection.get()
        if not prompt_entries["ids"]:
            console.print("[bold red]Knowledge base is empty.[/bold red]")
        else:
            for idx, doc in enumerate(prompt_entries["documents"]):
                console.print(f"Entry {idx + 1}: {doc}")

    # ------------Old Methods----------------

    def call(self, *args, **kwargs):
        """
        Synchronous wrapper for call_async method.
        Can be called from Jupyter notebooks or any synchronous context.
        """

        def run_async_in_new_loop():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            return new_loop.run_until_complete(self.call_async(*args, **kwargs))

        return self._executor.submit(run_async_in_new_loop).result()

    async def call_async(
        self,
        task: str,
        expected_output: Optional[str] = None,
        mode: str = "",
        prune_after=1,
        **kwargs,
    ):
        # Retrieve relevant prompt knowledge
        prompt_knowledge = await self.prompt_kmu.retrieve_knowledge(
            task, n_results=self.retrieval_limit
        )
        prompt_knowledge_str = (
            " ".join(
                [
                    item
                    for sublist in prompt_knowledge["documents"]
                    for item in (sublist if isinstance(sublist, list) else [sublist])
                ]
            )
            if prompt_knowledge["documents"]
            else "No relevant prompt knowledge found."
        )

        # Generate prompt using Meta-Prompt Agent
        prompt_input = f"""
        Main Goal: {self.main_role}
        Task: {task}
        Relevant Prompt Knowledge: {prompt_knowledge_str}
        """
        system_prompt, user_prompt = self.meta_prompt_agent.prompt(prompt_input)
        formatted_prompt = self.llm.format_prompt(system_prompt, user_prompt)
        generated_prompt = await self.llm.generate_async(
            formatted_prompt, PromptGenerationOutput
        )

        # Create or update Operational Agent with the generated prompt
        await self.create_operational_agent(generated_prompt.prompt)

        general_knowledge_input = f"""
        Main Goal: {self.main_role}
        Task: {task}
        Task Solving Prompt: {generated_prompt.prompt}
        """
        # Retrieve relevant general knowledge
        general_knowledge = await self.general_kmu.retrieve_knowledge(
            general_knowledge_input, n_results=self.retrieval_limit
        )
        general_knowledge_str = (
            " ".join(
                [
                    item
                    for sublist in general_knowledge["documents"]
                    for item in (sublist if isinstance(sublist, list) else [sublist])
                ]
            )
            if general_knowledge["documents"]
            else "No relevant general knowledge found."
        )

        # Execute task with relevant general knowledge
        oa_input = f"""
        Task: {task}
        Relevant Knowledge: {general_knowledge_str}

        Use the above relevant knowledge to inform your response. If no relevant knowledge is provided, rely on your general understanding.
        """
        system_prompt, user_prompt = self.operational_agent.prompt(oa_input)
        formatted_prompt = self.llm.format_prompt(system_prompt, user_prompt)
        oa_response = await self.llm.generate_async(
            formatted_prompt, OperationalAgentOutput
        )

        right_answer = None
        if mode == "training":
            feedback, right_answer = await self._get_feedback(
                oa_response=oa_response,
                expected_output=expected_output,
                knowledge=general_knowledge_str,
                task=task,
            )

            # Store prompt feedback
            await self.store_prompt_feedback(
                generated_prompt.prompt, task, oa_response.confidence, feedback.feedback
            )

            system_prompt, user_prompt = self.knowledge_insight_agent.prompt(
                f"Overall Goal: {self.main_role}\nAnalyze feedback: {feedback}\nTask: {task}\nResponse: {oa_response.response}\nReasoning: {oa_response.reasoning}"
            )
            formatted_prompt = self.llm.format_prompt(system_prompt, user_prompt)
            knowledge_update = await self.llm.generate_async(
                formatted_prompt, KnowledgeInsightOutput
            )

            if knowledge_update.new_knowledge:
                tasks = []  # List to store all the async tasks
                for knowledge in knowledge_update.new_knowledge:
                    # Append each save_knowledge task to the list
                    tasks.append(self.general_kmu.save_knowledge(knowledge))

                # Use asyncio.gather to run all tasks concurrently
                await asyncio.gather(*tasks)

                # Print message after all tasks are complete

            # Flatten and check IDs
            prompt_ids = prompt_knowledge.get("ids", [])
            general_ids = general_knowledge.get("ids", [])

            if prompt_ids and general_ids:
                flattened_prompt_ids = [
                    item for sublist in prompt_ids for item in sublist
                ]
                flattened_general_ids = [
                    item for sublist in general_ids for item in sublist
                ]
                if self.training_iterations % prune_after == 0:
                    if flattened_prompt_ids and flattened_general_ids:
                        full_feedback_input = f"""
                        Overall Goal: {self.main_role}
                        Task: {task}
                        Agent Response: {oa_response.response}
                        Expected Output: {expected_output}
                        
                        Feedback for the agent's response: {feedback.feedback}
                        """
                        self.pruning_queue.append(
                            {
                                "feedback": full_feedback_input,
                                "prompt_ids": flattened_prompt_ids,
                                "general_ids": flattened_general_ids,
                            }
                        )

            if len(self.pruning_queue) >= self.pruning_queue_size:
                await self.prune_knowledge_bases()
            self.training_iterations += 1
            return {
                "response": oa_response.response,
                "reasoning": oa_response.reasoning,
                "confidence": oa_response.confidence,
                "feedback": feedback.dict(),
                "knowledge_update": knowledge_update.new_knowledge,
                "right_answer": right_answer,
            }

        return {
            "response": oa_response.response,
            "reasoning": oa_response.reasoning,
            "confidence": oa_response.confidence,
            "knowledge_update": "None",
            "right_answer": right_answer,
        }
