import asyncio
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Dict, List, Optional, Type

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
    if verbosity:
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
                role="Execute tasks based on the provided prompt. Think step by step based on the context and execute the task. Use the knowledge provided to solve the task.",
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
            You are responsible for generating and optimizing prompts for the Operational Agent by effectively synthesizing general and task-specific knowledge. To create an effective prompt, follow these instructions:

            1. **Role Definition**: Clearly define the agent's role by specifying the broader goal and the expected actions. Highlight how this role contributes to achieving the main goal.
            2. **Task-Specific Insights**: Extract relevant knowledge from previous similar tasks (Prompt-Specific Knowledge) to adapt the prompt for the current scenario.
            3. **Knowledge Integration**: Integrate general knowledge that could be useful in providing contextual understanding for the task. Combine insights to provide an enriched prompt.
            4. **Feedback Utilization**: If available, use feedback from past task execution to enhance this prompt. Identify areas for improvement and adjust accordingly to refine the prompt’s quality.
            5. **Output Requirements**: Ensure the prompt includes specific instructions that help to solve the current task effectively, specifying the criteria for success.

            Output Format: 
            - **Role**: Define the agent’s role.
            - **Function**: Detailed description of the responsibilities.
            - **Instructions**: Task-specific guidance based on retrieved knowledge.
            - **Knowledge to Consider**: Integrate relevant general knowledge or insights from prior prompts.
            """,
        )

        self.response_comparison_agent = Agent(
            name="ResponseComparison",
            role="Compare the agent's response to the expected output",
            function="""
                        You are responsible for comparing the Operational Agent's response to the expected output for the given task. Follow these steps:

            1. **Exact Match**: First, compare the response and the expected output verbatim. If they match, mark it as correct.
            2. **Semantic Similarity**: If there is no verbatim match, evaluate whether the response and expected output convey the same meaning. Use contextual cues to determine if the key information aligns.
            3. **Goal Alignment Check**: Evaluate if the response aligns with the overall main goal of the task, even if the format differs. Assess how well the response satisfies the broader objectives.
            4. **Detailed Analysis**: Highlight the key similarities and differences between the response and expected output.
            5. **Output Result**: Provide a boolean indicating whether the response is correct, along with a comprehensive explanation.

            Make sure your comparison is thorough, factoring in both correctness and goal alignment.
            """,
        )

        self.general_feedback_agent = Agent(
            name="GeneralFeedbackAnalysis",
            role="Provide comprehensive feedback on the operational agent's performance, focusing on the reasoning behind the response, the relevance of the retrieved knowledge, and the overall alignment with the main goal.",
            function="""
            You are responsible for evaluating the response of the Operational Agent, even when no expected output is provided.
            Assess the relevance of the retrieved knowledge, how effectively it was utilized, and whether it aligns with the task at hand and the main goal.
            Analyze the reasoning process used to derive the response. Identify any gaps in reasoning or areas where the connection between knowledge and task execution could be stronger.
            Provide detailed, actionable feedback on the strengths and weaknesses of the response, emphasizing any knowledge gaps and opportunities for refining or correcting the reasoning used.
            Highlight potential improvements or new approaches that align the reasoning more closely with the main goal, considering the quality of the retrieved knowledge and its application.
            Deliver a performance score between 0 and 1, and offer suggestions that will help the Operational Agent refine its reasoning in future tasks.
            Reinforce the importance of effective reasoning and the correct use of knowledge, suggesting how to adaptively learn from this feedback.
            Important: Prefer giving detailed and comprehensive yet not verbose feedback that is not already there in the knowledge, but will help and is correct.
            """,
        )

        self.positive_feedback_agent = Agent(
            name="PositiveFeedbackAnalysis",
            role="Analyze successful responses to reinforce correct application of knowledge and reasoning, and to reinforce the learning.",
            function="""
            When the response is correct, you are tasked with evaluating why the correct outcome was achieved and reinforcing the positive aspects of the reasoning and knowledge use. Specifically, you need to:
            - **Identify Key Elements of Success**: Analyze and identify the key factors that led to the successful response. Evaluate the role of retrieved knowledge and the reasoning process. What knowledge was most relevant and why? What reasoning steps were crucial for reaching the correct outcome?
            - **Reinforce Correct Knowledge Use**: Reinforce why the retrieved knowledge was appropriate for the task. Explain how the retrieved information contributed to the success of the task and how the correct application of this knowledge was instrumental in achieving the desired output.
            - **Generalize Successful Approach**: Provide insights into how this approach can be generalized and used for similar tasks in the future. Highlight specific reasoning pathways and decision-making processes that were effective, so that they can be replicated.
            - **Update Knowledge Base with Successful Patterns**: Suggest updating the knowledge base with the successful strategies and patterns observed. Reinforce any positive aspects of the prompt and general knowledge that were particularly effective in producing the correct output. Emphasize new, specific, reusable knowledge entries that can guide similar tasks in the future.
            - **Emphasize Correct Reasoning and Knowledge Pathways**: Describe why the reasoning taken by the agent aligned well with the broader goal. Explain why the knowledge retrieved was not only appropriate but also ideal for the task and how these specific actions can be used to improve adaptability and generalization for other similar tasks.
            - **Provide Actionable Recommendations for Future Success**: Offer specific points that must be repeated to ensure consistent success. Describe how to best utilize this correct knowledge and effective reasoning pattern in similar future scenarios, so the agent has a better chance of replicating the same success.
            - **Feedback for Consistency**: Ensure that all relevant reasoning steps are documented and stored in the knowledge base, which can reinforce consistency across future task execution.
            Important: Prefer giving detailed, comprehensive feedback that includes analysis not already in the knowledge base, but which is valuable to reinforce success. Be explicit about what actions worked, which knowledge was beneficial, and why. This feedback should enhance the agent's future decision-making capabilities.
            """,
        )

        self.negative_feedback_agent = Agent(
            name="NegativeFeedbackAnalysis",
            role="Critically analyze incorrect responses, focusing on evaluating the retrieved knowledge, flaws in reasoning, and providing corrective feedback.",
            function="""
            When the response is incorrect, you are tasked with evaluating the root cause of the failure. You need to:
            - **Critically Evaluate Retrieved Knowledge**: Analyze why the retrieved knowledge was incorrect, insufficient, or improperly used for the given task. Explicitly determine if the knowledge itself was flawed or if there was a gap in the knowledge that prevented the correct reasoning.
            - **Reasoning and Application Flaws**: Identify specific flaws in how the retrieved knowledge was applied. Was the reasoning appropriate given the task at hand? How did the use (or misuse) of knowledge lead to the deviation from the expected output?
            - **Explain Impact on Outcome**: Explain how the incorrect or insufficient use of knowledge affected the overall outcome and why the correct answer wasn't reached. Was there a misinterpretation of the retrieved information, or was critical information missing?
            - **Provide Knowledge Correction Suggestions**: Suggest specific corrections to the existing knowledge. Highlight what new insights, corrections, or additional knowledge should be added to bridge the identified gaps. Make it explicit how these corrections can improve future performance for similar tasks.
            - **Improvement Strategies**: Provide actionable strategies to ensure similar mistakes are avoided in the future. Explain how to properly apply relevant knowledge and improve reasoning steps to ensure alignment with the main goal.
            - **New Knowledge Recommendations**: Recommend new knowledge entries that could help better address the knowledge gap. Be explicit about what was missing and what new information would have led to the correct output.
            Make sure to compare the output to the expected output. If it is incorrect, explain in detail why it went wrong, what knowledge or reasoning caused the deviation, and how these specific issues can be addressed in future iterations.
            Important: Your feedback must be critical yet constructive. Prefer detailed and comprehensive analysis that is not already in the knowledge base, focusing on gaps that, if filled, would have a substantial impact on improving the accuracy of the response. Be explicit about both the reasoning and the knowledge gaps, and focus on changes that need to be made to align with the overall main goal.
            """,
        )

        self.knowledge_insight_agent = Agent(
            name="KnowledgeInsight",
            role="""
            You are responsible for converting feedback (positive or negative) into generalizable knowledge entries for the knowledge management units (KMUs). 
            Your main objective is to analyze the feedback, understand what worked or failed, and derive broadly applicable knowledge that enhances the system's future performance.
            The knowledge you extract should be abstract enough to apply to a wide range of similar tasks, while still being actionable and relevant.
            """,
            function="""
            You are tasked with transforming feedback into meaningful and generalizable knowledge entries to be stored in the KMU. Your responsibilities include:
            - **Analyze Feedback, Tasks, and Responses**: Analyze the feedback provided (positive or negative) along with the context of the task and the response. Understand both what led to the success and where failures occurred.
            - **Distill Generalizable Insights**: Based on your analysis, detect patterns and derive general rules that could improve the agent's performance in similar future tasks. This might include extracting effective strategies, pitfalls to avoid, or any refined reasoning pathways that align with the overall goal.
            - **Extract Reusable Knowledge and Rules**: Formulate knowledge entries that can be used for future tasks. Ensure that each piece of knowledge is unique, diverse, and represents new learnings. Avoid storing direct input-output pairs; instead, focus on storing the reasoning process, patterns, and knowledge that help the system solve similar tasks more effectively.
            - **Differentiate Between Types of Knowledge**:
                - **Concepts**: Identify and store the fundamental principles that underpin the successful completion of the task.
                - **Ideas**: Capture innovative concepts that led to better task performance.
                - **Principles and Rules**: Store core principles, guidelines, or protocols that can improve performance across a range of similar tasks.
                - **Effective Prompts and Reasoning Paths**: Store prompts and reasoning that contributed positively to the task, along with the associated task contexts.
            - **Applicability and Context**: For each piece of knowledge, specify the contexts or scenarios where it is relevant. This will help in retrieving the right knowledge when needed in the future.
            - **Identify and Fill Gaps**: If there were knowledge gaps that led to negative outcomes, determine what knowledge could have prevented these issues and include it in the entries. Similarly, reinforce any effective knowledge that led to success.
            - **Ensure Knowledge Diversity and Quality**: Make sure the extracted knowledge is diverse and not redundant. It should add meaningful insights to the knowledge base that are not already present. Prioritize knowledge that is actionable and can directly improve task completion in the future.
            IMPORTANT:
            1. Do not memorize and store exact input-output pairs. Focus on the reasoning, patterns, and knowledge that support task completion.
            2. Extract knowledge that is geared towards achieving the main goal more effectively in the future.
            3. Reinforce successful reasoning pathways and highlight corrective knowledge where failures occurred to ensure overall system improvement.
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

    def inference(self, task: str, output_schema: Optional[Type] = str):
        """
        Synchronous wrapper for inference_async method.
        Can be called from Jupyter notebooks or any synchronous context.
        """

        def run_async_in_new_loop():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            return new_loop.run_until_complete(
                self.inference_async(task, output_schema)
            )

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

    async def inference_async(self, task: str, output_schema: Optional[Type] = str):
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

        # Dynamically get the OperationalAgentOutput schema based on output_schema
        OperationalAgentOutput = self.get_operational_agent_schema(output_schema)
        oa_response = await self.llm.generate_async(
            formatted_prompt, OperationalAgentOutput
        )

        # Dynamically get InferenceOutput schema based on OperationalAgentOutput
        InferenceOutput = self.get_inference_output_schema(OperationalAgentOutput)

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
        self,
        task: str,
        expected_output: Optional[str] = None,
        prune_after=1,
        output_schema: Optional[Type] = str,
    ):
        start_time = time.time()

        print_cond_rich(
            f"Starting [bold]Training[/bold] for task: [italic]{task}[/italic]",
            title="Training Start 🎯",
            style="yellow",
        )

        # Step 1: Run inference first
        inference_result = await self.inference_async(task, output_schema=output_schema)

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

        end_time = time.time()
        execution_time = end_time - start_time

        print_cond_rich(
            f"Training Completed for task: [italic]{task}[/italic] in [bold]{execution_time:.2f}[/bold] seconds",
            title="Training Complete 🏁",
            style="green",
        )

        # Dynamically get TrainingOutput schema based on InferenceOutput
        TrainingOutput = self.get_training_output_schema(type(inference_result))

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

    def get_operational_agent_schema(self, output_schema: Optional[Type] = str):
        """
        Dynamically creates OperationalAgentOutput with the response field's type
        depending on the output_schema provided.
        """

        class DynamicOperationalAgentOutput(BaseModel):
            reasoning: str = Field(
                ..., description="Detailed reasoning behind the response"
            )
            response: output_schema = Field(
                ..., description="Response from the Operational Agent"
            )
            confidence: float = Field(
                ..., description="Confidence in the response between 0 and 1"
            )

        return DynamicOperationalAgentOutput

    def get_inference_output_schema(
        self, operational_agent_output_schema: Type[BaseModel]
    ):
        """
        Dynamically creates InferenceOutput based on the operational_agent_output_schema.
        """

        class DynamicInferenceOutput(BaseModel):
            operational_agent_response: operational_agent_output_schema = Field(
                ..., description="Operational Agent response"
            )
            knowledge_used: str = Field(
                ..., description="Knowledge used by the Operational Agent"
            )

        return DynamicInferenceOutput

    def get_training_output_schema(self, inference_output_schema: Type[BaseModel]):
        """
        Dynamically creates TrainingOutput based on the inference_output_schema.
        """

        class DynamicTrainingOutput(BaseModel):
            inference_output: inference_output_schema = Field(
                ..., description="The result of the inference step."
            )
            feedback: FeedbackOutput = Field(
                ..., description="The feedback provided during training."
            )
            knowledge_update: List[str] = Field(
                ..., description="The updated knowledge entries."
            )
            right_answer: Optional[bool] = Field(
                None,
                description="Whether the Operational Agent's response was correct.",
            )

        return DynamicTrainingOutput

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
