import uuid
from threading import Lock
from typing import Dict, List, Optional

from pydantic import BaseModel, Field
from rich.console import Console

from agents import Agent

from .knowldge import KnowledgeManagementSystem

verbosity = False
disable_loging = True


## -- utility functions ---
def print_cond(text, verbose=verbosity):
    if verbose:
        print(text)


def generate_short_uuid(length=8):
    short_uuid = str(uuid.uuid4()).replace("-", "")[:length]
    return short_uuid


# pydantic models ---
class FeedbackOutput(BaseModel):
    performance_score: float = Field(
        ..., description="Performance score of the Operational Agent", ge=0.0, le=1.0
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
        ..., description="Confidence in the response", ge=0.0, le=1.0
    )


class MLU:
    def __init__(
        self,
        main_role,
        pruning_queue_size=1,
        compress_knowledge=True,
        collection_name=None,
        retrival_limit=10,
    ):
        self.main_role = main_role
        self.mlu_id = generate_short_uuid(5)
        self.pruning_queue_size = pruning_queue_size
        self.retrival_limit = retrival_limit

        self.general_kms = KnowledgeManagementSystem(
            main_goal=f"Extract relevant information that can be inferred from the task-specific feedback from the feedback agent to do the main goal better- Main goal: {main_role}",
            storage_goal=f"Store general knowledge needed by agents to do the main role - Main goal: {main_role} in the best way possible. Encourage storing knowledge that can be used in future tasks and not just for the current task.",
            retrieval_goal=f"Retrieve relevant general knowledge needed to solve the task based on the main role - Main goal: {main_role}",
            chroma_db_path="./.db",
            compress_knowldge=compress_knowledge,
            collection_name=collection_name,
        )
        self.prompt_kms = KnowledgeManagementSystem(
            main_goal=f"Extract relevant information needed to construct high-quality prompts based on the feedback from answering specific task-related query with a given old prompt. - Main goal:  {main_role} ",
            storage_goal=f"Store prompt/role/function of agent-related knowledge and prompt's effectiveness insights like strengths and weaknesses needed to perform the main role - Main goal: {main_role} better on given tasks. Encourage storing knowledge that can be used in future tasks and not just for the current task.",
            retrieval_goal=f"Retrieve relevant prompt insights needed for constructing better prompts to achieve the - Main goal: {main_role} for given tasks",
            chroma_db_path="./.db",
            compress_knowldge=compress_knowledge,
            collection_name=collection_name,
        )

        self.operational_agent = None
        self.pruning_queue = []
        self.training_iterations = 0

        # Initialize agents
        self.initialize_agents()
        self.create_operational_agent = self.default_create_operational_agent

    def set_analysis_agent(self, custom_analysis_agent):
        self.create_operational_agent = custom_analysis_agent

    def default_create_operational_agent(self, prompt):
        self.operational_agent = Agent(
            name="OperationalAgent",
            role="Execute tasks based on the provided prompt",
            function=prompt,
            output_model=OperationalAgentOutput,
            disable_logging=disable_loging,
        )

    def initialize_agents(self):
        class PromptGenerationOutput(BaseModel):
            prompt: str = Field(..., description="Generated prompt")

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
            output_model=PromptGenerationOutput,
            disable_logging=disable_loging,
        )

        class ResponseComparisonOutput(BaseModel):
            is_correct: bool = Field(
                ..., description="Whether the response matches the expected output"
            )
            explanation: str = Field(
                ..., description="Explanation of the comparison result"
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
            output_model=ResponseComparisonOutput,
            disable_logging=disable_loging,  # To disable sql for multithreading
        )

        class FeedbackOutput(BaseModel):
            performance_score: float = Field(
                ...,
                description="Performance score of the Operational Agent",
                ge=0.0,
                le=1.0,
            )
            feedback: str = Field(..., description="Detailed feedback on the response")
            improvement_suggestions: List[str] = Field(
                ..., description="Suggestions for improvement"
            )
            knowledge_gaps: List[str] = Field(
                ..., description="Identified knowledge gaps"
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
    Consider the overall goal of the MLU while providing feedback.
     - question and evaluate things you are not sure about, and provide feedback on the reasoning and the response.
    - ask about knowledge gaps and provide feedback on the reasoning and the response or lack of such knowledge.
    Consider the overall goal of the MLU while providing feedback.
    If no expected output is provided, evaluate the response based on its relevance and quality for the given task and overall goal.
    check if the reasoning and the reply makes sense given the main goal of what we are trying to achieve.""",
            output_model=FeedbackOutput,
            disable_logging=disable_loging,  # To disable sql for multithreading
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
    Consider the overall goal of the MLU while providing feedback.
     - question and evaluate things you are not sure about, and provide feedback on the reasoning and the response.
    - ask about knowledge gaps and provide feedback on the reasoning and the response or lack of such knowledge.
    Consider the overall goal of the MLU while providing feedback.
    If no expected output is provided, evaluate the response based on its relevance and quality for the given task.
    
    Make sure to compare the output to that of expected output, if it is the same give reasons as to what made the main agent reply this way and extract insights, if its wrong, extract and explain what caused it wrong warning signs and how to avoid it in future to make sure we use this for future new tasks.
    Do not give positive feedback for wrong expected output, make sure to give feedback on what went wrong and how to avoid it in future.
    """,
            output_model=FeedbackOutput,
            disable_logging=disable_loging,  # To disable sql for multithreading
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
    - Provide a performance score and detailed feedback
    Consider the overall goal of the MLU while providing feedback.
     - question and evaluate things you are not sure about, and provide feedback on the reasoning and the response.
    - ask about knowledge gaps and provide feedback on the reasoning and the response or lack of such knowledge.
    Consider the overall goal of the MLU while providing feedback.
    If no expected output is provided, evaluate the response based on its relevance and quality for the given task.
    
    Make sure to compare the output to that of expected output, if it is the same give reasons as to what made the main agent reply this way and extract insights, if its wrong, extract and explain what caused it wrong warning signs and how to avoid it in future to make sure we use this for future new tasks.
    Do not give positive feedback for wrong expected output, make sure to give feedback on what went wrong and how to avoid it in future.
    """,
            output_model=FeedbackOutput,
            disable_logging=disable_loging,  # To disable sql for multithreading
        )

        class KnowledgeInsightOutput(BaseModel):
            new_knowledge: List[str] = Field(
                ...,
                description="List of new knowledge entries, containing things like rules, ideas, concepts, principles, and their applicability",
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
    - Detecting patterns and formulating general rules based on the analysis.
    - Extracting key elements that influence the quality of task answers.
    - Formulating general, reusable knowledge entries that can be applied to future tasks.
    - Provide a clear list of knowledge to be added to the database, the list should be diverse and each element should be unique and contain new knowledge to be added.
    - This list may include the following in a combined format as well:Concepts: Fundamental principles that underpin the task structure to achieve the main goal.Ideas: Innovative concepts that can improve task performance.Principles: Core values or standards to uphold.Rules: Guidelines or protocols that should be followed for optimal task performance.
    - include things in format that will help us do the main goal better based on the task, feedback and reasoning.
    - Include the applicability: Contexts or scenarios where each piece of knowledge is relevant.
    
    derive general rules and principles from the feedback and reasoning that can be applied to future tasks.
    
    you are responsible for adding knowldge in a database that will help us do this the main goal better based on the task, feedback and reasoning.
    """,
            output_model=KnowledgeInsightOutput,
            disable_logging=disable_loging,  # To disable sql for multithreading
        )

    def _get_feedback(
        self,
        oa_response: OperationalAgentOutput,
        expected_output: Optional[str],
        knowledge: Optional[str],
        task: str,
    ) -> FeedbackOutput:
        if expected_output is not None and expected_output != "":
            comparison_result = self.response_comparison_agent(
                f"Compare Response: <{oa_response.response}>\n to Expected Output: <{expected_output}>\n for the Task: {task}",
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
            Agent Response: {oa_response.response}
            Agent Reasoning: {oa_response.reasoning}
            Agent Confidence: {oa_response.confidence}
            """
            feedback = self.general_feedback_agent(
                feedback_input,
            )
        return feedback, comparison_result.is_correct

    def prune_knowledge_bases(self):
        for entry in self.pruning_queue:
            if entry["prompt_ids"]:
                self.prompt_kms.prune(entry["feedback"], entry["prompt_ids"])
            if entry["general_ids"]:
                self.general_kms.prune(entry["feedback"], entry["general_ids"])
        self.pruning_queue = []

    def call(
        self,
        task: str,
        expected_output: Optional[str] = None,
        mode: str = "",
        prune_after=1,
        **kwargs,
    ):
        # Retrieve relevant prompt knowledge
        prompt_knowledge = self.prompt_kms.retrieve_knowledge(
            task, n_results=self.retrival_limit
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
        generated_prompt = self.meta_prompt_agent(
            prompt_input,
        ).prompt

        # Create or update Operational Agent with the generated prompt
        self.create_operational_agent(generated_prompt)

        general_knowledge_input = f"""
        Main Goal: {self.main_role}
        Task: {task}
        Task Solving Prompt: {generated_prompt}
        """
        # Retrieve relevant general knowledge
        general_knowledge = self.general_kms.retrieve_knowledge(
            general_knowledge_input, n_results=self.retrival_limit
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
        oa_response = self.operational_agent(
            oa_input,
        )

        right_answer = None  # need to return the right answer from the feedback agent and add it. we essentially compare ans to output only in training step when expected output is given.
        if mode == "training":
            feedback, right_answer = self._get_feedback(
                oa_response=oa_response,
                expected_output=expected_output,
                knowledge=general_knowledge_str,
                task=task,
            )

            # Store prompt feedback
            self.store_prompt_feedback(
                generated_prompt, task, oa_response.confidence, feedback.feedback
            )

            knowledge_update = self.knowledge_insight_agent(
                f"Overall Goal: {self.main_role}\nAnalyze feedback: {feedback}\nTask: {task}\nResponse: {oa_response.response}\nReasoning: {oa_response.reasoning}",
            )

            if knowledge_update.new_knowledge:
                for knowledge in knowledge_update.new_knowledge:
                    self.general_kms.save_knowledge(knowledge)

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
                if (
                    self.training_iterations % prune_after == 0
                ):  # Control the pruning frequency

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

            if (
                len(self.pruning_queue) >= self.pruning_queue_size
            ):  # Control the queue size
                self.prune_knowledge_bases()
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

    def store_prompt_feedback(self, prompt, task, performance, feedback):
        insight = f"Prompt: {prompt}\nTask: {task}\nPerformance: {performance}\nFeedback: {feedback}"
        self.prompt_kms.save_knowledge(insight)

    def print_knowledge_base(self):
        console = Console()
        console.print("General Knowledge Base:")
        general_entries = self.general_kms.collection.get()
        if not general_entries["ids"]:
            console.print("[bold red]Knowledge base is empty.[/bold red]")
        else:
            for idx, doc in enumerate(general_entries["documents"]):
                console.print(f"Entry {idx + 1}: {doc}")

        console.print("\nPrompt Knowledge Base:")
        prompt_entries = self.prompt_kms.collection.get()
        if not prompt_entries["ids"]:
            console.print("[bold red]Knowledge base is empty.[/bold red]")
        else:
            for idx, doc in enumerate(prompt_entries["documents"]):
                console.print(f"Entry {idx + 1}: {doc}")
