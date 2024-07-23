from typing import Any, List, Optional

from pydantic import BaseModel, Field

from agents import Agent


class Concept(BaseModel):
    name: str = Field(..., description="Name of the concept")
    definition: str = Field(..., description="Definition or explanation of the concept")


class LogicalArgument(BaseModel):
    premises: List[str] = Field(..., description="List of premises")
    conclusion: str = Field(..., description="Conclusion drawn from premises")


class ReasoningStep(BaseModel):
    concepts: List[Concept] = Field(
        default_factory=list, description="Concepts identified or used in this step"
    )
    argument: Optional[LogicalArgument] = Field(
        None, description="Logical argument constructed in this step"
    )
    explanation: str = Field(
        ..., description="Detailed explanation of the reasoning process"
    )


class Solution(BaseModel):
    answer: str = Field(..., description="The proposed answer to the problem")
    justification: str = Field(..., description="Justification for the proposed answer")


class Validity(BaseModel):
    is_valid: bool = Field(..., description="Whether the solution is logically valid")
    explanation: str = Field(..., description="Explanation of the validity assessment")


class Conversation(BaseModel):
    problem: str = Field(..., description="The original problem")
    reasoning_steps: List[ReasoningStep] = Field(
        default_factory=list, description="List of reasoning steps"
    )
    current_solution: Solution = Field(
        None, description="The current proposed solution"
    )
    validity_check: Validity = Field(
        None, description="The validity check of the current solution"
    )
    iterations: int = Field(0, description="Number of iterations taken")


disable_logging = True

concept_analyzer = Agent(
    name="ConceptAnalyzer",
    role="Analyze the problem and identify key concepts",
    function="""
    Analyze the given problem and previous reasoning steps. Identify and define key concepts that are crucial for understanding and solving the problem. For each concept:
    1. Provide a clear name for the concept.
    2. Give a concise definition or explanation of the concept in the context of the problem.
    Also, explain how these concepts relate to the problem and to each other.
    """,
    output_model=ReasoningStep,
    disable_logging=disable_logging,
)

logical_reasoner = Agent(
    name="LogicalReasoner",
    role="Construct logical arguments based on identified concepts",
    function="""
    Using the identified concepts and previous reasoning steps, construct a logical argument relevant to solving the problem. Your output should include:
    1. A list of clear, relevant premises based on the concepts and given information.
    2. A logical conclusion drawn from these premises.
    3. A detailed explanation of your reasoning process, including how you arrived at the premises and conclusion.
    """,
    output_model=ReasoningStep,
    disable_logging=disable_logging,
)

solution_formulator = Agent(
    name="SolutionFormulator",
    role="Formulate a solution based on the reasoning steps",
    function="""
    Review all the reasoning steps, including identified concepts and logical arguments. Your task is to:
    1. Propose a clear and concise answer to the original problem.
    2. Provide a detailed justification for this answer, explaining how it follows from the reasoning steps.
    3. Ensure that your solution addresses the key concepts and logical arguments presented in the reasoning steps.
    """,
    output_model=Solution,
    disable_logging=disable_logging,
)

validity_checker = Agent(
    name="ValidityChecker",
    role="Check the logical validity of the proposed solution",
    function="""
    Examine the proposed solution in the context of the original problem and all reasoning steps. Your task is to:
    1. Determine whether the solution logically follows from the reasoning steps and adequately addresses the problem.
    2. Check for any logical fallacies, inconsistencies, or unjustified leaps in reasoning.
    3. Provide a clear verdict on the validity of the solution (True/False).
    4. Give a detailed explanation of your validity assessment, including any issues found or strengths identified.
    5. Make sure to consider the overall coherence and logical flow of the reasoning process.
    6. Critically evaluate the solution based on the concepts and logical arguments presented and see if it the correct answer.
    7. If you are not confident in the validity of the solution, suggest improvements or further analysis.
    """,
    output_model=Validity,
    disable_logging=disable_logging,
)


def logic_concept_reasoning_network(
    problem: str, max_iterations: int = 5
) -> Conversation:
    conversation = Conversation(problem=problem)

    for iteration in range(max_iterations):
        conversation.iterations += 1

        # Concept Analysis
        concept_input = (
            f"Problem: {problem}\nPrevious reasoning: {conversation.reasoning_steps}"
        )
        concept_step = concept_analyzer(concept_input)
        conversation.reasoning_steps.append(concept_step)

        # Logical Reasoning
        logic_input = (
            f"Problem: {problem}\nPrevious reasoning: {conversation.reasoning_steps}"
        )
        logic_step = logical_reasoner(logic_input)
        conversation.reasoning_steps.append(logic_step)

        # Formulate Solution
        solution_input = (
            f"Problem: {problem}\nReasoning steps: {conversation.reasoning_steps}"
        )
        current_solution = solution_formulator(solution_input)
        conversation.current_solution = current_solution

        # Check Validity
        validity_input = f"Problem: {problem}\nReasoning steps: {conversation.reasoning_steps}\nProposed solution: {current_solution}"
        validity_result = validity_checker(validity_input)
        conversation.validity_check = validity_result

        if validity_result.is_valid:
            break
        else:
            # If not valid, add the validity check explanation as a new reasoning step to inform the next iteration
            invalid_step = ReasoningStep(
                concepts=[],
                argument=None,
                explanation=f"Validity check failed: {validity_result.explanation}",
            )
            conversation.reasoning_steps.append(invalid_step)

    class Answer(BaseModel):
        response: str
        reasoning: Conversation

        def __getattribute__(self, name: str) -> Any:
            try:
                return super().__getattribute__(name)
            except:
                return ""

    return Answer(response=current_solution.answer, reasoning=conversation)
