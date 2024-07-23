from typing import Any, List, Optional

from pydantic import BaseModel, Field

from agents import Agent


class Pattern(BaseModel):
    description: str = Field(..., description="Description of the identified pattern")
    examples: List[str] = Field(..., description="Examples illustrating the pattern")


class Transformation(BaseModel):
    input_pattern: str = Field(..., description="Input pattern")
    output_pattern: str = Field(..., description="Output pattern")
    rule: str = Field(..., description="Transformation rule")


class LogicalRule(BaseModel):
    premise: str = Field(..., description="Premise of the logical rule")
    conclusion: str = Field(..., description="Conclusion of the logical rule")


class Solution(BaseModel):
    content: str = Field(..., description="Proposed solution")
    explanation: str = Field(
        ..., description="Explanation of how the solution was derived"
    )


class ValidationResult(BaseModel):
    is_consistent: bool = Field(
        ..., description="Whether the solution is consistent with patterns and rules"
    )
    feedback: str = Field(..., description="Feedback on the solution's consistency")


class Conversation(BaseModel):
    prompt: str = Field(
        ..., description="The original prompt containing problem and examples"
    )
    patterns: List[Pattern] = Field(default_factory=list)
    transformations: List[Transformation] = Field(default_factory=list)
    logical_rules: List[LogicalRule] = Field(default_factory=list)
    current_solution: Optional[Solution] = None
    validation_result: Optional[ValidationResult] = None
    iterations: int = Field(0, description="Number of iterations taken")


class Answer(BaseModel):
    response: str
    reasoning: Conversation

    def __getattribute__(self, name: str) -> Any:
        try:
            return super().__getattribute__(name)
        except:
            return ""


pattern_extractor = Agent(
    name="PatternExtractor",
    role="Extract patterns from examples",
    function="""
    Analyze the given prompt, focusing on the examples provided. Identify recurring patterns in the examples.
    For each pattern:
    1. Provide a clear description of the pattern.
    2. List the examples that illustrate this pattern.
    Ensure that you capture both obvious and subtle patterns that might be relevant to solving the problem.
    """,
    output_model=List[Pattern],
)

transformation_analyzer = Agent(
    name="TransformationAnalyzer",
    role="Analyze transformations between patterns",
    function="""
    Based on the identified patterns, analyze how inputs are transformed into outputs in the examples.
    For each transformation:
    1. Describe the input pattern.
    2. Describe the output pattern.
    3. Formulate a rule that explains how the input is transformed into the output.
    Focus on generalizable transformation rules that could be applied to new inputs.
    """,
    output_model=List[Transformation],
)

logical_inference_engine = Agent(
    name="LogicalInferenceEngine",
    role="Infer logical rules from patterns and transformations",
    function="""
    Using the identified patterns and transformations, infer logical rules that govern the problem-solving process.
    For each logical rule:
    1. State the premise (condition or scenario).
    2. State the conclusion (what can be inferred or deduced).
    Ensure that these logical rules are consistent with the examples and can be applied to solve new problems.
    """,
    output_model=List[LogicalRule],
)

solution_synthesizer = Agent(
    name="SolutionSynthesizer",
    role="Synthesize a solution based on patterns, transformations, and logical rules",
    function="""
    Using the patterns, transformations, and logical rules derived from the examples:
    1. Propose a solution to the problem presented in the prompt.
    2. Provide a detailed explanation of how you applied the patterns, transformations, and logical rules to arrive at this solution.
    3. Ensure that your solution is consistent with the examples provided in the prompt.
    """,
    output_model=Solution,
)

consistency_validator = Agent(
    name="ConsistencyValidator",
    role="Validate the consistency of the proposed solution",
    function="""
    Examine the proposed solution in the context of the original prompt, identified patterns, transformations, and logical rules. Your task is to:
    1. Determine whether the solution is consistent with the patterns observed in the examples.
    2. Verify that the solution correctly applies the identified transformations.
    3. Check if the solution adheres to the inferred logical rules.
    4. Provide a clear verdict on the consistency of the solution (True/False).
    5. Give detailed feedback, including any inconsistencies found or strengths identified.
    6. If the solution is inconsistent, suggest areas for improvement in the next iteration.
    """,
    output_model=ValidationResult,
)


def pattern_based_logical_reasoning_network(
    prompt: str,
    max_iterations: int = 5,
) -> Answer:
    conversation = Conversation(prompt=prompt)

    for iteration in range(max_iterations):
        conversation.iterations += 1

        # Pattern Extraction
        patterns = pattern_extractor(
            f"Prompt: {prompt}\nCurrent patterns: {conversation.patterns}"
        )
        conversation.patterns = patterns

        # Transformation Analysis
        transformations = transformation_analyzer(
            f"Prompt: {prompt}\nPatterns: {patterns}\nCurrent transformations: {conversation.transformations}",
        )
        conversation.transformations = transformations

        # Logical Inference
        logical_rules = logical_inference_engine(
            f"Prompt: {prompt}\nPatterns: {patterns}\nTransformations: {transformations}\nCurrent logical rules: {conversation.logical_rules}",
        )
        conversation.logical_rules = logical_rules

        # Solution Synthesis
        solution = solution_synthesizer(
            f"Prompt: {prompt}\nPatterns: {patterns}\nTransformations: {transformations}\nLogical Rules: {logical_rules}",
        )
        conversation.current_solution = solution

        # Consistency Validation
        validation_result = consistency_validator(
            f"Prompt: {prompt}\nPatterns: {patterns}\nTransformations: {transformations}\nLogical Rules: {logical_rules}\nProposed Solution: {solution}",
        )
        conversation.validation_result = validation_result

        if validation_result.is_consistent:
            break

    return Answer(
        response=conversation.current_solution.content, reasoning=conversation
    )
