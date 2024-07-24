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
    Answer: str = Field(
        ..., description="Final answer. Give only the concise answer and nothing else"
    )
    explanation: str = Field(
        ..., description="Explanation of how the answer was derived"
    )


class ValidationResult(BaseModel):
    is_consistent: bool = Field(
        ...,
        description="Whether the answer is consistent with patterns and rules. AND is the answer correct?",
    )
    feedback: str = Field(..., description="Feedback on the answer's consistency")


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
    role="Extract detailed patterns from examples",
    function="""
    Analyze the given prompt and examples to identify recurring patterns. For each pattern:
    1. Provide a detailed description of the pattern, including any contextual factors.
    2. List and explain the examples that illustrate this pattern.
    3. Capture both simple and complex patterns, including any subtle variations.
    4. Ensure that patterns are well-documented and illustrated with examples.
    """,
    output_model=List[Pattern],
)


transformation_analyzer = Agent(
    name="TransformationAnalyzer",
    role="Analyze and document transformations between patterns",
    function="""
    Examine identified patterns to understand how inputs are transformed into outputs. For each transformation:
    1. Describe the input pattern in detail.
    2. Describe the output pattern in detail.
    3. Formulate precise transformation rules that explain the relationship between input and output.
    4. Document edge cases, exceptions, and variability in transformations.
    5. Ensure rules are generalizable and applicable to various scenarios.
    """,
    output_model=List[Transformation],
)


logical_inference_engine = Agent(
    name="LogicalInferenceEngine",
    role="Infer robust logical rules from patterns and transformations",
    function="""
    Use identified patterns and transformations to infer logical rules that govern the problem-solving process. For each logical rule:
    1. State the premise clearly, including any conditions or scenarios.
    2. State the conclusion logically and concisely.
    3. Ensure the rule is consistent with the examples provided.
    4. Formulate rules that capture both straightforward and complex logical relationships.
    5. Validate rules through hypothetical scenarios to ensure applicability.
    """,
    output_model=List[LogicalRule],
)


solution_synthesizer = Agent(
    name="SolutionSynthesizer",
    role="Synthesize comprehensive solutions using patterns, transformations, and logical rules",
    function="""
    Integrate identified patterns, transformations, and logical rules to propose a solution. For the solution:
    1. Clearly state the proposed answer.
    2. Provide a detailed explanation of how the solution was derived using the identified patterns, transformations, and logical rules.
    3. Ensure the solution is consistent with the examples and hypothetical scenarios.
    4. Validate the solution against both provided examples and hypothetical scenarios to ensure robustness in your mind.
    """,
    output_model=Solution,
)


consistency_validator = Agent(
    name="ConsistencyValidator",
    role="Validate the solution's consistency with patterns, transformations, and logical rules",
    function="""
    Examine the proposed solution in context of the original prompt, identified patterns, transformations, and logical rules. Your task is to:
    1. Determine whether the solution is consistent with the identified patterns.
    2. Verify that the solution correctly applies the identified transformations.
    3. Check if the solution adheres to the inferred logical rules.
    4. Provide a clear verdict on the consistency and check to see if the answer is the correct solution (True/False).
       Do not give True if the answer is not correct.
    5. Offer detailed feedback, highlighting any inconsistencies and suggesting areas for improvement.
    6. Suggest potential improvements or alternative approaches for inconsistent solutions.
    Important: Do not give True if you think the answer is not correct
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

    return Answer(response=conversation.current_solution.Answer, reasoning=conversation)
