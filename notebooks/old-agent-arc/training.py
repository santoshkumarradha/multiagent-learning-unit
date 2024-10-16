import sys

sys.path.append("../")
import os

from rich import print

openai_key = open("../openai.key").read().strip()
os.environ["OPENAI_API_KEY"] = openai_key


from multi_agent_llm import KMU, RCLU, OpenAILLM

llm = OpenAILLM(model_name="gpt-4o-mini")


# Perform reasoning
from pydantic import BaseModel, Field


class OperationalAgentOutput(BaseModel):
    explanation: str = Field(..., description="Explanation of logic of transformation")
    answer: str = Field(..., description="Final answer")


clu = RCLU(
    llm,
    main_goal="Find the hidden rule and learn the transformation logic of the cryptographic puzzle.",
    name="PuzzleSolverRCLU1",
    retrieval_memory_count=50,
    compress_knowledge=False,
)

# Perform training
query = """Given:
"Neural networks transform data efficiently" → "eeraf"
"Artificial intelligence automates decisions" → "rnue" 
Query:
What is "Gradient descent optimizes loss functions" → ?"""

expected_output = "repou"


iterations = 50
for i in range(iterations):
    print(f"Iteration {i+1}/{iterations}")
    training_result = clu.train(
        query, expected_output, OperationalAgentOutput, verbose=False
    )
    print(f"Training result: {training_result}")
