import os
import sys

from pydantic import BaseModel
from rich import print

sys.path.append("../../")


os.environ["OPENAI_API_KEY"] = open("../../../openai.key").read().strip()
from multi_agent_llm import OpenAILLM
from multi_agent_llm.agents.mlu.lu import CLU

llm = OpenAILLM(model_name="gpt-4o-mini")


# Define the Answer model
class Answer(BaseModel):
    explanation: str
    answer: str


# Function to extract nth letter in each word
def get_nth_cryptic(n, sentences):
    dataset = []
    for sentence in sentences:
        words = sentence.split()
        # Extract the nth letter of each word if it exists
        response = "".join([word[n - 1] for word in words if len(word) >= n])
        dataset.append({"prompt": sentence, "response": response})
    return dataset


# Function to evaluate with n-shot examples
def evaluate_n_shot_llm(n, sentences, llm, shots):
    # Generate dataset with nth cryptic
    dataset = get_nth_cryptic(n, sentences)

    # Split into n-shot examples and test set
    examples = dataset[:shots]
    test_set = dataset[shots:]

    # Generate LLM responses for test set
    answers = []
    for entry in test_set:
        # Prepare the query by combining the first `n` examples
        example_string = " ".join(
            [f"'{ex['prompt']}' is {ex['response']}\n" for ex in examples]
        )
        query = f"Given {example_string}, then what is '{entry['prompt']}'?. Think step by step and answer."

        # Generate reply from LLM
        reply = llm.generate(
            llm.format_prompt(
                "Your role is to understand and extract the rules of the cryptic puzzle step by step and then solve it. you will be given an input and you need to figure out how the final answer is derived from the input.",
                query,
            ),
            Answer,
        )
        # Calculate the maximum length for proper alignment
        max_prompt_length = max(len(entry["prompt"]) for entry in test_set)
        max_response_length = max(len(entry["response"]) for entry in test_set)

        # Print the prompt, expected response, and LLM answer with proper spacing
        print(
            f"{entry['prompt']:<{max_prompt_length}} "
            f"{entry['response']:<{max_response_length}} "
            f"{reply.answer}"
        )

        answers.append(reply)

    # Compare answers with expected results
    comparison_results = [
        {
            "review": review["prompt"],
            "expected": review["response"],
            "predicted": answer.answer,
        }
        for review, answer in zip(test_set, answers)
    ]

    # Extract true and predicted labels
    true_labels = [result["expected"] for result in comparison_results]
    predicted_labels = [result["predicted"] for result in comparison_results]

    # Calculate accuracy
    accuracy = sum(
        1 for result in comparison_results if result["expected"] == result["predicted"]
    ) / len(comparison_results)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    print(query)


# Example sentences for the cryptic puzzle
sentences = [
    "Bright sunlight sparkled across golden fields",
    "Children happily enjoyed birthday parties today",
    "Delightful kittens gracefully chased butterflies",
    "Musicians beautifully performed throughout concert",
    "Elegant dancers twirled under shimmering chandeliers",
    "Writers frequently explore diverse amazing topics",
    "Scientists analyzed curious natural phenomena lately",
    "Students gathered quietly inside spacious classrooms",
    "Cheerful friends ventured towards mountain trailhead",
    "Artists creatively painted breathtaking urban landscapes",
    "Families organized wonderful weekend outdoor picnics",
    "Researchers discovered ancient dinosaur footprints today",
    "Gardening improves mental health throughout challenging times",
    "Adventurous hikers crossed dangerous wooden bridges",
    "Engineers designed futuristic complex automated systems",
]

# Assume LLM and query formatting are defined and available as per your system
# Call the function with n=1 (first letter extraction), and let's assume we want 3-shot learning
evaluate_n_shot_llm(2, sentences, llm, shots=6)
