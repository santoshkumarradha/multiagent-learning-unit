import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import cycle

from tqdm import tqdm

sys.path.append("../../")
from multi_agent_llm import OpenAILLM
from multi_agent_llm.agents.mlu.lu import CLU
from pydantic import BaseModel


# Define the Answer model for LLM response
class Answer(BaseModel):
    explanation: str
    answer: str


# Function to extract nth letter in each word
def get_nth_cryptic(n, sentences):
    """Generates a dataset where for each sentence, we extract the nth letter of each word."""
    dataset = []
    for sentence in sentences:
        words = sentence.split()
        # Extract the nth letter of each word if it exists
        response = "".join([word[n - 1] for word in words if len(word) >= n])
        dataset.append({"prompt": sentence, "response": response})
    return dataset


# Function to train and test using CLU with cyclic training
def cyclic_train_and_test_clu(
    clu, sentences, n_seen, n, train_indices, test_indices, results_file
):
    """Train CLU with first n_seen examples in cyclic manner, and test on the rest."""
    # Generate dataset with nth cryptic
    dataset = get_nth_cryptic(n, sentences)

    # Cycle over the training examples
    cycle_train_indices = cycle(train_indices)

    # Training Phase
    print(f"\n=== Training Phase ===")
    for _ in tqdm(range(n_seen), desc="Training Samples", leave=False):
        idx = next(cycle_train_indices)
        question_data = dataset[idx]
        task = f"Solve this puzzle: {question_data['prompt']}\n"

        # Train with CLU
        expected_output = Answer(
            explanation=f"The correct answer is {question_data['response']} because...<figure out the explanation on your own>",
            answer=question_data["response"],
        )
        clu.training(task=task, expected_output=expected_output)

    # Testing Phase
    print(f"\n=== Testing Phase (Parallel Inference) ===")
    correct_count = 0
    total_tested = 0

    # Threaded inference using ThreadPoolExecutor
    with ThreadPoolExecutor() as executor:
        future_to_idx = {
            executor.submit(run_inference, clu, dataset[idx], idx): idx
            for idx in test_indices
        }

        for future in tqdm(
            as_completed(future_to_idx),
            total=len(future_to_idx),
            desc="Testing Samples",
            leave=False,
        ):
            idx = future_to_idx[future]
            question_data = dataset[idx]
            try:
                inferred_answer = future.result()
                # Check if the inference is correct
                is_correct = inferred_answer == question_data["response"]
                correct_count += int(is_correct)
                total_tested += 1

                print(
                    f"Prompt: Solve this puzzle: {question_data['prompt']}, Inferred Answer: {inferred_answer}, Expected: {question_data['response']}, Correct: {is_correct}"
                )
            except Exception as exc:
                print(f"Generated an exception: {exc}")

    # Calculate accuracy
    accuracy = correct_count / total_tested if total_tested > 0 else 0

    # Save accuracy and training indices to JSON file
    results_data = []
    if os.path.exists(results_file):
        with open(results_file, "r") as file:
            results_data = json.load(file)

    # Add new result to the data
    results_data.append(
        {
            "Run_ID": len(results_data) + 1,
            "Accuracy": accuracy,
            "Train_Indices": train_indices,
        }
    )

    # Write updated data back to the JSON file
    with open(results_file, "w") as file:
        json.dump(results_data, file, indent=4)

    print(f"\nFinal Accuracy: {accuracy * 100:.2f}%")


def run_inference(clu, question_data, idx):
    """Function to run inference for a single test example."""
    task = f"Solve this puzzle: {question_data['prompt']}\n"
    inferred_output = clu.inference(task=task, output_schema=Answer)
    return inferred_output.operational_agent_response.response.answer


if __name__ == "__main__":
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

    main_role = """Your role is to understand and extract the rules of the cryptic puzzle step by step and then solve it."""

    llm = OpenAILLM(model_name="gpt-4o-mini")
    clu = CLU(
        llm=llm,
        main_role=main_role,
        collection_name="crypt-01",
        compress_knowledge=False,  # Store longer text in KMU
    )
    results_file = "accuracy_results.json"

    # Call the function with 3 training examples (n_seen=3), nth letter (n=2)
    cyclic_train_and_test_clu(
        clu,
        sentences,
        n_seen=4,
        n=2,
        train_indices=list(range(4)),
        test_indices=range(4, len(sentences)),
        results_file=results_file,
    )
    cyclic_train_and_test_clu(
        clu,
        sentences,
        n_seen=4 * 2,
        n=2,
        train_indices=list(range(4)),
        test_indices=range(4, len(sentences)),
        results_file=results_file,
    )
    cyclic_train_and_test_clu(
        clu,
        sentences,
        n_seen=4 * 4,
        n=2,
        train_indices=list(range(4)),
        test_indices=range(4, len(sentences)),
        results_file=results_file,
    )
