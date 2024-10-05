import asyncio
import json
import os
import random
import sys
from itertools import cycle

from tqdm import tqdm

sys.path.append("../../")
from multi_agent_llm import OpenAILLM
from multi_agent_llm.agents.mlu.lu import CLU
from pydantic import BaseModel, Field


# Define the Answer model for LLM response
class Answer(BaseModel):
    explanation: str = Field(..., description="Detailed explanation for the answer")
    answer: str = Field(..., description=" Final Answer to the question")


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


# Function to train and test using CLU with cyclic training and n_shot examples
def cyclic_train_and_test_clu(
    clu, sentences, n_seen, n, train_indices, test_indices, results_file, n_shot=3
):
    """Train CLU with first n_seen examples in cyclic manner, and test on the rest."""
    # Generate dataset with nth cryptic
    dataset = get_nth_cryptic(n, sentences)

    # Ensure n_shot is less than n_seen
    assert n_shot < n_seen, "n_shot must be less than n_seen."

    # Cycle over the training examples
    cycle_train_indices = cycle(train_indices)

    # Training Phase
    print(f"\n=== Training Phase ===")
    for _ in tqdm(range(n_seen), desc="Training Samples", leave=False):
        idx = next(cycle_train_indices)

        # Prepare n_shot examples
        available_indices = [i for i in train_indices if i != idx]
        shot_indices = random.sample(
            available_indices, min(n_shot, len(available_indices))
        )
        shot_examples = [dataset[i] for i in shot_indices]

        # Construct the prompt with n_shot examples
        question_data = dataset[idx]
        shot_examples_str = "\n".join(
            [f"'{ex['prompt']}' = '{ex['response']}'" for ex in shot_examples]
        )
        task = (
            f"Solve this puzzle:\n"
            f"if {shot_examples_str},\n"
            f"then what is '{question_data['prompt']}'?"
        )

        # Train with CLU
        expected_output = Answer(
            explanation=f"The correct answer is {question_data['response']} because...<figure out the explanation on your own>",
            answer=question_data["response"],
        )
        clu.training(task=task, expected_output=expected_output)

    # Testing Phase
    print(f"\n=== Testing Phase (Parallel Inference with n_shot examples) ===")
    asyncio.run(
        run_parallel_inference(
            clu, dataset, train_indices, test_indices, n_shot, results_file
        )
    )


async def run_parallel_inference(
    clu, dataset, train_indices, test_indices, n_shot, results_file
):
    correct_count = 0
    total_tested = 0
    tasks = []

    for idx in test_indices:
        tasks.append(
            run_inference_with_n_shot(clu, dataset, idx, train_indices, n_shot)
        )

    for coro in tqdm(
        asyncio.as_completed(tasks),
        total=len(tasks),
        desc="Testing Samples",
        leave=False,
    ):
        idx, inferred_answer = await coro
        question_data = dataset[idx]

        # Check if the inference is correct
        is_correct = inferred_answer == question_data["response"]
        correct_count += int(is_correct)
        total_tested += 1

        print(
            f"Prompt: Solve this puzzle: {question_data['prompt']}, Inferred Answer: {inferred_answer}, Expected: {question_data['response']}, Correct: {is_correct}"
        )

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


async def run_inference_with_n_shot(clu, dataset, idx, train_indices, n_shot):
    """Function to run inference for a single test example with n_shot examples as context."""
    # Prepare n_shot examples from the training set
    available_indices = [i for i in train_indices if i != idx]
    shot_indices = random.sample(available_indices, min(n_shot, len(available_indices)))
    shot_examples = [dataset[i] for i in shot_indices]

    # Construct the inference prompt with n_shot examples
    question_data = dataset[idx]
    shot_examples_str = "\n".join(
        [f"'{ex['prompt']}' = '{ex['response']}'" for ex in shot_examples]
    )
    task = (
        f"Solve this puzzle:\n"
        f"if {shot_examples_str},\n"
        f"then what is '{question_data['prompt']}'?"
    )

    # Run inference asynchronously
    inferred_output = await clu.inference_async(task=task, output_schema=Answer)
    return idx, inferred_output.operational_agent_response.response


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

    main_role = """"find the hidden rule and give me the transformation logic along with transformation"""

    llm = OpenAILLM(model_name="gpt-4o-mini")
    clu = CLU(
        llm=llm,
        main_role=main_role,
        collection_name="crypt-n-shot-04",
        compress_knowledge=False,  # Store longer text in KMU
    )
    results_file = "accuracy_results_n_shot.json"

    # Call the function with
    cyclic_train_and_test_clu(
        clu,
        sentences,
        n_seen=25,
        n=2,
        train_indices=list(range(4)),
        test_indices=range(4, len(sentences)),
        results_file=results_file,
        n_shot=2,  # Number of n_shot examples to include
    )

    # Call the function with
    cyclic_train_and_test_clu(
        clu,
        sentences,
        n_seen=25,
        n=2,
        train_indices=list(range(4)),
        test_indices=range(4, len(sentences)),
        results_file=results_file,
        n_shot=2,  # Number of n_shot examples to include
    )
