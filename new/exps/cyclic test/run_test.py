import random
import sys
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
def cyclic_train_and_test_clu(clu, sentences, n_seen, n, train_indices, test_indices):
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
    print(f"\n=== Testing Phase ===")
    correct_count = 0
    total_tested = 0
    for idx in tqdm(test_indices, desc="Testing Samples", leave=False):
        question_data = dataset[idx]
        task = f"Solve this puzzle: {question_data['prompt']}\n"

        # Test with CLU inference
        inferred_output = clu.inference(task=task, output_schema=Answer)
        inferred_answer = inferred_output.operational_agent_response.response.answer

        # Check if the inference is correct
        is_correct = inferred_answer == question_data["response"]
        correct_count += int(is_correct)
        total_tested += 1

        print(
            f"Prompt: {task}, Inferred Answer: {inferred_answer}, Expected: {question_data['response']}, Correct: {is_correct}"
        )

    # Calculate and print accuracy
    accuracy = correct_count / total_tested if total_tested > 0 else 0
    print(f"\nFinal Accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    # Example sentences for the cryptic puzzle
    sentences = [
        "The quick brown fox jumps over the lazy dog",
        "Hello world",
        "I love programming",
        "How are you today",
        "Python is fun",
        "Let's go to the park",
        "She sells sea shells",
        "This is a test",
        "Open the door please",
        "Machine learning is fascinating",
    ]

    main_role = """Your role is to understand and extract the rules of the cryptic puzzle step by step and then solve it."""

    llm = OpenAILLM(model_name="gpt-4o-mini")
    clu = CLU(llm=llm, main_role=main_role, collection_name="crypt-01")

    # Call the function with 3 training examples (n_seen=3), nth letter (n=2)
    cyclic_train_and_test_clu(
        clu,
        sentences,
        n_seen=9,
        n=2,
        train_indices=[1, 2, 3, 1, 2, 3, 4, 1, 2, 3, 4],
        test_indices=range(4, len(sentences)),
    )
