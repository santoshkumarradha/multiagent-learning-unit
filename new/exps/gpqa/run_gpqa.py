import csv
import random
import sys
from typing import List

sys.path.append("../../")
from multi_agent_llm import OpenAILLM
from multi_agent_llm.agents.mlu.lu import CLU
from pydantic import BaseModel, Field
from tqdm import tqdm


class GPQAQuestion(BaseModel):
    question: str
    options: List[str]
    correct_answer: str


class QA(BaseModel):
    explanation: str = Field(..., description="Explanation for the answer")
    answer: str = Field(
        ...,
        description="Final answer (A/B/C/D/Uncertain) without any additional explanation.",
    )


class GPQADataset:
    def __init__(self, file_path: str):
        self.questions = self._load_questions(file_path)

    def _load_questions(self, file_path: str) -> List[GPQAQuestion]:
        questions = []
        with open(file_path, "r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                options = [
                    row["Correct Answer"],
                    row["Incorrect Answer 1"],
                    row["Incorrect Answer 2"],
                    row["Incorrect Answer 3"],
                ]
                question = GPQAQuestion(
                    question=row["Question"],
                    options=options,
                    correct_answer=row["Correct Answer"],
                )
                questions.append(question)
        return questions

    def __len__(self) -> int:
        return len(self.questions)

    def __getitem__(self, index: int) -> GPQAQuestion:
        return self.questions[index]


def shuffle_options_and_set_answer(question_data: GPQAQuestion):
    """Randomize the options and determine the new correct answer label."""
    options = question_data.options.copy()
    correct_answer = question_data.correct_answer

    # Shuffle the options
    random.shuffle(options)

    # Determine the new label for the correct answer
    correct_label = chr(options.index(correct_answer) + ord("A"))

    return options, correct_label


def format_options(options: List[str]) -> str:
    """Format the options into labeled form (A. option, B. option, etc.)"""
    labeled_options = [f"{chr(65 + i)}. {option}" for i, option in enumerate(options)]
    return "\n".join(labeled_options)


def train_and_test_mlu(
    clu: CLU,
    dataset: GPQADataset,
    train_indices: List[int],
    test_indices: List[int],
    n: int,
    training_steps: int,
    csv_file: str,
):
    with open(csv_file, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["Phase", "Index", "Task", "Inferred Answer", "Correct Answer", "Accuracy"]
        )

        for step in tqdm(range(training_steps), desc="Training Steps", leave=True):
            print(f"\n=== Training Step {step + 1} ===")

            # Training in batches of n
            for i in tqdm(
                range(0, len(train_indices), n),
                desc="Training Sub-Iteration",
                leave=False,
            ):
                batch_train_indices = train_indices[i : i + n]

                # Training Phase
                print("\n=== Training Phase ===")
                for idx in tqdm(
                    batch_train_indices, desc="Training on Samples", leave=False
                ):
                    question_data = dataset[idx]
                    task = f"Question: {question_data.question}\n"

                    # Shuffle options and set the correct label
                    shuffled_options, correct_label = shuffle_options_and_set_answer(
                        question_data
                    )
                    task += f"Options:\n{format_options(shuffled_options)}\n\nPick the correct option after thinking step by step."

                    # Prepare the expected output with the shuffled correct answer
                    expected_output = QA(
                        explanation=f"The correct answer is {correct_label} because of {question_data.correct_answer}.",
                        answer=correct_label,
                    )

                    clu.training(task=task, expected_output=expected_output)
                    writer.writerow(["Training", idx, task, "", correct_label, ""])

                # Testing Phase (Inference) after every n training samples
                print("\n=== Testing Phase (Inference) ===")
                correct_count = 0
                for idx in tqdm(test_indices, desc="Inference on Samples", leave=False):
                    question_data = dataset[idx]
                    task = f"Question: {question_data.question}\n"

                    # Shuffle options and set the correct label
                    shuffled_options, correct_label = shuffle_options_and_set_answer(
                        question_data
                    )
                    task += f"Options:\n{format_options(shuffled_options)}\n\nPick the correct option after thinking step by step."

                    # Perform inference
                    inferred_output = clu.inference(task=task, output_schema=QA)

                    inferred_answer = (
                        inferred_output.operational_agent_response.response.answer
                    )

                    # Determine correctness and write to CSV
                    is_correct = inferred_answer == correct_label
                    correct_count += int(is_correct)
                    writer.writerow(
                        [
                            "Inference",
                            idx,
                            task,
                            inferred_answer,
                            correct_label,
                            is_correct,
                        ]
                    )

                # Calculate and print accuracy after each inference step
                accuracy = correct_count / len(test_indices)
                print(f"Accuracy after step {step + 1}: {accuracy * 100:.2f}%")
                writer.writerow(["Accuracy", "", "", "", "", accuracy])


if __name__ == "__main__":
    # Set up CLU and dataset
    main_role = """Your role is to engage in meta-learning to master the process of solving complex scientific reasoning tasks by learning from examples. You will observe and analyze various scientific questions, along with their reasoning and correct answers, to extract the underlying patterns and strategies required to solve similar tasks in the future. The core of your approach should focus on:

    Learning from Examples: Study the provided examples of scientific reasoning, including both the questions and their step-by-step solutions, to internalize the key principles and problem-solving approaches.
    Identifying Patterns: Recognize the recurring structures, logical steps, and scientific principles that frequently appear in questions, enabling you to generalize solutions to new problems.
    Meta-Cognition: Develop an understanding of how to reason effectively by reflecting on the thought processes used to arrive at correct answers, and adapt these processes to a wide range of scientific questions.
    Applying Adaptive Reasoning: Use the patterns and strategies learned from prior examples to approach new questions with flexibility, applying appropriate reasoning steps based on the unique characteristics of each task.
    Self-Improvement: Continuously improve by comparing your reasoning and answers to provided solutions, identifying areas where your reasoning can be refined or extended, and updating your approach to improve performance on future tasks.
    """

    llm = OpenAILLM(model_name="gpt-4o-mini")
    clu = CLU(llm=llm, main_role=main_role, collection_name="gpqa-test-01")

    # Load the dataset
    dataset = GPQADataset(file_path="./gpqa_diamond.csv")

    # Define train and test indices
    train_indices = list(range(20))  # Example: first 20 indices for training
    test_indices = list(range(20, 30))  # Example: next 10 indices for testing

    # Train and test the MLU with n=4 and training_steps=5
    train_and_test_mlu(
        clu,
        dataset,
        train_indices,
        test_indices,
        n=5,
        training_steps=10,
        csv_file="mlu_results.csv",
    )
