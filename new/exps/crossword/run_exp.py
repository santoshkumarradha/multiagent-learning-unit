import csv
import json
import sys
from typing import Dict, List

from tqdm import tqdm

sys.path.append("../../")
# Assuming necessary imports for multi-agent LLM
from multi_agent_llm import OpenAILLM
from multi_agent_llm.agents.mlu.lu import CLU
from pydantic import BaseModel, Field


# Define the crossword puzzle structure
class Crossword:
    def __init__(self, clues: List[str], solution: List[List[str]]):
        # Split clues into horizontal and vertical based on indices
        self.h_clues = {f"h{i+1}": clues[i] for i in range(5)}  # First 5 are horizontal
        self.v_clues = {
            f"v{i+1}": clues[i + 5] for i in range(5)
        }  # Next 5 are vertical
        self.solution = solution

    def clues(self):
        puzzle = "\n"
        # Display clues in organized format
        for label, clue in self.h_clues.items():
            puzzle += f"{label}. {clue}\n"
        for label, clue in self.v_clues.items():
            puzzle += f"{label}. {clue}\n"
        return puzzle


# Define the format for the puzzle answer
class PuzzleAnswer(BaseModel):
    reasoning: str = Field(..., description="Explanation of the solution")
    crossword_solution: List[List[str]] = Field(
        ...,
        description="5 horizontal and 5 vertical words each in a 5x5 array of characters",
    )


class CrosswordDataset:
    def __init__(self, file_path: str):
        self.crosswords = self._load_crosswords(file_path)

    def _load_crosswords(self, file_path: str) -> List[Crossword]:
        crosswords = []
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
            for puzzle in data:
                clues = puzzle[0]  # List of 10 clues (5 horizontal and 5 vertical)
                solution_flat = puzzle[1]  # Flattened list of 25 characters
                # Convert flattened solution into 5x5 array
                solution = [solution_flat[i : i + 5] for i in range(0, 25, 5)]
                crosswords.append(Crossword(clues=clues, solution=solution))
        return crosswords

    def __len__(self) -> int:
        return len(self.crosswords)

    def __getitem__(self, index: int) -> Crossword:
        return self.crosswords[index]


def calculate_crossword_metrics(answers, actual_solutions):
    correct_letters_percentages = []
    correct_words_percentages = []
    correct_games_percentages = []

    for index, (grid, actual_sol) in enumerate(zip(answers, actual_solutions)):
        try:
            if len(actual_sol) != 5 or any(len(row) != 5 for row in actual_sol):
                print(f"Skipping: Correct solution at index {index} is not a 5x5 grid.")
                continue

            if len(grid) != 5 or any(len(row) != 5 for row in grid):
                print(f"Skipping: Grid at index {index} is not a 5x5 grid.")
                continue

            correct_letters = sum(
                grid[i][j] == actual_sol[i][j] for i in range(5) for j in range(5)
            )
            correct_letters_percentage = (correct_letters / 25) * 100
            correct_letters_percentages.append(correct_letters_percentage)

            correct_words = 0
            for i in range(5):
                if grid[i] == actual_sol[i]:
                    correct_words += 1

            for j in range(5):
                if [grid[i][j] for i in range(5)] == [
                    actual_sol[i][j] for i in range(5)
                ]:
                    correct_words += 1

            correct_words_percentage = (correct_words / 10) * 100
            correct_words_percentages.append(correct_words_percentage)

            correct_games = 1 if correct_letters == 25 else 0
            correct_games_percentage = (correct_games / 1) * 100
            correct_games_percentages.append(correct_games_percentage)

        except Exception as e:
            print(f"Error processing grid at index {index}: {e}")
            continue

    return {
        "Correct Letters Percentage": correct_letters_percentages,
        "Correct Words Percentage": correct_words_percentages,
        "Correct Games Percentage": correct_games_percentages,
    }


def inference_phase(clu, dataset, indices, phase, writer):
    correct_count = 0
    inferred_solutions = []
    actual_solutions = []

    try:
        for idx in tqdm(indices, desc=f"{phase} on Samples"):
            crossword_data = dataset[idx]
            task = crossword_data.clues()

            inferred_output = clu.inference(task=task, output_schema=PuzzleAnswer)
            inferred_solution = (
                inferred_output.operational_agent_response.response.crossword_solution
            )
            inferred_solutions.append(inferred_solution)
            actual_solutions.append(crossword_data.solution)

            is_correct = inferred_solution == crossword_data.solution
            correct_count += int(is_correct)
            writer.writerow(
                [
                    phase,
                    idx,
                    task,
                    inferred_solution,
                    crossword_data.solution,
                    is_correct,
                ]
            )
        accuracy = correct_count / len(indices)
        print(f"Accuracy during {phase}: {accuracy * 100:.2f}%")
        metrics = calculate_crossword_metrics(inferred_solutions, actual_solutions)
        print(f"Metrics during {phase}: {metrics}")
        return accuracy, metrics

    except KeyboardInterrupt:
        print("Execution interrupted! Cleaning up...")
        sys.exit(0)  # Ensure clean exit on interrupt


def train_phase(clu, dataset, train_indices, writer):
    try:
        for idx in tqdm(train_indices, desc="Training on Samples"):
            crossword_data = dataset[idx]
            task = crossword_data.clues()

            expected_output = PuzzleAnswer(
                reasoning="<the dataset does not contain explanation, please give your own reasoning>",
                crossword_solution=crossword_data.solution,
            )
            clu.training(task=task, expected_output=expected_output)
            writer.writerow(["Training", idx, task, "", crossword_data.solution, ""])

    except KeyboardInterrupt:
        print("Execution interrupted! Cleaning up...")
        sys.exit(0)  # Ensure clean exit on interrupt


def train_and_test_crossword(
    clu: CLU,
    dataset: CrosswordDataset,
    train_indices: List[int],
    test_indices: List[int],
    csv_file: str,
    metadata_file: str,
):
    with open(csv_file, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["Phase", "Index", "Task", "Inferred Answer", "Correct Answer", "Accuracy"]
        )

        print("\n=== Initial Testing Phase (Inference Before Training) ===")
        initial_accuracy, initial_metrics = inference_phase(
            clu, dataset, test_indices, "Inference Before Training", writer
        )

        print("\n=== Training Phase ===")
        train_phase(clu, dataset, train_indices, writer)

        print("\n=== Final Testing Phase (Inference After Training) ===")
        final_accuracy, final_metrics = inference_phase(
            clu, dataset, test_indices, "Inference After Training", writer
        )

    with open(metadata_file, mode="w", newline="", encoding="utf-8") as meta_file:
        meta_writer = csv.writer(meta_file)
        meta_writer.writerow(
            ["Phase", "Accuracy", "Correct Letters", "Correct Words", "Correct Games"]
        )

        # Before training, handle case of empty metrics
        try:
            meta_writer.writerow(
                [
                    "Before Training",
                    initial_accuracy,
                    sum(initial_metrics["Correct Letters Percentage"])
                    / (len(initial_metrics["Correct Letters Percentage"]) or 1),
                    sum(initial_metrics["Correct Words Percentage"])
                    / (len(initial_metrics["Correct Words Percentage"]) or 1),
                    sum(initial_metrics["Correct Games Percentage"])
                    / (len(initial_metrics["Correct Games Percentage"]) or 1),
                ]
            )
        except ZeroDivisionError:
            print("No data to calculate metrics for Before Training phase.")

        # After training, handle case of empty metrics
        try:
            meta_writer.writerow(
                [
                    "After Training",
                    final_accuracy,
                    sum(final_metrics["Correct Letters Percentage"])
                    / (len(final_metrics["Correct Letters Percentage"]) or 1),
                    sum(final_metrics["Correct Words Percentage"])
                    / (len(final_metrics["Correct Words Percentage"]) or 1),
                    sum(final_metrics["Correct Games Percentage"])
                    / (len(final_metrics["Correct Games Percentage"]) or 1),
                ]
            )
        except ZeroDivisionError:
            print("No data to calculate metrics for After Training phase.")


if __name__ == "__main__":
    # Set up CLU and dataset
    main_role = """Your role is to learn how to solve crossword puzzles by reasoning through horizontal and vertical clues. You will understand and learn how to solve the puzzle by analyzing both sets of clues and deducing the correct words for each horizontal and vertical section. you will learn how to use logical steps and cross-references between the clues to ensure the solution satisfies all given hints. Explanation in training is not given, but you will be tested on your ability to solve the puzzle correctly."""

    llm = OpenAILLM(model_name="gpt-4o-mini")
    clu = CLU(llm=llm, main_role=main_role, collection_name="crossword-test-03")

    dataset = CrosswordDataset(file_path="./mini0505_0_100_5.json")

    # Define train and test indices
    train_indices = [0, 1, 2, 3, 4]  # Train on these indices
    test_indices = [
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
    ]  # Test on these indices

    # Run training and testing phases
    train_and_test_crossword(
        clu,
        dataset,
        train_indices,
        test_indices,
        csv_file="crossword_results_1.csv",
        metadata_file="crossword_metadata.csv",
    )
