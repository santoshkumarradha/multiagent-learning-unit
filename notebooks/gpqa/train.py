# %%
import asyncio
import concurrent.futures
import csv
import json
import os
import random
import sys
from datetime import datetime
from queue import Queue
from threading import Lock
from typing import List

import pandas as pd
from pydantic import BaseModel, Field
from tqdm import tqdm
from tqdm.asyncio import tqdm

sys.path.append("../../")

random.seed(1993)

openai_key = open("../../openai.key").read().strip()
os.environ["OPENAI_API_KEY"] = openai_key

from multi_agent_llm import OpenAILLM
from multi_agent_llm.agents.clu.oa import DefaultOperationalAgent
from multi_agent_llm.agents.clu.split_clu import CLU


def convert_dict_to_str(d: dict) -> dict:
    new_dict = {}

    for key, value in d.items():
        # Convert the key to a string
        new_key = str(key)

        # If the value is a dictionary, recursively convert it
        if isinstance(value, dict):
            new_value = convert_dict_to_str(value)
        # If the value is a list, apply str() to each item in the list
        elif isinstance(value, list):
            new_value = [str(item) for item in value]
        # Otherwise, just convert the value to a string
        else:
            new_value = str(value)

        # Add the new key-value pair to the new dictionary
        new_dict[new_key] = new_value

    return new_dict


# %%
# Function to load CSV file into a list of questions
def load_csv_to_questions(file_path: str) -> List[BaseModel]:
    """
    Load the CSV file and return a list of questions.
    """
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
            question = row["Question"]
            explanation = row["Explanation"]
            questions.append(
                {
                    "question": question,
                    "options": options,
                    "correct_answer": "A",  # Assuming the correct answer is always A
                    "explanation": explanation,
                }
            )
    return questions


# %%
class Reasoning(BaseModel):
    reasoning: str = Field(..., description="Explanation for this step of reasoning")
    correct_answer: str = Field(
        ..., description="Final answer (A/B/C/D) without any additional explanation."
    )


class QA(BaseModel):
    steps: List[str] = Field(..., description="List of steps in the reasoning process")
    answer: str = Field(
        ..., description="Final answer (A/B/C/D) without any additional explanation."
    )


def train_clu_batch(clu_instance, train_data: List[dict], batch_size: int):
    def process_question(clu_instance, question):
        question_str = f"""
        Question: {question['question']}
        Options:
        A. {question['options'][0]}
        B. {question['options'][1]}
        C. {question['options'][2]}
        D. {question['options'][3]}
        Pick the correct option and explain your reasoning. think step by step.
        """
        expected_output = Reasoning(
            reasoning=question["explanation"],
            correct_answer="A",
        )
        result = clu_instance.train(task=question_str, expected_output=expected_output)
        return result

    lock = Lock()
    queue = Queue()

    # Populate the queue with data indices
    for i in range(len(train_data)):
        queue.put(i)

    def worker():
        while not queue.empty():
            index = queue.get()
            question = train_data[index]
            result = process_question(clu_instance, question)
            with lock:
                tqdm.write(f"Training: Question {index + 1}/{len(train_data)}")
            queue.task_done()

    with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
        futures = [executor.submit(worker) for _ in range(batch_size)]
        concurrent.futures.wait(futures)


def evaluate_clu(
    clu_instance, test_data: List[dict], batch_size: int, output_file: str
):
    correct_answers = 0
    total_questions = len(test_data)
    results = []

    def process_question(clu_instance, question):
        question_str = f"""
        Question: {question['question']}
        Options:
        A. {question['options'][0]}
        B. {question['options'][1]}
        C. {question['options'][2]}
        D. {question['options'][3]}
        Pick the correct option and explain your reasoning. Make sure you give final answer as A/B/C/D only and nothing else. Give the steps in the steps part and only alphabet in the answer part without any additional explanation or answers.
        """
        result = clu_instance.inference(question_str, response_schema=QA)
        return result

    lock = Lock()
    queue = Queue()

    # Populate the queue with data indices
    for i in range(len(test_data)):
        queue.put(i)

    def worker():
        nonlocal correct_answers
        while not queue.empty():
            index = queue.get()
            question = test_data[index]
            result = process_question(clu_instance, question)
            with lock:
                answer = result["response"].answer.strip().upper()
                if answer == question["correct_answer"]:
                    correct_answers += 1
                results.append(
                    {
                        "question": question["question"],
                        "predicted_answer": answer,
                        "correct_answer": question["correct_answer"],
                        "full_response": convert_dict_to_str(result),
                    }
                )
                tqdm.write(f"Testing: Question {index + 1}/{len(test_data)}")
            queue.task_done()

    with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
        futures = [executor.submit(worker) for _ in range(batch_size)]
        concurrent.futures.wait(futures)

    # Calculate accuracy
    accuracy = (correct_answers / total_questions) * 100
    results.append({"accuracy": accuracy})
    tqdm.write(f"Accuracy: {accuracy:.2f}%")

    # Save results to file
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)


# Run the training and evaluation
if __name__ == "__main__":

    llm = OpenAILLM(model_name="gpt-4o-mini")
    file_path = "./gpqa_diamond.csv"
    dataset = load_csv_to_questions(file_path)

    # Split data into train and test
    n_train = 10
    random.seed(1990)
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    train_data = [dataset[i] for i in train_indices]
    test_data = [dataset[i] for i in test_indices]

    # CLU Setup
    main_role = """
You are tasked with developing advanced meta-learning abilities to solve complex problems. Your goal is not just to understand specific problems but to learn how to approach and solve all kinds of problems by practicing with examples, learning from mistakes, and incorporating feedback to continuously improve your problem-solving techniques.

Key objectives:
- Develop a general problem-solving framework that can be applied to various types of challenges. Instead of simply memorizing answers, focus on building techniques and strategies that allow you to tackle new and unfamiliar problems effectively.
- Practice analyzing problems step-by-step, breaking them into smaller, manageable components, and systematically reasoning through each part to find solutions.
- Learn how to adapt your reasoning and approach based on feedback and past mistakes, continuously refining your techniques for solving increasingly complex challenges.
- Strengthen your analytical thinking skills by experimenting with different approaches and understanding which techniques are most effective in different scenarios.
- Build a deep understanding of how to learn from examples, extract generalizable principles, and apply those principles to a wide range of problems, even ones you have never encountered before.

Your ultimate goal is to develop into an expert meta-learner, capable of reasoning through and solving any problem by continuously learning, adapting, and improving your problem-solving skills through practice, feedback, and iteration.
"""
    collection_name = "gpqa_v6"
    clu = CLU(
        main_role=main_role,
        operational_agent=DefaultOperationalAgent(llm, verbose=False),
        collection_name=collection_name,  # "role-play-v1-dialogue"(Holds the speaking style)
        compress_knowledge=False,
        retrival_limit=15,
        llm=llm,
        pruning_queue_size=2,
        exploration_rate=0.1,
        verbose=False,
    )

    training_cycles = 10
    for _ in range(training_cycles):
        train_clu_batch(clu, train_data, batch_size=5)

    output_file = f"gpqa_results_CoT_col-{collection_name}_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.json"
    evaluate_clu(clu, test_data, batch_size=20, output_file=output_file)
