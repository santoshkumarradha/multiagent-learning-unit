import concurrent.futures
import pprint
import random
from typing import Callable, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from dtaidistance import dtw
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def dtw_2d_distance_old(matrix1, matrix2):
    n, m = len(matrix1), len(matrix2)
    p, q = len(matrix1[0]), len(matrix2[0])
    dtw_matrix = np.full((n + 1, m + 1, p + 1, q + 1), float("inf"))

    dtw_matrix[0, 0, 0, 0] = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            for k in range(1, p + 1):
                for l in range(1, q + 1):
                    cost = abs(matrix1[i - 1][k - 1] - matrix2[j - 1][l - 1])
                    dtw_matrix[i, j, k, l] = cost + min(
                        dtw_matrix[i - 1, j, k, l],  # insertion in row
                        dtw_matrix[i, j - 1, k, l],  # insertion in column
                        dtw_matrix[i, j, k - 1, l],  # insertion in depth
                        dtw_matrix[i, j, k, l - 1],  # insertion in depth
                        dtw_matrix[i - 1, j - 1, k - 1, l - 1],  # match
                    )

    return dtw_matrix[n, m, p, q]


def dtw_2d_distance(matrix1, matrix2):
    # Flatten the 2D matrices to 1D
    matrix1_flat = np.array(matrix1).flatten()
    matrix2_flat = np.array(matrix2).flatten()

    # Calculate the DTW distance
    distance = dtw.distance(matrix1_flat, matrix2_flat)

    return distance


def process_item(item, mlu, prompt_key, response_key, mode, analysis_agent, logging):
    if analysis_agent is not None:
        mlu.set_analysis_agent(analysis_agent)
    prompt = item[prompt_key]
    truth = item[response_key]
    reply = mlu.call(prompt, expected_output=f"{truth}", mode=mode)

    response = {
        "response": reply["response"],
        "right_answer": reply["right_answer"],
        "knowledge_base": reply.get("knowledge_update", "No knowledge added"),
    }
    # Pretty-print the knowledge update
    knowledge_update = reply.get("knowledge_update", "No knowledge added")
    print("\nLearned Knowledge Update:")
    pprint.pprint(knowledge_update)

    dtw_score = None
    if logging:
        try:
            res_str = response["response"].split(":")[-1]
            res = np.array(eval(res_str))
            actual_array = eval(truth)
        except:
            print("Response is not parsable")
            res = actual_array = np.zeros((1, 1))
        try:
            res_diff = res - np.array(actual_array)
            title = "Difference"
        except:
            res_diff = np.zeros((len(res), len(res[0])))
            title = "Diff not possible"

        try:
            dtw_score = dtw_2d_distance(res, actual_array)
            print(f"DTW Score: {dtw_score}")
        except:
            print("DTW not possible")

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

        ax3.imshow(res_diff, cmap="plasma")
        ax3.set_title(title)

        ax2.imshow(res, cmap="plasma")
        ax2.set_title("Predicted Array")

        ax1.imshow(actual_array, cmap="plasma")
        ax1.set_title("Actual Array")

        plt.tight_layout()
        plt.show()

    return response["right_answer"], dtw_score


def train_and_evaluate(
    dataset: List[Dict[str, str]],
    mlu,
    prompt_key: str,
    response_key: str,
    epochs=5,
    test_size=0.4,
    train_batch_size=1,
    test_batch_size=1,
    initial_serial_items=3,
    logging=True,
    training_analysis_agent: Optional[Callable] = None,
    testing_analysis_agent: Optional[Callable] = None,
):
    results = []
    train_dtw_scores = []
    test_dtw_scores = []

    # Split data into training and testing sets
    if test_size == 0:
        train_data = dataset
        test_data = []
    elif test_size == 1:
        train_data = []
        test_data = dataset
    else:
        train_data, test_data = train_test_split(
            dataset, test_size=test_size, random_state=42
        )

    for epoch in range(epochs):
        train_correct_answers = 0
        test_correct_answers = 0
        total_train_data = len(train_data)
        total_test_data = len(test_data)

        print(f"\nStarting Epoch {epoch + 1}\n")

        # Training phase
        for idx, item in enumerate(tqdm(train_data, desc="Training", colour="green")):
            if idx < initial_serial_items:
                right_answer, dtw_score = process_item(
                    item,
                    mlu,
                    prompt_key,
                    response_key,
                    mode="training",
                    analysis_agent=training_analysis_agent,
                    logging=logging,
                )
                train_dtw_scores.append(dtw_score)
                if right_answer:
                    train_correct_answers += 1
            else:
                batch_start = idx
                batch_end = min(idx + train_batch_size, total_train_data)
                batch_dtw_scores = []
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = [
                        executor.submit(
                            process_item,
                            train_data[i],
                            mlu,
                            prompt_key,
                            response_key,
                            mode="training",
                            analysis_agent=training_analysis_agent,
                            logging=logging,
                        )
                        for i in range(batch_start, batch_end)
                    ]
                    for future in concurrent.futures.as_completed(futures):
                        right_answer, dtw_score = future.result()
                        batch_dtw_scores.append(dtw_score)
                        if right_answer:
                            train_correct_answers += 1
                train_dtw_scores.append(batch_dtw_scores)
                idx = batch_end - 1

        if total_train_data > 0:
            train_accuracy = train_correct_answers / total_train_data
            print(
                f"\nEpoch {epoch + 1} Train Accuracy: {train_accuracy * 100:.2f}% ({train_correct_answers}/{total_train_data}) 🎓\n"
            )
        else:
            train_accuracy = 0

        # Testing phase
        for idx in tqdm(
            range(0, total_test_data, test_batch_size), desc="Testing", colour="blue"
        ):
            batch_start = idx
            batch_end = min(idx + test_batch_size, total_test_data)
            batch_dtw_scores = []
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(
                        process_item,
                        test_data[i],
                        mlu,
                        prompt_key,
                        response_key,
                        mode="inference",
                        analysis_agent=testing_analysis_agent,
                        logging=logging,
                    )
                    for i in range(batch_start, batch_end)
                ]
                for future in concurrent.futures.as_completed(futures):
                    right_answer, dtw_score = future.result()
                    batch_dtw_scores.append(dtw_score)
                    if right_answer:
                        test_correct_answers += 1
            test_dtw_scores.append(batch_dtw_scores)

        if total_test_data > 0:
            test_accuracy = test_correct_answers / total_test_data
        else:
            test_accuracy = 0

        results.append(
            {
                "epoch": epoch + 1,
                "train_correct_answers": train_correct_answers,
                "train_accuracy": train_accuracy,
                "test_correct_answers": test_correct_answers,
                "test_accuracy": test_accuracy,
            }
        )

        print(
            f"\nEpoch {epoch + 1} Test Accuracy: {test_accuracy * 100:.2f}% ({test_correct_answers}/{total_test_data}) 🧪\n"
        )

    return results, train_dtw_scores, test_dtw_scores
