import json
import os


def convert_json_format(directory):
    output_data = []

    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            filepath = os.path.join(directory, filename)
            with open(filepath, "r") as file:
                data = json.load(file)
                item = {
                    "training_pairs": [
                        (pair["input"], pair["output"]) for pair in data["train"]
                    ],
                    "test_pair": (data["test"][0]["input"], data["test"][0]["output"]),
                }
                output_data.append(item)

    return output_data


def format_matrix(matrix):
    # return '\n'.join([','.join(map(str, row)) for row in matrix])
    return str(matrix)


def generate_string(data):
    natural_language_output = []

    for item in data:
        training_descriptions = []
        for i, (input_matrix, output_matrix) in enumerate(item["training_pairs"]):
            formatted_input = format_matrix(input_matrix)
            formatted_output = format_matrix(output_matrix)
            training_description = f"if image_{i+1}:\n{formatted_input} is transformed as \n{formatted_output}"
            training_descriptions.append(training_description)

        test_input_matrix, test_output_matrix = item["test_pair"]
        formatted_test_input = format_matrix(test_input_matrix)
        formatted_test_output = format_matrix(test_output_matrix)

        query = (
            "\n".join(training_descriptions)
            + f"\nthen what is \n{formatted_test_input}\n transformed to?"
        )
        response = formatted_test_output

        natural_language_output.append({"query": query, "reply": response})

    return natural_language_output


import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt


def plot_sample_matrices(sample):
    fig = plt.figure(figsize=(8, 8))
    outer = gridspec.GridSpec(2, 2, wspace=0.2, hspace=0.2)

    training_pairs = sample["training_pairs"]
    test_pair = sample["test_pair"]

    # Plotting training pairs
    for i, (input_matrix, output_matrix) in enumerate(training_pairs):
        row, col = divmod(i, 2)
        inner = gridspec.GridSpecFromSubplotSpec(
            1, 2, subplot_spec=outer[i], wspace=0.1, hspace=0.1
        )

        ax1 = plt.Subplot(fig, inner[0])
        ax1.imshow(input_matrix, cmap="viridis")
        ax1.set_title(f"Tr In {i + 1}")
        ax1.axis("off")
        fig.add_subplot(ax1)

        ax2 = plt.Subplot(fig, inner[1])
        ax2.imshow(output_matrix, cmap="viridis")
        ax2.set_title(f"Tr Out {i + 1}")
        ax2.axis("off")
        fig.add_subplot(ax2)

    # Plotting test pair
    row, col = divmod(len(training_pairs), 2)
    inner = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer[row * 2 + col], wspace=0.1, hspace=0.1
    )

    ax1 = plt.Subplot(fig, inner[0])
    ax1.imshow(test_pair[0], cmap="viridis")
    ax1.set_title("Test In")
    ax1.axis("off")
    fig.add_subplot(ax1)

    ax2 = plt.Subplot(fig, inner[1])
    ax2.imshow(test_pair[1], cmap="viridis")
    ax2.set_title("Test Out")
    ax2.axis("off")
    fig.add_subplot(ax2)

    plt.tight_layout()
    plt.show()
