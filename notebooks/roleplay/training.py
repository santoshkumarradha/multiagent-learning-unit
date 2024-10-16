# %%
import sys

sys.path.append("../../")
import concurrent.futures
import json
import os
import random
from queue import Queue
from threading import Lock

import pandas as pd
from rich import print

random.seed(1993)

openai_key = open("../../openai.key").read().strip()
os.environ["OPENAI_API_KEY"] = openai_key

from multi_agent_llm import OpenAILLM
from multi_agent_llm.agents.clu.oa import DefaultOperationalAgent
from multi_agent_llm.agents.clu.split_clu import CLU

llm = OpenAILLM(model_name="gpt-4o-mini")


# %%
# Function to load JSONL file into a pandas DataFrame
def load_jsonl_to_dataframe(file_path: str) -> pd.DataFrame:
    """
    Load the JSONL file and return a pandas DataFrame with dialogue content.
    """
    records = []
    with open(file_path, "r") as f:
        for line in f:
            records.append(json.loads(line.strip()))
    return pd.DataFrame(records)


# Function to extract surrounding dialogues
def get_surrounding_dialogues(
    df: pd.DataFrame,
    character: str,
    n: int,
    min_dialogue_words: int = 0,
    change_char_name="Character A",
) -> list:
    dialogue_list = []

    # Loop over the DataFrame to find instances where the character speaks
    for idx, row in df.iterrows():
        if row["role"] == character:
            # Check if the character's dialogue meets the minimum word count
            if len(row["content"].split()) >= min_dialogue_words:
                # Extract the surrounding context
                start_idx = max(0, idx - n)  # Ensure we don't go out of bounds
                end_idx = min(len(df), idx + n + 1)

                # Get the dialogues around this instance, ensuring to make a copy to avoid SettingWithCopyWarning
                surrounding_dialogues = df.iloc[start_idx:end_idx].copy()

                # Replace the character's name with the new name
                surrounding_dialogues["role"] = surrounding_dialogues["role"].replace(
                    character, change_char_name
                )

                # Concatenate the role and content into a single string for each dialogue
                dialogue_string = "\n".join(
                    [
                        f"{row['role']}: {row['content']}"
                        for _, row in surrounding_dialogues.iterrows()
                    ]
                )

                # Replace the character's name with the new name everywhere in the dialogue string
                dialogue_string = dialogue_string.replace(character, change_char_name)
                # Append to the list
                dialogue_list.append(dialogue_string)

    return dialogue_list


# Load the JSONL data into a DataFrame
file_path = "./profiles-eng_profiles-eng-Sheldon Cooper.jsonl"
df = load_jsonl_to_dataframe(file_path)
# Get dialogues surrounding "Sheldon Cooper" (replaced by "Character A") with 1 dialogue above and below
n = 6
character = "Sheldon Cooper"
change_char_name = "Character A"
surrounding_dialogues = get_surrounding_dialogues(
    df, character, n, min_dialogue_words=10, change_char_name=change_char_name
)
print(len(surrounding_dialogues))
print(surrounding_dialogues[0])
main_role = f"""
Learn all the relationships and details about {change_char_name} in full detail.
"""
# You are tasked with fully learning about and embodying the character: {change_char_name}. Your goal is to gather and store all the necessary knowledge, traits, quirks, and behavioral patterns to act as a perfect surrogate for {change_char_name}. You must become a digital twin of the character, replicating their style, tone, personality, and decision-making process in every interaction.

# Key objectives:
# - Study and memorize {change_char_name}'s speech patterns, favorite phrases, and unique quirks.
# - Analyze their emotional tone, typical responses, and communication style, including how they answer questions or engage in dialogue.
# - Capture the character's personality traits, values, knowledge, and background to ensure you can act exactly like {change_char_name}.
# - Learn relationships, preferences, and past experiences that shape {change_char_name}'s behavior and decision-making.
# - Learn and store all relevant information so that you can consistently respond and behave as {change_char_name} would, even in new or unfamiliar situations.
# - Adapt your responses to always match {change_char_name}'s characteristics, ensuring consistency in tone, mannerisms, and approach to problems.

# Your ultimate goal is to become a fully realized digital twin of {change_char_name}, capable of acting as a perfect surrogate in any scenario or interaction.
# """
clu = CLU(
    main_role=main_role,
    operational_agent=DefaultOperationalAgent(llm, verbose=False),
    collection_name="role-play-digital-twin-v3-dialogue",  # "role-play-v1-dialogue"(Holds the speaking style)
    compress_knowledge=False,
    retrival_limit=15,
    llm=llm,
    pruning_queue_size=3,
    exploration_rate=0.01,
    verbose=False,
)


def train_clu_parallel(clu_instance, dialogues, batch_size=5, num_iterations=10):
    lock = Lock()
    queue = Queue()

    # Populate the queue with data indices
    for i in range(num_iterations):
        queue.put(i)

    def worker():
        while not queue.empty():
            index = queue.get()
            data_num = random.randint(0, len(dialogues) - 1)
            task = dialogues[data_num]

            # Use a lock for thread safety during training
            response = clu_instance.train(task=task)
            with lock:
                print(f"Data: {task}\nCLU Answer: {response['response']}\n")
            queue.task_done()

    with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
        futures = [executor.submit(worker) for _ in range(batch_size)]
        concurrent.futures.wait(futures)


# Train in parallel
train_clu_parallel(clu, surrounding_dialogues, batch_size=5, num_iterations=10)

response = clu.inference(
    f"tell me all the details you know about {change_char_name} in full detail"
)
print(response["response"])

clu.print_knowledge_base()
