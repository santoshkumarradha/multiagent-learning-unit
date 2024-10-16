import concurrent
import os
import uuid
from threading import Lock
from typing import Any, Dict, List, Optional, Union

import chromadb
from chromadb.utils import embedding_functions
from pydantic import BaseModel, Field
from rich import print
from rich.console import Console
from rich.layout import Layout
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

custom_theme = Theme(
    {
        "agent_name": "bold cyan",
        "task": "green",
        "response": "yellow",
    }
)

console = Console(
    theme=custom_theme, width=80
)  # Set a fixed width for consistent formatting


def format_agent_output(name: str, task: str, response: str):
    agent_info = f"[agent_name]{name}[/agent_name]\n"
    task_section = "[task]Task:[/task]\n" + task + "\n"
    response_section = "[response]Response:[/response]\n" + response

    full_content = agent_info + task_section + response_section
    return full_content


class SearchTerms(BaseModel):
    terms: List[str] = Field(..., description="List of search terms")


class ProcessedKnowledge(BaseModel):
    content: str = Field(..., description="Processed content")
    tags: List[str] = Field(..., description="List of tags")


class PruningSuggestions(BaseModel):
    new_entries: List[str] = Field(
        ..., description="List of new or modified knowledge entries"
    )
    indices_to_update: List[int] = Field(
        ..., description="List of indices of entries to be updated or deleted"
    )


class KMU:
    def __init__(
        self,
        main_goal: str,
        storage_goal: str,
        retrieval_goal: str,
        llm: Any,
        openai_api_key: Optional[str] = None,
        persist_directory: Optional[str] = None,
        collection_name: Optional[str] = None,
        compress_knowledge: bool = False,
        verbose: bool = False,
    ):
        self.main_goal = main_goal
        self.storage_goal = storage_goal
        self.retrieval_goal = retrieval_goal
        self.llm = llm
        self.verbose = verbose
        self.persist_directory = persist_directory
        # Set OpenAI API key
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError(
                "OpenAI API key must be provided or set as an environment variable."
            )

        # Initialize ChromaDB client
        if self.persist_directory:
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=chromadb.Settings(allow_reset=True),
            )
        else:
            self.client = chromadb.Client(chromadb.Settings(allow_reset=True))

        # Create OpenAI embedding function
        self.openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=self.openai_api_key, model_name="text-embedding-3-small"
        )

        # Create or get collection
        if collection_name is None:
            collection_name = f"knowledge_base_{uuid.uuid4().hex}"

        self.collection = self.client.get_or_create_collection(
            name=collection_name, embedding_function=self.openai_ef
        )

        if compress_knowledge:
            self.compress_knowledge = "Make sure that you condense this information to much smaller sentences or a single small paragraph to extract all relevant information in a compact way. Don't be verbose; make it pointed yet detailed. Include all points."

    def reset_client(self, persist_directory: Optional[str] = None):
        """Needed becasue of a weird bug in ChromaDB where the client needs to be reset after a while. else we get the message "Delete of nonexisting embedding ID" """
        self.client.stop()
        if persist_directory:
            self.client = chromadb.PersistentClient(
                path=persist_directory, settings=chromadb.Settings(allow_reset=True)
            )
        else:
            self.client = chromadb.Client(chromadb.Settings(allow_reset=True))
        # Create or get collection
        if collection_name is None:
            collection_name = f"knowledge_base_{uuid.uuid4().hex}"
        self.collection = self.client.get_or_create_collection(
            name=collection_name, embedding_function=self.openai_ef
        )

    def _generate_response(
        self, name: str, role: str, function: str, task: str, response_model: BaseModel
    ):
        system_prompt = f"""
        You are: {name}
        Your role: {role}
        Your function: {function}
        Based on your role and function, do the task you are given.
        """

        user_prompt = task

        # if self.verbose:
        #     console.print(f"[agent_name]Generating response for {name}[/agent_name]")

        # Generate response using LLM
        result = self.llm.generate(
            self.llm.format_prompt(
                system_prompt=system_prompt, user_prompt=user_prompt
            ),
            schema=response_model,
        )

        if self.verbose:
            formatted_output = format_agent_output(name, task, str(result))
            console.print(
                Panel(
                    formatted_output,
                    title=name,
                    border_style="cyan",
                    expand=False,
                )
            )

        return result

    def save_knowledge(self, content: str) -> str:
        # Use the knowledge processor to process the given content
        processed = self._generate_response(
            name="KnowledgeProcessor",
            role="Process and align knowledge with the main goal",
            function=f"""
            Main Goal: {self.main_goal}
            Storage Goal: {self.storage_goal}

            Your task is to process the given content to align it with the main goal and storage goal.
            Modify the content if necessary and generate relevant tags. Choose tags that are aligned with the main goal.
            Modify the content so that we extract relevant and concise information based on the main goal and storage goal.
            Prefer detailed but concise information that is relevant to the main goal and storage goal.
            {"IMPORTANT:" + self.compress_knowledge if hasattr(self, 'compress_knowledge') else ""}
            """,
            task=f"Content: {content}",
            response_model=ProcessedKnowledge,
        )
        entry_id = str(uuid.uuid4())
        self.collection.add(
            documents=[processed.content + ",".join(processed.tags)],
            ids=[entry_id],
            metadatas=[{"tags": ",".join(processed.tags)}],
        )
        return entry_id

    def retrieve_knowledge(
        self, query: str, n_results: int = 5
    ) -> Dict[str, List[Any]]:
        # Use the search agent to generate search terms for the query
        search_output = self._generate_response(
            name="SearchAgent",
            role="Generate search terms for knowledge retrieval",
            function=f"""
            Main Goal: {self.main_goal}
            Retrieval Goal: {self.retrieval_goal}

            Your task is to generate a list of search terms based on the given query and goals.
            These terms will be used to search the knowledge base effectively.
            Output a list of search terms.
            """,
            task=f"Query: {query}",
            response_model=SearchTerms,
        )
        combined_query = " ".join(search_output.terms)
        results = self.collection.query(
            query_texts=[combined_query], n_results=n_results
        )
        return results

    def prune(self, feedback: str, ids: List[str]):
        # Retrieve existing entries by their IDs
        existing_entries = self.collection.get(ids=ids)
        existing_content = "\n".join(existing_entries["documents"])
        existing_tags = [metadata["tags"] for metadata in existing_entries["metadatas"]]

        # Use the pruning agent to suggest modifications based on feedback
        pruning_input = f"""
        Feedback: {feedback}
        Existing Content:
        {existing_content}
        Existing Tags:
        {existing_tags}
        """

        pruning_suggestions = self._generate_response(
            name="PruningAgent",
            role="Analyze feedback and suggest knowledge base modifications",
            function=f"""
            Main goal: {self.main_goal}
            Analyze the provided feedback and suggest modifications to the existing knowledge entries.
            Your task is to:
            1. Identify which entries need to be modified or removed based on the feedback.
            2. Provide new or modified entries that reflect the feedback and improve the existing knowledge.
            3. Specify the indices of the entries that should be updated or removed.

            Your response should include:
            1. A list of new or modified knowledge entries.
            2. A list of indices (0-based) indicating which of the original entries should be updated or removed.

            Remember:
            - You don't need to change everything, only modify what needs improvement.
            - Aim to improve the quality and relevance of the knowledge base.
            1. Modifications of the original knowledge 
            2. Addition of new information to original knowledge
            3. Removal of old knowledge
            4. Completely new addition of knowledge
            It should aim to improve the quality and relevance of the knowledge base.
            Always club together similar knowledge and remove redundant knowledge.
            - Club together similar knowledge and remove redundant information.
            {"IMPORTANT:" + self.compress_knowledge if hasattr(self, 'compress_knowledge') else ""}
            """,
            task=pruning_input,
            response_model=PruningSuggestions,
        )

        # Validate and process the indices
        valid_indices = []
        for index in pruning_suggestions.indices_to_update:
            try:
                if 0 <= index < len(ids):
                    valid_indices.append(index)
                else:
                    print(
                        f"Warning: Index {index} is out of range and will be ignored."
                    )
            except:
                print(f"Warning: Invalid index {index} will be ignored.")

        # Delete the entries at valid indices
        ids_to_delete = [ids[i] for i in valid_indices]
        self.collection.delete(ids=ids_to_delete)
        self.reset_client(self.persist_directory)

        # Add the new entries using threading with lock
        def save_entry(entry):
            return self.save_knowledge(entry)

        new_ids = []
        lock = Lock()

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_entry = {
                executor.submit(save_entry, entry): entry
                for entry in pruning_suggestions.new_entries
            }
            for future in concurrent.futures.as_completed(future_to_entry):
                entry_id = future.result()
                with lock:
                    new_ids.append(entry_id)

        print(
            f"Updated {len(valid_indices)} entries and added {len(new_ids)} new entries."
        )
        return new_ids
