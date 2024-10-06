import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, List, Optional

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from pydantic import BaseModel, Field
from rich.console import Console
from rich.table import Table

console = Console()


class KnowledgeManagementUnit:
    def __init__(
        self,
        llm,
        main_goal: str,
        storage_goal: str,
        retrieval_goal: str,
        name: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        persist_directory: str = "./chroma_db",
    ):
        self.llm = llm
        self.main_goal = main_goal
        self.storage_goal = storage_goal
        self.retrieval_goal = retrieval_goal
        self.name = name or str(uuid.uuid4())
        self.lock = threading.Lock()

        # Set OpenAI API key
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError(
                "OpenAI API key must be provided or set as an environment variable."
            )

        # Initialize ChromaDB client
        if persist_directory:
            self.client = chromadb.PersistentClient(path=persist_directory)

        else:
            self.client = chromadb.Client(Settings(allow_reset=True))

        # Create OpenAI embedding function
        self.openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=self.openai_api_key, model_name="text-embedding-3-small"
        )

        # Create or get collection with OpenAI embedding function

        self.collection = self.client.create_collection(
            name=self.name, embedding_function=self.openai_ef, get_or_create=True
        )

    def _prompt_agent(self, agent_type: str, compress: bool = True, **kwargs) -> str:
        # Helper method to prompt agents
        system_prompt = f"""You are the {agent_type} Agent in the Knowledge Management Unit.
        Main Goal: {self.main_goal}
        Storage Goal: {self.storage_goal}
        Retrieval Goal: {self.retrieval_goal}
        
        Your task is to {agent_type.lower()} knowledge based on these goals."""

        user_prompt = f"Given the following information, please {agent_type.lower()} the knowledge appropriately: {kwargs}"
        if compress and agent_type == "Alignment":
            system_prompt += "\n\nImportant: Condense and compress the knowledge to its most essential form and dont be verbose."

        class AgentResponse(BaseModel):
            response: str = Field(
                ...,
                description=f"The {agent_type.lower()}d knowledge or action to take",
            )

        result = self.llm.generate(
            self.llm.format_prompt(
                system_prompt=system_prompt, user_prompt=user_prompt
            ),
            schema=AgentResponse,
        )

        return result.response

    def replace_knowledge(
        self, old_ids: List[str], new_knowledge: List[str], compress: bool = True
    ) -> List[str]:
        with self.lock:
            # Remove old entries
            if old_ids:
                self.collection.delete(ids=old_ids)

        # Add new entries in parallel
        new_ids = []
        with ThreadPoolExecutor() as executor:
            # Submit save tasks to the executor
            future_to_entry = {
                executor.submit(self.save, entry, compress): entry
                for entry in new_knowledge
                if entry.strip()
            }

            # Collect results as they complete
            for future in as_completed(future_to_entry):
                try:
                    new_id = future.result()
                    new_ids.append(new_id)
                except Exception as e:
                    print(f"An error occurred while saving an entry: {e}")

        return new_ids

    def save(self, knowledge: str, compress=True) -> str:
        with self.lock:
            # Align knowledge with goals
            aligned_knowledge = self._prompt_agent(
                "Alignment", knowledge=knowledge, compress=compress
            )

            # Generate a unique ID for the knowledge
            knowledge_id = str(uuid.uuid4())

            # Add the aligned knowledge to the collection
            self.collection.add(
                documents=[aligned_knowledge],
                metadatas=[{"original": knowledge}],
                ids=[knowledge_id],
            )

            return knowledge_id

    def retrieve(self, query: str, n: int = 5) -> List[str]:
        # Generate search terms based on the query and goals
        search_terms = self._prompt_agent("Retrieval", query=query)

        # Query the collection
        results = self.collection.query(query_texts=[search_terms], n_results=n)

        return results["documents"][0] if results["documents"] else []

    def prune(self, feedback: str, knowledge_ids: List[str]) -> List[str]:
        # Retrieve existing entries
        existing_entries = self.collection.get(ids=knowledge_ids)

        # Prune and update knowledge based on feedback
        pruned_knowledge = self._prompt_agent(
            "Pruning", feedback=feedback, existing_entries=existing_entries
        )

        # Remove old entries
        self.collection.delete(ids=knowledge_ids)

        # Add updated entries
        new_ids = []
        for entry in pruned_knowledge.split(
            "\n"
        ):  # Assuming the agent returns entries separated by newlines
            if entry.strip():
                new_id = self.save(entry)
                new_ids.append(new_id)

        return new_ids

    def print_knowledge(self, verbose: bool = False):
        collection_data = self.collection.get()

        table = Table(title=f"Knowledge in {self.name}")
        table.add_column("ID", style="cyan")
        table.add_column("Knowledge", style="magenta")
        if verbose:
            table.add_column("Metadata", style="green")

        for i, doc in enumerate(collection_data["documents"]):
            if verbose:
                table.add_row(
                    collection_data["ids"][i], doc, str(collection_data["metadatas"][i])
                )
            else:
                table.add_row(collection_data["ids"][i], doc)

        console.print(table)


# Example usage
def example_usage():
    # Define a mock LLM class for demonstration
    class MockLLM:
        def generate(self, prompt, schema):
            # This is a simplistic mock. In reality, you'd use a real LLM here.
            return schema(response="Mocked response based on: " + prompt)

        def format_prompt(self, system_prompt, user_prompt):
            return f"{system_prompt}\n{user_prompt}"

    # Initialize the KMU
    llm = MockLLM()
    kmu = KnowledgeManagementUnit(
        llm,
        main_goal="Maintain a knowledge base about machine learning",
        storage_goal="Store concise and relevant information",
        retrieval_goal="Retrieve accurate and up-to-date information",
        name="ML_Knowledge",
        openai_api_key="your_openai_api_key_here",  # Replace with your actual OpenAI API key
    )

    # Save some knowledge
    id1 = kmu.save(
        "Neural networks are a type of machine learning model inspired by the human brain."
    )
    id2 = kmu.save(
        "Gradient descent is an optimization algorithm used to minimize the loss function in machine learning models."
    )

    # Retrieve knowledge
    results = kmu.retrieve("How do neural networks work?")
    print("Retrieved:", results)

    # Prune knowledge
    new_ids = kmu.prune("The information about neural networks is outdated.", [id1])
    print("New IDs after pruning:", new_ids)


if __name__ == "__main__":
    example_usage()
