import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from pydantic import BaseModel, Field
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from .kmu import KnowledgeManagementUnit

console = Console()


class ReinforceKnowledgeManagementUnit(KnowledgeManagementUnit):
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
            api_key=self.openai_api_key, model_name="text-embedding-ada-002"
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
            system_prompt += "\n\nImportant: Condense and compress the knowledge to its most essential form and don't be verbose."

        class AgentResponse(BaseModel):
            response: str = Field(
                ...,
                description=f"The {agent_type.lower()}d knowledge or action to take",
            )

        prompt = self.llm.format_prompt(
            system_prompt=system_prompt, user_prompt=user_prompt
        )
        result = self.llm.generate(prompt, schema=AgentResponse)

        return result.response

    def save(self, knowledge: str, metadata: Dict[str, Any], compress=True) -> str:
        with self.lock:
            # Align knowledge with goals
            aligned_knowledge = self._prompt_agent(
                "Alignment", knowledge=knowledge, compress=compress
            )

            # Generate a unique ID for the knowledge
            knowledge_id = str(uuid.uuid4())

            # Add the aligned knowledge to the collection with metadata
            self.collection.add(
                documents=[aligned_knowledge],
                metadatas=[metadata],
                ids=[knowledge_id],
            )

            return knowledge_id

    def retrieve(
        self,
        query: str,
        n: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        # Generate search terms based on the query and goals
        search_terms = self._prompt_agent("Retrieval", query=query)

        # Query the collection with metadata filters
        results = self.collection.query(
            query_texts=[search_terms], n_results=n, where=metadata_filter
        )

        return results["documents"][0] if results["documents"] else []

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
                    collection_data["ids"][i],
                    doc,
                    str(collection_data["metadatas"][i]),
                )
            else:
                table.add_row(collection_data["ids"][i], doc)

        console.print(table)
