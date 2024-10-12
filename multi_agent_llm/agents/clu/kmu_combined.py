import os
import threading
import uuid
from typing import Any, Dict, List, Optional

import chromadb
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
            self.client = chromadb.Client()

        # Create OpenAI embedding function
        self.openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=self.openai_api_key, model_name="text-embedding-3-small"
        )

        # Create or get collection with OpenAI embedding function
        self.collection = self.client.create_collection(
            name=self.name, embedding_function=self.openai_ef, get_or_create=True
        )

    def save(
        self,
        knowledge: str,
        metadata: Dict[str, Any],
        compress: bool = True,
        align_knowledge: bool = True,
    ) -> str:
        with self.lock:
            if align_knowledge:
                # Align knowledge with goals
                aligned_knowledge = self._prompt_agent(
                    "Alignment", knowledge=knowledge, compress=False
                )
            else:
                aligned_knowledge = knowledge

            # Check for existing entries with the same task_id
            task_id = metadata.get("task_id")
            existing_entries = self.collection.get(where={"task_id": task_id})

            if existing_entries and existing_entries["documents"]:
                # Combine strategies
                existing_knowledge = existing_entries["documents"][0]
                combined_knowledge = self._combine_knowledge_entries(
                    existing_knowledge, aligned_knowledge
                )
                # Remove old entry
                self.collection.delete(ids=existing_entries["ids"])
                aligned_knowledge = combined_knowledge

            # Generate a unique ID for the knowledge
            knowledge_id = str(uuid.uuid4())

            # Add the aligned knowledge to the collection
            self.collection.add(
                documents=[aligned_knowledge],
                metadatas=[metadata],
                ids=[knowledge_id],
            )

            return knowledge_id

    def _combine_knowledge_entries(
        self, existing_knowledge: str, new_knowledge: str
    ) -> str:
        # Combine existing and new knowledge entries
        system_prompt = f"""You are a Knowledge Combining Agent.
Your task is to merge the following knowledge entries into a single, coherent entry without losing any important information."""

        user_prompt = f"""Existing Knowledge Entry:
{existing_knowledge}

New Knowledge Entry:
{new_knowledge}

Combined Knowledge Entry:"""

        class CombinedKnowledgeOutput(BaseModel):
            combined_knowledge: str = Field(
                ..., description="The combined knowledge entry"
            )

        query = self.llm.format_prompt(
            system_prompt=system_prompt, user_prompt=user_prompt
        )
        result = self.llm.generate(query, schema=CombinedKnowledgeOutput)

        return result.combined_knowledge

    def retrieve(self, query: str, task_id: str, n: int = 5) -> List[str]:
        # Generate search terms based on the query and goals
        search_terms = self._prompt_agent("Retrieval", query=query)

        # Query the collection with task_id filter
        results = self.collection.query(
            query_texts=[search_terms],
            n_results=n,
        )

        return results["documents"][0] if results["documents"] else []

    def _prompt_agent(self, agent_type: str, compress: bool = True, **kwargs) -> str:
        # Helper method to prompt agents
        system_prompt = f"""You are the {agent_type} Agent in the Knowledge Management Unit.
Main Goal: {self.main_goal}
Storage Goal: {self.storage_goal}
Retrieval Goal: {self.retrieval_goal}

Your task is to {agent_type.lower()} knowledge based on these goals."""

        user_prompt = f"Given the following information, please {agent_type.lower()} the knowledge appropriately: {kwargs}"
        if agent_type == "Alignment":
            system_prompt += (
                "\n\nImportant: Condense and align the knowledge with the goals."
            )

        class AgentResponse(BaseModel):
            response: str = Field(
                ...,
                description=f"The {agent_type.lower()}d knowledge",
            )

        result = self.llm.generate(
            self.llm.format_prompt(
                system_prompt=system_prompt, user_prompt=user_prompt
            ),
            schema=AgentResponse,
        )

        return result.response

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
