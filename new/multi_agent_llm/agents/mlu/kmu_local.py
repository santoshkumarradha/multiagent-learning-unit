import asyncio
import json
import os
import uuid
from typing import Any, Dict, List, Optional, Union

import faiss
import numpy as np
import openai
from pydantic import BaseModel, Field

from ...agent_class import Agent
from ...llm import LLMBase


class ProcessedKnowledge(BaseModel):
    content: str = Field(..., description="Processed content")
    tags: List[str] = Field(..., description="List of tags")


class PruningSuggestions(BaseModel):
    new_entries: List[str] = Field(..., description="List of new entries")


class KnowledgeManagementUnit:
    def __init__(
        self,
        llm: LLMBase,
        main_goal: str,
        storage_goal: str,
        retrieval_goal: str,
        knowledge_file: str = "knowledge_store.json",
        compress_knowledge=False,
        dimension: int = 1536,  # OpenAI embedding dimension
    ):
        self.llm = llm
        self.main_goal = main_goal
        self.storage_goal = storage_goal
        self.retrieval_goal = retrieval_goal
        self.dimension = dimension
        self.knowledge_file = knowledge_file

        # Initialize FAISS index
        self.index = faiss.IndexFlatL2(dimension)

        self.compress_knowledge = (
            "Make sure that you condense this information to much smaller sentences or small single paragraph to extract all the relevant information in a compact way from this content. Don't be verbose. Make it pointed to be detailed yet short. Include all points."
            if compress_knowledge
            else ""
        )

        self.knowledge_processor = self._create_knowledge_processor()
        self.pruning_agent = self._create_pruning_agent()

        # Load existing knowledge from file
        self.knowledge_store = self._load_knowledge_store()
        self._load_faiss_index()

    def _load_knowledge_store(self) -> Dict[str, Any]:
        try:
            with open(self.knowledge_file, "r") as f:
                if f.read().strip():
                    f.seek(0)
                    return json.load(f)
                else:
                    return {}
        except FileNotFoundError:
            return {}

    def _save_knowledge_store(self):
        with open(self.knowledge_file, "w") as f:
            json.dump(self.knowledge_store, f)

    def _load_faiss_index(self):
        if self.knowledge_store:
            embeddings = [entry["embedding"] for entry in self.knowledge_store.values()]
            if embeddings:
                self.index.add(np.array(embeddings, dtype=np.float32))

    def _get_openai_embedding(self, text: str) -> List[float]:
        client = openai.Client()
        response = client.embeddings.create(
            input=[text], model="text-embedding-3-small"
        )
        return response.data[0].embedding

    def _create_knowledge_processor(self):
        return Agent(
            name="KnowledgeProcessor",
            role="Process and align knowledge with the main goal",
            function=f"""
            Main Goal: {self.main_goal}
            Storage Goal: {self.storage_goal}

            You have received content that must be processed to meet both the main and storage goals:
            1. Extract concise and relevant information that best serves the main goal.
            2. Integrate knowledge from general and prompt-specific spaces to enhance content quality and relevance.
            3. Modify the content to remove redundancies while retaining key insights.
            4. Tag the processed content with keywords that align with the core themes of the main goal.

            IMPORTANT: Ensure the processed content is compact yet retains all crucial information for future retrieval and goal alignment.
            {"IMPORTANT: " + self.compress_knowledge if self.compress_knowledge else ""}
            Output the processed content and a list of tags.
            """,
        )

    def _create_pruning_agent(self):
        return Agent(
            name="PruningAgent",
            role="Analyze feedback and suggest knowledge base modifications",
            function=f"""
            Main Goal: {self.main_goal}
            You are provided with feedback on the existing knowledge:
            1. Analyze the provided feedback, categorizing it as positive or negative.
            2. Based on positive feedback, identify knowledge to reinforce and improve (e.g., add supporting details or emphasize key points).
            3. Based on negative feedback, modify or remove knowledge to better align with the main and storage goals.
            4. Suggest new knowledge entries where gaps are identified, ensuring these align closely with the existing structure and goals.
            5. Always club together similar knowledge entries and eliminate redundancy to ensure the knowledge base remains efficient.

            Output your suggestions as a list of new or modified entries.
            {"IMPORTANT: " + self.compress_knowledge if self.compress_knowledge else ""}
            """,
        )

    async def save_knowledge(self, content: str) -> str:
        system_prompt, user_prompt = self.knowledge_processor.prompt(
            f"Content: {content}"
        )
        formatted_prompt = self.llm.format_prompt(system_prompt, user_prompt)
        processed = await self.llm.generate_async(formatted_prompt, ProcessedKnowledge)

        entry_id = str(uuid.uuid4())
        embedding = self._get_openai_embedding(processed.content)

        # Add to FAISS index and persistent store
        self.index.add(np.array([embedding], dtype=np.float32))
        self.knowledge_store[entry_id] = {
            "content": processed.content,
            "tags": processed.tags,
            "embedding": embedding,
        }
        self._save_knowledge_store()
        return entry_id

    async def retrieve_knowledge(
        self, query: str, n_results: int = 5
    ) -> Dict[str, List[Any]]:
        query_embedding = self._get_openai_embedding(query)

        # Perform search in FAISS index
        distances, indices = self.index.search(
            np.array([query_embedding], dtype=np.float32), n_results
        )
        results = {
            "documents": [],
            "tags": [],
            "distances": distances.tolist(),
        }

        for idx in indices[0]:
            if idx != -1:  # Valid result
                entry_id = list(self.knowledge_store.keys())[idx]
                results["documents"].append(self.knowledge_store[entry_id]["content"])
                results["tags"].append(self.knowledge_store[entry_id]["tags"])

        return results

    async def prune(self, feedback: str, ids: List[str]):
        existing_entries = [self.knowledge_store[i] for i in ids]
        existing_content = "\n".join([entry["content"] for entry in existing_entries])

        pruning_input = f"""
        Feedback: {feedback}
        Existing Content:
        {existing_content}
        """

        system_prompt, user_prompt = self.pruning_agent.prompt(pruning_input)
        formatted_prompt = self.llm.format_prompt(system_prompt, user_prompt)
        pruning_suggestions = await self.llm.generate_async(
            formatted_prompt, PruningSuggestions
        )

        # Delete the old entries
        for i in ids:
            del self.knowledge_store[i]

        # Add the new entries
        new_ids = []
        for new_entry in pruning_suggestions.new_entries:
            new_id = await self.save_knowledge(new_entry)
            new_ids.append(new_id)

        self._save_knowledge_store()
        print(f"Pruned {len(ids)} old entries and added {len(new_ids)} new entries.")
        return new_ids
