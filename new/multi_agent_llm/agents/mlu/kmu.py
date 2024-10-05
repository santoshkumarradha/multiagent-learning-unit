import asyncio
import os
import uuid
from typing import Any, Dict, List, Optional, Union

import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from chromadb.utils import embedding_functions
from pydantic import BaseModel, Field

from ...agent_class import Agent
from ...llm import LLMBase


class SearchTerms(BaseModel):
    terms: List[str] = Field(..., description="List of search terms")


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
        chroma_db: Union[str, chromadb.PersistentClient] = "./.db",
        compress_knowledge=False,
        chroma_db_path: Optional[str] = None,
        collection_name: str = None,
    ):
        self.llm = llm
        self.main_goal = main_goal
        self.storage_goal = storage_goal
        self.retrieval_goal = retrieval_goal
        if collection_name is None:
            collection_name = f"knowledge_base_{uuid.uuid4().hex}"
        if chroma_db_path:
            chroma_db = chroma_db_path
        if isinstance(chroma_db, str):
            self.client = chromadb.PersistentClient(path=chroma_db)
        else:
            self.client = chroma_db

        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"), model_name="text-embedding-3-small"
        )
        self.embedding_function = (
            openai_ef  # embedding_functions.DefaultEmbeddingFunction()
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name, embedding_function=self.embedding_function
        )

        self.compress_knowledge = (
            "Make sure that you condense this information to much smaller sentences or small single paragraph to extract all the relevant information in a compact way from this content. Don't be verbose. Make it pointed to be detailed yet short. Include all points."
            if compress_knowledge
            else ""
        )

        self.search_agent = self._create_search_agent()
        self.knowledge_processor = self._create_knowledge_processor()
        self.pruning_agent = self._create_pruning_agent()

    def _create_search_agent(self):
        return Agent(
            name="SearchAgent",
            role="Generate search terms for knowledge retrieval",
            function=f"""
            Main Goal: {self.main_goal}
            Retrieval Goal: {self.retrieval_goal}

            You are provided with a query that aligns with a broader knowledge objective. 
            1. Based on the query and goals, identify key terms that maximize information retrieval. 
            2. Use feedback from previous searches (if available) to refine your approach and avoid redundant terms.
            3. Prioritize specific terms that align with the storage goal, making retrieval efficient and comprehensive.

            Output a concise and ranked list of search terms that best achieve the retrieval goal.
            """,
        )

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
            
            Output your suggestions as a list of modified/new entries added to the knowledge base.
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
        self.collection.add(
            documents=[processed.content + "\ntags: " + ",".join(processed.tags)],
            ids=[entry_id],
            metadatas=[{"tags": ",".join(processed.tags)}],
        )
        return entry_id

    async def retrieve_knowledge(
        self, query: str, n_results: int = 5
    ) -> Dict[str, List[Any]]:
        system_prompt, user_prompt = self.search_agent.prompt(f"Query: {query}")
        formatted_prompt = self.llm.format_prompt(system_prompt, user_prompt)
        search_output = await self.llm.generate_async(formatted_prompt, SearchTerms)

        combined_query = " ".join(search_output.terms)
        results = self.collection.query(
            query_texts=[combined_query], n_results=n_results
        )
        return results

    async def prune(self, feedback: str, ids: List[str]):
        existing_entries = self.collection.get(ids=ids)
        existing_content = "\n".join(existing_entries["documents"])
        existing_tags = [metadata["tags"] for metadata in existing_entries["metadatas"]]

        pruning_input = f"""
        Feedback: {feedback}
        Existing Content:
        {existing_content}
        Existing Tags:
        {existing_tags}
        """

        system_prompt, user_prompt = self.pruning_agent.prompt(pruning_input)
        formatted_prompt = self.llm.format_prompt(system_prompt, user_prompt)
        pruning_suggestions = await self.llm.generate_async(
            formatted_prompt, PruningSuggestions
        )

        # Delete the old entries
        self.collection.delete(ids=ids)

        # Add the new entries
        new_ids = []
        for new_entry in pruning_suggestions.new_entries:
            new_id = await self.save_knowledge(new_entry)
            new_ids.append(new_id)

        print(f"Pruned {len(ids)} old entries and added {len(new_ids)} new entries.")
        return new_ids
