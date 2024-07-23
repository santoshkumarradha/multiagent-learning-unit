import uuid
from typing import Any, Dict, List, Optional, Union

import chromadb
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from agents import Agent


class SearchTerms(BaseModel):
    terms: List[str] = Field(..., description="List of search terms")


class ProcessedKnowledge(BaseModel):
    content: str = Field(..., description="Processed content")
    tags: List[str] = Field(..., description="List of tags")


class KnowledgeManagementSystem:
    def __init__(
        self,
        main_goal: str,
        storage_goal: str,
        retrieval_goal: str,
        chroma_db: Union[str, chromadb.PersistentClient] = "./.db",
        compress_knowldge=False,
        chroma_db_path: Optional[str] = None,
        collection_name: str = None,
    ):
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
        self.collection = self.client.get_or_create_collection(collection_name)
        self.sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

        if compress_knowldge:
            self.compress_knowldge = "Make sure that you condense this information to much smaller sentences or small single paragraph to extract the all the relevant information in a compact way from this content. dont be verbose. Make it pointed to be detailed yet short. include all points"

        self.search_agent = Agent(
            name="SearchAgent",
            role="Generate search terms for knowledge retrieval",
            function=f"""
            Main Goal: {self.main_goal}
            Retrieval Goal: {self.retrieval_goal}

            Your task is to generate a list of search terms based on the given query and goals.
            These terms will be used to search the knowledge base effectively.

            Output a list of search terms.
            """,
            output_model=SearchTerms,
        )

        self.knowledge_processor = Agent(
            name="KnowledgeProcessor",
            role="Process and align knowledge with the main goal",
            function=f"""
            Main Goal: {self.main_goal}
            Storage Goal: {self.storage_goal}

            Your task is to process the given content to align it with the main goal and storage goal.
            Modify the content if necessary and generate relevant tags. Choose tags that are aligned with the main goal.
            Modify the content so that we extract relevant and concise information based on the main goal and storage goal.
            Prefer detailed but concise information that is relevant to the main goal and storage goal.
            
            {"IMPORTANT:"+self.compress_knowldge if compress_knowldge else ""}

            Output the processed content and a list of tags.
            """,
            output_model=ProcessedKnowledge,
        )

        class PruningSuggestions(BaseModel):
            new_entries: List[str] = Field(..., description="List of new entries")

        self.pruning_agent = Agent(
            name="PruningAgent",
            role="Analyze feedback and suggest knowledge base modifications",
            function=f"""
            Main goal: {main_goal}
            Analyze the provided feedback and rewrite the knowldge that has been used to be more relevant for the main goal and storage goal
            and provide new modified entries that reflect the feedback of the existing knowldge.
            Your new knowldge, based on feedback  can include 
            1. modifications of the original knowldge 
            2. Addition of new information to original knowldge
            3. removal of old knowldge
            4. Completely new addition of new knowldge
            It should aim to improve the quality and relevance of the knowledge base.
            Always club together the similar knowldge and remove the redundant knowldge.
            
            Output your suggestions as a list of modified/new entries added to the knowledge base.
            {"IMPORTANT:"+self.compress_knowldge if compress_knowldge else ""}
            """,
            output_model=PruningSuggestions,
            disable_logging=True,
        )

    def save_knowledge(self, content: str) -> str:
        processed = self.knowledge_processor(
            f"Content: {content}",
        )
        entry_id = str(uuid.uuid4())
        embedding = self.sentence_model.encode([processed.content])[0]
        self.collection.add(
            documents=[processed.content],
            embeddings=[embedding.tolist()],
            ids=[entry_id],
            metadatas=[{"tags": ",".join(processed.tags)}],
        )
        return [item for sublist in entry_id for item in sublist]

    def retrieve_knowledge(
        self, query: str, n_results: int = 5
    ) -> Dict[str, List[Any]]:
        search_output = self.search_agent(
            f"Query: {query}",
        )
        combined_query = " ".join(search_output.terms)
        query_embedding = self.sentence_model.encode([combined_query])[0]
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()], n_results=n_results
        )
        return results

    def prune(self, feedback: str, ids: List[str]):
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

        pruning_suggestions = self.pruning_agent(
            pruning_input,
        )

        # Delete the old entries
        self.collection.delete(ids=ids)

        # Add the new entries
        new_ids = []
        for new_entry in pruning_suggestions.new_entries:
            new_id = self.save_knowledge(new_entry)
            new_ids.append(new_id)

        print(f"Pruned {len(ids)} old entries and added {len(new_ids)} new entries.")
        return new_ids
