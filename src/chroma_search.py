import chromadb
from chromadb.utils import embedding_functions
import os
from typing import Dict, Any, List, Optional

class ChromaManager:
    def __init__(self, path: str = "data/chroma", collection_name: str = "chapters"):
        os.makedirs(path, exist_ok=True)
        self.client = chromadb.PersistentClient(path=path)
        self.embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embed_fn,
        )

    def add_chapter(self, chapter_id: str, text: str, metadata: Dict[str, Any]):
        """Adds a chapter with its text and metadata."""
        # Ensure metadata values are serializable
        for key, value in metadata.items():
            if not isinstance(value, (str, int, float, bool)):
                metadata[key] = str(value)
        
        self.collection.add(
            documents=[text], 
            metadatas=[metadata], 
            ids=[chapter_id]
        )
        print(f"Added chapter {chapter_id} to ChromaDB.")

    def query_chapters(
        self,
        query_text: str,
        n_results: int = 3,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Queries for chapters with optional metadata filtering."""
        query_params = {
            "query_texts": [query_text],
            "n_results": n_results,
            "include": ["metadatas", "documents", "distances"]
        }
        
        if filter_metadata:
            query_params["where"] = filter_metadata
        
        return self.collection.query(**query_params)

# Global instance for backward compatibility
db_manager = ChromaManager()

def add_chapter(id: str, text: str, metadata: dict):
    """Backward compatibility function."""
    db_manager.add_chapter(id, text, metadata)

def query_similar(text_query: str, n_results: int = 3):
    """Backward compatibility function."""
    return db_manager.query_chapters(text_query, n_results)
