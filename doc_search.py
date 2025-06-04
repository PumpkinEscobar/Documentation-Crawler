import json
import numpy as np
import faiss
from typing import List, Dict, Any, Optional
import tiktoken
from dataclasses import dataclass
from openai import AsyncOpenAI
import asyncio
from pathlib import Path

@dataclass
class SearchResult:
    content: str
    url: str
    source: str
    title: str
    relevance_score: float
    metadata: Dict[str, Any]

class DocumentationSearch:
    def __init__(self, embeddings_file: str, index_file: str):
        self.embeddings_file = embeddings_file
        self.index_file = index_file
        
        with open(embeddings_file, 'r') as f:
            self.embeddings_data = json.load(f)
        
        self.embedding_model = self.embeddings_data['metadata']['model']
        self.index = faiss.read_index(index_file)
        self.client = AsyncOpenAI()

    async def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Search for relevant documentation"""
        try:
            # Get query embedding using new API
            response = await self.client.embeddings.create(
                model=self.embedding_model,
                input=query
            )
            query_embedding = np.array([response.data[0].embedding], dtype=np.float32)
            
            # Search using FAISS
            scores, indices = self.index.search(query_embedding, top_k)
            
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx == -1:  # Invalid index
                    continue
                    
                chunk = self.embeddings_data['chunks'][idx]
                results.append(SearchResult(
                    content=chunk['content'],
                    url=chunk['url'],
                    source=chunk['source'],
                    title=chunk['metadata']['title'],
                    relevance_score=float(score),
                    metadata=chunk['metadata']
                ))
            
            return results
        except Exception as e:
            print(f"Error during search: {e}")
            return []

    async def semantic_search(self, query: str, filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Search with semantic understanding and optional filters"""
        results = await self.search(query, top_k=20)
        
        if filters:
            filtered_results = []
            for result in results:
                if all(
                    getattr(result, key, result.metadata.get(key)) == value
                    for key, value in filters.items()
                ):
                    filtered_results.append(result)
            results = filtered_results
        
        return results[:5]  # Return top 5 after filtering

    def get_context_window(self, result: SearchResult, window_size: int = 2) -> str:
        """Get context around a search result"""
        # Find the chunk in our data
        target_chunk_id = None
        for i, chunk in enumerate(self.embeddings_data['chunks']):
            if chunk['content'] == result.content:
                target_chunk_id = i
                break
        
        if target_chunk_id is None:
            return result.content
        
        # Get surrounding chunks
        start_idx = max(0, target_chunk_id - window_size)
        end_idx = min(len(self.embeddings_data['chunks']), target_chunk_id + window_size + 1)
        
        context_chunks = []
        for i in range(start_idx, end_idx):
            chunk = self.embeddings_data['chunks'][i]
            prefix = ">>> " if i == target_chunk_id else "    "
            context_chunks.append(f"{prefix}{chunk['content']}")
        
        return "\n".join(context_chunks)

    async def multi_query_search(self, queries: List[str], top_k: int = 3) -> Dict[str, List[SearchResult]]:
        """Search with multiple related queries"""
        results = {}
        for query in queries:
            results[query] = await self.search(query, top_k)
        return results

def create_search_instance(data_dir: str = ".") -> DocumentationSearch:
    """Create a DocumentationSearch instance with default paths"""
    embeddings_file = Path(data_dir) / "embeddings.json"
    index_file = Path(data_dir) / "docs_search.index"
    
    if not embeddings_file.exists() or not index_file.exists():
        raise FileNotFoundError(
            "Documentation search files not found. Please run the crawler first."
        )
    
    return DocumentationSearch(str(embeddings_file), str(index_file))

async def main():
    """Example usage"""
    try:
        search = create_search_instance()
        
        # Basic search
        results = await search.search(
            "How do I handle rate limits in the OpenAI API?",
            top_k=3
        )
        
        print("\nSearch Results:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result.title} ({result.source})")
            print(f"URL: {result.url}")
            print(f"Relevance: {result.relevance_score:.3f}")
            print(f"Content: {result.content[:200]}...")
        
        # Filtered search
        filtered_results = await search.semantic_search(
            "GPT-4 capabilities",
            filters={"source": "openai", "type": "main_content"}
        )
        
        print("\nFiltered Results:")
        for i, result in enumerate(filtered_results, 1):
            print(f"\n{i}. {result.title}")
            print(f"Content: {result.content[:200]}...")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 