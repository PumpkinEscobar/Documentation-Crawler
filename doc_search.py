import json
import numpy as np
import faiss
from typing import List, Dict, Any, Optional
import tiktoken
from dataclasses import dataclass
import openai
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
        # Load embeddings and chunks
        with open(embeddings_file, 'r') as f:
            self.embeddings_data = json.load(f)
            
        # Load FAISS index
        self.index = faiss.read_index(index_file)
        
        # Initialize embedding model
        self.embedding_model = self.embeddings_data['metadata']['model']
        self.encoding = tiktoken.get_encoding("cl100k_base")
        
    async def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Search for relevant documentation chunks"""
        try:
            # Get query embedding
            response = await openai.Embedding.acreate(
                input=[query],
                model=self.embedding_model
            )
            query_embedding = np.array([response.data[0].embedding]).astype('float32')
            
            # Search index
            scores, indices = self.index.search(query_embedding, top_k)
            
            # Format results
            results = []
            for score, idx in zip(scores[0], indices[0]):
                chunk_data = self.embeddings_data['chunks'][idx]
                result = SearchResult(
                    content=chunk_data['content'],
                    url=chunk_data['url'],
                    source=chunk_data['source'],
                    title=chunk_data['metadata']['title'],
                    relevance_score=float(score),
                    metadata=chunk_data['metadata']
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            print(f"Error during search: {e}")
            return []

    async def semantic_search(self, query: str, filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Search with optional filters"""
        results = await self.search(query)
        
        if filters:
            filtered_results = []
            for result in results:
                matches_filters = all(
                    result.metadata.get(key) == value
                    for key, value in filters.items()
                )
                if matches_filters:
                    filtered_results.append(result)
            return filtered_results
            
        return results

    def get_context_window(self, result: SearchResult, window_size: int = 2) -> str:
        """Get surrounding context for a search result"""
        try:
            chunk_index = result.metadata['chunk_index']
            source = result.source
            section = result.metadata['section']
            
            # Find adjacent chunks
            context_chunks = []
            for i in range(max(0, chunk_index - window_size), chunk_index + window_size + 1):
                chunk_id = f"{source}_{section}_{i}"
                for chunk in self.embeddings_data['chunks']:
                    if chunk['chunk_id'] == chunk_id:
                        context_chunks.append(chunk['content'])
                        break
            
            return "\n\n".join(context_chunks)
            
        except Exception as e:
            print(f"Error getting context: {e}")
            return result.content

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