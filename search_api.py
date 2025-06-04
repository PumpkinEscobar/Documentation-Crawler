from fastapi import FastAPI, HTTPException, Query
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import uvicorn
from doc_search import DocumentationSearch, create_search_instance, SearchResult
from dataclasses import asdict

app = FastAPI(
    title="AI Documentation Search API",
    description="API for searching through curated AI documentation",
    version="1.0.0"
)

# Initialize search instance
try:
    search_instance = create_search_instance()
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Please run the crawler first to generate the search index.")
    exit(1)

class SearchQuery(BaseModel):
    query: str
    top_k: Optional[int] = 5
    filters: Optional[Dict[str, Any]] = None

class SearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    metadata: Dict[str, Any]

@app.post("/search", response_model=SearchResponse)
async def search(query: SearchQuery):
    """
    Search through the documentation with an optional filter.
    
    Example:
    ```json
    {
        "query": "How do I handle rate limits in GPT-4?",
        "top_k": 3,
        "filters": {
            "source": "openai",
            "type": "main_content"
        }
    }
    ```
    """
    try:
        if query.filters:
            results = await search_instance.semantic_search(query.query, query.filters)
        else:
            results = await search_instance.search(query.query, query.top_k)
            
        return SearchResponse(
            results=[asdict(r) for r in results],
            metadata={
                "total_results": len(results),
                "query": query.query,
                "filters_applied": bool(query.filters)
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/context/{source}/{section}/{chunk_id}")
async def get_context(
    source: str,
    section: str,
    chunk_id: int,
    window_size: int = Query(default=2, ge=0, le=5)
):
    """Get the context window around a specific chunk of documentation"""
    try:
        # Create a dummy SearchResult to use with get_context_window
        for chunk in search_instance.embeddings_data['chunks']:
            if chunk['chunk_id'] == f"{source}_{section}_{chunk_id}":
                result = SearchResult(
                    content=chunk['content'],
                    url=chunk['url'],
                    source=chunk['source'],
                    title=chunk['metadata']['title'],
                    relevance_score=1.0,
                    metadata=chunk['metadata']
                )
                context = search_instance.get_context_window(result, window_size)
                return {"context": context}
                
        raise HTTPException(status_code=404, detail="Chunk not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/multi-search")
async def multi_search(queries: List[str], top_k: int = Query(default=3, ge=1, le=10)):
    """Search with multiple related queries"""
    try:
        results = await search_instance.multi_query_search(queries, top_k)
        return {
            "results": {
                query: [asdict(r) for r in query_results]
                for query, query_results in results.items()
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sources")
def list_sources():
    """List all available documentation sources"""
    try:
        sources = set(
            chunk['source']
            for chunk in search_instance.embeddings_data['chunks']
        )
        return {"sources": list(sources)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    """Check if the search service is running and index is loaded"""
    return {
        "status": "healthy",
        "index_loaded": bool(search_instance.index),
        "embedding_model": search_instance.embedding_model
    }

def main():
    """Run the API server"""
    uvicorn.run(
        "search_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

if __name__ == "__main__":
    main() 