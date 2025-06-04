from fastapi import FastAPI, HTTPException, Query
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import uvicorn
import os
import json
import re
from datetime import datetime
from pathlib import Path
from doc_search import DocumentationSearch, create_search_instance, SearchResult
from dataclasses import asdict

app = FastAPI(
    title="AI Documentation Search API",
    description="API for searching through curated AI documentation",
    version="1.0.0"
)

# Security headers middleware
@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self'; object-src 'none'"
    return response

# Initialize search instance
try:
    search_instance = create_search_instance()
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Please run the crawler first to generate the search index.")
    search_instance = None

# Knowledge base for wizard functionality
class WizardKnowledgeBase:
    def __init__(self, knowledge_dir="knowledge_base"):
        self.knowledge_dir = knowledge_dir
        self.knowledge_cache = {}
        self.providers = ["openai", "anthropic", "google", "meta", "xai"]
        self.content_types = ["api_docs", "guides", "examples", "best_practices"]
        self.load_knowledge()
    
    def load_knowledge(self):
        """Load and index knowledge files"""
        if not os.path.exists(self.knowledge_dir):
            os.makedirs(self.knowledge_dir)
            return
        
        for root, dirs, files in os.walk(self.knowledge_dir):
            for file in files:
                if file.endswith(('.txt', '.md', '.json', '.yaml')):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            relative_path = os.path.relpath(file_path, self.knowledge_dir)
                            self.knowledge_cache[relative_path] = {
                                'content': content,
                                'last_modified': os.path.getmtime(file_path),
                                'size': len(content),
                                'type': self._classify_content(relative_path, content)
                            }
                    except Exception:
                        continue
    
    def _classify_content(self, file_path, content):
        """Classify content type based on path and content"""
        path_lower = file_path.lower()
        content_lower = content.lower()
        
        if 'api' in path_lower or 'endpoint' in content_lower:
            return 'api_docs'
        elif 'guide' in path_lower or 'tutorial' in path_lower:
            return 'guides'
        elif 'example' in path_lower or 'cookbook' in path_lower:
            return 'examples'
        elif 'best_practice' in path_lower or 'pattern' in path_lower:
            return 'best_practices'
        else:
            return 'general'
    
    def search(self, query, provider=None, content_type=None, limit=10):
        """Search knowledge base with optional filters"""
        query_terms = query.lower().split()
        results = []
        
        for file_path, data in self.knowledge_cache.items():
            content_lower = data['content'].lower()
            
            # Apply filters
            if provider and provider.lower() not in file_path.lower():
                continue
            if content_type and data['type'] != content_type:
                continue
            
            # Calculate relevance score
            score = 0
            for term in query_terms:
                score += content_lower.count(term)
            
            if score > 0:
                snippet = self._extract_snippet(data['content'], query_terms)
                results.append({
                    'file': file_path,
                    'score': score,
                    'snippet': snippet,
                    'type': data['type'],
                    'size': data['size'],
                    'last_modified': datetime.fromtimestamp(data['last_modified']).isoformat()
                })
        
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:limit]
    
    def _extract_snippet(self, content, query_terms, snippet_length=300):
        """Extract relevant snippet around query terms"""
        content_lower = content.lower()
        
        first_pos = len(content)
        for term in query_terms:
            pos = content_lower.find(term)
            if pos != -1 and pos < first_pos:
                first_pos = pos
        
        if first_pos == len(content):
            return content[:snippet_length] + "..."
        
        start = max(0, first_pos - snippet_length // 2)
        end = min(len(content), start + snippet_length)
        
        snippet = content[start:end]
        if start > 0:
            snippet = "..." + snippet
        if end < len(content):
            snippet = snippet + "..."
        
        return snippet

# Initialize knowledge base
kb = WizardKnowledgeBase()

class SearchQuery(BaseModel):
    query: str
    top_k: Optional[int] = 5
    filters: Optional[Dict[str, Any]] = None

class SearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    metadata: Dict[str, Any]

# Enhanced search endpoints (from search_api.py)
@app.post("/search", response_model=SearchResponse)
async def semantic_search(query: SearchQuery):
    """Enhanced semantic search through documentation"""
    if not search_instance:
        raise HTTPException(status_code=503, detail="Search index not available")
    
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

# Simple search endpoints (from wizard_api.py)
@app.get("/api/search")
async def simple_search(
    query: str = Query(..., description="Search query"),
    provider: Optional[str] = Query(None, description="Filter by provider"),
    content_type: Optional[str] = Query(None, description="Filter by content type"), 
    limit: int = Query(10, ge=1, le=50, description="Maximum results")
):
    """Simple knowledge base search"""
    try:
        results = kb.search(query, provider, content_type, limit)
        return {
            'query': query,
            'filters': {'provider': provider, 'content_type': content_type},
            'total_results': len(results),
            'results': results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/content/{file_path:path}")
async def get_file_content(file_path: str):
    """Get full content of a specific file"""
    try:
        if file_path in kb.knowledge_cache:
            return {
                'file': file_path,
                'content': kb.knowledge_cache[file_path]['content']
            }
        raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/providers")
def get_providers():
    """Get list of available providers"""
    return {
        'providers': kb.providers,
        'content_types': kb.content_types
    }

@app.get("/api/stats")
def get_stats():
    """Get knowledge base statistics"""
    try:
        stats = {
            'total_files': len(kb.knowledge_cache),
            'content_types': {},
            'providers': {},
            'total_size': 0
        }
        
        for file_path, data in kb.knowledge_cache.items():
            content_type = data['type']
            stats['content_types'][content_type] = stats['content_types'].get(content_type, 0) + 1
            
            for provider in kb.providers:
                if provider in file_path.lower():
                    stats['providers'][provider] = stats['providers'].get(provider, 0) + 1
                    break
            
            stats['total_size'] += data['size']
        
        return stats
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
    if not search_instance:
        raise HTTPException(status_code=503, detail="Search index not available")
        
    try:
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
    if not search_instance:
        raise HTTPException(status_code=503, detail="Search index not available")
        
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
    if not search_instance:
        return {"sources": []}
        
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
    """Check if the search service is running"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "search_index_loaded": bool(search_instance and search_instance.index),
        "knowledge_files": len(kb.knowledge_cache),
        "embedding_model": search_instance.embedding_model if search_instance else None
    }

def main():
    """Run the consolidated API server"""
    port = int(os.environ.get('PORT', 8000))
    debug_mode = os.environ.get('DEBUG', '0').lower() in ['true', '1']
    
    uvicorn.run(
        "search_api:app",
        host="0.0.0.0",
        port=port,
        reload=debug_mode
    )

if __name__ == "__main__":
    main() 