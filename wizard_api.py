from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import re
from datetime import datetime
import logging

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Apply secure logging formatter
class SecureFormatter(logging.Formatter):
    def format(self, record):
        message = super().format(record)
        patterns = [
            r'api[_-]?key[\"\\s:=]+[a-zA-Z0-9-_]+',
            r'token[\"\\s:=]+[a-zA-Z0-9-_]+',
            r'bearer\\s+[a-zA-Z0-9-_]+',
        ]
        for pattern in patterns:
            message = re.sub(pattern, '[REDACTED]', message, flags=re.IGNORECASE)
        return message

for handler in logging.root.handlers:
    handler.setFormatter(SecureFormatter(handler.formatter._fmt))

@app.after_request
def add_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self'; object-src 'none'; frame-ancestors 'self'"
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains' # if served over HTTPS
    # Consider more restrictive CSP if your app requires it
    return response

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
            logger.warning(f"Created knowledge directory: {self.knowledge_dir}")
            return
        
        for root, dirs, files in os.walk(self.knowledge_dir):
            for file in files:
                if file.endswith(('.txt', '.md', '.json', '.yaml')):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            # Create searchable index entry
                            relative_path = os.path.relpath(file_path, self.knowledge_dir)
                            self.knowledge_cache[relative_path] = {
                                'content': content,
                                'last_modified': os.path.getmtime(file_path),
                                'size': len(content),
                                'type': self._classify_content(relative_path, content)
                            }
                    except Exception as e:
                        logger.error(f"Error loading {file_path}: {e}")
        
        logger.info(f"Loaded {len(self.knowledge_cache)} knowledge files")
    
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
                # Extract relevant snippet
                snippet = self._extract_snippet(data['content'], query_terms)
                results.append({
                    'file': file_path,
                    'score': score,
                    'snippet': snippet,
                    'type': data['type'],
                    'size': data['size'],
                    'last_modified': datetime.fromtimestamp(data['last_modified']).isoformat()
                })
        
        # Sort by relevance and limit results
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:limit]
    
    def _extract_snippet(self, content, query_terms, snippet_length=300):
        """Extract relevant snippet around query terms"""
        content_lower = content.lower()
        
        # Find first occurrence of any query term
        first_pos = len(content)
        for term in query_terms:
            pos = content_lower.find(term)
            if pos != -1 and pos < first_pos:
                first_pos = pos
        
        if first_pos == len(content):
            return content[:snippet_length] + "..."
        
        # Extract snippet around the term
        start = max(0, first_pos - snippet_length // 2)
        end = min(len(content), start + snippet_length)
        
        snippet = content[start:end]
        if start > 0:
            snippet = "..." + snippet
        if end < len(content):
            snippet = snippet + "..."
        
        return snippet
    
    def get_content(self, file_path):
        """Get full content of a specific file"""
        if file_path in self.knowledge_cache:
            return self.knowledge_cache[file_path]['content']
        return None

# Initialize knowledge base
kb = WizardKnowledgeBase()

@app.route('/api/search', methods=['GET'])
def search_knowledge():
    """Search the knowledge base"""
    try:
        query = request.args.get('query', '')
        provider = request.args.get('provider', None)
        content_type = request.args.get('content_type', None)
        limit = int(request.args.get('limit', 10))
        
        if not query:
            return jsonify({'error': 'Query parameter is required'}), 400
        
        results = kb.search(query, provider, content_type, limit)
        
        return jsonify({
            'query': query,
            'filters': {
                'provider': provider,
                'content_type': content_type
            },
            'total_results': len(results),
            'results': results
        })
    
    except Exception as e:
        logger.error(f"Search error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/content/<path:file_path>', methods=['GET'])
def get_content(file_path):
    """Get full content of a specific file"""
    try:
        content = kb.get_content(file_path)
        if content is None:
            return jsonify({'error': 'File not found'}), 404
        
        return jsonify({
            'file': file_path,
            'content': content
        })
    
    except Exception as e:
        logger.error(f"Content retrieval error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/providers', methods=['GET'])
def get_providers():
    """Get list of available providers"""
    return jsonify({
        'providers': kb.providers,
        'content_types': kb.content_types
    })

@app.route('/api/stats', methods=['GET'])
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
            # Count by content type
            content_type = data['type']
            stats['content_types'][content_type] = stats['content_types'].get(content_type, 0) + 1
            
            # Count by provider
            for provider in kb.providers:
                if provider in file_path.lower():
                    stats['providers'][provider] = stats['providers'].get(provider, 0) + 1
                    break
            
            stats['total_size'] += data['size']
        
        return jsonify(stats)
    
    except Exception as e:
        logger.error(f"Stats error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'knowledge_files': len(kb.knowledge_cache)
    })

if __name__ == '__main__':
    # Securely get port from environment or default
    try:
        port = int(os.environ.get('PORT', "5000"))
        if not (1024 <= port <= 65535):
            logger.warning(f"Port {port} is outside the recommended range (1024-65535). Using default 5000.")
            port = 5000
    except ValueError:
        logger.warning("Invalid PORT environment variable. Using default 5000.")
        port = 5000

    # Ensure debug is False in production environments
    debug_mode = os.environ.get('FLASK_DEBUG', '0').lower() in ['true', '1']
    if os.environ.get("ENVIRONMENT") == "production" and debug_mode:
        logger.warning("Flask debug mode is enabled in a production environment! Forcing debug mode to False.")
        debug_mode = False
        
    app.run(host='0.0.0.0', port=port, debug=debug_mode) 