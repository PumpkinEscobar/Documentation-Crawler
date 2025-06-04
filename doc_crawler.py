from playwright.async_api import async_playwright
import requests
import json
import yaml
import sys
import time
import os
from typing import Optional, Dict, List, Any, Tuple
from urllib.parse import urljoin, urlparse
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
import asyncio
import re
import numpy as np
import tiktoken
from tqdm import tqdm
import openai

# Secure configuration management
class SecureConfig:
    def __init__(self):
        self._api_keys = {}
        
    def get_api_key(self, service: str) -> Optional[str]:
        """Securely retrieve API key for a service"""
        if service not in self._api_keys:
            env_var = f"{service.upper()}_API_KEY"
            key = os.getenv(env_var)
            if key:
                self._api_keys[service] = key
        return self._api_keys.get(service)
    
    def validate_api_key(self, service: str) -> bool:
        """Validate that API key exists for service"""
        return self.get_api_key(service) is not None

# Global secure config instance
secure_config = SecureConfig()

# Input validation functions
def validate_url(url: str) -> bool:
    """Validate URL for security"""
    from urllib.parse import urlparse
    
    if not url or not isinstance(url, str):
        raise ValueError("Invalid URL format")
    
    parsed = urlparse(url)
    if parsed.scheme != 'https':
        raise ValueError("Only HTTPS URLs allowed")
    
    # Whitelist allowed domains
    allowed_domains = [
        'platform.openai.com',
        'docs.anthropic.com', 
        'ai.google.dev',
        'huggingface.co'
    ]
    
    if not any(domain in parsed.netloc for domain in allowed_domains):
        raise ValueError(f"Domain {parsed.netloc} not in allowlist")
    
    return True

@dataclass
class ModelInfo:
    id: str
    description: str
    capabilities: list[str]
    token_limit: int
    recommended_use: str
    category: str

@dataclass
class DocumentChunk:
    content: str
    source: str
    url: str
    chunk_id: str
    metadata: Dict[str, Any]

class ModelDataProcessor:
    def __init__(self):
        self.model_categories = {
            'gpt-4': 'text-generation',
            'gpt-3.5': 'text-generation',
            'dall-e': 'image-generation',
            'whisper': 'speech-to-text',
            'tts': 'text-to-speech',
            'text-embedding': 'embeddings'
        }
        
        self.model_capabilities = {
            'gpt-4': ['text', 'chat', 'vision', 'analysis'],
            'gpt-3.5': ['text', 'chat'],
            'dall-e': ['image'],
            'whisper': ['audio', 'transcription'],
            'tts': ['audio', 'speech'],
            'text-embedding': ['embeddings', 'vector']
        }
        
        self.token_limits = {
            'gpt-4': 128000,
            'gpt-4-turbo': 128000,
            'gpt-3.5-turbo': 16385,
            'text-embedding-3-large': 8191,
            'text-embedding-3-small': 8191
        }
        
        self.model_descriptions = {
            'gpt-4': 'Most capable GPT-4 model, ideal for tasks requiring deep understanding and complex problem-solving',
            'gpt-4-turbo': 'Optimized GPT-4 model balancing speed and capabilities',
            'gpt-3.5-turbo': 'Fast and cost-effective model for most chat and text generation tasks',
            'dall-e-3': 'Advanced image generation model with high fidelity and artistic capabilities',
            'text-embedding-3-large': 'High-performance text embedding model for semantic search and analysis',
            'whisper': 'Robust speech recognition model supporting multiple languages',
            'tts': 'Natural-sounding text-to-speech synthesis'
        }

    def should_include_model(self, model_id: str) -> bool:
        """Determine if a model should be included in the cleaned output"""
        # Exclude internal, preview, and mini variants unless they're major releases
        exclusion_patterns = [
            r'-preview-\d{4}',  # Date-specific previews
            r'-internal$',
            r'-(nano|mini)(?!-\d{4})',  # Mini/nano variants except dated releases
            r'search|audio|transcribe|latest',  # Specialized internal variants
            r'moderation|codex'  # Internal tools
        ]
        
        for pattern in exclusion_patterns:
            if re.search(pattern, model_id):
                return False
                
        # Include only the latest version of major models
        major_models = [
            'gpt-4', 'gpt-4-turbo', 'gpt-3.5-turbo',
            'dall-e-3', 'text-embedding-3',
            'whisper', 'tts'
        ]
        
        return any(model_id.startswith(model) for model in major_models)

    def get_model_category(self, model_id: str) -> str:
        """Determine the primary category of a model"""
        for prefix, category in self.model_categories.items():
            if model_id.startswith(prefix):
                return category
        return 'other'

    def get_model_capabilities(self, model_id: str) -> list[str]:
        """Get the capabilities of a model"""
        for prefix, capabilities in self.model_capabilities.items():
            if model_id.startswith(prefix):
                return capabilities
        return []

    def get_token_limit(self, model_id: str) -> int:
        """Get the token limit for a model"""
        for prefix, limit in self.token_limits.items():
            if model_id.startswith(prefix):
                return limit
        return 4096  # Default limit

    def get_model_description(self, model_id: str) -> str:
        """Get the description for a model"""
        for prefix, description in self.model_descriptions.items():
            if model_id.startswith(prefix):
                return description
        return "General purpose AI model"

    def get_recommended_use(self, model_id: str) -> str:
        """Get recommended use cases for a model"""
        category = self.get_model_category(model_id)
        capabilities = self.get_model_capabilities(model_id)
        
        if 'text-generation' in category:
            if 'vision' in capabilities:
                return "Complex tasks requiring multimodal understanding (text, images, code)"
            elif model_id.startswith('gpt-4'):
                return "Advanced reasoning, analysis, and generation tasks"
            else:
                return "General purpose text generation and chat applications"
        elif 'image-generation' in category:
            return "High-quality image generation and editing"
        elif 'embeddings' in category:
            return "Semantic search, text similarity, and clustering applications"
        elif 'speech-to-text' in category:
            return "Audio transcription and voice recognition"
        elif 'text-to-speech' in category:
            return "High-quality voice synthesis and audio generation"
        return "General AI applications"

    def process_model_data(self, models_data: list) -> dict:
        """Process and clean model data"""
        processed_models = {}
        
        for model in models_data:
            model_id = model['id']
            
            if not self.should_include_model(model_id):
                continue
                
            model_info = ModelInfo(
                id=model_id,
                description=self.get_model_description(model_id),
                capabilities=self.get_model_capabilities(model_id),
                token_limit=self.get_token_limit(model_id),
                recommended_use=self.get_recommended_use(model_id),
                category=self.get_model_category(model_id)
            )
            
            processed_models[model_id] = asdict(model_info)
            
        return processed_models

@dataclass
class DocumentSection:
    title: str
    content: str
    url: str
    last_updated: str
    subsections: List[Dict[str, Any]]
    related_topics: List[str]
    code_examples: List[Dict[str, str]]
    authority_level: str  # 'official', 'community', 'experimental'
    last_verified: str  # ISO date when the content was last verified
    api_version: Optional[str]  # API version this documentation applies to
    example_count: int  # Number of code examples
    source_quality: float  # Score from 0-1 based on various quality metrics
    prerequisites: List[str]  # Required knowledge/setup
    platform_compatibility: List[str]  # Supported platforms/environments

    def __post_init__(self):
        if not self.last_verified:
            self.last_verified = datetime.now().isoformat()
        if not self.authority_level:
            self.authority_level = 'community'
        if not self.example_count:
            self.example_count = len(self.code_examples)
        if not self.source_quality:
            # Calculate quality score based on multiple factors
            self.source_quality = self._calculate_quality_score()
            
    def _calculate_quality_score(self) -> float:
        score = 0.5  # Base score
        
        # Adjust based on authority level
        auth_scores = {'official': 0.3, 'community': 0.1, 'experimental': 0.0}
        score += auth_scores.get(self.authority_level, 0.0)
        
        # Adjust based on content quality indicators
        if len(self.code_examples) > 0:
            score += 0.1
        if len(self.related_topics) > 0:
            score += 0.05
        if len(self.prerequisites) > 0:
            score += 0.05
            
        return min(1.0, score)  # Cap at 1.0

class VectorProcessor:
    def __init__(self, embedding_model: str = "text-embedding-3-small"):
        self.embedding_model = embedding_model
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.max_tokens = 8191  # Max tokens for embedding-3-small
        self.chunk_overlap = 50  # Number of tokens to overlap between chunks
        self.logger = logging.getLogger(__name__)
        
    def create_chunks(self, text: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """Split text into chunks suitable for embedding"""
        chunks = []
        tokens = self.encoding.encode(text)
        
        # Calculate chunk size to maintain reasonable length
        chunk_size = min(self.max_tokens, 1000)  # Smaller chunks for better granularity
        
        # Split into overlapping chunks
        for i in range(0, len(tokens), chunk_size - self.chunk_overlap):
            chunk_tokens = tokens[i:i + chunk_size]
            chunk_text = self.encoding.decode(chunk_tokens)
            
            chunk = DocumentChunk(
                content=chunk_text,
                source=metadata['source'],
                url=metadata['url'],
                chunk_id=f"{metadata['source']}_{metadata['section']}_{i}",
                metadata={
                    **metadata,
                    'chunk_index': i,
                    'token_count': len(chunk_tokens)
                }
            )
            chunks.append(chunk)
            
        return chunks

    async def get_embeddings(self, chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """Get embeddings for text chunks using OpenAI API"""
        if not secure_config.validate_api_key("openai"):
            self.logger.error("OpenAI API key not found for embeddings. Skipping.")
            return None
        
        openai.api_key = secure_config.get_api_key("openai") # Temporarily set for the SDK call

        try:
            embeddings_data = {
                'chunks': [],
                'embeddings': [],
                'metadata': {
                    'model': self.embedding_model,
                    'total_chunks': len(chunks),
                    'timestamp': datetime.now().isoformat()
                }
            }
            
            # Process chunks in batches to avoid rate limits
            batch_size = 50
            for i in tqdm(range(0, len(chunks), batch_size), desc="Generating embeddings"):
                batch = chunks[i:i + batch_size]
                
                # Get embeddings from OpenAI
                response = await openai.Embedding.acreate(
                    input=[chunk.content for chunk in batch],
                    model=self.embedding_model
                )
                
                # Store chunk data and embeddings
                for chunk, embedding_data in zip(batch, response.data):
                    embeddings_data['chunks'].append(asdict(chunk))
                    embeddings_data['embeddings'].append(embedding_data.embedding)
                
                # Respect rate limits
                await asyncio.sleep(0.1)
            
            return embeddings_data
            
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {str(e)}")
            return None
        finally:
            openai.api_key = None # Clear the API key after use

class DocumentationCrawler:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.browser_options = {
            'args': [
                '--no-sandbox', 
                '--disable-setuid-sandbox',
                '--disable-dev-shm-usage',  # Security: prevent shared memory issues
                '--disable-gpu',  # Security: disable GPU acceleration
                '--no-first-run',  # Security: skip first run setup
                '--disable-extensions'  # Security: disable extensions
            ]
        }
        self.setup_logging()
        self.results = {}
        self.browser = None
        
        # Validated source configuration
        self.sources = {
            'openai': 'https://platform.openai.com/docs',
            'anthropic': 'https://docs.anthropic.com',
            'google-ai': 'https://ai.google.dev/docs',
            'huggingface': 'https://huggingface.co/docs'
        }
        
        # Validate URLs at initialization
        for source, url in self.sources.items():
            try:
                validate_url(url)
            except ValueError as e:
                self.logger.error(f"Invalid URL for {source}: {e}")
                del self.sources[source]

    async def crawl_all(self):
        """Crawl all configured documentation sources"""
        try:
            # Security: Use secure browser options
            playwright = await async_playwright().start()
            self.browser = await playwright.chromium.launch(
                headless=True,  # Security: always run headless
                args=self.browser_options['args']
            )
            
            context = await self.browser.new_context(
                ignore_https_errors=False,  # Security: enforce HTTPS validation
                viewport={'width': 1280, 'height': 720},
                user_agent='DocumentationCrawler/1.0 (Security-Hardened)'
            )
            
            for source, base_url in self.sources.items():
                # Security: Validate each URL before crawling
                try:
                    validate_url(base_url)
                except ValueError as e:
                    self.logger.error(f"Skipping {source} due to security validation: {e}")
                    continue
                    
                self.logger.info(f"Crawling {source} documentation...")
                page = await context.new_page()
                await self._setup_page_handlers(page)
                
                try:
                    await self._crawl_source(page, source, base_url)
                except Exception as e:
                    self.logger.error(f"Error crawling {source}: {str(e)}")
                finally:
                    await page.close()
            
            await context.close()
            await self.browser.close()
            await playwright.stop()
            self.logger.info("Documentation crawl completed successfully!")
            return True
            
        except Exception as e:
            self.logger.error(f"Error during crawl: {str(e)}")
            return False

    async def _crawl_source(self, page, source: str, base_url: str):
        """Crawl a specific documentation source"""
        try:
            response = await page.goto(base_url, wait_until='networkidle', timeout=60000)
            if not response or response.status >= 400:
                raise Exception(f"Failed to load {base_url}: HTTP {response.status}")
            
            # Extract main content
            content = await self._extract_content(page)
            if not content:
                raise Exception("No content extracted")
            
            # Store results
            self.results[source] = {
                'url': base_url,
                'content': content,
                'topics': await self._extract_topics(page),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            raise Exception(f"Error crawling {source}: {str(e)}")

    async def _extract_content(self, page) -> str:
        """Extract content with simplified strategy"""
        selectors = [
            'article', 'main', '.content', '.documentation',
            'section', '.markdown-body', '.doc-content'
        ]
        
        for selector in selectors:
            try:
                elements = await page.query_selector_all(selector)
                if elements:
                    content = []
                    for element in elements:
                        text = await element.text_content()
                        if text:
                            content.append(text.strip())
                    return '\n\n'.join(content)
            except Exception:
                continue
        
        return ""

    async def _extract_topics(self, page) -> List[str]:
        """Extract topics with simplified strategy"""
        topics = set()
        selectors = ['.tags a', '.topics a', 'meta[name="keywords"]']
        
        for selector in selectors:
            try:
                elements = await page.query_selector_all(selector)
                for element in elements:
                    text = await element.text_content()
                    if text:
                        topics.add(text.strip().lower())
            except:
                continue
        
        return list(topics)

    async def _setup_page_handlers(self, page):
        """Set up basic page handlers"""
        await page.route('**/*', self._handle_request)

    async def _handle_request(self, route):
        """Handle requests with simplified strategy"""
        if route.request.resource_type in ['image', 'media', 'font', 'stylesheet']:
            await route.abort()
        else:
            await route.continue_()

    def _default_config(self) -> Dict:
        """Return default configuration"""
        return {
            'log_level': 'INFO',
            'timeout': 60000,
            'max_retries': 3
        }

    def setup_logging(self):
        """Setup basic logging"""
        logging.basicConfig(
            level=self.config.get('log_level', 'INFO'),
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

def main():
    crawler = DocumentationCrawler()
    asyncio.run(crawler.crawl_all())

if __name__ == "__main__":
    main() 
