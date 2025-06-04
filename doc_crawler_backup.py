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

# Configure OpenAI API
openai.api_key = os.getenv('OPENAI_API_KEY')
if not openai.api_key:
    raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")

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

class DocumentationCrawler:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.browser_options = self.config.get('browser_options', {
            'args': ['--no-sandbox', '--disable-setuid-sandbox']
        })
        self.setup_logging()
        self.knowledge_map = {}
        self.vector_processor = VectorProcessor()
        self.model_processor = ModelDataProcessor()
        self.results = {}
        self.browser = None
        self.reputable_sources = {
            'openai': {
                'authority_level': 'official',
                'base_url': 'https://platform.openai.com/docs',
                'api_required': True
            },
            'anthropic': {
                'authority_level': 'official',
                'base_url': 'https://docs.anthropic.com',
                'api_required': True
            },
            'google-ai': {
                'authority_level': 'official',
                'base_url': 'https://ai.google.dev/docs',
                'api_required': True
            },
            'huggingface': {
                'authority_level': 'official',
                'base_url': 'https://huggingface.co/docs',
                'api_required': True,
                'key_env': 'HUGGINGFACE_ACCESS_TOKEN',
                'key_type': 'read'
            },
            'grok': {
                'authority_level': 'official',
                'base_url': 'https://grok.x.ai/docs',
                'api_required': True,
                'beta': True
            }
        }
        
        # Keywords to identify and skip military/aviation content
        self.exclusion_keywords = [
            'military-grade',
            'weapons-system',
            'warfare-specific',
            'combat-operations',
            'aircraft-control',
            'flight-systems',
            'aerospace-defense',
            'missile-guidance',
            'tactical-operations',
            'strategic-command'
        ]
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('crawler.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _default_config(self) -> Dict:
        return {
            'sources': {
                'openai': {
                    'api': {
                        'base_url': 'https://api.openai.com/v1',
                        'key_env': 'OPENAI_API_KEY',
                        'endpoints': [
                            '/chat/completions',
                            '/completions',
                            '/embeddings',
                            '/fine-tuning',
                            '/models',
                            '/assistants',
                            '/images'
                        ]
                    },
                    'docs': {
                        'base_url': 'https://platform.openai.com',
                        'sections': [
                            'docs/introduction',
                            'docs/models',
                            'docs/guides',
                            'docs/tutorials',
                            'docs/api-reference',
                            'docs/plugins',
                            'docs/safety-best-practices'
                        ]
                    }
                },
                'anthropic': {
                    'docs': {
                        'base_url': 'https://docs.anthropic.com',
                        'sections': [
                            'claude',
                            'api',
                            'prompt-engineering',
                            'enterprise',
                            'security',
                            'best-practices'
                        ]
                    }
                },
                'google-ai': {
                    'docs': {
                        'base_url': 'https://ai.google.dev',
                        'sections': [
                            'docs',
                            'docs/palm-api',
                            'docs/gemini-api',
                            'docs/multimodal',
                            'docs/embeddings',
                            'docs/safety'
                        ]
                    }
                },
                'grok': {
                    'docs': {
                        'base_url': 'https://grok.x.ai',
                        'sections': [
                            'docs/api',
                            'docs/models',
                            'docs/tutorials',
                            'docs/best-practices',
                            'docs/examples'
                        ]
                    }
                },
                'huggingface': {
                    'docs': {
                        'base_url': 'https://huggingface.co',
                        'sections': [
                            'docs/hub',
                            'docs/transformers',
                            'docs/tokenizers',
                            'docs/datasets',
                            'docs/accelerate',
                            'docs/peft',
                            'docs/optimum'
                        ]
                    }
                }
            },
            'output': {
                'doc_index': 'doc_index.yaml',
                'knowledge_base': 'knowledge_base.json',
                'embeddings': 'embeddings.json',
                'search_index': 'docs_search.index'
            }
        }

    async def crawl_all(self):
        """Crawl all configured documentation sources"""
        self.logger.info("Starting comprehensive documentation crawl...")
        
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(
                    headless=True,
                    args=self.browser_options['args']
                )
                self.browser = browser

            # Create a persistent context with proper viewport and user agent
            context = await self.browser.new_context(
                viewport={'width': 1920, 'height': 1080},
                user_agent='Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
                ignore_https_errors=True,
                extra_http_headers={
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                    'Accept-Encoding': 'gzip, deflate, br',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1',
                    'Sec-Fetch-Dest': 'document',
                    'Sec-Fetch-Mode': 'navigate',
                    'Sec-Fetch-Site': 'none',
                    'Sec-Fetch-User': '?1'
                }
            )

            # Set up authentication for each source
            await self._setup_authentication(context)

            # OpenAI API Documentation
            if api_key := os.getenv('OPENAI_API_KEY'):
                await self.crawl_openai_api(api_key)
            else:
                self.logger.warning("OpenAI API key not found, skipping API documentation")

            # Web Documentation
            for source, config in self.config['sources'].items():
                self.logger.info(f"Crawling {source} documentation...")
                
                # Create a new page for each source
                page = await context.new_page()
                await self._setup_page_handlers(page)
                
                # Wait for authentication to take effect
                await asyncio.sleep(2)
                
                # Crawl each section with proper error handling
                for section in config['docs']['sections']:
                    url = urljoin(config['docs']['base_url'], section)
                    try:
                        await self._crawl_section(page, url, source, 'docs', section)
                    except Exception as e:
                        self.logger.error(f"Error crawling {url}: {str(e)}")
                        continue
                
                await page.close()
            
            # Generate Knowledge Map
            self.generate_knowledge_map()
            
            # Process embeddings
            await self.process_embeddings()
            
            # Save Results
            self.save_results()
            
            # Cleanup
            await context.close()
            await self.browser.close()
            
            self.logger.info("Documentation crawl completed successfully!")
            return True
            
        except Exception as e:
            self.logger.error(f"Error during crawl: {str(e)}")
            return False

    async def _setup_authentication(self, context):
        """Set up authentication for all sources at once"""
        # OpenAI Authentication
        if api_key := os.getenv('OPENAI_API_KEY'):
            await context.add_cookies([{
                'name': 'openai-auth-token',
                'value': api_key,
                'domain': 'platform.openai.com',
                'path': '/',
                'httpOnly': True,
                'secure': True
            }])
            
        # Anthropic Authentication
        if api_key := os.getenv('ANTHROPIC_API_KEY'):
            await context.add_cookies([{
                'name': 'anthropic-auth',
                'value': api_key,
                'domain': '.anthropic.com',
                'path': '/',
                'httpOnly': True,
                'secure': True
            }])
            
        # Hugging Face Authentication
        if token := os.getenv('HUGGINGFACE_ACCESS_TOKEN'):
            await context.add_cookies([{
                'name': '_oauth_state',
                'value': token,
                'domain': '.huggingface.co',
                'path': '/',
                'httpOnly': True,
                'secure': True
            }])
            
        # Google AI Authentication
        if api_key := os.getenv('GOOGLE_AI_KEY'):
            await context.add_cookies([{
                'name': 'goog-api-key',
                'value': api_key,
                'domain': '.ai.google.dev',
                'path': '/',
                'httpOnly': True,
                'secure': True
            }])
            
        # Grok Authentication
        if api_key := os.getenv('GROK_API_KEY'):
            await context.add_cookies([{
                'name': 'grok-auth',
                'value': api_key,
                'domain': '.x.ai',
                'path': '/',
                'httpOnly': True,
                'secure': True
            }])

    async def _setup_page_handlers(self, page):
        """Set up page event handlers and interceptors"""
        # Increase timeout for all operations
        page.set_default_timeout(120000)  # 2 minutes
        
        # Handle JavaScript dialogs automatically
        page.on('dialog', lambda dialog: dialog.accept())
        
        # Filter out common React errors
        def handle_error(err):
            error_str = str(err)
            if any(skip in error_str for skip in [
                'React error #418',
                'React error #423',
                'hydration',
                'Hydration failed',
                'ResizeObserver loop',
                'Error loading chunk'
            ]):
                return
            self.logger.error(f'Page error: {error_str}')
        
        page.on('pageerror', handle_error)
        
        # Set up request interception
        await page.route('**/*', self._handle_request)
        
        # Handle React hydration issues and add retry logic
        await page.add_init_script("""
            // Handle React hydration issues
            window.addEventListener('error', function(event) {
                if (event.message.includes('React') || 
                    event.message.includes('hydration') ||
                    event.message.includes('Hydration')) {
                    event.stopPropagation();
                    event.preventDefault();
                }
            });
            
            // Add retry logic for failed resource loads
            window.addEventListener('error', function(event) {
                if (event.target && (event.target.tagName === 'SCRIPT' || event.target.tagName === 'LINK')) {
                    const src = event.target.src || event.target.href;
                    if (src && !window.__retryCount) {
                        window.__retryCount = {};
                    }
                    if (src && (!window.__retryCount[src] || window.__retryCount[src] < 3)) {
                        window.__retryCount[src] = (window.__retryCount[src] || 0) + 1;
                        setTimeout(() => {
                            const elem = document.createElement(event.target.tagName);
                            for (const attr of event.target.attributes) {
                                elem.setAttribute(attr.name, attr.value);
                            }
                            event.target.parentNode.replaceChild(elem, event.target);
                        }, 1000);
                    }
                }
            }, true);
            
            // Override fetch with timeout and retry
            const originalFetch = window.fetch;
            window.fetch = async function(url, options = {}) {
                const maxRetries = 3;
                let lastError;
                
                for (let i = 0; i < maxRetries; i++) {
                    try {
                        const controller = new AbortController();
                        const timeoutId = setTimeout(() => controller.abort(), 30000);
                        options.signal = controller.signal;
                        
                        const response = await Promise.race([
                            originalFetch(url, options),
                            new Promise((_, reject) => 
                                setTimeout(() => reject(new Error('Timeout')), 30000)
                            )
                        ]);
                        
                        clearTimeout(timeoutId);
                        return response;
                    } catch (error) {
                        lastError = error;
                        await new Promise(resolve => setTimeout(resolve, Math.pow(2, i) * 1000));
                    }
                }
                throw lastError;
            };
        """)

    async def _handle_request(self, route):
        """Handle intercepted requests"""
        if route.request.resource_type in ['image', 'media', 'font']:
            await route.abort()  # Skip non-essential resources
        else:
            await route.continue_()

    async def _crawl_section(self, page, url: str, source: str, doc_type: str, section: str):
        """Crawl a specific documentation section with enhanced content extraction"""
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                # Navigate to the page and wait for load
                response = await page.goto(url, wait_until='networkidle', timeout=60000)
                if not response:
                    raise Exception("No response from server")
                    
                # Handle different status codes
                if response.status == 401 or response.status == 403:
                    self.logger.error(f"Authentication failed for {url}. Checking authentication...")
                    await self._setup_authentication(await page.context())
                    # Retry with new authentication
                    response = await page.goto(url, wait_until='networkidle', timeout=60000)
                    if response.status >= 400:
                        raise Exception(f"Authentication retry failed with HTTP {response.status}")
                elif response.status >= 400:
                    raise Exception(f"HTTP {response.status} error")
                
                # Wait for dynamic content with increased timeout
                try:
                    await self._wait_for_content(page)
                except Exception as e:
                    self.logger.warning(f"Content wait warning on attempt {attempt + 1}: {str(e)}")
                    # If content wait fails, try to proceed anyway
                    pass
                
                # Scroll to load lazy content
                await self._scroll_for_content(page)
                
                # Extract content with retry
                content = None
                for content_attempt in range(3):
                    try:
                        content = await self._extract_content(page)
                        if content:
                            break
                    except Exception as e:
                        self.logger.warning(f"Content extraction attempt {content_attempt + 1} failed: {str(e)}")
                        await asyncio.sleep(1)
                
                if not content:
                    raise Exception("No content extracted")
                
                # Process and store the content
                doc_section = await self._process_section_content(page, url, content)
                
                # Store results with semantic organization
                section_key = f"{source}_{doc_type}_{section}"
                self.results[section_key] = {
                    'type': 'web_documentation',
                    'source': source,
                    'section': section,
                    'data': asdict(doc_section),
                    'semantic_context': {
                        'topics': await self._extract_topics(page),
                        'related_concepts': await self._extract_related_concepts(page),
                        'key_terms': await self._extract_key_terms(page)
                    }
                }
                
                # Log success
                self.logger.info(f"Successfully crawled {url}")
                break
                
            except Exception as e:
                if attempt == max_retries - 1:
                    self.logger.error(f"Failed to crawl {url} after {max_retries} attempts: {str(e)}")
                    return
                self.logger.warning(f"Attempt {attempt + 1} failed for {url}: {str(e)}")
                # Exponential backoff
                await asyncio.sleep(retry_delay * (2 ** attempt))

    async def _scroll_for_content(self, page):
        """Scroll the page to load lazy-loaded content"""
        try:
            # Get initial page height
            height = await page.evaluate('document.documentElement.scrollHeight')
            
            # Scroll in increments
            viewport_height = await page.evaluate('window.innerHeight')
            for i in range(0, height, viewport_height):
                await page.evaluate(f'window.scrollTo(0, {i})')
                await page.wait_for_timeout(100)  # Small delay for content to load
                
            # Scroll back to top
            await page.evaluate('window.scrollTo(0, 0)')
            
            # Wait a bit for any final dynamic content
            await page.wait_for_timeout(1000)
            
        except Exception as e:
            self.logger.debug(f"Error during scroll: {str(e)}")

    async def _wait_for_content(self, page):
        """Wait for content to load with multiple selector strategies"""
        selectors = [
            'article', 'main', '.content', '.documentation', '.doc-content',
            '#content', '.markdown-body', '.notion-page-content',
            'div[role="main"]', '.main-content', '.article-content',
            # Modern web app selectors
            '[data-testid="page-content"]',
            '[data-testid="documentation"]',
            '[data-testid="main-content"]',
            # Common documentation patterns
            '.docs-content',
            '.api-docs',
            '.reference-content',
            # Framework-specific selectors
            '.chakra-container',
            '.mui-container',
            '.ant-layout-content',
            # Documentation-specific elements
            'article h1',
            '.api-reference',
            '.method-list-group'
        ]
        
        try:
            # Wait for network to be idle first
            await page.wait_for_load_state('networkidle', timeout=10000)
            
            # Try primary content selectors
            for selector in selectors:
                try:
                    await page.wait_for_selector(selector, timeout=5000)
                    return
                except:
                    continue
                    
            # Fallback to any text content
            await page.wait_for_selector('h1, h2, h3, p', timeout=5000)
            
            # Additional wait for dynamic content
            await page.wait_for_timeout(2000)
            
        except Exception as e:
            self.logger.warning(f"Content wait warning: {str(e)}")

    async def _extract_content(self, page) -> str:
        """Extract content with improved reliability"""
        content = ""
        
        # Try multiple content extraction strategies
        strategies = [
            # Strategy 1: Main content containers
            lambda: page.query_selector_all('article, main, .content, .documentation'),
            # Strategy 2: Semantic HTML5 elements
            lambda: page.query_selector_all('section, article, aside, nav'),
            # Strategy 3: Common documentation classes
            lambda: page.query_selector_all('.markdown-body, .doc-content, .article-content'),
            # Strategy 4: Modern web app selectors
            lambda: page.query_selector_all('[data-testid="page-content"], [data-testid="documentation"]'),
            # Strategy 5: API documentation specific
            lambda: page.query_selector_all('.api-reference, .method-list-group, .endpoint-list'),
            # Strategy 6: Headers and paragraphs
            lambda: page.query_selector_all('h1, h2, h3, h4, h5, h6, p'),
            # Strategy 7: Code blocks and examples
            lambda: page.query_selector_all('pre, code, .highlight, .example'),
            # Strategy 8: Framework-specific content
            lambda: page.query_selector_all('.chakra-container, .mui-container, .ant-layout-content')
        ]
        
        for strategy in strategies:
            try:
                elements = await strategy()
                for element in elements:
                    try:
                        # Try to get text content
                        text = await element.text_content()
                        if text and text.strip():
                            content += text.strip() + "\n\n"
                            
                        # Try to get code content
                        if await element.get_attribute('class') and any(cls in await element.get_attribute('class') for cls in ['highlight', 'code', 'example']):
                            code = await element.inner_text()
                            if code and code.strip():
                                content += "```\n" + code.strip() + "\n```\n\n"
                                
                        # Try to get API endpoint information
                        if await element.get_attribute('class') and 'endpoint' in await element.get_attribute('class'):
                            endpoint_info = await self._extract_endpoint_info(element)
                            if endpoint_info:
                                content += endpoint_info + "\n\n"
                    except Exception as e:
                        self.logger.debug(f"Error extracting element content: {str(e)}")
                        continue
                        
                if content:
                    break
            except Exception as e:
                self.logger.debug(f"Content extraction strategy failed: {str(e)}")
                continue
        
        return content.strip()

    async def _extract_endpoint_info(self, element) -> Optional[str]:
        """Extract API endpoint information"""
        try:
            method = await element.query_selector('.http-method, .method')
            path = await element.query_selector('.endpoint-path, .path')
            description = await element.query_selector('.description, .endpoint-description')
            
            info = []
            if method:
                info.append(f"Method: {await method.text_content()}")
            if path:
                info.append(f"Path: {await path.text_content()}")
            if description:
                info.append(f"Description: {await description.text_content()}")
                
            return "\n".join(info) if info else None
            
        except Exception as e:
            self.logger.debug(f"Error extracting endpoint info: {str(e)}")
            return None

    def generate_knowledge_map(self):
        """Generate structured knowledge map from crawled documentation"""
        self.knowledge_map = {
            'metadata': {
                'last_updated': datetime.now().isoformat(),
                'sources': list(self.config['sources'].keys()),
                'total_sections': 0,
                'total_code_examples': 0
            },
            'core_concepts': {  # New section for fundamental AI/LLM concepts
                'model_architectures': {},
                'training_approaches': {},
                'deployment_patterns': {},
                'prompt_engineering': {},
                'safety_practices': {}
            },
            'api_reference': {  # Restructured API documentation
                'models': {},
                'endpoints': {},
                'parameters': {},
                'rate_limits': {},
                'authentication': {}
            },
            'implementation_guides': {  # Practical implementation details
                'code_examples': {},
                'best_practices': {},
                'integration_patterns': {},
                'error_handling': {}
            },
            'use_cases': {  # Real-world applications
                'enterprise': {},
                'research': {},
                'product_development': {}
            },
            'cross_references': {}  # Maintained for linking related concepts
        }
        
        # Define relevant content categories
        self.content_categories = {
            'model_architectures': [
                'transformer', 'attention', 'embedding', 'tokenization',
                'fine-tuning', 'architecture', 'neural network'
            ],
            'training_approaches': [
                'training', 'fine-tuning', 'transfer learning', 'few-shot',
                'zero-shot', 'prompt engineering', 'dataset'
            ],
            'deployment_patterns': [
                'deployment', 'scaling', 'optimization', 'inference',
                'latency', 'throughput', 'monitoring'
            ],
            'prompt_engineering': [
                'prompt', 'instruction', 'context', 'few-shot',
                'chain-of-thought', 'template', 'system message'
            ],
            'safety_practices': [
                'safety', 'ethics', 'bias', 'fairness', 'security',
                'privacy', 'responsible AI'
            ]
        }
        
        for key, content in self.results.items():
            source = content['source']
            doc_type = content['type']
            
            if doc_type == 'api_documentation':
                self._process_api_documentation(source, content)
            elif doc_type == 'web_documentation':
                self._process_web_documentation(source, content)
                
        # Post-process to ensure quality
        self._cleanup_knowledge_map()
        
    def _process_api_documentation(self, source: str, content: Dict):
        """Process API documentation with improved categorization"""
        if 'openai_models' in content:
            models_data = content['content']
            
            # Organize models by capability and use case
            for model_id, model_info in models_data.items():
                model_entry = {
                    'id': model_id,
                    'description': model_info['description'],
                    'capabilities': model_info['capabilities'],
                    'token_limit': model_info['token_limit'],
                    'recommended_use': model_info['recommended_use']
                }
                
                # Add to appropriate categories
                if 'text-generation' in model_info['category']:
                    self.knowledge_map['api_reference']['models']['text_generation'] = \
                        self.knowledge_map['api_reference']['models'].get('text_generation', [])
                    self.knowledge_map['api_reference']['models']['text_generation'].append(model_entry)
                elif 'embeddings' in model_info['category']:
                    self.knowledge_map['api_reference']['models']['embeddings'] = \
                        self.knowledge_map['api_reference']['models'].get('embeddings', [])
                    self.knowledge_map['api_reference']['models']['embeddings'].append(model_entry)
                
        elif 'endpoint' in content:
            endpoint_info = {
                'path': content['endpoint'],
                'description': content.get('description', ''),
                'authentication': content.get('metadata', {}).get('authentication_required', True),
                'rate_limits': content.get('metadata', {}).get('rate_limits', {})
            }
            self.knowledge_map['api_reference']['endpoints'][content['endpoint']] = endpoint_info
            
    def _process_web_documentation(self, source: str, content: Dict):
        """Process web documentation with improved categorization"""
        section_data = content['data']
        section_content = section_data['content'].lower()
        
        # Categorize content based on relevant terms
        for category, terms in self.content_categories.items():
            if any(term.lower() in section_content for term in terms):
                category_path = self._determine_category_path(category, section_content)
                
                if category_path:
                    entry = {
                        'title': section_data['title'],
                        'url': section_data['url'],
                        'source': source,
                        'last_updated': section_data['last_updated'],
                        'quality_score': section_data.get('source_quality', 0.5),
                        'code_examples': len(section_data['code_examples']),
                        'key_concepts': self._extract_key_concepts(section_content)
                    }
                    
                    # Add to appropriate category
                    current_dict = self.knowledge_map
                    for path_part in category_path.split('.'):
                        if path_part not in current_dict:
                            current_dict[path_part] = {}
                        current_dict = current_dict[path_part]
                    
                    if 'entries' not in current_dict:
                        current_dict['entries'] = []
                    current_dict['entries'].append(entry)
                    
    def _determine_category_path(self, base_category: str, content: str) -> Optional[str]:
        """Determine the specific category path based on content analysis"""
        if base_category == 'model_architectures':
            if 'transformer' in content:
                return 'core_concepts.model_architectures.transformer'
            elif 'embedding' in content:
                return 'core_concepts.model_architectures.embedding'
        elif base_category == 'training_approaches':
            if 'fine-tuning' in content:
                return 'core_concepts.training_approaches.fine_tuning'
            elif 'few-shot' in content:
                return 'core_concepts.training_approaches.few_shot'
        elif base_category == 'prompt_engineering':
            if 'chain-of-thought' in content:
                return 'core_concepts.prompt_engineering.chain_of_thought'
            elif 'system message' in content:
                return 'core_concepts.prompt_engineering.system_messages'
        
        return f'core_concepts.{base_category}.general'
        
    def _extract_key_concepts(self, content: str) -> List[str]:
        """Extract key concepts from content using improved pattern matching"""
        key_concepts = []
        
        # Common patterns in AI/LLM documentation
        patterns = [
            r'(?:called|known as|termed)\s+([A-Z][a-zA-Z\s-]+)',  # Term definitions
            r'(?:concept of|technique called)\s+([A-Z][a-zA-Z\s-]+)',  # Techniques
            r'"([A-Z][a-zA-Z\s-]+)"(?=\s+is|refers to|means)',  # Quoted terms
            r'`([A-Z][a-zA-Z\s-]+)`',  # Code-styled terms
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, content)
            key_concepts.extend(match.group(1) for match in matches)
            
        return list(set(key_concepts))
        
    def _cleanup_knowledge_map(self):
        """Clean up the knowledge map to ensure quality and relevance"""
        def cleanup_section(section: Dict) -> Dict:
            if isinstance(section, dict):
                # Remove empty categories
                return {k: cleanup_section(v) for k, v in section.items() 
                       if v and (isinstance(v, dict) or isinstance(v, list))}
            return section
            
        # Clean up each main section
        for section in ['core_concepts', 'api_reference', 'implementation_guides', 'use_cases']:
            if section in self.knowledge_map:
                self.knowledge_map[section] = cleanup_section(self.knowledge_map[section])

    async def process_embeddings(self):
        """Process all crawled content into embeddings"""
        all_chunks = []
        
        # Process each documentation section
        for key, content in self.results.items():
            if content['type'] == 'web_documentation':
                section_data = content['data']
                
                # Process main content
                chunks = self.vector_processor.create_chunks(
                    section_data['content'],
                    {
                        'source': content['source'],
                        'section': content['section'],
                        'url': section_data['url'],
                        'title': section_data['title'],
                        'type': 'main_content'
                    }
                )
                all_chunks.extend(chunks)
                
                # Process subsections
                for subsection in section_data['subsections']:
                    chunks = self.vector_processor.create_chunks(
                        subsection['content'],
                        {
                            'source': content['source'],
                            'section': content['section'],
                            'url': section_data['url'],
                            'title': subsection['title'],
                            'type': 'subsection'
                        }
                    )
                    all_chunks.extend(chunks)
                
                # Process code examples
                for i, example in enumerate(section_data['code_examples']):
                    chunks = self.vector_processor.create_chunks(
                        example['code'],
                        {
                            'source': content['source'],
                            'section': content['section'],
                            'url': section_data['url'],
                            'language': example['language'],
                            'type': 'code_example',
                            'example_index': i
                        }
                    )
                    all_chunks.extend(chunks)
        
        # Generate embeddings
        embeddings_data = await self.vector_processor.get_embeddings(all_chunks)
        
        if embeddings_data:
            # Save embeddings
            with open(self.config['output']['embeddings'], 'w') as f:
                json.dump(embeddings_data, f)
            self.logger.info(f"Embeddings saved to {self.config['output']['embeddings']}")
            
            # Create FAISS index for fast similarity search
            self._create_search_index(embeddings_data)

    def _create_search_index(self, embeddings_data: Dict[str, Any]):
        """Create FAISS index for fast similarity search"""
        try:
            import faiss
            import numpy as np
            
            # Check if we have any embeddings
            if not embeddings_data['embeddings']:
                self.logger.warning("No embeddings to index, skipping search index creation")
                return
            
            # Convert embeddings to numpy array
            embeddings = np.array(embeddings_data['embeddings']).astype('float32')
            
            # Create and train index
            dimension = len(embeddings[0])
            index = faiss.IndexFlatL2(dimension)
            
            # Add vectors to index
            if len(embeddings) > 0:
                index.add(embeddings)
                
                # Save index
                faiss.write_index(index, self.config['output']['search_index'])
                self.logger.info(f"Search index saved to {self.config['output']['search_index']}")
            else:
                self.logger.warning("No vectors to add to search index")
            
        except ImportError:
            self.logger.warning("FAISS not installed, skipping search index creation")
        except Exception as e:
            self.logger.error(f"Error creating search index: {str(e)}")

    def save_results(self):
        """Save results and knowledge map"""
        # Save detailed documentation
        with open(self.config['output']['knowledge_base'], 'w') as f:
            json.dump(self.results, f, indent=2)
            
        # Save structured index
        with open(self.config['output']['doc_index'], 'w') as f:
            yaml.dump(self.knowledge_map, f, default_flow_style=False)
            
        self.logger.info(f"Results saved to {self.config['output']['knowledge_base']}")
        self.logger.info(f"Documentation index saved to {self.config['output']['doc_index']}")

    def _should_skip_content(self, content: str, url: str) -> bool:
        """Determine if content should be skipped based on exclusion criteria"""
        # Check URL first
        url_lower = url.lower()
        if any(kw in url_lower for kw in self.exclusion_keywords):
            logging.info(f"Skipping URL due to exclusion keywords: {url}")
            return True
            
        # Check content
        content_lower = content.lower()
        excluded_terms = [kw for kw in self.exclusion_keywords if kw in content_lower]
        if excluded_terms:
            logging.info(f"Skipping content with excluded terms: {', '.join(excluded_terms)}")
            return True
            
        return False
        
    def _is_reputable_source(self, source: str) -> Tuple[bool, str]:
        """Check if source is in our list of reputable sources"""
        source_normalized = source.replace('_', '-').lower()
        for source_name, info in self.reputable_sources.items():
            source_name_normalized = source_name.replace('_', '-').lower()
            if source_normalized == source_name_normalized:
                return True, info['authority_level']
        return False, 'community'

    async def _extract_topics(self, page) -> List[str]:
        """Extract topic information from the page"""
        topics = set()
        
        # Look for topic indicators
        selectors = [
            '.tags a', '.topics a', '.categories a',
            'meta[name="keywords"]',
            '.breadcrumb a',
            'nav a',
            # Additional topic indicators
            '[data-testid="topic"]',
            '.topic-tag',
            '.category-label'
        ]
        
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

    async def _extract_related_concepts(self, page) -> List[str]:
        """Extract related concepts and terms"""
        concepts = set()
        
        # Look for related content sections
        selectors = [
            '.related a', '.see-also a', '.related-content a',
            'aside a', '.sidebar a',
            'div[class*="related"] a',
            # Additional related content indicators
            '[data-testid="related"]',
            '.related-topics',
            '.similar-content'
        ]
        
        for selector in selectors:
            try:
                elements = await page.query_selector_all(selector)
                for element in elements:
                    text = await element.text_content()
                    if text:
                        concepts.add(text.strip())
            except:
                continue
                
        return list(concepts)

    async def _extract_key_terms(self, page) -> List[str]:
        """Extract key terminology and concepts"""
        terms = set()
        
        # Look for term definitions and key concepts
        selectors = [
            'dt', '.term', '.definition', 
            'strong', '.key-term',
            'h3 code', 'h4 code',  # Often used for API terms
            # Additional term indicators
            '[data-testid="term"]',
            '.glossary-term',
            '.concept-definition'
        ]
        
        for selector in selectors:
            try:
                elements = await page.query_selector_all(selector)
                for element in elements:
                    text = await element.text_content()
                    if text:
                        terms.add(text.strip())
            except:
                continue
                
        return list(terms)

    async def extract_subsections(self, page) -> List[Dict[str, Any]]:
        """Extract subsections from the page"""
        subsections = []
        
        # Common subsection patterns
        patterns = [
            ('h2', 'h3'),  # Main sections
            ('h3', 'h4'),  # Sub-sections
            ('h4', 'h5'),  # Deep sections
            ('.section-title', '.section-content'),  # Custom sections
            ('[data-testid="section"]', '[data-testid="content"]')  # Modern web apps
        ]
        
        for heading_sel, content_sel in patterns:
            try:
                headings = await page.query_selector_all(heading_sel)
                for heading in headings:
                    title = await heading.text_content()
                    if not title:
                        continue
                        
                    # Get content until next heading
                    content = []
                    next_el = heading
                    while next_el:
                        next_el = await next_el.evaluate_handle('node => node.nextElementSibling')
                        if not next_el:
                            break
                            
                        # Stop if we hit another heading
                        next_tag = await next_el.evaluate('node => node.tagName.toLowerCase()')
                        if next_tag.startswith('h'):
                            break
                            
                        text = await next_el.text_content()
                        if text:
                            content.append(text.strip())
                            
                    if content:
                        subsections.append({
                            'title': title.strip(),
                            'content': '\n\n'.join(content),
                            'level': int(heading_sel[1]) if heading_sel.startswith('h') else 2
                        })
                        
            except Exception as e:
                self.logger.debug(f"Error extracting subsections: {str(e)}")
                continue
                
        return subsections

    async def extract_related_topics(self, page) -> List[str]:
        """Extract related topics from the page"""
        topics = set()
        
        # Look for related topics in various places
        selectors = [
            '.related-topics a',
            '.see-also a',
            'aside .topic-link',
            '.sidebar .topic',
            '[data-testid="related-topic"]',
            '.topic-tag'
        ]
        
        for selector in selectors:
            try:
                elements = await page.query_selector_all(selector)
                for element in elements:
                    text = await element.text_content()
                    if text:
                        topics.add(text.strip())
            except:
                continue
                
        return list(topics)

    async def extract_code_examples(self, page) -> List[Dict[str, str]]:
        """Extract code examples from the page"""
        examples = []
        
        # Look for code blocks
        selectors = [
            'pre code',
            '.highlight',
            '.example-code',
            '[data-testid="code-example"]',
            '.code-block'
        ]
        
        for selector in selectors:
            try:
                elements = await page.query_selector_all(selector)
                for element in elements:
                    # Get the code
                    code = await element.text_content()
                    if not code:
                        continue
                        
                    # Try to determine the language
                    lang = 'text'
                    class_attr = await element.get_attribute('class')
                    if class_attr:
                        # Common class patterns for language
                        lang_patterns = ['language-', 'lang-', 'highlight-']
                        for pattern in lang_patterns:
                            if pattern in class_attr:
                                lang = class_attr.split(pattern)[1].split()[0]
                                break
                                
                    examples.append({
                        'code': code.strip(),
                        'language': lang,
                        'source_line': await element.evaluate('node => node.closest("[data-line-number]")?.getAttribute("data-line-number")') or ''
                    })
                    
            except Exception as e:
                self.logger.debug(f"Error extracting code examples: {str(e)}")
                continue
                
        return examples

    async def _process_section_content(self, page, url: str, content: str) -> DocumentSection:
        """Process and structure section content"""
        try:
            title = await page.title()
            subsections = await self.extract_subsections(page)
            related_topics = await self.extract_related_topics(page)
            code_examples = await self.extract_code_examples(page)
            
            # Calculate source quality
            quality_score = self._calculate_quality_score(
                has_code=bool(code_examples),
                has_structure=bool(subsections),
                content_length=len(content)
            )
            
            return DocumentSection(
                title=title,
                content=content,
                url=url,
                last_updated=datetime.now().isoformat(),
                subsections=subsections,
                related_topics=related_topics,
                code_examples=code_examples,
                authority_level=self.reputable_sources.get(url.split('/')[2], {}).get('authority_level', 'community'),
                last_verified=datetime.now().isoformat(),
                api_version=self._extract_api_version(url, title),
                example_count=len(code_examples),
                source_quality=quality_score,
                prerequisites=self._extract_prerequisites(content),
                platform_compatibility=self._extract_platform_info(content)
            )
            
        except Exception as e:
            self.logger.error(f"Error processing section content: {str(e)}")
            return DocumentSection(
                title="Error processing content",
                content=content,
                url=url,
                last_updated=datetime.now().isoformat(),
                subsections=[],
                related_topics=[],
                code_examples=[],
                authority_level='community',
                last_verified=datetime.now().isoformat(),
                api_version=None,
                example_count=0,
                source_quality=0.1,
                prerequisites=[],
                platform_compatibility=[]
            )

    def _calculate_quality_score(self, has_code: bool, has_structure: bool, content_length: int) -> float:
        """Calculate a quality score for the content"""
        score = 0.5  # Base score
        
        # Add points for having code examples
        if has_code:
            score += 0.2
            
        # Add points for having proper structure
        if has_structure:
            score += 0.1
            
        # Add points based on content length
        if content_length > 5000:
            score += 0.2
        elif content_length > 2000:
            score += 0.1
            
        return min(1.0, score)  # Cap at 1.0

    def _extract_api_version(self, url: str, title: str) -> Optional[str]:
        """Extract API version information"""
        # Common version patterns
        patterns = [
            r'v\d+(\.\d+)*',
            r'api[- ]v\d+',
            r'version \d+\.\d+',
            r'\d{4}-\d{2}-\d{2}'
        ]
        
        for pattern in patterns:
            # Check URL
            url_match = re.search(pattern, url, re.I)
            if url_match:
                return url_match.group(0)
                
            # Check title
            title_match = re.search(pattern, title, re.I)
            if title_match:
                return title_match.group(0)
                
        return None

    def _extract_prerequisites(self, content: str) -> List[str]:
        """Extract prerequisites from content"""
        prereqs = []
        
        # Common prerequisite indicators
        patterns = [
            r'requires? (.*?)(?:\.|$)',
            r'prerequisites?:?\s*(.*?)(?:\.|$)',
            r'before you begin:?\s*(.*?)(?:\.|$)',
            r'you\'ll need:?\s*(.*?)(?:\.|$)'
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, content, re.I)
            for match in matches:
                req = match.group(1).strip()
                if req and len(req) < 100:  # Avoid overly long matches
                    prereqs.append(req)
                    
        return list(set(prereqs))

    def _extract_platform_info(self, content: str) -> List[str]:
        """Extract platform compatibility information"""
        platforms = set()
        
        # Common platform indicators
        indicators = [
            'windows', 'linux', 'mac', 'ios', 'android',
            'docker', 'kubernetes', 'aws', 'azure', 'gcp',
            'chrome', 'firefox', 'safari', 'edge'
        ]
        
        # Look for platform mentions
        content_lower = content.lower()
        for platform in indicators:
            if platform in content_lower:
                platforms.add(platform)
                
        return list(platforms)

    async def crawl_openai_api(self, api_key: str):
        """Crawl OpenAI API documentation"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            # Get Models
            response = requests.get(
                f"{self.config['sources']['openai']['api']['base_url']}/models",
                headers=headers
            )
            
            if response.status_code == 200:
                raw_models = response.json()['data']
                
                # Process and clean model data
                model_processor = ModelDataProcessor()
                processed_models = model_processor.process_model_data(raw_models)
                
                # Store processed models with metadata
                self.results['openai_models'] = {
                    'type': 'api_documentation',
                    'source': 'openai',
                    'content': processed_models,
                    'metadata': {
                        'last_updated': datetime.now().isoformat(),
                        'category': 'models',
                        'total_models': len(processed_models),
                        'categories': list(set(model['category'] for model in processed_models.values())),
                        'capabilities': list(set(cap for model in processed_models.values() for cap in model['capabilities']))
                    }
                }
                
            # Get Endpoints Documentation
            for endpoint in self.config['sources']['openai']['api']['endpoints']:
                self.results[f'openai_endpoint_{endpoint}'] = {
                    'type': 'api_documentation',
                    'source': 'openai',
                    'endpoint': endpoint,
                    'description': f"Documentation for {endpoint} endpoint",
                    'metadata': {
                        'last_updated': datetime.now().isoformat(),
                        'category': 'api_endpoint',
                        'authentication_required': True,
                        'rate_limits': True
                    }
                }
                
        except Exception as e:
            self.logger.error(f"Error crawling OpenAI API: {str(e)}")

def main():
    crawler = DocumentationCrawler()
    asyncio.run(crawler.crawl_all())

if __name__ == "__main__":
    main() 