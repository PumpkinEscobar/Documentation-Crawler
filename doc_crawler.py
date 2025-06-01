from playwright.sync_api import sync_playwright
import requests
import json
import yaml
import sys
import time
import os
from typing import Optional, Dict, List
from urllib.parse import urljoin, urlparse
import logging

class DocumentationCrawler:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.results = {}
        self.knowledge_map = {}
        self.setup_logging()
        
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
                            '/models'
                        ]
                    },
                    'docs': {
                        'base_url': 'https://platform.openai.com/docs',
                        'sections': [
                            'introduction',
                            'models',
                            'guides',
                            'tutorials',
                            'api-reference'
                        ]
                    },
                    'cookbook': {
                        'base_url': 'https://github.com/openai/openai-cookbook',
                        'sections': ['examples', 'techniques']
                    }
                },
                'anthropic': {
                    'docs': {
                        'base_url': 'https://docs.anthropic.com',
                        'sections': ['claude', 'api']
                    }
                },
                'google_ai': {
                    'docs': {
                        'base_url': 'https://cloud.google.com/ai',
                        'sections': ['docs', 'guides', 'tutorials']
                    }
                },
                'meta_ai': {
                    'docs': {
                        'base_url': 'https://ai.meta.com/tools',
                        'sections': ['documentation', 'research']
                    }
                }
            },
            'output': {
                'doc_index': 'doc_index.yaml',
                'knowledge_base': 'knowledge_base.json'
            }
        }

    async def crawl_all(self):
        """Crawl all configured documentation sources"""
        self.logger.info("Starting comprehensive documentation crawl...")
        
        try:
            # OpenAI API Documentation
            if api_key := os.getenv('OPENAI_API_KEY'):
                await self.crawl_openai_api(api_key)
            else:
                self.logger.warning("OpenAI API key not found, skipping API documentation")

            # Web Documentation
            async with sync_playwright() as p:
                browser = await p.chromium.launch(
                    headless=True,
                    args=['--disable-blink-features=AutomationControlled']
                )
                
                context = await browser.new_context(
                    viewport={'width': 1920, 'height': 1080},
                    user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                )
                
                for source, config in self.config['sources'].items():
                    self.logger.info(f"Crawling {source} documentation...")
                    await self.crawl_documentation(context, source, config)
                
                await browser.close()

            # Generate Knowledge Map
            self.generate_knowledge_map()
            
            # Save Results
            self.save_results()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error during crawl: {str(e)}")
            return False

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
                models = response.json()['data']
                self.results['openai_models'] = {
                    'type': 'api_documentation',
                    'source': 'openai',
                    'content': models
                }
                
            # Get Endpoints Documentation
            for endpoint in self.config['sources']['openai']['api']['endpoints']:
                self.results[f'openai_endpoint_{endpoint}'] = {
                    'type': 'api_documentation',
                    'source': 'openai',
                    'endpoint': endpoint,
                    'description': f"Documentation for {endpoint} endpoint"
                }
                
        except Exception as e:
            self.logger.error(f"Error crawling OpenAI API: {str(e)}")

    async def crawl_documentation(self, context, source: str, config: Dict):
        """Crawl web-based documentation"""
        for doc_type, doc_config in config.items():
            if doc_type == 'api':
                continue
                
            base_url = doc_config['base_url']
            
            try:
                page = await context.new_page()
                
                for section in doc_config['sections']:
                    url = f"{base_url}/{section}"
                    
                    try:
                        await page.goto(url, wait_until='networkidle', timeout=60000)
                        content = await page.content()
                        
                        self.results[f"{source}_{doc_type}_{section}"] = {
                            'type': 'web_documentation',
                            'source': source,
                            'section': section,
                            'url': url,
                            'content': content
                        }
                        
                    except Exception as e:
                        self.logger.error(f"Error crawling {url}: {str(e)}")
                        continue
                        
                await page.close()
                
            except Exception as e:
                self.logger.error(f"Error processing {source} {doc_type}: {str(e)}")

    def generate_knowledge_map(self):
        """Generate structured knowledge map from crawled documentation"""
        self.knowledge_map = {
            'metadata': {
                'last_updated': time.strftime("%Y-%m-%d %H:%M:%S"),
                'sources': list(self.config['sources'].keys())
            },
            'documentation': {}
        }
        
        for key, content in self.results.items():
            source = content['source']
            if source not in self.knowledge_map['documentation']:
                self.knowledge_map['documentation'][source] = {
                    'api_documentation': {},
                    'web_documentation': {},
                    'examples': {}
                }
                
            doc_type = content['type']
            if doc_type == 'api_documentation':
                self.knowledge_map['documentation'][source]['api_documentation'][key] = {
                    'type': 'endpoint' if 'endpoint' in content else 'model',
                    'path': content.get('endpoint', ''),
                    'description': content.get('description', '')
                }
            elif doc_type == 'web_documentation':
                self.knowledge_map['documentation'][source]['web_documentation'][content['section']] = {
                    'url': content['url'],
                    'last_crawled': time.strftime("%Y-%m-%d %H:%M:%S")
                }

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

def main():
    crawler = DocumentationCrawler()
    if crawler.crawl_all():
        print("Documentation crawl successful!")
    else:
        print("Documentation crawl failed. Check crawler.log for details.")

if __name__ == "__main__":
    main() 