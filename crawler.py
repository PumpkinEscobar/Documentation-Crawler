from playwright.sync_api import sync_playwright
from urllib.parse import urljoin, urlparse
import json
import sys
import time
import os
import requests
from typing import Optional

class DocsCrawler:
    def __init__(self, base_url, max_depth=1):
        self.base_url = base_url
        self.max_depth = max_depth
        self.visited = set()
        self.results = {}

    def crawl(self):
        print(f"Starting crawl of {self.base_url}")
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            
            try:
                # Start with the main page
                self._crawl_page(page, self.base_url, 0)
                return True
            except Exception as e:
                print(f"Error during crawl: {e}", file=sys.stderr)
                return False
            finally:
                browser.close()

    def _crawl_page(self, page, url, depth):
        if depth > self.max_depth or url in self.visited:
            return
        
        self.visited.add(url)
        print(f"Crawling {url}")

        try:
            page.goto(url, wait_until="networkidle")
            title = page.title()
            
            # Get all links
            links = []
            for a in page.query_selector_all("a"):
                href = a.get_attribute("href")
                if href:
                    full_url = urljoin(url, href.split("#")[0])
                    if full_url.startswith(self.base_url):
                        links.append(full_url)

            # Store the results
            self.results[url] = {
                "title": title,
                "links": list(set(links)),
                "crawled_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }

            # Crawl linked pages
            for link in links:
                if link not in self.visited:
                    self._crawl_page(page, link, depth + 1)

        except Exception as e:
            print(f"Error crawling {url}: {e}", file=sys.stderr)

    def save_results(self, filename):
        with open(filename, 'w') as f:
            json.dump({
                "metadata": {
                    "base_url": self.base_url,
                    "pages_crawled": len(self.results),
                    "max_depth": self.max_depth,
                    "crawl_date": time.strftime("%Y-%m-%d %H:%M:%S")
                },
                "pages": self.results
            }, f, indent=2)
        print(f"Results saved to {filename}")

class OpenAIDocsCrawler:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set it as OPENAI_API_KEY environment variable or pass it to the constructor.")
            
        self.base_url = "https://api.openai.com/v1"
        self.results = {}
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def crawl(self):
        """Crawl OpenAI's API endpoints and documentation"""
        print("Starting OpenAI API documentation crawler...")
        
        try:
            # Get list of available models
            print("Fetching available models...")
            models = self._get_models()
            
            # Get API endpoints documentation
            print("Fetching API endpoints documentation...")
            self._get_endpoints_docs()
            
            # Get model-specific information
            print("Fetching model-specific information...")
            for model in models:
                self._get_model_info(model['id'])
                
            return True
            
        except Exception as e:
            print(f"Error during crawl: {e}", file=sys.stderr)
            return False

    def _get_models(self):
        """Get list of available models"""
        response = requests.get(
            f"{self.base_url}/models",
            headers=self.headers
        )
        
        if response.status_code == 200:
            models = response.json()['data']
            self.results['models'] = {
                'title': 'Available Models',
                'content': models,
                'crawled_at': time.strftime("%Y-%m-%d %H:%M:%S")
            }
            return models
        else:
            print(f"Error fetching models: {response.text}", file=sys.stderr)
            return []

    def _get_model_info(self, model_id: str):
        """Get detailed information about a specific model"""
        try:
            response = requests.get(
                f"{self.base_url}/models/{model_id}",
                headers=self.headers
            )
            
            if response.status_code == 200:
                model_info = response.json()
                self.results[f'model_{model_id}'] = {
                    'title': f'Model: {model_id}',
                    'content': model_info,
                    'crawled_at': time.strftime("%Y-%m-%d %H:%M:%S")
                }
            else:
                print(f"Error fetching model {model_id}: {response.text}", file=sys.stderr)
                
        except Exception as e:
            print(f"Error processing model {model_id}: {e}", file=sys.stderr)

    def _get_endpoints_docs(self):
        """Get documentation for various API endpoints"""
        endpoints = [
            {'path': '/chat/completions', 'title': 'Chat Completions'},
            {'path': '/completions', 'title': 'Completions'},
            {'path': '/embeddings', 'title': 'Embeddings'},
            {'path': '/moderations', 'title': 'Moderations'},
            {'path': '/fine-tuning', 'title': 'Fine-tuning'},
            {'path': '/files', 'title': 'Files'}
        ]
        
        for endpoint in endpoints:
            try:
                # Make an OPTIONS request to get endpoint documentation
                # Note: This is a placeholder as the actual documentation might need
                # to be fetched differently based on OpenAI's API structure
                self.results[f"endpoint_{endpoint['path']}"] = {
                    'title': endpoint['title'],
                    'path': endpoint['path'],
                    'description': f"Documentation for {endpoint['title']} API endpoint",
                    'crawled_at': time.strftime("%Y-%m-%d %H:%M:%S")
                }
            except Exception as e:
                print(f"Error fetching docs for {endpoint['path']}: {e}", file=sys.stderr)

    def save_results(self, filename="openai_api_docs.json"):
        """Save the crawled documentation to a JSON file"""
        with open(filename, 'w') as f:
            json.dump({
                "metadata": {
                    "base_url": self.base_url,
                    "sections_crawled": len(self.results),
                    "crawl_date": time.strftime("%Y-%m-%d %H:%M:%S")
                },
                "documentation": self.results
            }, f, indent=2)
        print(f"Results saved to {filename}")

def main():
    try:
        crawler = OpenAIDocsCrawler()
        if crawler.crawl():
            crawler.save_results()
            print("OpenAI API documentation crawl successful!")
        else:
            print("Failed to crawl OpenAI API documentation.")
    except ValueError as e:
        print(f"Error: {e}")
        print("Please set your OpenAI API key using:")
        print("export OPENAI_API_KEY='your-api-key-here'")

if __name__ == "__main__":
    main() 