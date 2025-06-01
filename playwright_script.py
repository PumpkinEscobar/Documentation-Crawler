# Generate a Playwright-based script for crawling OpenAI's docs (SPA-aware version)
# Note: This script is to be run locally in a Python environment with Playwright installed

playwright_script = """
# playwright_crawler.py

from playwright.sync_api import sync_playwright
from urllib.parse import urljoin, urlparse
import json

BASE_URL = "https://platform.openai.com/docs"

def is_valid_link(href):
    if not href:
        return False
    parsed = urlparse(href)
    return parsed.path.startswith("/docs") and not parsed.fragment

def crawl_openai_docs():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(BASE_URL, wait_until="networkidle")

        links = set()
        anchors = page.query_selector_all("a")

        for a in anchors:
            href = a.get_attribute("href")
            if href:
                full_url = urljoin(BASE_URL, href.split("#")[0])
                if BASE_URL in full_url and is_valid_link(href):
                    links.add(full_url)

        browser.close()
        return sorted(links)

if __name__ == "__main__":
    urls = crawl_openai_docs()
    with open("openai_docs_index.json", "w") as f:
        json.dump(urls, f, indent=2)
    print(f"Found {len(urls)} documentation URLs. Saved to 'openai_docs_index.json'")
"""

with open("/mnt/data/playwright_crawler.py", "w") as f:
    f.write(playwright_script)

"/mnt/data/playwright_crawler.py is ready to download and run in your local Python environment."
