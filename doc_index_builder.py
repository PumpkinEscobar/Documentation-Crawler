import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

# Base config
base_url = "https://platform.openai.com/docs"
visited = set()
discovered_links = set()

# Helper to fetch and parse a page
def fetch_links(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except Exception as e:
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    anchors = soup.find_all("a", href=True)

    links = set()
    for a in anchors:
        href = a["href"]
        if href.startswith("http"):
            full_url = href
        else:
            full_url = urljoin(base_url, href)

        if base_url in full_url and urlparse(full_url).path.startswith("/docs"):
            links.add(full_url.split("#")[0])  # remove anchor fragments

    return links

# Simple BFS to avoid overcrawling
to_visit = {base_url}
max_depth = 2
depth = 0

while to_visit and depth < max_depth:
    next_level = set()
    for url in to_visit:
        if url in visited:
            continue
        visited.add(url)
        new_links = fetch_links(url)
        discovered_links.update(new_links)
        next_level.update(new_links - visited)
    to_visit = next_level
    depth += 1

# Convert to sorted list
discovered_links = sorted(discovered_links)
import pandas as pd
import ace_tools as tools

df = pd.DataFrame(discovered_links, columns=["OpenAI Documentation URLs"])
tools.display_dataframe_to_user(name="OpenAI Docs Index (Basic Crawl)", dataframe=df)
