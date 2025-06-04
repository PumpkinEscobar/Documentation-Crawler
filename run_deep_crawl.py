#!/usr/bin/env python3
# run_deep_crawl.py

import asyncio
import json
from doc_crawler import DocumentationCrawler

async def main():
    print("🏔️  Starting Deep Knowledge Mining...")
    print("Targeting implementation guides, tutorials, and examples")
    
    # Load deep crawl configuration
    with open('deep_crawler_config.json', 'r') as f:
        config = json.load(f)
    
    crawler = DocumentationCrawler(config)
    
    # Run enhanced crawl
    success = await crawler.crawl_all()
    
    if success:
        print("✅ Deep crawl completed!")
        print("Check knowledge_base.json for the treasure trove!")
    else:
        print("❌ Deep crawl had issues")

if __name__ == "__main__":
    asyncio.run(main())
