
# Add these sections to your doc_crawler.py

async def crawl_implementation_guides(self, provider_config):
    """Specifically target implementation guides and tutorials"""
    
    priority_urls = provider_config.get('priority_sections', [])
    valuable_content = {}
    
    for url in priority_urls:
        try:
            print(f"🎯 Targeting: {url}")
            page = await self.context.new_page()
            
            await page.goto(url, wait_until='networkidle', timeout=60000)
            
            # Extract rich content
            content = await self._extract_rich_content(page)
            
            if self._is_valuable_content(content):
                content_type = self._classify_content_type(content, url)
                
                valuable_content[url] = {
                    'type': content_type,
                    'content': content,
                    'code_examples': await self._extract_code_examples(page),
                    'headings': await self._extract_headings(page),
                    'links': await self._extract_internal_links(page),
                    'url': url,
                    'crawled_at': datetime.now().isoformat()
                }
                
                print(f"✅ Valuable content found: {content_type}")
            else:
                print(f"⚠️  Low-value content, skipping")
                
            await page.close()
            
        except Exception as e:
            print(f"❌ Error crawling {url}: {e}")
            continue
    
    return valuable_content

async def _extract_rich_content(self, page):
    """Extract rich, formatted content preserving structure"""
    
    # Try multiple content extraction strategies
    content_selectors = [
        'article',
        'main', 
        '.markdown-body',
        '.content',
        '.documentation',
        '.docs-content'
    ]
    
    for selector in content_selectors:
        try:
            element = await page.query_selector(selector)
            if element:
                # Get both text and HTML structure
                text_content = await element.text_content()
                html_content = await element.inner_html()
                
                if len(text_content) > 500:  # Substantial content
                    return {
                        'text': text_content,
                        'html': html_content,
                        'length': len(text_content)
                    }
        except:
            continue
    
    return None

async def _extract_code_examples(self, page):
    """Extract all code examples from the page"""
    
    code_examples = []
    
    # Look for code blocks
    code_selectors = [
        'pre code',
        '.highlight',
        '.code-block',
        '.example-code'
    ]
    
    for selector in code_selectors:
        try:
            elements = await page.query_selector_all(selector)
            for element in elements:
                code = await element.text_content()
                if code and len(code.strip()) > 10:
                    
                    # Try to determine language
                    language = 'text'
                    class_attr = await element.get_attribute('class')
                    if class_attr and 'language-' in class_attr:
                        language = class_attr.split('language-')[1].split()[0]
                    
                    code_examples.append({
                        'code': code.strip(),
                        'language': language
                    })
        except:
            continue
    
    return code_examples

def _classify_content_type(self, content, url):
    """Classify the type of content based on content and URL"""
    
    text = content.get('text', '').lower()
    url_lower = url.lower()
    
    if any(keyword in text for keyword in ['tutorial', 'step by step', 'getting started']):
        return 'tutorial'
    elif any(keyword in text for keyword in ['guide', 'how to', 'implementation']):
        return 'implementation_guide'  
    elif any(keyword in text for keyword in ['example', 'sample', 'demo']):
        return 'code_example'
    elif any(keyword in url_lower for keyword in ['best-practices', 'guidelines']):
        return 'best_practices'
    elif 'prompt' in text and 'engineering' in text:
        return 'prompt_engineering'
    else:
        return 'documentation'

def _is_valuable_content(self, content):
    """Determine if content is valuable for AI wizard training"""
    
    if not content:
        return False
    
    text = content.get('text', '')
    
    # Must have substantial content
    if len(text) < 500:
        return False
    
    # Should contain AI/LLM related keywords
    valuable_keywords = [
        'api', 'model', 'prompt', 'completion', 'embedding',
        'tutorial', 'example', 'implementation', 'guide',
        'python', 'javascript', 'code', 'function'
    ]
    
    keyword_count = sum(1 for keyword in valuable_keywords if keyword in text.lower())
    
    return keyword_count >= 3
