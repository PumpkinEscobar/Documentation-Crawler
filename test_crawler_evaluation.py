from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class DocumentSection:
    title: str
    content: str
    url: str
    last_updated: str
    subsections: List[Dict[str, str]]
    related_topics: List[str]
    code_examples: List[Dict[str, str]]
    authority_level: str
    last_verified: str
    api_version: Optional[str]
    example_count: int
    source_quality: float
    prerequisites: List[str]
    platform_compatibility: List[str]

class SimpleDocumentationEvaluator:
    def __init__(self):
        self.reputable_sources = {
            'openai': {'authority_level': 'official'},
            'anthropic': {'authority_level': 'official'},
            'google-ai': {'authority_level': 'official'},
            'huggingface': {'authority_level': 'official'},
            'grok': {'authority_level': 'official'}
        }
        
        self.exclusion_keywords = [
            'military-grade',
            'weapons-system',
            'tactical-operations'
        ]

    def is_reputable_source(self, source: str) -> Tuple[bool, str]:
        """Check if source is reputable"""
        source_normalized = source.replace('_', '-').lower()
        for source_name, info in self.reputable_sources.items():
            if source_normalized == source_name.lower():
                return True, info['authority_level']
        return False, 'community'

    def calculate_quality_score(self, has_code: bool, has_structure: bool, content_length: int) -> float:
        """Calculate content quality score"""
        score = 0.5  # Base score
        if has_code:
            score += 0.2
        if has_structure:
            score += 0.1
        if content_length > 5000:
            score += 0.2
        elif content_length > 2000:
            score += 0.1
        return min(1.0, score)

    def should_skip_content(self, content: str, url: str) -> bool:
        """Check if content should be excluded"""
        url_lower = url.lower()
        if any(kw in url_lower for kw in self.exclusion_keywords):
            return True
        content_lower = content.lower()
        excluded_terms = [kw for kw in self.exclusion_keywords if kw in content_lower]
        return bool(excluded_terms)

def test_evaluation_system():
    evaluator = SimpleDocumentationEvaluator()
    
    # Test Case 1: High-Quality Content
    high_quality_content = """
    # Introduction to AI Models
    
    This comprehensive guide covers the fundamentals of AI models.
    
    ## Prerequisites
    - Python 3.8+
    - Basic understanding of machine learning
    
    ## Code Example
    ```python
    import openai
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello!"}]
    )
    ```
    
    ## Platform Compatibility
    - Works on Linux, Windows, and MacOS
    - Docker support available
    """
    
    # Test Case 2: Low-Quality Content
    low_quality_content = """
    AI stuff
    some random text
    no structure
    """
    
    # Test Case 3: Potentially Problematic Content
    problematic_content = """
    AI models for military-grade applications
    tactical-operations implementation
    """
    
    print("Testing Quality Assessment System...")
    
    # Test source reputation
    print("\n1. Testing Source Reputation:")
    reputable = evaluator.is_reputable_source("openai")
    non_reputable = evaluator.is_reputable_source("unknown-source")
    print(f"OpenAI reputation: {reputable}")
    print(f"Unknown source reputation: {non_reputable}")
    
    # Test content quality scoring
    print("\n2. Testing Content Quality Scoring:")
    high_quality_score = evaluator.calculate_quality_score(
        has_code=True,
        has_structure=True,
        content_length=len(high_quality_content)
    )
    low_quality_score = evaluator.calculate_quality_score(
        has_code=False,
        has_structure=False,
        content_length=len(low_quality_content)
    )
    print(f"High quality content score: {high_quality_score}")
    print(f"Low quality content score: {low_quality_score}")
    
    # Test content exclusion
    print("\n3. Testing Content Exclusion:")
    should_exclude = evaluator.should_skip_content(problematic_content, "example.com/ai")
    print(f"Should exclude problematic content: {should_exclude}")
    
    # Test full document processing
    print("\n4. Testing Full Document Processing:")
    doc_section = DocumentSection(
        title="AI Models Guide",
        content=high_quality_content,
        url="https://example.com/guide",
        last_updated=datetime.now().isoformat(),
        subsections=[{"title": "Introduction", "content": "AI basics"}],
        related_topics=["Machine Learning", "Neural Networks"],
        code_examples=[{"code": "import openai", "language": "python"}],
        authority_level="official",
        last_verified=datetime.now().isoformat(),
        api_version="v1",
        example_count=1,
        source_quality=0.9,
        prerequisites=["Python 3.8+"],
        platform_compatibility=["Linux", "Windows", "MacOS"]
    )
    
    print("\nDocument Section Details:")
    print(f"Title: {doc_section.title}")
    print(f"Authority Level: {doc_section.authority_level}")
    print(f"Source Quality: {doc_section.source_quality}")
    print(f"Prerequisites: {doc_section.prerequisites}")
    print(f"Platform Compatibility: {doc_section.platform_compatibility}")

if __name__ == "__main__":
    test_evaluation_system() 