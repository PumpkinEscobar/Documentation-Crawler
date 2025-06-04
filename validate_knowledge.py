#!/usr/bin/env python3
"""
AI Knowledge Quality Validator
Analyzes and validates the quality of crawled documentation
"""

import json
import yaml
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple
from urllib.parse import urlparse

class KnowledgeValidator:
    def __init__(self, knowledge_file: str = "wizard_knowledge.json"):
        self.knowledge_file = Path(knowledge_file)
        self.quality_report = {
            'timestamp': datetime.now().isoformat(),
            'total_entries': 0,
            'quality_scores': {},
            'issues_found': [],
            'recommendations': [],
            'source_analysis': {}
        }
        
    def validate(self):
        """Run comprehensive validation"""
        print("🔍 AI Knowledge Quality Validation")
        print("=" * 50)
        
        if not self.knowledge_file.exists():
            print(f"❌ Knowledge file not found: {self.knowledge_file}")
            return
            
        try:
            with open(self.knowledge_file, 'r') as f:
                self.knowledge_data = json.load(f)
        except Exception as e:
            print(f"❌ Error loading knowledge file: {e}")
            return
            
        self._analyze_structure()
        self._validate_sources()
        self._check_content_quality()
        self._detect_hallucinations()
        self._verify_official_sources()
        self._generate_report()
        
    def _analyze_structure(self):
        """Analyze the structure of the knowledge base"""
        print("\n📊 Structure Analysis")
        print("-" * 30)
        
        metadata = self.knowledge_data.get('metadata', {})
        sources = metadata.get('sources', [])
        total_docs = metadata.get('total_documents', 0)
        
        print(f"📚 Total documents: {total_docs}")
        print(f"🌐 Sources claimed: {len(sources)}")
        print(f"📅 Last crawl: {metadata.get('last_updated', 'Unknown')}")
        
        self.quality_report['total_entries'] = total_docs
        self.quality_report['sources_claimed'] = sources
        
        # Check if structure makes sense
        if total_docs == 0:
            self.quality_report['issues_found'].append("No documents found in knowledge base")
        
        if not sources:
            self.quality_report['issues_found'].append("No sources listed in metadata")
            
    def _validate_sources(self):
        """Validate the authenticity of sources"""
        print("\n🔗 Source Validation")
        print("-" * 30)
        
        # Define legitimate AI documentation sources
        legitimate_sources = {
            'openai': {
                'domains': ['platform.openai.com', 'api.openai.com', 'cookbook.openai.com'],
                'patterns': [r'/docs/', r'/api/', r'/guides/', r'/models/']
            },
            'anthropic': {
                'domains': ['docs.anthropic.com', 'console.anthropic.com'],
                'patterns': [r'/docs/', r'/api/', r'/claude/']
            },
            'google': {
                'domains': ['ai.google.dev', 'cloud.google.com', 'developers.google.com'],
                'patterns': [r'/docs/', r'/ai/', r'/palm/', r'/gemini/']
            },
            'meta': {
                'domains': ['ai.meta.com', 'research.facebook.com'],
                'patterns': [r'/resources/', r'/llama/', r'/research/']
            },
            'xai': {
                'domains': ['docs.x.ai', 'x.ai'],
                'patterns': [r'/docs/', r'/api/', r'/grok/']
            }
        }
        
        source_quality = {}
        
        # Analyze each entry for source legitimacy
        for key, entry in self.knowledge_data.items():
            if key == 'metadata':
                continue
                
            if isinstance(entry, dict) and 'url' in entry:
                url = entry['url']
                domain = urlparse(url).netloc.lower()
                path = urlparse(url).path.lower()
                
                # Check if source is legitimate
                is_legitimate = False
                source_type = 'unknown'
                
                for provider, config in legitimate_sources.items():
                    if any(legit_domain in domain for legit_domain in config['domains']):
                        if any(re.search(pattern, path) for pattern in config['patterns']):
                            is_legitimate = True
                            source_type = provider
                            break
                
                source_quality[key] = {
                    'url': url,
                    'domain': domain,
                    'legitimate': is_legitimate,
                    'source_type': source_type,
                    'content_length': len(str(entry.get('content', '')))
                }
                
                if not is_legitimate:
                    self.quality_report['issues_found'].append(f"Questionable source: {url}")
                    print(f"⚠️  Questionable: {domain}")
                else:
                    print(f"✅ Verified: {domain} ({source_type})")
        
        self.quality_report['source_analysis'] = source_quality
        
    def _check_content_quality(self):
        """Check content quality indicators"""
        print("\n📝 Content Quality Analysis")
        print("-" * 30)
        
        quality_issues = []
        
        for key, entry in self.knowledge_data.items():
            if key == 'metadata':
                continue
                
            if isinstance(entry, dict):
                content = str(entry.get('content', ''))
                title = entry.get('title', '')
                
                # Check for common quality issues
                issues = []
                
                # Too short content
                if len(content) < 100:
                    issues.append("Very short content")
                
                # Too much repetition
                if self._has_excessive_repetition(content):
                    issues.append("Excessive repetition detected")
                
                # Nonsensical patterns
                if self._has_nonsensical_patterns(content):
                    issues.append("Nonsensical patterns found")
                
                # Missing key information for AI docs
                if not self._has_ai_keywords(content):
                    issues.append("Missing AI/LLM keywords")
                
                # Malformed HTML/JSON
                if self._has_malformed_content(content):
                    issues.append("Malformed content structure")
                
                if issues:
                    quality_issues.append({
                        'entry': key,
                        'title': title,
                        'issues': issues,
                        'url': entry.get('url', 'No URL')
                    })
                    print(f"⚠️  {key}: {', '.join(issues)}")
                else:
                    print(f"✅ {key}: Quality content")
        
        self.quality_report['content_quality_issues'] = quality_issues
        
    def _detect_hallucinations(self):
        """Detect potential AI hallucinations or fabricated content"""
        print("\n🧠 Hallucination Detection")
        print("-" * 30)
        
        # Common hallucination patterns in AI documentation
        suspicious_patterns = [
            r'As an AI language model',
            r'I cannot access',
            r'I don\'t have real-time',
            r'Please note that',
            r'fictional API endpoint',
            r'example\.com',
            r'placeholder.*key',
            r'lorem ipsum',
            r'This is a sample',
            r'TODO:|FIXME:|NOTE:'
        ]
        
        hallucination_flags = []
        
        for key, entry in self.knowledge_data.items():
            if key == 'metadata':
                continue
                
            content = str(entry.get('content', '')).lower()
            
            for pattern in suspicious_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    hallucination_flags.append({
                        'entry': key,
                        'pattern': pattern,
                        'url': entry.get('url', 'No URL')
                    })
                    print(f"🚨 Potential hallucination in {key}: {pattern}")
        
        if not hallucination_flags:
            print("✅ No obvious hallucinations detected")
            
        self.quality_report['hallucination_flags'] = hallucination_flags
        
    def _verify_official_sources(self):
        """Verify content is from official sources"""
        print("\n🏛️  Official Source Verification")
        print("-" * 30)
        
        # Check for official indicators
        official_indicators = [
            'official documentation',
            'api reference',
            'developer guide',
            'technical documentation',
            'release notes',
            'changelog'
        ]
        
        unofficial_indicators = [
            'blog post',
            'tutorial by',
            'community guide',
            'third-party',
            'unofficial',
            'personal website'
        ]
        
        source_verification = {}
        
        for key, entry in self.knowledge_data.items():
            if key == 'metadata':
                continue
                
            content = str(entry.get('content', '')).lower()
            title = str(entry.get('title', '')).lower()
            
            official_score = sum(1 for indicator in official_indicators 
                                if indicator in content or indicator in title)
            unofficial_score = sum(1 for indicator in unofficial_indicators 
                                 if indicator in content or indicator in title)
            
            verification_status = 'unknown'
            if official_score > unofficial_score:
                verification_status = 'likely_official'
            elif unofficial_score > 0:
                verification_status = 'likely_unofficial'
            
            source_verification[key] = {
                'status': verification_status,
                'official_indicators': official_score,
                'unofficial_indicators': unofficial_score
            }
            
            if verification_status == 'likely_unofficial':
                print(f"⚠️  Potentially unofficial: {key}")
                self.quality_report['issues_found'].append(f"Potentially unofficial source: {key}")
            else:
                print(f"✅ Official-looking: {key}")
        
        self.quality_report['source_verification'] = source_verification
        
    def _has_excessive_repetition(self, content: str) -> bool:
        """Check for excessive repetition in content"""
        words = content.lower().split()
        if len(words) < 10:
            return False
            
        # Check for repeated phrases
        for i in range(len(words) - 5):
            phrase = ' '.join(words[i:i+3])
            if content.lower().count(phrase) > 3:
                return True
        return False
        
    def _has_nonsensical_patterns(self, content: str) -> bool:
        """Check for nonsensical patterns"""
        nonsensical_patterns = [
            r'[a-zA-Z]{50,}',  # Very long words
            r'(\w)\1{10,}',    # Repeated characters
            r'[^\w\s]{20,}',   # Long sequences of special characters
            r'\d{20,}',        # Very long numbers
        ]
        
        for pattern in nonsensical_patterns:
            if re.search(pattern, content):
                return True
        return False
        
    def _has_ai_keywords(self, content: str) -> bool:
        """Check if content has relevant AI/LLM keywords"""
        ai_keywords = [
            'api', 'model', 'prompt', 'token', 'completion', 'embedding',
            'chat', 'gpt', 'claude', 'llm', 'ai', 'machine learning',
            'neural', 'transformer', 'endpoint', 'parameter', 'response'
        ]
        
        content_lower = content.lower()
        return any(keyword in content_lower for keyword in ai_keywords)
        
    def _has_malformed_content(self, content: str) -> bool:
        """Check for malformed HTML/JSON content"""
        # Check for unmatched tags or brackets
        if content.count('<') != content.count('>'):
            return True
        if content.count('{') != content.count('}'):
            return True
        if content.count('[') != content.count(']'):
            return True
        return False
        
    def _generate_report(self):
        """Generate comprehensive quality report"""
        print("\n📋 Quality Report Summary")
        print("=" * 50)
        
        total_issues = len(self.quality_report['issues_found'])
        legitimate_sources = sum(1 for analysis in self.quality_report['source_analysis'].values() 
                               if analysis['legitimate'])
        total_sources = len(self.quality_report['source_analysis'])
        
        print(f"📊 Overall Quality Score: {self._calculate_quality_score()}/100")
        print(f"🔗 Legitimate Sources: {legitimate_sources}/{total_sources}")
        print(f"⚠️  Issues Found: {total_issues}")
        print(f"🧠 Hallucination Flags: {len(self.quality_report['hallucination_flags'])}")
        
        if total_issues > 0:
            print(f"\n⚠️  Main Issues:")
            for issue in self.quality_report['issues_found'][:5]:
                print(f"   • {issue}")
        
        # Generate recommendations
        self._generate_recommendations()
        
        if self.quality_report['recommendations']:
            print(f"\n💡 Recommendations:")
            for rec in self.quality_report['recommendations'][:3]:
                print(f"   • {rec}")
        
        # Save detailed report
        report_file = f"quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(self.quality_report, f, indent=2)
        print(f"\n📄 Detailed report saved: {report_file}")
        
    def _calculate_quality_score(self) -> int:
        """Calculate overall quality score 0-100"""
        score = 100
        
        # Deduct for issues
        score -= len(self.quality_report['issues_found']) * 5
        score -= len(self.quality_report['hallucination_flags']) * 10
        
        # Deduct for illegitimate sources
        if self.quality_report['source_analysis']:
            illegitimate_ratio = sum(1 for analysis in self.quality_report['source_analysis'].values() 
                                   if not analysis['legitimate']) / len(self.quality_report['source_analysis'])
            score -= int(illegitimate_ratio * 30)
        
        return max(0, score)
        
    def _generate_recommendations(self):
        """Generate improvement recommendations"""
        recommendations = []
        
        if len(self.quality_report['hallucination_flags']) > 0:
            recommendations.append("Review and remove entries with hallucination patterns")
        
        illegitimate_sources = [key for key, analysis in self.quality_report['source_analysis'].items() 
                              if not analysis['legitimate']]
        if illegitimate_sources:
            recommendations.append(f"Remove {len(illegitimate_sources)} entries from questionable sources")
        
        if self.quality_report['total_entries'] < 50:
            recommendations.append("Increase crawl depth to gather more comprehensive documentation")
        
        if len(self.quality_report['content_quality_issues']) > 5:
            recommendations.append("Implement content filtering to remove low-quality entries")
        
        self.quality_report['recommendations'] = recommendations

def main():
    """Run knowledge validation"""
    validator = KnowledgeValidator()
    validator.validate()

if __name__ == "__main__":
    main()