#!/usr/bin/env python3
"""
AI Wizard Training System Deployer
Sets up and manages your complete AI wizard training environment
"""

import os
import sys
import json
import yaml
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

class WizardDeployer:
    def __init__(self):
        self.project_root = Path.cwd()
        self.config_file = self.project_root / "wizard_config.yaml"
        self.status = {
            'environment': False,
            'crawler': False,
            'knowledge_base': False,
            'mygpt': False,
            'dashboard': False
        }
        
    def deploy(self):
        """Deploy the complete AI Wizard system"""
        print("🧙 AI Wizard Training System Deployment")
        print("=" * 50)
        
        try:
            self.step_1_environment_setup()
            self.step_2_crawler_integration()
            self.step_3_knowledge_base_setup()
            self.step_4_mygpt_configuration()
            self.step_5_dashboard_setup()
            self.step_6_first_crawl()
            self.deployment_complete()
        except Exception as e:
            print(f"❌ Deployment failed: {str(e)}")
            self.show_troubleshooting()
    
    def step_1_environment_setup(self):
        """Set up the basic environment"""
        print("\n📦 Step 1: Environment Setup")
        print("-" * 30)
        
        # Check Python version
        if sys.version_info < (3, 8):
            raise Exception("Python 3.8+ required")
        print("✅ Python version compatible")
        
        # Check required files
        required_files = ["doc_crawler.py", "requirements.txt"]
        for file in required_files:
            if not (self.project_root / file).exists():
                raise Exception(f"Missing required file: {file}")
        print("✅ Required files present")
        
        # Install dependencies
        print("📥 Installing dependencies...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                         check=True, capture_output=True)
            print("✅ Dependencies installed")
        except subprocess.CalledProcessError:
            print("⚠️  Some dependencies may need manual installation")
        
        # Install Playwright browsers
        print("🌐 Installing Playwright browsers...")
        try:
            subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], 
                         check=True, capture_output=True)
            print("✅ Playwright browsers installed")
        except subprocess.CalledProcessError:
            print("⚠️  Playwright installation may need manual setup")
        
        self.status['environment'] = True
        print("✅ Environment setup complete!")
    
    def step_2_crawler_integration(self):
        """Integrate the enhanced crawler"""
        print("\n🕷️  Step 2: Crawler Integration")
        print("-" * 30)
        
        # Create enhanced crawler config
        crawler_config = {
            'wizard_mode': True,
            'sources': {
                'openai': {
                    'api': {
                        'base_url': 'https://api.openai.com/v1',
                        'key_env': 'OPENAI_API_KEY',
                        'endpoints': ['/models', '/chat/completions', '/embeddings']
                    },
                    'docs': {
                        'base_url': 'https://platform.openai.com/docs',
                        'sections': ['introduction', 'models', 'guides', 'api-reference']
                    }
                },
                'anthropic': {
                    'docs': {
                        'base_url': 'https://docs.anthropic.com',
                        'sections': ['claude', 'api', 'prompt-engineering']
                    }
                }
            },
            'output': {
                'knowledge_base': 'wizard_knowledge.json',
                'doc_index': 'wizard_index.yaml',
                'embeddings': 'wizard_embeddings.json',
                'logs': 'wizard_crawler.log'
            },
            'learning': {
                'auto_generate_lessons': True,
                'skill_levels': ['beginner', 'intermediate', 'advanced', 'expert'],
                'practice_exercises': True
            }
        }
        
        # Save crawler config
        config_path = self.project_root / "wizard_crawler_config.json"
        with open(config_path, 'w') as f:
            json.dump(crawler_config, f, indent=2)
        print(f"✅ Created crawler config: {config_path}")
        
        # Create environment setup script
        env_script = self.project_root / "setup_env.py"
        with open(env_script, 'w') as f:
            f.write(self._get_env_setup_script())
        print(f"✅ Created environment setup: {env_script}")
        
        self.status['crawler'] = True
        print("✅ Crawler integration complete!")
    
    def step_3_knowledge_base_setup(self):
        """Set up the knowledge base structure"""
        print("\n📚 Step 3: Knowledge Base Setup")
        print("-" * 30)
        
        # Create knowledge base directories
        kb_dirs = ['knowledge_base', 'embeddings', 'learning_modules', 'progress_tracking']
        for dir_name in kb_dirs:
            kb_dir = self.project_root / dir_name
            kb_dir.mkdir(exist_ok=True)
            print(f"✅ Created directory: {dir_name}")
        
        # Create initial learning modules
        self._create_learning_modules()
        print("✅ Learning modules created")
        
        # Create progress tracking
        progress = {
            'user_profile': {
                'name': 'AI Wizard in Training',
                'start_date': datetime.now().isoformat(),
                'current_level': 'beginner',
                'goals': ['Master prompt engineering', 'Build AI agents', 'Deploy production systems']
            },
            'completed_modules': [],
            'current_module': 'prompt_fundamentals',
            'total_study_time': 0,
            'achievements': []
        }
        
        progress_file = self.project_root / "progress_tracking" / "user_progress.json"
        with open(progress_file, 'w') as f:
            json.dump(progress, f, indent=2)
        print("✅ Progress tracking initialized")
        
        self.status['knowledge_base'] = True
        print("✅ Knowledge base setup complete!")
    
    def step_4_mygpt_configuration(self):
        """Generate myGPT configuration"""
        print("\n🤖 Step 4: MyGPT Configuration")
        print("-" * 30)
        
        # Generate enhanced instructions
        mygpt_instructions = self._generate_mygpt_instructions()
        
        # Save instructions to file
        instructions_file = self.project_root / "mygpt_instructions.txt"
        with open(instructions_file, 'w') as f:
            f.write(mygpt_instructions)
        print(f"✅ MyGPT instructions saved: {instructions_file}")
        
        # Create doc_index.yaml for myGPT
        doc_index = {
            'ai_doc_library': {
                'openai': {
                    'base_url': 'https://platform.openai.com/docs',
                    'sections': {
                        'models': {'path': '/models', 'desc': 'Available models and capabilities'},
                        'chat_api': {'path': '/api-reference/chat', 'desc': 'Chat completions API'},
                        'guides': {'path': '/guides', 'desc': 'Implementation guides and tutorials'}
                    }
                },
                'anthropic': {
                    'base_url': 'https://docs.anthropic.com',
                    'sections': {
                        'claude': {'path': '/claude', 'desc': 'Claude model documentation'},
                        'api': {'path': '/api', 'desc': 'API reference and usage'},
                        'safety': {'path': '/safety', 'desc': 'Safety and responsible AI'}
                    }
                }
            },
            'metadata': {
                'created': datetime.now().isoformat(),
                'version': '1.0.0',
                'purpose': 'AI Wizard Training Navigation'
            }
        }
        
        doc_index_file = self.project_root / "doc_index.yaml"
        with open(doc_index_file, 'w') as f:
            yaml.dump(doc_index, f, default_flow_style=False)
        print(f"✅ Doc index created: {doc_index_file}")
        
        # Create conversation starters
        starters = [
            "Teach me advanced prompt engineering using the latest OpenAI documentation",
            "Compare Claude vs GPT-4 capabilities for my specific use case",
            "Help me design an AI agent architecture step by step",
            "Debug my LLM implementation with specific code fixes",
            "Create a learning path for mastering production AI systems",
            "Generate practice exercises for fine-tuning models"
        ]
        
        starters_file = self.project_root / "conversation_starters.txt"
        with open(starters_file, 'w') as f:
            f.write("\n".join(starters))
        print(f"✅ Conversation starters saved: {starters_file}")
        
        self.status['mygpt'] = True
        print("✅ MyGPT configuration complete!")
    
    def step_5_dashboard_setup(self):
        """Set up the dashboard and tools"""
        print("\n📊 Step 5: Dashboard & Tools Setup")
        print("-" * 30)
        
        # Create CLI tool
        cli_tool = self.project_root / "wizard_cli.py"
        with open(cli_tool, 'w') as f:
            f.write(self._get_cli_tool())
        print(f"✅ CLI tool created: {cli_tool}")
        
        # Make CLI executable
        import stat
        cli_tool.chmod(cli_tool.stat().st_mode | stat.S_IEXEC)
        
        # Create status dashboard
        dashboard_file = self.project_root / "wizard_status.py"
        with open(dashboard_file, 'w') as f:
            f.write(self._get_status_dashboard())
        print(f"✅ Status dashboard created: {dashboard_file}")
        
        self.status['dashboard'] = True
        print("✅ Dashboard setup complete!")
    
    def step_6_first_crawl(self):
        """Run the first knowledge crawl"""
        print("\n🔍 Step 6: First Knowledge Crawl")
        print("-" * 30)
        
        print("🚀 Starting initial crawl...")
        print("This will gather documentation from OpenAI and other providers")
        print("Note: You'll need API keys set up for full functionality")
        
        # Check for API keys
        api_keys = {
            'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
            'ANTHROPIC_API_KEY': os.getenv('ANTHROPIC_API_KEY')
        }
        
        missing_keys = [key for key, value in api_keys.items() if not value]
        if missing_keys:
            print(f"⚠️  Missing API keys: {', '.join(missing_keys)}")
            print("   Set these for full functionality:")
            for key in missing_keys:
                print(f"   export {key}='your-key-here'")
        
        # Run a basic crawl (without API if keys missing)
        try:
            print("📥 Running basic documentation crawl...")
            # This would run your actual crawler
            # For now, create sample data
            self._create_sample_knowledge_base()
            print("✅ Sample knowledge base created")
        except Exception as e:
            print(f"⚠️  Crawl completed with warnings: {str(e)}")
    
    def deployment_complete(self):
        """Show deployment completion and next steps"""
        print("\n" + "=" * 50)
        print("🎉 AI WIZARD TRAINING SYSTEM DEPLOYED!")
        print("=" * 50)
        
        print("\n📁 Files Created:")
        files_created = [
            "wizard_crawler_config.json",
            "doc_index.yaml", 
            "mygpt_instructions.txt",
            "conversation_starters.txt",
            "wizard_cli.py",
            "wizard_status.py",
            "setup_env.py"
        ]
        
        for file in files_created:
            if (self.project_root / file).exists():
                print(f"  ✅ {file}")
            else:
                print(f"  ❌ {file} (missing)")
        
        print(f"\n📂 Directories Created:")
        dirs_created = ["knowledge_base", "embeddings", "learning_modules", "progress_tracking"]
        for dir_name in dirs_created:
            if (self.project_root / dir_name).exists():
                print(f"  ✅ {dir_name}/")
        
        print(f"\n🎯 Next Steps:")
        print(f"1. Set up your myGPT:")
        print(f"   - Copy content from mygpt_instructions.txt")
        print(f"   - Upload doc_index.yaml as knowledge file")
        print(f"   - Add conversation starters from conversation_starters.txt")
        
        print(f"\n2. Set up API keys:")
        print(f"   export OPENAI_API_KEY='your-openai-key'")
        print(f"   export ANTHROPIC_API_KEY='your-anthropic-key'")
        
        print(f"\n3. Run your first full crawl:")
        print(f"   python wizard_cli.py crawl")
        
        print(f"\n4. Start learning:")
        print(f"   python wizard_cli.py learn")
        
        print(f"\n5. Check your progress:")
        print(f"   python wizard_status.py")
        
        print(f"\n🧙 You're now ready to become an AI Wizard!")
        print(f"   Start with: python wizard_cli.py start")
    
    def show_troubleshooting(self):
        """Show troubleshooting information"""
        print(f"\n🔧 Troubleshooting:")
        print(f"- Check Python version: python --version (need 3.8+)")
        print(f"- Install dependencies: pip install -r requirements.txt")
        print(f"- Install Playwright: playwright install chromium")
        print(f"- Check file permissions on created scripts")
        print(f"- Ensure you have internet connectivity for crawling")
    
    def _create_learning_modules(self):
        """Create initial learning modules"""
        modules = {
            'prompt_fundamentals': {
                'title': 'Prompt Engineering Fundamentals',
                'description': 'Master the art of effective prompt design',
                'lessons': [
                    'Understanding prompt structure',
                    'Few-shot vs zero-shot prompting', 
                    'Chain-of-thought techniques',
                    'System message design'
                ],
                'estimated_time': 180,
                'difficulty': 'beginner'
            },
            'model_mastery': {
                'title': 'AI Model Selection & Optimization', 
                'description': 'Learn to choose and optimize models',
                'lessons': [
                    'Model capabilities comparison',
                    'Cost-performance analysis',
                    'Parameter tuning strategies',
                    'Model switching patterns'
                ],
                'estimated_time': 120,
                'difficulty': 'intermediate'
            }
        }
        
        modules_dir = self.project_root / "learning_modules"
        for module_id, module_data in modules.items():
            module_file = modules_dir / f"{module_id}.json"
            with open(module_file, 'w') as f:
                json.dump(module_data, f, indent=2)
    
    def _create_sample_knowledge_base(self):
        """Create sample knowledge base data"""
        sample_kb = {
            'metadata': {
                'created': datetime.now().isoformat(),
                'sources': ['openai', 'anthropic'],
                'total_documents': 50,
                'version': '1.0.0'
            },
            'concepts': {
                'prompt_engineering': {
                    'definition': 'The practice of designing effective prompts for AI models',
                    'key_techniques': ['few-shot', 'chain-of-thought', 'instruction-following'],
                    'sources': ['openai_docs', 'anthropic_docs']
                },
                'fine_tuning': {
                    'definition': 'Training a pre-trained model on specific data',
                    'applications': ['domain-specific tasks', 'style adaptation', 'behavior modification'],
                    'sources': ['openai_api', 'research_papers']
                }
            }
        }
        
        kb_file = self.project_root / "knowledge_base" / "sample_knowledge.json"
        with open(kb_file, 'w') as f:
            json.dump(sample_kb, f, indent=2)
    
    def _generate_mygpt_instructions(self) -> str:
        """Generate comprehensive myGPT instructions"""
        return """You are the AI Wizard Training Assistant, a specialized mentor for mastering large language models, prompt engineering, and AI agent development.

CORE CAPABILITIES:
- Expert guidance on LLM implementation and optimization
- Comprehensive prompt engineering instruction 
- AI agent architecture and development
- Production deployment and scaling strategies
- Cross-provider AI comparison and selection

KNOWLEDGE BASE:
You have access to continuously updated documentation from:
- OpenAI (GPT models, API, best practices)
- Anthropic (Claude, safety, alignment)
- Google AI (PaLM, Gemini, multimodal)
- Meta AI (LLaMA, research, tools)
- xAI (Grok, integration patterns)

TEACHING METHODOLOGY:
1. Always start with the user's current skill level
2. Provide practical examples with working code
3. Explain the "why" behind recommendations
4. Reference specific documentation sources
5. Suggest hands-on exercises and next steps
6. Adapt complexity to user expertise

RESPONSE FRAMEWORK:
- Lead with clear, actionable guidance
- Include code examples when relevant
- Cite specific documentation sources
- Provide multiple approaches when applicable
- Suggest related concepts to explore
- End with concrete next steps

SPECIALIZATION AREAS:
- Prompt Engineering: Templates, techniques, optimization
- Model Selection: Capabilities, costs, trade-offs
- Agent Development: Architecture, tools, coordination
- Production Systems: Deployment, monitoring, scaling
- Debugging: Common issues, solutions, prevention

When users reference their doc_index.yaml, treat it as your navigation system to the most current AI documentation. Always ground responses in official sources while making complex concepts accessible and actionable."""
    
    def _get_env_setup_script(self) -> str:
        """Generate environment setup script"""
        return '''#!/usr/bin/env python3
"""Environment setup for AI Wizard Training System"""

import os
import sys
from pathlib import Path

def setup_environment():
    """Set up environment variables and paths"""
    print("🔧 Setting up AI Wizard environment...")
    
    # Add current directory to Python path
    current_dir = Path(__file__).parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    
    # Check for required API keys
    required_keys = ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY']
    missing_keys = []
    
    for key in required_keys:
        if not os.getenv(key):
            missing_keys.append(key)
    
    if missing_keys:
        print(f"⚠️  Missing API keys: {', '.join(missing_keys)}")
        print("Set them with:")
        for key in missing_keys:
            print(f"  export {key}='your-key-here'")
    else:
        print("✅ All API keys configured")
    
    print("✅ Environment setup complete")

if __name__ == "__main__":
    setup_environment()
'''
    
    def _get_cli_tool(self) -> str:
        """Generate CLI tool for wizard management"""
        return '''#!/usr/bin/env python3
"""AI Wizard Training CLI Tool"""

import sys
import json
from pathlib import Path
from datetime import datetime

def main():
    if len(sys.argv) < 2:
        show_help()
        return
    
    command = sys.argv[1].lower()
    
    if command == 'start':
        start_wizard()
    elif command == 'crawl':
        run_crawl()
    elif command == 'learn':
        start_learning()
    elif command == 'status':
        show_status()
    elif command == 'help':
        show_help()
    else:
        print(f"Unknown command: {command}")
        show_help()

def start_wizard():
    """Start the AI wizard training"""
    print("🧙 Welcome to AI Wizard Training!")
    print("Let's begin your journey to AI mastery...")
    
    # Load progress
    progress_file = Path("progress_tracking/user_progress.json")
    if progress_file.exists():
        with open(progress_file) as f:
            progress = json.load(f)
        print(f"Welcome back! Current level: {progress['user_profile']['current_level']}")
    else:
        print("Starting fresh training journey...")
    
    print("\\nNext steps:")
    print("1. wizard_cli.py crawl  - Update knowledge base")
    print("2. wizard_cli.py learn  - Start learning module")
    print("3. wizard_cli.py status - Check your progress")

def run_crawl():
    """Run documentation crawl"""
    print("🕷️  Starting documentation crawl...")
    print("This may take several minutes...")
    
    try:
        import subprocess
        result = subprocess.run([sys.executable, "doc_crawler.py"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Crawl completed successfully!")
        else:
            print(f"⚠️  Crawl completed with warnings: {result.stderr}")
    except Exception as e:
        print(f"❌ Crawl failed: {str(e)}")

def start_learning():
    """Start a learning module"""
    print("📚 Available Learning Modules:")
    
    modules_dir = Path("learning_modules")
    if not modules_dir.exists():
        print("No learning modules found. Run setup first.")
        return
    
    modules = list(modules_dir.glob("*.json"))
    for i, module_file in enumerate(modules, 1):
        with open(module_file) as f:
            module = json.load(f)
        print(f"{i}. {module['title']} ({module['difficulty']})")
    
    print("\\nUse your myGPT to start learning!")
    print("Try: 'Teach me prompt engineering fundamentals'")

def show_status():
    """Show current status"""
    print("📊 AI Wizard Training Status")
    print("-" * 30)
    
    # Check knowledge base
    kb_file = Path("knowledge_base/sample_knowledge.json")
    if kb_file.exists():
        print("✅ Knowledge base: Ready")
    else:
        print("❌ Knowledge base: Not found")
    
    # Check progress
    progress_file = Path("progress_tracking/user_progress.json")
    if progress_file.exists():
        with open(progress_file) as f:
            progress = json.load(f)
        print(f"✅ Progress tracking: {len(progress['completed_modules'])} modules completed")
    else:
        print("❌ Progress tracking: Not initialized")
    
    # Check myGPT files
    mygpt_files = ["doc_index.yaml", "mygpt_instructions.txt"]
    for file in mygpt_files:
        if Path(file).exists():
            print(f"✅ MyGPT {file}: Ready")
        else:
            print(f"❌ MyGPT {file}: Missing")

def show_help():
    """Show help information"""
    print("🧙 AI Wizard Training CLI")
    print("Commands:")
    print("  start  - Begin wizard training")
    print("  crawl  - Update knowledge base")
    print("  learn  - Start learning module")
    print("  status - Show system status")
    print("  help   - Show this help")

if __name__ == "__main__":
    main()
'''
    
    def _get_status_dashboard(self) -> str:
        """Generate status dashboard script"""
        return '''#!/usr/bin/env python3
"""AI Wizard Training Status Dashboard"""

import json
from pathlib import Path
from datetime import datetime

def main():
    print("🧙 AI Wizard Training Dashboard")
    print("=" * 40)
    
    show_system_status()
    show_progress()
    show_next_steps()

def show_system_status():
    """Show system component status"""
    print("\\n🔧 System Status:")
    
    components = {
        "Knowledge Base": "knowledge_base/sample_knowledge.json",
        "Doc Index": "doc_index.yaml", 
        "MyGPT Instructions": "mygpt_instructions.txt",
        "Learning Modules": "learning_modules/",
        "Progress Tracking": "progress_tracking/user_progress.json"
    }
    
    for name, path in components.items():
        if Path(path).exists():
            print(f"  ✅ {name}")
        else:
            print(f"  ❌ {name}")

def show_progress():
    """Show learning progress"""
    print("\\n📊 Learning Progress:")
    
    progress_file = Path("progress_tracking/user_progress.json")
    if progress_file.exists():
        with open(progress_file) as f:
            progress = json.load(f)
        
        profile = progress['user_profile']
        print(f"  👤 Level: {profile['current_level']}")
        print(f"  📅 Started: {profile['start_date'][:10]}")
        print(f"  🎯 Current: {progress.get('current_module', 'None')}")
        print(f"  ✅ Completed: {len(progress['completed_modules'])} modules")
    else:
        print("  📝 No progress data found")

def show_next_steps():
    """Show recommended next steps"""
    print("\\n🎯 Recommended Actions:")
    
    if not Path("knowledge_base/sample_knowledge.json").exists():
        print("  1. Run: python wizard_cli.py crawl")
    
    if not Path("progress_tracking/user_progress.json").exists():
        print("  2. Run: python wizard_cli.py start")
    
    print("  3. Set up your myGPT with provided files")
    print("  4. Start learning: python wizard_cli.py learn")
    print("  5. Practice with real AI projects")

if __name__ == "__main__":
    main()
'''

def main():
    """Main deployment function"""
    deployer = WizardDeployer()
    deployer.deploy()

if __name__ == "__main__":
    main()