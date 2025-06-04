#!/usr/bin/env python3
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
    print("\n🔧 System Status:")
    
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
    print("\n📊 Learning Progress:")
    
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
    print("\n🎯 Recommended Actions:")
    
    if not Path("knowledge_base/sample_knowledge.json").exists():
        print("  1. Run: python wizard_cli.py crawl")
    
    if not Path("progress_tracking/user_progress.json").exists():
        print("  2. Run: python wizard_cli.py start")
    
    print("  3. Set up your myGPT with provided files")
    print("  4. Start learning: python wizard_cli.py learn")
    print("  5. Practice with real AI projects")

if __name__ == "__main__":
    main()
