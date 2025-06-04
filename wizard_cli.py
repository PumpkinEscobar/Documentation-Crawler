#!/usr/bin/env python3
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
    
    print("\nNext steps:")
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
    
    print("\nUse your myGPT to start learning!")
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
    mygpt_files = ["mygpt_doc_index.yaml", "mygpt_instructions.txt"]
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
