#!/usr/bin/env python3
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
