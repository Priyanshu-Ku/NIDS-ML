"""
Project Verification Script

This script verifies that the NIDS-ML project is set up correctly.
Run this script to check if all files and directories are in place.

Usage:
    python verify_setup.py
"""

import os
import sys
from pathlib import Path

# Color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_colored(text, color):
    """Print colored text."""
    print(f"{color}{text}{RESET}")

def check_file(filepath, description):
    """Check if a file exists."""
    if os.path.exists(filepath):
        print_colored(f"✓ {description}: {filepath}", GREEN)
        return True
    else:
        print_colored(f"✗ {description}: {filepath} (MISSING)", RED)
        return False

def check_directory(dirpath, description):
    """Check if a directory exists."""
    if os.path.isdir(dirpath):
        print_colored(f"✓ {description}: {dirpath}", GREEN)
        return True
    else:
        print_colored(f"✗ {description}: {dirpath} (MISSING)", RED)
        return False

def main():
    """Main verification function."""
    print_colored("\n" + "="*60, BLUE)
    print_colored("  NIDS-ML PROJECT SETUP VERIFICATION", BLUE)
    print_colored("="*60 + "\n", BLUE)
    
    base_dir = Path(__file__).parent
    checks_passed = 0
    total_checks = 0
    
    # Check directories
    print_colored("Checking Directories:", YELLOW)
    print_colored("-" * 60, YELLOW)
    
    directories = [
        (base_dir / "data", "Data directory"),
        (base_dir / "data" / "raw", "Raw data directory"),
        (base_dir / "data" / "processed", "Processed data directory"),
        (base_dir / "src", "Source code directory"),
        (base_dir / "models", "Models directory"),
        (base_dir / "logs", "Logs directory"),
    ]
    
    for dirpath, desc in directories:
        total_checks += 1
        if check_directory(dirpath, desc):
            checks_passed += 1
    
    # Check Python module files
    print_colored("\nChecking Python Modules:", YELLOW)
    print_colored("-" * 60, YELLOW)
    
    modules = [
        (base_dir / "src" / "__init__.py", "Package init file"),
        (base_dir / "src" / "config.py", "Configuration module"),
        (base_dir / "src" / "utils.py", "Utilities module"),
        (base_dir / "src" / "data_preprocessing.py", "Data preprocessing module"),
        (base_dir / "src" / "feature_selection.py", "Feature selection module"),
        (base_dir / "src" / "model_training.py", "Model training module"),
        (base_dir / "src" / "shap_explainability.py", "SHAP explainability module"),
        (base_dir / "src" / "realtime_detection.py", "Real-time detection module"),
        (base_dir / "src" / "dashboard.py", "Dashboard module"),
    ]
    
    for filepath, desc in modules:
        total_checks += 1
        if check_file(filepath, desc):
            checks_passed += 1
    
    # Check documentation files
    print_colored("\nChecking Documentation:", YELLOW)
    print_colored("-" * 60, YELLOW)
    
    docs = [
        (base_dir / "README.md", "Main README"),
        (base_dir / "QUICKSTART.md", "Quick start guide"),
        (base_dir / "ARCHITECTURE.md", "Architecture documentation"),
        (base_dir / "PROJECT_SUMMARY.md", "Project summary"),
        (base_dir / "requirements.txt", "Requirements file"),
        (base_dir / ".gitignore", "Git ignore file"),
        (base_dir / "LICENSE", "License file"),
    ]
    
    for filepath, desc in docs:
        total_checks += 1
        if check_file(filepath, desc):
            checks_passed += 1
    
    # Summary
    print_colored("\n" + "="*60, BLUE)
    print_colored(f"Verification Complete: {checks_passed}/{total_checks} checks passed", 
                 GREEN if checks_passed == total_checks else YELLOW)
    print_colored("="*60 + "\n", BLUE)
    
    if checks_passed == total_checks:
        print_colored("✓ All checks passed! Project is ready.", GREEN)
        print_colored("\nNext Steps:", YELLOW)
        print("  1. Create virtual environment: python -m venv venv")
        print("  2. Activate environment: .\\venv\\Scripts\\Activate.ps1 (Windows)")
        print("  3. Install dependencies: pip install -r requirements.txt")
        print("  4. Download dataset to data/raw/")
        print("  5. Start with Part 2 - Implementation\n")
        return 0
    else:
        print_colored("⚠ Some checks failed. Please review the missing items.", RED)
        return 1

if __name__ == "__main__":
    sys.exit(main())
