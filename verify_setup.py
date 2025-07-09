#!/usr/bin/env python3
"""
Quick verification script to check if the project structure is correctly set up.
"""

import os
import sys
from pathlib import Path

def check_project_structure():
    """Verify that all expected files and directories exist."""
    
    project_root = Path(__file__).parent
    print(f"Checking project structure at: {project_root.absolute()}")
    print("=" * 60)
    
    # Expected directories
    expected_dirs = [
        "src",
        "src/utils", 
        "src/training",
        "src/evaluation",
        "notebooks",
        "examples", 
        "configs",
        "data",
        "models"
    ]
    
    # Expected files
    expected_files = [
        "README.md",
        "requirements.txt",
        "setup.py",
        "CONTRIBUTING.md",
        ".gitignore",
        
        # Source files
        "src/__init__.py",
        "src/utils/__init__.py",
        "src/utils/model_utils.py",
        "src/utils/data_utils.py", 
        "src/training/__init__.py",
        "src/training/sft_trainer.py",
        "src/training/dpo_trainer.py",
        "src/training/rl_trainer.py",
        "src/evaluation/__init__.py",
        "src/evaluation/metrics.py",
        "src/evaluation/benchmark.py",
        
        # Examples
        "examples/run_sft.py",
        "examples/run_dpo.py", 
        "examples/run_rl.py",
        "examples/run_benchmark.py",
        
        # Notebooks
        "notebooks/01_supervised_fine_tuning.ipynb",
        "notebooks/02_direct_preference_optimization.ipynb",
        "notebooks/03_online_reinforcement_learning.ipynb",
        
        # Configs
        "configs/sft_config.yaml",
        "configs/dpo_config.yaml",
        "configs/rl_config.yaml"
    ]
    
    # Check directories
    print("📁 DIRECTORIES")
    print("-" * 30)
    missing_dirs = []
    for dir_path in expected_dirs:
        full_path = project_root / dir_path
        if full_path.exists() and full_path.is_dir():
            print(f"✅ {dir_path}")
        else:
            print(f"❌ {dir_path}")
            missing_dirs.append(dir_path)
    
    print(f"\n📄 FILES")
    print("-" * 30)
    missing_files = []
    for file_path in expected_files:
        full_path = project_root / file_path
        if full_path.exists() and full_path.is_file():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}")
            missing_files.append(file_path)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    
    total_dirs = len(expected_dirs)
    total_files = len(expected_files)
    found_dirs = total_dirs - len(missing_dirs)
    found_files = total_files - len(missing_files)
    
    print(f"Directories: {found_dirs}/{total_dirs} ({'✅' if missing_dirs == [] else '❌'})")
    print(f"Files: {found_files}/{total_files} ({'✅' if missing_files == [] else '❌'})")
    
    if missing_dirs or missing_files:
        print("\n⚠️  MISSING ITEMS:")
        for item in missing_dirs + missing_files:
            print(f"   - {item}")
        return False
    else:
        print("\n🎉 All expected files and directories are present!")
        print("📦 Project structure is correctly set up!")
        return True

def check_imports():
    """Test if core modules can be imported."""
    print("\n" + "=" * 60)
    print("🐍 IMPORT TESTING")
    print("=" * 60)
    
    # Add src to path
    src_path = Path(__file__).parent / "src"
    sys.path.insert(0, str(src_path))
    
    modules_to_test = [
        "utils.model_utils",
        "utils.data_utils", 
        "training.sft_trainer",
        "training.dpo_trainer",
        "training.rl_trainer",
        "evaluation.metrics",
        "evaluation.benchmark"
    ]
    
    success_count = 0
    for module in modules_to_test:
        try:
            __import__(module)
            print(f"✅ {module}")
            success_count += 1
        except ImportError as e:
            print(f"❌ {module} - {e}")
    
    print(f"\nImports: {success_count}/{len(modules_to_test)}")
    return success_count == len(modules_to_test)

def main():
    """Run all verification checks."""
    print("🔍 POST-TRAINING LLMS PROJECT VERIFICATION")
    print("=" * 60)
    print("This script verifies that the project structure is correctly set up")
    print("and that all core modules can be imported successfully.")
    print("=" * 60)
    
    structure_ok = check_project_structure()
    imports_ok = check_imports()
    
    print("\n" + "=" * 60)
    print("🏁 FINAL RESULT") 
    print("=" * 60)
    
    if structure_ok and imports_ok:
        print("🎉 SUCCESS! The project is correctly set up and ready to use.")
        print("\n📚 Next steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Try the notebooks in the 'notebooks/' directory")
        print("3. Run example scripts in the 'examples/' directory")
        print("4. Read the README.md for detailed usage instructions")
        return 0
    else:
        print("❌ ISSUES DETECTED! Please fix the missing components.")
        print("\n🔧 Common solutions:")
        print("1. Make sure you're running this from the project root")
        print("2. Check that all files were created correctly")
        print("3. Install required dependencies")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
