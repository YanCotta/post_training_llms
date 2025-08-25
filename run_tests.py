#!/usr/bin/env python3
"""
Test runner for the post_training_llms configuration system.

This script runs all tests for the unified configuration system
and provides a summary of the results.
"""

import sys
import subprocess
from pathlib import Path


def run_tests():
    """Run all tests for the configuration system."""
    print("🚀 Running Configuration System Tests")
    print("=" * 50)
    
    # Check if pytest is available
    try:
        import pytest
    except ImportError:
        print("❌ pytest is not installed. Please install it first:")
        print("   pip install pytest")
        return 1
    
    # Get the tests directory
    tests_dir = Path(__file__).parent / "tests"
    if not tests_dir.exists():
        print("❌ Tests directory not found. Please ensure tests/ exists.")
        return 1
    
    # Run the tests
    print(f"📁 Running tests from: {tests_dir}")
    print()
    
    try:
        # Run pytest with verbose output
        result = subprocess.run([
            sys.executable, "-m", "pytest",
            str(tests_dir),
            "-v",
            "--tb=short",
            "--strict-markers",
            "--disable-warnings"
        ], capture_output=True, text=True)
        
        # Print test output
        if result.stdout:
            print(result.stdout)
        
        if result.stderr:
            print("⚠️  Test warnings/errors:")
            print(result.stderr)
        
        # Print summary
        print("\n" + "=" * 50)
        if result.returncode == 0:
            print("🎉 All tests passed successfully!")
            print("✅ Configuration system is working correctly.")
        else:
            print("❌ Some tests failed.")
            print(f"Exit code: {result.returncode}")
        
        return result.returncode
        
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return 1


def run_specific_test(test_name):
    """Run a specific test by name."""
    print(f"🧪 Running specific test: {test_name}")
    print("=" * 50)
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest",
            f"tests/test_{test_name}.py",
            "-v",
            "--tb=short"
        ], capture_output=True, text=True)
        
        if result.stdout:
            print(result.stdout)
        
        if result.stderr:
            print("⚠️  Test warnings/errors:")
            print(result.stderr)
        
        return result.returncode
        
    except Exception as e:
        print(f"❌ Error running test {test_name}: {e}")
        return 1


def main():
    """Main entry point for the test runner."""
    if len(sys.argv) > 1:
        # Run specific test
        test_name = sys.argv[1]
        return run_specific_test(test_name)
    else:
        # Run all tests
        return run_tests()


if __name__ == "__main__":
    sys.exit(main())
