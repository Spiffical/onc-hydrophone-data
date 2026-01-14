#!/usr/bin/env python3
"""
Run all tests in the tests/ directory.

Usage:
    python tests/run_all_tests.py [--verbose]
"""

import os
import sys
import unittest
from pathlib import Path

# Add parent to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def run_all_tests(verbose=False):
    """Discover and run all tests."""
    print("\n" + "="*70)
    print("🧪 RUNNING ALL TESTS FOR onc-hydrophone-data")
    print("="*70 + "\n")
    
    # Get the tests directory
    tests_dir = Path(__file__).parent
    
    # Discover tests
    loader = unittest.TestLoader()
    suite = loader.discover(str(tests_dir), pattern='test_*.py')
    
    # Count tests
    test_count = suite.countTestCases()
    print(f"Discovered {test_count} tests\n")
    
    # Run tests
    verbosity = 2 if verbose else 1
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "="*70)
    print("📊 TEST SUMMARY")
    print("="*70)
    print(f"  Tests run: {result.testsRun}")
    print(f"  Failures:  {len(result.failures)}")
    print(f"  Errors:    {len(result.errors)}")
    print(f"  Skipped:   {len(result.skipped)}")
    
    if result.wasSuccessful():
        print("\n✅ ALL TESTS PASSED!")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED!")
        
        if result.failures:
            print("\nFailures:")
            for test, traceback in result.failures:
                print(f"  - {test}")
        
        if result.errors:
            print("\nErrors:")
            for test, traceback in result.errors:
                print(f"  - {test}")
        
        return 1


if __name__ == '__main__':
    verbose = '--verbose' in sys.argv or '-v' in sys.argv
    sys.exit(run_all_tests(verbose))
