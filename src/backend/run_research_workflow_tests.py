#!/usr/bin/env python3
"""
Test runner for Research Workflow Process Framework tests.
"""

import asyncio
import sys
import logging
from tests.test_run_research_workflow import TestRunResearchWorkflow

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


async def main():
    """Run all research workflow tests."""
    print("🚀 Dedicated Test Suite for run_research_workflow Function")
    print("=" * 70)

    # Create test instance
    test_instance = TestRunResearchWorkflow()

    # Run all tests
    passed, failed = await test_instance.run_all_tests()

    # Summary
    print("\n" + "=" * 70)
    print("📊 TEST SUMMARY")
    print("=" * 70)
    print(f"Total Tests: {passed + failed}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(
        f"Success Rate: {(passed / (passed + failed) * 100):.1f}%"
        if (passed + failed) > 0
        else "0%"
    )

    if failed == 0:
        print(
            "\n🎉 All tests passed! The run_research_workflow function is working correctly."
        )
        sys.exit(0)
    else:
        print(
            f"\n❌ {failed} test(s) failed. Please check the run_research_workflow implementation."
        )
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
