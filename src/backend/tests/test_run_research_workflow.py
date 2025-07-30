#!/usr/bin/env python3
"""
Simple end-to-end test for the run_research_workflow function with "vaping material" topic.
"""

import asyncio
import logging
import sys
import os

# Add the parent directory to the path so we can import from patterns
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from semantic_kernel.kernel import Kernel
from semantic_kernel.functions import KernelPlugin

from patterns.search_plugin import AzureSearchPlugin
from patterns.research_workflow import ResearchWorkflow


class TestRunResearchWorkflowEndToEnd:
    """Simple end-to-end test for run_research_workflow method."""

    def __init__(self):
        """Initialize test instance."""
        pass

    def create_workflow(self):
        """Create workflow with real kernel and search plugin."""
        kernel = Kernel()
        search_plugin = AzureSearchPlugin()
        kernel.add_plugin(
            KernelPlugin.from_object(
                plugin_instance=search_plugin, plugin_name="azureSearch"
            )
        )
        return ResearchWorkflow(kernel)

    async def test_vaping_material_research(self):
        """Test run_research_workflow with 'vaping material' topic for debugging."""
        print("🔬 Testing run_research_workflow with 'vaping material' topic")
        print("=" * 60)

        # Set up detailed logging
        logging.basicConfig(level=logging.INFO)
        workflow_logger = logging.getLogger("patterns.research_workflow")
        workflow_logger.setLevel(logging.DEBUG)

        try:
            # Create workflow
            workflow = self.create_workflow()

            # Test topic
            topic = "vaping material"

            print(f"🔍 Starting research for topic: '{topic}'")
            print("-" * 40)

            # Run the workflow
            result = await workflow.run_research_workflow(topic)

            print("\n" + "=" * 60)
            print("📊 FINAL RESULT:")
            print("=" * 60)
            print(result)
            print("\n" + "=" * 60)

            # Show final state for debugging
            if workflow.final_state:
                print("🔍 FINAL STATE DEBUG INFO:")
                print("=" * 60)
                print(f"Original Topic: {workflow.final_state.original_topic}")
                print(f"Current Query: {workflow.final_state.current_query}")
                print(f"Iteration Count: {workflow.final_state.iteration_count}")
                print(
                    f"Search Queries Used: {workflow.final_state.search_queries_used}"
                )
                print(
                    f"Total Papers Found: {len(workflow.final_state.accumulated_papers)}"
                )
                print(f"Seen Paper IDs: {len(workflow.final_state.seen_paper_ids)}")

                if workflow.final_state.accumulated_papers:
                    print("\n📄 PAPERS FOUND:")
                    for i, paper in enumerate(
                        workflow.final_state.accumulated_papers[:5]
                    ):  # Show first 5
                        print(f"  {i + 1}. {paper.title} (Score: {paper.search_score})")

                    if len(workflow.final_state.accumulated_papers) > 5:
                        print(
                            f"  ... and {len(workflow.final_state.accumulated_papers) - 5} more papers"
                        )

                print(
                    f"\n✅ Final Research Content Length: {len(workflow.final_state.final_research_content)} characters"
                )
            else:
                print("⚠️  No final state available")

            # Basic validation
            assert result is not None, "Result is None"
            assert isinstance(result, str), "Result is not a string"
            assert len(result) > 0, "Result is empty"

            print("\n✅ Test completed successfully!")
            return True

        except Exception as e:
            print(f"\n❌ Test failed with error: {e}")
            import traceback

            traceback.print_exc()
            return False

    async def run_all_tests(self):
        """Run the single end-to-end test."""
        print("🔬 Running Single End-to-End Test for 'vaping material'")
        print("=" * 60)

        passed = 0
        failed = 0

        try:
            result = await self.test_vaping_material_research()
            if result:
                passed = 1
            else:
                failed = 1
        except Exception as e:
            print(f"❌ Test failed: {e}")
            failed = 1

        print(f"\n📊 Test Results: {passed} passed, {failed} failed")
        return passed, failed


# Main execution
if __name__ == "__main__":

    async def main():
        test_instance = TestRunResearchWorkflowEndToEnd()
        await test_instance.run_all_tests()

    asyncio.run(main())
