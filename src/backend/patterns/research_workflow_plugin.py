"""
Research Workflow Plugin using Semantic Kernel Process Framework.

This module implements an end-to-end research workflow that searches,
validates, and summarizes scientific literature using a structured process.
"""

import asyncio
import json
import logging
from enum import Enum
from typing import List, Set, Dict, Any
from pydantic import BaseModel, Field
from semantic_kernel.functions import kernel_function
from semantic_kernel.processes.process_builder import ProcessBuilder
from semantic_kernel.processes.kernel_process.kernel_process_step import (
    KernelProcessStep,
)
from semantic_kernel.processes.kernel_process.kernel_process_step_context import (
    KernelProcessStepContext,
)
from semantic_kernel.processes.kernel_process.kernel_process_step_state import (
    KernelProcessStepState,
)
from semantic_kernel.processes.local_runtime.local_event import KernelProcessEvent
from semantic_kernel.processes.local_runtime.local_kernel_process import start

logger = logging.getLogger(__name__)


# Data Models
class Paper(BaseModel):
    """Represents a research paper found during search."""

    id: str
    title: str
    content_snippet: str = ""
    relevance_score: float = 0.0
    document_url: str = ""


class Summary(BaseModel):
    """Represents a summarized research finding."""

    paper_id: str
    title: str
    summary_text: str
    document_url: str = ""
    citation_number: int = 0


class ResearchState(BaseModel):
    """State object that flows through the research process."""

    topic: str = ""
    search_query: str = ""
    found_papers: List[Paper] = Field(default_factory=list)
    reviewed_ids: Set[str] = Field(default_factory=set)
    summaries: List[Summary] = Field(default_factory=list)
    total_sources_found: int = 0
    research_summary: str = ""
    iteration_count: int = 0
    max_iterations: int = 3


class ResearchEvents(str, Enum):
    """Events that trigger transitions in the research workflow."""

    StartResearch = "startResearch"
    QueryPrepared = "queryPrepared"
    PapersFound = "papersFound"
    ValidationPassed = "validationPassed"
    ValidationFailed = "validationFailed"
    ResearchComplete = "researchComplete"


# Process Steps
class GetQueryStep(KernelProcessStep[ResearchState]):
    """Initial step that extracts and prepares the research query."""

    @staticmethod
    def activate(context: KernelProcessStepContext, state: ResearchState) -> None:
        logger.info(f"Starting research workflow for topic: {state.topic}")

        # Prepare initial search query
        state.search_query = f"scientific research on {state.topic}"
        state.iteration_count = 0

        # Emit event to proceed to search
        context.emit_event(
            KernelProcessEvent(id=ResearchEvents.QueryPrepared, data=state)
        )


class SearchPapersStep(KernelProcessStep[ResearchState]):
    """Step that searches for papers using Azure AI Search."""

    @staticmethod
    async def activate(context: KernelProcessStepContext, state: ResearchState) -> None:
        logger.info(f"Searching for papers with query: {state.search_query}")

        try:
            # Use the azureSearch plugin to find papers
            search_function = context.kernel.get_function("azureSearch", "search")
            search_results_json = await search_function.invoke(
                context.kernel, query=state.search_query
            )

            # Parse search results
            search_results = json.loads(str(search_results_json.value))

            if "error" in search_results:
                logger.error(f"Search failed: {search_results['error']}")
                context.emit_event(
                    KernelProcessEvent(id=ResearchEvents.ValidationFailed, data=state)
                )
                return

            # Convert search results to Paper objects
            new_papers = []
            results = search_results.get("results", [])

            for i, result in enumerate(results):
                paper_id = result.get("chunk_id", f"paper_{i}")

                # Skip if already reviewed
                if paper_id in state.reviewed_ids:
                    continue

                paper = Paper(
                    id=paper_id,
                    title=result.get("title", f"Document {i + 1}"),
                    content_snippet=result.get("chunk", "")[:300] + "...",
                    relevance_score=result.get("@search.score", 0.0),
                    document_url=result.get("parent_id", ""),
                )

                new_papers.append(paper)
                state.reviewed_ids.add(paper_id)

            state.found_papers.extend(new_papers)
            state.total_sources_found = len(state.found_papers)

            logger.info(
                f"Found {len(new_papers)} new papers, total: {state.total_sources_found}"
            )

            # Emit event to validate results
            context.emit_event(
                KernelProcessEvent(id=ResearchEvents.PapersFound, data=state)
            )

        except Exception as e:
            logger.error(f"Error in search step: {str(e)}")
            context.emit_event(
                KernelProcessEvent(id=ResearchEvents.ValidationFailed, data=state)
            )


class ValidateResultsStep(KernelProcessStep[ResearchState]):
    """Step that validates if we have sufficient research results."""

    @staticmethod
    def activate(context: KernelProcessStepContext, state: ResearchState) -> None:
        logger.info(f"Validating results: {len(state.found_papers)} papers found")

        state.iteration_count += 1

        # Check if we have sufficient results or reached max iterations
        min_papers = 3
        has_sufficient_papers = len(state.found_papers) >= min_papers
        reached_max_iterations = state.iteration_count >= state.max_iterations

        if has_sufficient_papers or reached_max_iterations:
            logger.info("Validation passed - proceeding to summarization")
            context.emit_event(
                KernelProcessEvent(id=ResearchEvents.ValidationPassed, data=state)
            )
        else:
            logger.info("Validation failed - need more papers, searching again")
            # Modify search query for next iteration
            state.search_query = f"{state.topic} research findings studies analysis"
            context.emit_event(
                KernelProcessEvent(id=ResearchEvents.ValidationFailed, data=state)
            )


class SummarizeResultsStep(KernelProcessStep[ResearchState]):
    """Final step that creates summaries and research output."""

    @staticmethod
    def activate(context: KernelProcessStepContext, state: ResearchState) -> None:
        logger.info(f"Summarizing {len(state.found_papers)} papers")

        # Create summaries for each paper
        summaries = []
        for i, paper in enumerate(state.found_papers, 1):
            summary = Summary(
                paper_id=paper.id,
                title=paper.title,
                summary_text=paper.content_snippet,
                document_url=paper.document_url,
                citation_number=i,
            )
            summaries.append(summary)

        state.summaries = summaries

        # Create research summary
        state.research_summary = f"Scientific research on {state.topic} based on {len(summaries)} sources from the research index."

        logger.info("Research workflow completed successfully")

        # Emit completion event
        context.emit_event(
            KernelProcessEvent(id=ResearchEvents.ResearchComplete, data=state)
        )


class ResearchWorkflowPlugin:
    """Main plugin class that orchestrates the research workflow."""

    @kernel_function(
        name="run_research",
        description="Run the end-to-end literature research workflow; returns structured JSON with research findings.",
    )
    async def run_research(self, context: KernelProcessStepContext, topic: str) -> str:
        """
        Execute the complete research workflow for a given topic.

        Args:
            context: Kernel process step context
            topic: Research topic to investigate

        Returns:
            JSON string containing research summary, sources, and total count
        """
        try:
            logger.info(f"Starting research workflow for topic: {topic}")

            # 1. Build the process
            process = ProcessBuilder[ResearchState]("ResearchWorkflow")

            # Add steps
            get_query_step = process.add_step(GetQueryStep)
            search_step = process.add_step(SearchPapersStep)
            validate_step = process.add_step(ValidateResultsStep)
            summarize_step = process.add_step(SummarizeResultsStep)

            # Define the workflow transitions
            process.on_input_event(ResearchEvents.StartResearch).send_event_to(
                get_query_step
            )
            get_query_step.on_event(ResearchEvents.QueryPrepared).send_event_to(
                search_step
            )
            search_step.on_event(ResearchEvents.PapersFound).send_event_to(
                validate_step
            )
            validate_step.on_event(ResearchEvents.ValidationPassed).send_event_to(
                summarize_step
            )
            validate_step.on_event(ResearchEvents.ValidationFailed).send_event_to(
                search_step
            )
            summarize_step.on_event(ResearchEvents.ResearchComplete).stop_process()

            # Build the kernel process
            kernel_process = process.build()

            # 2. Create initial state and event
            initial_state = ResearchState(topic=topic)
            initial_event = KernelProcessEvent(
                id=ResearchEvents.StartResearch, data=initial_state
            )

            # 3. Start the process
            await start(
                process=kernel_process,
                kernel=context.kernel,
                initial_event=initial_event,
            )

            # 4. Extract final state from the summarize step
            final_state = summarize_step.state

            # 5. Format the response as expected by agents
            sources = []
            for summary in final_state.summaries:
                sources.append(
                    {
                        "citation_number": summary.citation_number,
                        "title": summary.title,
                        "content_snippet": summary.summary_text,
                        "relevance_score": 1.0,  # Default relevance
                        "document_url": summary.document_url,
                    }
                )

            response = {
                "research_summary": final_state.research_summary,
                "sources": sources,
                "total_sources_found": final_state.total_sources_found,
            }

            logger.info(f"Research workflow completed with {len(sources)} sources")
            return json.dumps(response)

        except Exception as e:
            logger.error(f"Error in research workflow: {str(e)}")
            # Return error response
            error_response = {
                "research_summary": f"Error conducting research on {topic}: {str(e)}",
                "sources": [],
                "total_sources_found": 0,
            }
            return json.dumps(error_response)
