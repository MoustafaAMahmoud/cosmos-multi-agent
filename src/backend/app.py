"""
FastAPI backend application for blog post generation using AI debate orchestration.

This module initializes a FastAPI application that exposes endpoints for generating
blog posts using a debate pattern orchestrator, with appropriate logging, tracing,
and metrics configurations.
"""

# Apply Azure AI compatibility patches before any other imports
try:
    import azure_ai_patch
except:
    pass

import json
import logging
import os
from fastapi import FastAPI, Body
from fastapi.responses import StreamingResponse
from patterns.debate import DebateOrchestrator
from utils.util import (
    load_dotenv_from_azd,
    set_up_tracing,
    set_up_metrics,
    set_up_logging,
)
from fastapi.middleware.cors import CORSMiddleware

load_dotenv_from_azd()
set_up_tracing()
set_up_metrics()
set_up_logging()

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:   %(name)s   %(message)s",
)
logger = logging.getLogger(__name__)
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(
    logging.WARNING
)
logging.getLogger("azure.monitor.opentelemetry.exporter.export").setLevel(
    logging.WARNING
)

# Choose patterns to use
debate_orchestrator = DebateOrchestrator()

app = FastAPI()
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
logger.info(
    "Diagnostics: %s",
    os.getenv("SEMANTICKERNEL_EXPERIMENTAL_GENAI_ENABLE_OTEL_DIAGNOSTICS"),
)

import uuid
from typing import Dict, List, Optional, Any
from pydantic import BaseModel


# Define request and response models
class ResearchRequest(BaseModel):
    question: str = "Help me with my research question"
    user_id: str = "default_user"
    include_debate_details: bool = False
    maximum_iterations: int = 10  # Optional parameter to control conversation length


class DebateStatusUpdate(BaseModel):
    type: str = "status"
    message: str
    agent: Optional[str] = None


class DebateMessage(BaseModel):
    role: str
    name: Optional[str] = None
    content: str


class DebateResponse(BaseModel):
    type: str = "response"
    final_answer: Dict[str, Any]
    debate_details: Optional[List[DebateMessage]] = None


orchestrator = DebateOrchestrator()


@app.post("/api/v1/research-support")
async def http_research_support(request_body: ResearchRequest = Body(...)):
    """
    Process a research query using the debate orchestrator.

    Args:
        request_body (ResearchRequest): Request body containing:
            - question (str): The user's research question
            - user_id (str): Identifier for the user making the request
            - include_debate_details (bool): Whether to include full debate transcript
            - maximum_iterations (int): Maximum number of agent conversation turns

    Returns:
        StreamingResponse: A streaming response with status updates and final answer.
        Each chunk is a JSON object with a "type" field:
        - "status": Status updates during processing
        - "response": Final response with answer and optional debate details
    """
    logger.info("Research support request received: %s", request_body.dict())

    # Generate a unique conversation ID if not provided
    user_id = request_body.user_id or f"user_{uuid.uuid4()}"

    # Create conversation message
    conversation_messages = [
        {"role": "user", "name": "user", "content": request_body.question}
    ]

    # Store all debate messages if details are requested
    debate_messages = [] if request_body.include_debate_details else None

    async def stream_response():
        """
        Asynchronous generator that streams debate orchestrator responses.

        Yields:
            JSON strings for status updates and final response
        """
        # Create a fresh agent group chat for this conversation
        # This ensures we're using the right agents for this specific question
        orchestrator = DebateOrchestrator()

        async for chunk in orchestrator.process_conversation(
            user_id,
            conversation_messages,
            maximum_iterations=request_body.maximum_iterations,
        ):
            # If the chunk is JSON, it's the final response
            if chunk.startswith("{"):
                try:
                    final_response = json.loads(chunk)

                    # Extract the actual research content from the debate transcript
                    # The final response might have the wrong content (APPROVED/REJECTED/REVIEW message)
                    # So we need to find the actual research response
                    if "debate_transcript" in final_response:
                        # Find the research agent's actual response (not critic messages)
                        research_content = None
                        for msg in reversed(final_response["debate_transcript"]):  # Search from latest
                            if (msg.get("name") == "ResearchAgent" and 
                                msg.get("content") and 
                                not msg["content"].startswith("APPROVED") and
                                not msg["content"].startswith("REJECTED") and
                                not msg["content"].startswith("REVIEW") and
                                not msg["content"].startswith("CONTINUE_RESEARCH") and
                                "## Research Summary" in msg["content"]):
                                research_content = msg["content"]
                                break
                        
                        # Also check if the main content is a critic review and needs replacement
                        main_content = final_response.get("content", "")
                        if (main_content.startswith("REVIEW") or 
                            main_content.startswith("APPROVED") or 
                            main_content.startswith("REJECTED") or
                            "REVIEW RESULT:" in main_content):
                            # This is a critic message, replace with research content
                            if research_content:
                                final_response["content"] = research_content
                            else:
                                # Fallback: find any ResearchAgent message with content
                                for msg in reversed(final_response["debate_transcript"]):
                                    if (msg.get("name") == "ResearchAgent" and 
                                        msg.get("content") and
                                        len(msg["content"]) > 100):  # Substantial content
                                        final_response["content"] = msg["content"]
                                        break
                        elif research_content and "## Research Summary" not in main_content:
                            # Update if we have better research content
                            final_response["content"] = research_content

                    # Construct the final response object
                    response_obj = {"type": "response", "final_answer": final_response}

                    # Add debate details if requested
                    if request_body.include_debate_details and "debate_transcript" in final_response:
                        response_obj["debate_details"] = final_response["debate_transcript"]

                    yield json.dumps(response_obj) + "\n"
                except json.JSONDecodeError:
                    # If JSON parsing fails, treat it as a status update
                    status = {"type": "status", "message": chunk}
                    yield json.dumps(status) + "\n"
            else:
                # This is a status update
                # Parse the agent from the status (if in "AGENT: message" format)
                agent = None
                message = chunk
                if ": " in chunk:
                    parts = chunk.split(": ", 1)
                    if len(parts) == 2:
                        agent, message = parts

                status = {"type": "status", "message": message, "agent": agent}
                yield json.dumps(status) + "\n"

    return StreamingResponse(stream_response(), media_type="application/json")