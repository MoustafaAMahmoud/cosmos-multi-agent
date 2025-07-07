"""
Utility module for Azure AI Accelerator application.

This module provides helper functions for:
- Environment configuration
- OpenTelemetry setup for observability (tracing, metrics, and logging)
- Agent creation from YAML definitions
- Workflow utilities for agent interactions
"""

from io import StringIO
from subprocess import run, PIPE
import os
import logging
from dotenv import load_dotenv
import yaml

# Commented out OpenTelemetry imports to avoid Azure AI package conflicts
# from opentelemetry.sdk.resources import Resource
# from opentelemetry._logs import set_logger_provider
# from opentelemetry.metrics import set_meter_provider
# from opentelemetry.trace import set_tracer_provider

# from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
# from opentelemetry.sdk._logs.export import (
#     BatchLogRecordProcessor,
#     # ConsoleLogExporter
# )
# from opentelemetry.sdk.metrics import MeterProvider
# from opentelemetry.sdk.metrics.view import DropAggregation, View
# from opentelemetry.sdk.metrics.export import (
#     PeriodicExportingMetricReader,
#     # ConsoleMetricExporter
# )
# from opentelemetry.sdk.trace import TracerProvider
# from opentelemetry.sdk.trace.export import (
#     BatchSpanProcessor,
#     # ConsoleSpanExporter
# )
# from opentelemetry.semconv.resource import ResourceAttributes

# from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
# from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
# from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

# from azure.monitor.opentelemetry.exporter import (
#     AzureMonitorLogExporter,
#     AzureMonitorMetricExporter,
#     AzureMonitorTraceExporter,
# )

from semantic_kernel.connectors.ai.function_choice_behavior import (
    FunctionChoiceBehavior,
)
from semantic_kernel.connectors.ai.open_ai import AzureChatPromptExecutionSettings

from semantic_kernel.functions import KernelArguments
from semantic_kernel.agents import ChatCompletionAgent


def load_dotenv_from_azd():
    """
    Loads environment variables from Azure Developer CLI (azd) or .env file.

    Attempts to load environment variables using the azd CLI first.
    If that fails, falls back to loading from a .env file in the current directory.
    """
    result = run("azd env get-values", stdout=PIPE, stderr=PIPE, shell=True, text=True)
    if result.returncode == 0:
        logging.info(f"Found AZD environment. Loading...")
        load_dotenv(stream=StringIO(result.stdout))
    else:
        logging.info(f"AZD environment not found. Trying to load from .env file...")
        load_dotenv()


# Commented out telemetry resource to avoid import conflicts
# telemetry_resource = Resource.create(
#     {
#         ResourceAttributes.SERVICE_NAME: os.getenv(
#             "AZURE_RESOURCE_GROUP", "ai-accelerator"
#         )
#     }
# )

# Set endpoint to the local Aspire Dashboard endpoint to enable local telemetry - DISABLED by default
local_endpoint = None
# local_endpoint = "http://localhost:4317"


def set_up_tracing():
    """
    Sets up exporters for Azure Monitor and optional local telemetry.
    COMMENTED OUT: Tracing disabled to avoid Azure AI package import conflicts.
    """
    logging.info("Tracing setup skipped (commented out to avoid import conflicts)")
    pass
    
    # Original tracing setup commented out:
    # if not os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING"):
    #     logging.info(
    #         "APPLICATIONINSIGHTS_CONNECTION_STRING is not set skipping observability setup."
    #     )
    #     return
    #
    # exporters = []
    # exporters.append(
    #     AzureMonitorTraceExporter.from_connection_string(
    #         os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")
    #     )
    # )
    # if local_endpoint:
    #     exporters.append(OTLPSpanExporter(endpoint=local_endpoint))
    #
    # tracer_provider = TracerProvider(resource=telemetry_resource)
    # for trace_exporter in exporters:
    #     tracer_provider.add_span_processor(BatchSpanProcessor(trace_exporter))
    # set_tracer_provider(tracer_provider)


def set_up_metrics():
    """
    Configures metrics collection with OpenTelemetry.
    COMMENTED OUT: Metrics disabled to avoid Azure AI package import conflicts.
    """
    logging.info("Metrics setup skipped (commented out to avoid import conflicts)")
    pass


def set_up_logging():
    """
    Configures logging with OpenTelemetry.
    COMMENTED OUT: Logging setup disabled to avoid Azure AI package import conflicts.
    """
    logging.info("OpenTelemetry logging setup skipped (commented out to avoid import conflicts)")
    pass


async def describe_next_action(kernel, settings, messages):
    """
    Determines the next action in the research agent conversation workflow.

    Args:
        kernel: The Semantic Kernel instance
        settings: Execution settings for the prompt
        messages: Conversation history between agents

    Returns:
        str: A brief summary of the next action, indicating which research agent is acting
    """
    # Get the last message to determine which agent just spoke
    last_message = messages[-1] if messages else {"name": "None"}
    last_agent = last_message.get("name", "Unknown")

    next_action = await kernel.invoke_prompt(
        function_name="describe_next_action",
        prompt=f"""
        Given the following conversation between research agents, describe the next action.

        Provide a brief summary (3-5 words) of what's happening next in the format: "AGENT: Action description"

        AGENTS:
        - ResearchAgent: Performs intelligent agentic retrieval and research analysis
        - CompanyResearchAgent: Searches company knowledge base for research
        - Critic-Team: Evaluates completeness of research solution

        If the last message is from Critic-Team with a score of 8 or higher, respond with "APPROVED: Solution complete"
        If a complete research solution has been reached, respond with "FINAL: Complete research provided"

        Last agent to speak: {last_agent}

        CONVERSATION HISTORY: {messages[-3:] if len(messages) >= 3 else messages}
        """,
        settings=settings,
    )

    return next_action
