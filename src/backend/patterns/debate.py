import os
import re
import json
import logging
from typing import ClassVar
import datetime
from utils.util import describe_next_action
from patterns.agent_manager import AgentManager

from semantic_kernel.kernel import Kernel
from semantic_kernel.agents import AgentGroupChat
from semantic_kernel.exceptions.agent_exceptions import AgentChatException
from semantic_kernel.agents.strategies.termination.termination_strategy import (
    TerminationStrategy,
)
from semantic_kernel.agents.strategies import KernelFunctionSelectionStrategy
from semantic_kernel.connectors.ai.open_ai import AzureChatPromptExecutionSettings

from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.contents.utils.author_role import AuthorRole
from semantic_kernel.core_plugins.time_plugin import TimePlugin
from semantic_kernel.functions import (
    KernelPlugin,
    KernelFunctionFromPrompt,
    KernelArguments,
)

from semantic_kernel.connectors.ai.azure_ai_inference import (
    AzureAIInferenceChatCompletion,
)
from azure.ai.inference.aio import ChatCompletionsClient
from azure.identity.aio import DefaultAzureCredential

from semantic_kernel.contents import ChatHistory
from semantic_kernel.agents.strategies import (
    SequentialSelectionStrategy,
    DefaultTerminationStrategy,
)

from semantic_kernel.agents.strategies.selection.selection_strategy import (
    SelectionStrategy,
)
from .search_plugin import AzureSearchPlugin
from .research_workflow import ResearchWorkflow
from .research_workflow_plugin import ResearchWorkflowPlugin
from opentelemetry.trace import get_tracer

from pydantic import Field
########################################


# This pattern demonstrates how a debate between equally skilled models
# can deliver an outcome that exceeds the capability of the model if
# the task is handled as a single request-response in its entirety.
# We focus each agent on the subset of the whole task and thus
# get better results.
class DebateOrchestrator:
    """
    Orchestrates a debate between AI agents to produce higher quality responses.

    This class sets up and manages a conversation between Writer and Critic agents using
    Semantic Kernel's Agent Group Chat functionality. The debate pattern improves response
    quality by allowing specialized agents to focus on different aspects of the task.
    """

    # --------------------------------------------
    # Constructor
    # --------------------------------------------
    def __init__(self):
        """
        Creates the DebateOrchestrator with necessary services and kernel configurations.

        Sets up Azure OpenAI connections for both executor and utility models,
        configures Semantic Kernel, and prepares execution settings for the agents.
        """

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.info("Semantic Orchestrator Handler init")

        self.logger.info("Creating - %s", os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"))

        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION")
        executor_deployment_name = os.getenv("EXECUTOR_AZURE_OPENAI_DEPLOYMENT_NAME")
        utility_deployment_name = os.getenv("UTILITY_AZURE_OPENAI_DEPLOYMENT_NAME")

        credential = DefaultAzureCredential()

        # Multi model setup - a service is an LLM in SK terms
        # Executor - gpt-4o
        # Utility  - gpt-4o-mini
        executor_service = AzureAIInferenceChatCompletion(
            ai_model_id="executor",
            service_id="executor",
            client=ChatCompletionsClient(
                endpoint=f"{str(endpoint).strip('/')}/openai/deployments/{executor_deployment_name}",
                api_version=api_version,
                credential=credential,
                credential_scopes=["https://cognitiveservices.azure.com/.default"],
            ),
        )

        utility_service = AzureAIInferenceChatCompletion(
            ai_model_id="utility",
            service_id="utility",
            client=ChatCompletionsClient(
                endpoint=f"{str(endpoint).strip('/')}/openai/deployments/{utility_deployment_name}",
                api_version=api_version,
                credential=credential,
                credential_scopes=["https://cognitiveservices.azure.com/.default"],
            ),
        )

        self.kernel = Kernel(
            services=[executor_service, utility_service],
            plugins=[
                KernelPlugin.from_object(
                    plugin_instance=AzureSearchPlugin(), plugin_name="azureSearch"
                ),
                KernelPlugin.from_object(
                    plugin_instance=ResearchWorkflowPlugin(),
                    plugin_name="researchWorkflow",
                ),
            ],
        )

        self.settings_executor = AzureChatPromptExecutionSettings(
            service_id="executor", temperature=0
        )
        self.settings_utility = AzureChatPromptExecutionSettings(
            service_id="utility", temperature=0
        )

        self.resourceGroup = os.getenv("AZURE_RESOURCE_GROUP")

        # Create the agent manager
        self.agent_manager = AgentManager(self.kernel, service_id="executor")

        # Create the research workflow
        self.research_workflow = ResearchWorkflow(self.kernel)

    # --------------------------------------------
    # Create Agent Group Chat
    # --------------------------------------------
    def create_agent_group_chat(
        self, agents_directory="agents/research", maximum_iterations=3
    ):
        """
        Creates and configures an agent group chat with Writer and Critic agents.

        Returns:
            AgentGroupChat: A configured group chat with specialized agents,
                           selection strategy and termination strategy.
        """

        self.logger.debug("Creating chat")

        # Load all agents from directory
        agents = self.agent_manager.load_agents_from_directory(agents_directory)

        if not agents:
            raise ValueError(f"No agents found in {agents_directory}")

        # Get critics for termination strategy
        critics = self.agent_manager.get_critics()
        if not critics:
            self.logger.warning(
                "No critic agents found. Using default termination strategy."
            )
            # Find any agent named "Critic-Team" if is_critic wasn't specified
            for agent in agents:
                if "critic" in agent.name.lower():
                    critics.append(agent)
                    self.logger.info(f"Using {agent.name} as critic based on name")

        # Create agent group chat with all loaded agents
        agent_group_chat = AgentGroupChat(
            agents=agents,
            selection_strategy=SequentialSelectionStrategy(),
            termination_strategy=DefaultTerminationStrategy(
                maximum_iterations=maximum_iterations
            ),
        )

        return agent_group_chat

    async def process_conversation(
        self, user_id, conversation_messages, maximum_iterations=3
    ):
        """
        Processes a conversation by orchestrating interactions between Cosmos DB specialist agents.

        Manages the entire conversation flow from initialization to response collection, uses OpenTelemetry
        for tracing, and provides status updates throughout the conversation.

        Args:
            user_id: Unique identifier for the user, used in session tracking.
            conversation_messages: List of dictionaries with role, name and content
                                representing the conversation history.
            maximum_iterations: Maximum number of conversation turns.

        Yields:
            Status updates during processing and the final response in JSON format.
        """

        try:
            # Extract user query
            user_query = None
            for msg in conversation_messages:
                if msg.get("role") == "user":
                    user_query = msg.get("content")

            if not user_query:
                self.logger.warning("No user query found in conversation messages")
                user_query = "Tell me about Azure Cosmos DB"

            # Setup OpenTelemetry tracing
            tracer = get_tracer(__name__)

            # Create a unique session ID for tracing purposes
            current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
            session_id = f"{user_id}-{current_time}"

            # Start the traced conversation session
            with tracer.start_as_current_span(session_id):
                # Initial status message
                yield "Evaluating your research question..."

                # Status updates for workflow steps
                yield "Starting comprehensive research analysis..."

                yield "Searching scientific literature databases..."

                yield "Processing research findings..."

                yield "Adding analytical insights..."

                yield "Validating research quality..."

                yield "Finalizing comprehensive research summary..."

                # Run the research workflow
                try:
                    research_result = (
                        await self.research_workflow.run_research_workflow(user_query)
                    )

                    # Create final response structure
                    final_response = {
                        "role": "assistant",
                        "content": research_result,
                        "name": "ResearchWorkflow",
                        "debate_transcript": [],  # Empty for now since we're using workflow
                    }

                except Exception as e:
                    self.logger.error(f"Error in research workflow: {str(e)}")
                    final_response = {
                        "role": "assistant",
                        "content": f"I encountered an issue while processing your research request: {str(e)}. Please try again with a more specific question.",
                        "name": "ResearchWorkflow",
                        "debate_transcript": [],
                    }

            # Final message is formatted as JSON to indicate the final response
            yield json.dumps(final_response)

        except Exception as e:
            # Log the error
            self.logger.error(f"Error in process_conversation: {str(e)}", exc_info=True)

            # Return a user-friendly error message
            error_response = {
                "role": "assistant",
                "content": "I encountered an issue while processing your request. Please try again with a more specific question about Azure Cosmos DB.",
                "error": str(e),
            }
            yield json.dumps(error_response)
