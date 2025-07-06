import os
import re
import json
import logging
from typing import ClassVar, Optional, Any, Callable, Dict, List
import datetime
import asyncio
import time
from functools import wraps
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
from azure.core.exceptions import HttpResponseError

from semantic_kernel.contents import ChatHistory
from semantic_kernel.agents.strategies import (
    SequentialSelectionStrategy,
    DefaultTerminationStrategy,
)

from semantic_kernel.agents.strategies.selection.selection_strategy import (
    SelectionStrategy,
)

try:
    from .simple_agentic_retrieval import SimpleAgenticRetrieval

    AGENTIC_RETRIEVAL_AVAILABLE = True
except ImportError as e:
    # Handle Azure AI package import failures gracefully
    import logging

    logging.error(f"Azure AI agentic retrieval is not available: {e}")
    AGENTIC_RETRIEVAL_AVAILABLE = False
    AGENTIC_RETRIEVAL_ERROR = str(e)
# Commented out tracing to avoid Azure AI package import issues
# from opentelemetry.trace import get_tracer

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

        # Multi model setup for cost optimization - a service is an LLM in SK terms
        #
        # Why two models?
        # - Executor (gpt-4o): High-capability model for research agents and complex reasoning
        # - Utility (gpt-4o-mini): Cost-effective model for background tasks like status updates
        #
        # This dual-model approach reduces costs by using the expensive model only for
        # actual research tasks while using the cheaper model for descriptive/utility functions.
        #
        # Executor - gpt-4o (primary research and agent conversations)
        # Utility  - gpt-4o-mini (status updates, progress descriptions)
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

        # Create the simple agentic retrieval plugin
        if AGENTIC_RETRIEVAL_AVAILABLE:
            self.logger.info("[DebateOrchestrator] Creating SimpleAgenticRetrieval...")
            try:
                self.agentic_plugin = SimpleAgenticRetrieval()
                self.logger.info(
                    "[DebateOrchestrator] SimpleAgenticRetrieval created successfully"
                )
            except Exception as e:
                self.logger.error(
                    f"[DebateOrchestrator] Failed to create SimpleAgenticRetrieval: {e}"
                )
                self.agentic_plugin = None
        else:
            self.logger.error(
                "[DebateOrchestrator] Azure AI agentic retrieval is not available"
            )
            self.logger.error(f"[DebateOrchestrator] Error: {AGENTIC_RETRIEVAL_ERROR}")
            self.logger.error(
                "[DebateOrchestrator] The API will not function properly without Azure AI packages"
            )
            self.agentic_plugin = None

        # Create kernel with or without agentic plugin
        plugins = []
        if self.agentic_plugin:
            plugins.append(
                KernelPlugin.from_object(
                    plugin_instance=self.agentic_plugin, plugin_name="agenticSearch"
                )
            )
        else:
            self.logger.warning(
                "[DebateOrchestrator] No agentic search plugin available - agents will not be able to search"
            )

        self.kernel = Kernel(
            services=[executor_service, utility_service],
            plugins=plugins,
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
        self,
        user_id: str,
        conversation_messages: List[Dict[str, Any]],
        maximum_iterations: int = 3,
    ):
        """
        Processes a conversation by orchestrating interactions between Research KB specialist agents.

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

        def clean_message_for_json(message_dict: Dict[str, Any]) -> Dict[str, Any]:
            """
            Create a clean, JSON-serializable version of a message dictionary.
            """
            return {
                "role": message_dict.get("role", "assistant"),
                "name": message_dict.get("name", "unknown"),
                "content": message_dict.get("content", ""),
                # Add any other simple fields you need, but avoid complex objects
            }

        try:
            # Create the agent group chat with specialized research agents
            self.agent_group_chat = self.create_agent_group_chat(
                agents_directory="agents/research",
                maximum_iterations=maximum_iterations,
            )

            # Extract user query
            user_query = None
            for msg in conversation_messages:
                if msg.get("role") == "user":
                    user_query = msg.get("content")

            if not user_query:
                self.logger.warning("No user query found in conversation messages")
                user_query = "Please help me with my research question"

            # Set the user query on the agentic retrieval plugin for context
            if self.agentic_plugin:
                self.agentic_plugin.set_current_query(user_query)
                self.logger.info(f"Set user query on agentic plugin: {user_query}")
            else:
                self.logger.warning(
                    "No agentic plugin available - search functionality will not work"
                )

            # Format user message for add_chat_messages
            user_messages = [
                ChatMessageContent(
                    role=AuthorRole(m.get("role")),
                    name=m.get("name"),
                    content=m.get("content"),
                )
                for m in conversation_messages
                if m.get("role") == "user"
            ]

            # If we have any user messages, add them to the chat
            if user_messages:
                try:
                    await self.agent_group_chat.add_chat_messages(user_messages)
                    self.logger.info(
                        f"Added {len(user_messages)} user messages to chat"
                    )
                except Exception as e:
                    self.logger.warning(f"Error adding chat messages: {e}")

            # Setup OpenTelemetry tracing - commented out to avoid import issues
            # tracer = get_tracer(__name__)

            # Create a unique session ID for tracing purposes
            current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
            session_id = f"{user_id}-{current_time}"

            # Track all messages exchanged during this conversation
            messages: List[Dict[str, Any]] = []

            # List to store clean debate messages for the response
            clean_debate_transcript: List[Dict[str, Any]] = []

            # Track iterations to prevent infinite loops
            iteration_count = 0

            # Store the final response during the conversation
            final_response = None
            research_agents: List[str] = []

            # Store all retrieval references from the conversation
            all_references: Dict[str, Any] = {}

            # Start the conversation session (tracing commented out)
            # with tracer.start_as_current_span(session_id):

            # Initial status message
            yield "Processing your research query..."

            # Get agent names for later use
            research_agents = [
                agent.name
                for agent in self.agent_manager.get_all_agents()
                if "Research" in agent.name
            ]

            # Process each message in the conversation with rate limit protection
            message_count = 0
            async for agent_message in self.agent_group_chat.invoke():
                # Add progressive delay between messages to prevent rate limits
                if message_count > 0:
                    await asyncio.sleep(0.5)  # 500ms delay between agent messages
                message_count += 1

                # Log the message
                msg_dict = agent_message.to_dict()
                self.logger.info("Agent: %s", msg_dict)

                # Add to messages collection
                message_dict = agent_message.to_dict()
                messages.append(message_dict)

                # Add clean version to debate transcript
                clean_message = clean_message_for_json(message_dict)
                clean_debate_transcript.append(clean_message)

                # Check for search errors and interrupt API only if we don't have a successful response yet
                message_content = message_dict.get("content", "")
                if any(
                    error_phrase in message_content
                    for error_phrase in [
                        "I cannot retrieve information from the knowledge base",
                        "Azure AI packages not properly installed",
                        "Missing required environment variables",
                        "AZURE_AI_FOUNDRY_PROJECT",
                        "AZURE_AI_SEARCH_ENDPOINT",
                    ]
                ):
                    self.logger.error(
                        f"Search error detected in agent response: {message_content}"
                    )
                    # Only return error immediately if we don't have a successful research response yet
                    if not final_response:
                        error_response = {
                            "role": "assistant",
                            "content": "I cannot retrieve information from the knowledge base at this time. Please check the search service configuration.",
                            "error": "Search service unavailable",
                            "name": agent_message.name or "ResearchAgent",
                        }
                        yield json.dumps(error_response)
                        return  # Exit the function immediately
                    else:
                        # We have a successful response, so just log the error and continue to return the successful result
                        self.logger.info(
                            "Search error occurred after successful research completion - continuing with successful results"
                        )
                        break  # Exit the conversation loop to return the successful response

                # Store potential final response from research agents
                # Only store non-APPROVED/REJECTED messages as final response
                if (
                    agent_message.name in research_agents
                    and not message_content.startswith("APPROVED")
                    and not message_content.startswith("REJECTED")
                    and "## Research Summary" in message_content
                ):
                    final_response = clean_message_for_json(message_dict)

                # Increment iteration count
                iteration_count += 1

                # Generate descriptive status for the client
                try:
                    next_action = await describe_next_action(
                        self.kernel, self.settings_utility, messages
                    )
                    self.logger.info("%s", next_action)
                except Exception as e:
                    self.logger.warning(f"Failed to describe next action: {e}")
                    next_action = "Processing research..."

                # Yield status update
                yield f"{next_action}"

                # Check for termination conditions
                if (
                    "APPROVED:" in next_action
                    or "FINAL:" in next_action
                    or "Solution complete" in next_action
                    or iteration_count >= maximum_iterations
                ):
                    self.logger.info(
                        f"Conversation terminating: {next_action} (iteration {iteration_count})"
                    )
                    break

                # Safety check - prevent infinite loops
                if iteration_count >= maximum_iterations * 2:
                    self.logger.warning(
                        f"Force terminating after {iteration_count} iterations"
                    )
                    break

            # Find the actual research response (not APPROVED messages)
            if not final_response:
                # Look for ResearchAgent messages with actual content
                for msg in reversed(messages):
                    if (
                        msg.get("name") == "ResearchAgent"
                        and msg.get("content")
                        and not msg["content"].startswith("APPROVED")
                        and "## Research Summary" in msg["content"]
                    ):
                        final_response = clean_message_for_json(msg)
                        break

                # If still no response, look for any non-APPROVED assistant message
                if not final_response:
                    for msg in reversed(messages):
                        if (
                            msg.get("role") == "assistant"
                            and msg.get("content")
                            and not msg["content"].startswith("APPROVED")
                            and not msg["content"].startswith("REJECTED")
                        ):
                            final_response = clean_message_for_json(msg)
                            break

                # Ultimate fallback
                if not final_response:
                    final_response = {
                        "role": "assistant",
                        "content": "I wasn't able to generate a complete response. Please try again with more specific requirements about your research question.",
                        "name": "ResearchAgent",
                    }

            # Add the clean transcript to the final response
            final_response["debate_transcript"] = clean_debate_transcript

            # Collect all sources from retrieval results following new pattern
            sources: List[Dict[str, Any]] = []
            if self.agentic_plugin and hasattr(
                self.agentic_plugin, "retrieval_results"
            ):
                citation_counter = 1
                processed_docs = set()

                # Define file extensions that qualify for citations
                valid_extensions = [
                    ".pdf",
                    ".docx",
                    ".doc",
                    ".txt",
                    ".xlsx",
                    ".xls",
                    ".pptx",
                    ".ppt",
                    ".md",
                    ".html",
                    ".htm",
                ]

                # Process all retrieval results
                for (
                    message_id,
                    retrieval_result,
                ) in self.agentic_plugin.retrieval_results.items():
                    try:
                        # Check if retrieval_result has references attribute
                        if (
                            hasattr(retrieval_result, "references")
                            and retrieval_result.references
                        ):
                            for reference in retrieval_result.references:
                                # Extract document information
                                doc_dict = (
                                    reference.as_dict()
                                    if hasattr(reference, "as_dict")
                                    else reference
                                )
                                doc_title = doc_dict.get(
                                    "document_title"
                                ) or doc_dict.get("title", "")

                                # Only process documents with valid file extensions
                                if doc_title and any(
                                    doc_title.lower().endswith(ext)
                                    for ext in valid_extensions
                                ):
                                    # Only add each unique document once
                                    if doc_title not in processed_docs:
                                        processed_docs.add(doc_title)

                                        # Extract content snippet
                                        content = doc_dict.get(
                                            "content_text", ""
                                        ) or doc_dict.get("content", "")
                                        content_snippet = (
                                            content[:200] + "..."
                                            if len(content) > 200
                                            else content
                                        )

                                        # Generate document URL
                                        document_url = ""
                                        if doc_title.endswith(".pdf") and hasattr(
                                            self.agentic_plugin,
                                            "_generate_blob_sas_url",
                                        ):
                                            document_url = self.agentic_plugin._generate_blob_sas_url(
                                                doc_title
                                            )
                                        elif doc_dict.get(
                                            "content_path", ""
                                        ).startswith("http"):
                                            document_url = doc_dict.get(
                                                "content_path", ""
                                            )

                                        # Get relevance score from Azure AI Search
                                        relevance_score = doc_dict.get(
                                            "@search.reranker_score",
                                            doc_dict.get(
                                                "@search.score",
                                                doc_dict.get("relevance_score", 0),
                                            ),
                                        )

                                        sources.append(
                                            {
                                                "citation_number": citation_counter,
                                                "title": doc_title,
                                                "filename": doc_title,
                                                "content_snippet": content_snippet,
                                                "relevance_score": relevance_score,
                                                "content_type": "image"
                                                if "images/"
                                                in doc_dict.get("content_path", "")
                                                else "text",
                                                "document_url": document_url,
                                            }
                                        )
                                        citation_counter += 1

                        # Also check response attribute for inline sources
                        elif (
                            hasattr(retrieval_result, "response")
                            and retrieval_result.response
                        ):
                            # The response might contain source information
                            self.logger.info("Checking response attribute for sources")
                            # This is a fallback - sources should be in references

                    except Exception as e:
                        self.logger.error(f"Error processing retrieval result: {e}")
                        continue

            # Log source collection results
            unique_docs = len(processed_docs) if "processed_docs" in locals() else 0
            self.logger.info(
                f"Collected {len(sources)} unique document citations from {unique_docs} total unique documents"
            )

            # Add comprehensive research summary
            if sources:
                research_summary = (
                    f"Research identified {len(sources)} unique document sources with valid filenames. "
                    f"Sources include {', '.join(set(source['title'].split('.')[-1].upper() for source in sources))} files "
                    f"covering multiple aspects of {user_query or 'the topic'}."
                )
            else:
                research_summary = "No documents with valid filenames were found for this research topic."

            # Build enhanced final response
            final_response["sources"] = sources
            final_response["total_sources_found"] = len(sources)
            final_response["research_summary"] = research_summary
            final_response["research_topic"] = user_query or "Unknown topic"

            # Final message is formatted as JSON to indicate the final response
            yield json.dumps(final_response)

        except Exception as e:
            # Log the error
            self.logger.error(f"Error in process_conversation: {str(e)}", exc_info=True)

            # Return a user-friendly error message
            error_response = {
                "role": "assistant",
                "content": "I encountered an issue while processing your request. Please try again with a more specific research question.",
                "error": str(e),
                "name": "ResearchAgent",
            }
            yield json.dumps(error_response)
