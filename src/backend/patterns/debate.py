import os
import re
import json
import logging
from typing import ClassVar, Optional, Any, Callable
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
from azure.core.exceptions import HttpResponseError

from semantic_kernel.contents import ChatHistory
from semantic_kernel.agents.strategies import (
    SequentialSelectionStrategy,
    DefaultTerminationStrategy,
)

from semantic_kernel.agents.strategies.selection.selection_strategy import (
    SelectionStrategy,
)
from .agentic_retrieval_plugin import AgenticRetrievalPlugin
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

        # Create the agentic retrieval plugin and store reference
        self.logger.info("[DebateOrchestrator] Creating AgenticRetrievalPlugin...")
        try:
            self.agentic_plugin = AgenticRetrievalPlugin(kernel=None)
            self.logger.info("[DebateOrchestrator] AgenticRetrievalPlugin created successfully")
        except Exception as e:
            self.logger.error(f"[DebateOrchestrator] Failed to create AgenticRetrievalPlugin: {e}")
            self.agentic_plugin = None

        self.kernel = Kernel(
            services=[executor_service, utility_service],
            plugins=[
                KernelPlugin.from_object(
                    plugin_instance=TimePlugin(), plugin_name="time"
                ),
                KernelPlugin.from_object(
                    plugin_instance=self.agentic_plugin, plugin_name="agenticSearch"
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

        def clean_message_for_json(message_dict):
            """
            Create a clean, JSON-serializable version of a message dictionary.
            """
            return {
                "role": message_dict.get("role", "assistant"),
                "name": message_dict.get("name", "unknown"),
                "content": message_dict.get("content", ""),
                # Add any other simple fields you need, but avoid complex objects
            }

        # Define retry decorator for rate limit handling
        async def retry_with_backoff(
            func: Callable,
            max_retries: int = 3,
            initial_delay: float = 2.0,
            backoff_factor: float = 2.0,
            max_delay: float = 30.0
        ):
            """Retry function with exponential backoff for rate limit errors."""
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return await func()
                except Exception as e:
                    error_str = str(e)
                    # Check if it's a rate limit error (429)
                    if "429" in error_str or "rate limit" in error_str.lower() or "RateLimitError" in str(type(e)):
                        last_exception = e
                        if attempt < max_retries - 1:
                            # Calculate delay with exponential backoff
                            delay = min(initial_delay * (backoff_factor ** attempt), max_delay)
                            self.logger.warning(
                                f"Rate limit hit (attempt {attempt + 1}/{max_retries}). "
                                f"Retrying in {delay:.1f} seconds..."
                            )
                            await asyncio.sleep(delay)
                            continue
                    # Re-raise if not a rate limit error
                    raise e
            
            # If we exhausted all retries, raise the last exception
            if last_exception:
                raise last_exception

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

            # Setup OpenTelemetry tracing
            tracer = get_tracer(__name__)

            # Create a unique session ID for tracing purposes
            current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
            session_id = f"{user_id}-{current_time}"

            # Track all messages exchanged during this conversation
            messages = []

            # List to store clean debate messages for the response
            clean_debate_transcript = []

            # Track iterations to prevent infinite loops
            iteration_count = 0

            # Store the final response during the conversation
            final_response = None
            research_agents = []
            
            # Store all retrieval references from the conversation
            all_references = {}

            # Start the traced conversation session
            with tracer.start_as_current_span(session_id):
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

                    # Store potential final response from research agents
                    if agent_message.name in research_agents:
                        final_response = clean_message_for_json(message_dict)

                    # Increment iteration count
                    iteration_count += 1

                    # Generate descriptive status for the client with retry logic
                    try:
                        async def get_next_action():
                            return await describe_next_action(
                                self.kernel, self.settings_utility, messages
                            )
                        
                        # Use retry logic for the describe_next_action call
                        next_action = await retry_with_backoff(
                            get_next_action,
                            max_retries=2,
                            initial_delay=1.0
                        )
                        self.logger.info("%s", next_action)
                    except Exception as e:
                        self.logger.warning(f"Failed to describe next action after retries: {e}")
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

            # Use the final response we collected during the conversation
            if not final_response:
                # Fallback to the last assistant message
                assistant_messages = [
                    msg for msg in messages if msg.get("role") == "assistant"
                ]
                if assistant_messages:
                    final_response = clean_message_for_json(assistant_messages[-1])
                else:
                    # Ultimate fallback if no messages found
                    final_response = {
                        "role": "assistant",
                        "content": "I wasn't able to generate a complete response. Please try again with more specific requirements about your research question.",
                        "name": "ResearchAgent",
                    }

            # Add the clean transcript to the final response
            final_response["debate_transcript"] = clean_debate_transcript
            
            # Collect all sources from the agentic retrieval plugin using exhaustive search results
            sources = []
            if self.agentic_plugin and hasattr(self.agentic_plugin, 'retrieval_results'):
                citation_counter = 1
                processed_docs = set()
                
                # Get all cached retrieval results from exhaustive research
                for conversation_id, results in self.agentic_plugin.retrieval_results.items():
                    for result in results:
                        doc_title = result.get('document_title', 'Unknown Document')
                        
                        # Only add each document once
                        if doc_title not in processed_docs:
                            processed_docs.add(doc_title)
                            content_snippet = result.get('content_text', '')[:200] + "..." if len(result.get('content_text', '')) > 200 else result.get('content_text', '')
                            
                            # Generate SAS URL for PDF documents
                            document_url = ""
                            if doc_title.endswith('.pdf') and self.agentic_plugin:
                                document_url = self.agentic_plugin._generate_blob_sas_url(doc_title)
                            elif result.get('content_path', '') and result.get('content_path', '').startswith("http"):
                                document_url = result.get('content_path', '')
                            
                            # Create topic relation summary for this source
                            topic_relation_summary = ""
                            if user_query and self.agentic_plugin:
                                topic_relation_summary = self.agentic_plugin._create_source_summary(result, user_query)
                            
                            sources.append({
                                'citation_number': citation_counter,
                                'title': doc_title,
                                'filename': doc_title,
                                'content_snippet': content_snippet,
                                'topic_relation_summary': topic_relation_summary,
                                'relevance_score': result.get('@search.reranker_score', result.get('@search.score', 0)),
                                'content_type': 'image' if result.get('content_path', '') and 'images/' in result.get('content_path', '') else 'text',
                                'document_url': document_url
                            })
                            citation_counter += 1
            
            # Add comprehensive research summary
            if sources:
                research_summary = (
                    f"Comprehensive research identified {len(sources)} relevant sources covering "
                    f"multiple aspects of the topic. The research includes technical documentation, "
                    f"academic studies, and practical applications."
                )
            else:
                research_summary = "Limited sources were found for this research topic."
            
            # Add enhanced sources response for exhaustive research
            final_response["sources"] = sources
            final_response["source_count"] = len(sources)
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
            }
            yield json.dumps(error_response)
