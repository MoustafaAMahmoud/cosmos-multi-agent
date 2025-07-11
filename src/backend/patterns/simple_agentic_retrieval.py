# Consolidated imports
import os
import logging
import json
from typing import Dict, Any, Optional, Tuple
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.core.credentials import AzureKeyCredential
from azure.ai.projects import AIProjectClient
from azure.ai.agents.models import (
    FunctionTool,
    ToolSet,
    ListSortOrder,
    AgentsNamedToolChoice,
    AgentsNamedToolChoiceType,
    FunctionName,
)
from azure.search.documents.agent import KnowledgeAgentRetrievalClient
from azure.search.documents.agent.models import (
    KnowledgeAgentRetrievalRequest,
    KnowledgeAgentMessage,
    KnowledgeAgentMessageTextContent,
    KnowledgeAgentIndexParams,
)
from azure.search.documents.indexes.models import (
    KnowledgeAgent,
    KnowledgeAgentAzureOpenAIModel,
    KnowledgeAgentTargetIndex,
    KnowledgeAgentRequestLimits,
    AzureOpenAIVectorizerParameters,
)
from azure.ai.agents.models import FunctionTool, ToolSet, ListSortOrder

from azure.search.documents.agent import KnowledgeAgentRetrievalClient
from azure.search.documents.indexes import SearchIndexClient
from semantic_kernel.functions import kernel_function


def load_environment_variables():
    """Load and return environment variables as a configuration dictionary."""
    load_dotenv(override=True)

    return {
        "project_endpoint": os.environ.get("AZURE_AI_FOUNDRY_PROJECT"),
        "agent_model": os.getenv("AGENT_MODEL"),
        "search_endpoint": os.getenv("AZURE_AI_SEARCH_ENDPOINT"),
        "search_api_key": os.getenv("AZURE_AI_SEARCH_API_KEY"),
        "search_service_name": os.getenv("AZURE_AI_SEARCH_SERVICE_NAME"),
        "index_name": os.getenv("AZURE_SEARCH_INDEX_NAME"),
        "semantic_config": os.getenv("AZURE_SEARCH_SEMANTIC_CONFIG"),
        "openai_endpoint": os.environ.get("AZURE_OPENAI_ENDPOINT"),
        "azure_openai_key": os.getenv("AZURE_OPENAI_API_KEY"),
        "azure_openai_gpt_deployment": os.getenv("AZURE_OPENAI_GPT_DEPLOYMENT"),
        "azure_openai_gpt_model": os.getenv("AZURE_OPENAI_GPT_MODEL"),
        "embedding_deployment": os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
        "embedding_model": os.getenv("AZURE_OPENAI_EMBEDDING_MODEL"),
        "agent_name": os.getenv("AZURE_SEARCH_AGENT_NAME"),
        "agent_instructions": os.getenv("AGENT_INSTRUCTIONS"),
    }


def check_environment(config: Dict[str, Any]):
    """Check if required environment variables are set."""
    required_vars = [
        "project_endpoint",
        "search_endpoint",
    ]

    missing_vars = [var for var in required_vars if not config.get(var)]
    if missing_vars:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing_vars)}"
        )


def get_credentials():
    """Initialize and return Azure credentials."""
    credential = DefaultAzureCredential()
    search_credential = AzureKeyCredential(os.environ["AZURE_AI_SEARCH_API_KEY"])
    return credential, search_credential


def create_ai_agent(
    agent_name: str,
    agent_model: str,
    model_instructions: str,
    azure_foundry_project_endpoint: str,
    azure_foundry_project_credential: Any,
) -> Tuple[AIProjectClient, Any]:
    """
    Create an Azure AI Agent for research Q&A and conversational tasks.

    Args:
        agent_name: Unique name for the AI agent (e.g., 'research-qa-agent')

        agent_model: Model identifier for the agent.  (e.g., 'gpt-4.1')

        model_instructions: System instructions that define the agent's behavior.
            Example: "You are a helpful research assistant that provides accurate answers."

        azure_foundry_project_endpoint: The Azure AI Foundry project endpoint URL.
            Format: 'https://{project-name}.services.ai.azure.com/api/projects/{project-name}"'


        azure_foundry_project_credential: Authentication credential for Azure.
            - DefaultAzureCredential() for managed identity

    Returns:
        Tuple[AIProjectClient, Agent]: A tuple containing:
            - project_client: The AIProjectClient instance for further operations
            - agent: The created agent object with properties (id, name, model, etc.)

    Raises:
        ValueError: If required parameters are invalid
        Exception: For Azure service errors

    Example:
        ```python
        from azure.identity import DefaultAzureCredential

        project_client, agent = create_ai_agent(
            agent_name="research-assistant",
            agent_model="gpt-4.1",
            model_instructions="You are a helpful research assistant...",
            azure_foundry_project_endpoint="https://myproject.eastus.inference.ml.azure.com",
            azure_foundry_project_credential=DefaultAzureCredential()
        )
        ```
    """
    # Validate inputs
    if not agent_name:
        raise ValueError("agent_name must be provided")

    if not agent_model:
        raise ValueError("agent_model must be provided")

    if not model_instructions:
        raise ValueError("model_instructions must be provided")

    # Initialize the project client
    project_client = AIProjectClient(
        endpoint=azure_foundry_project_endpoint,
        credential=azure_foundry_project_credential,
    )

    # Create the agent
    agent = project_client.agents.create_agent(
        model=agent_model,
        name=agent_name,
        instructions=model_instructions,
    )

    print(f"AI agent '{agent_name}' created or updated successfully")
    return project_client, agent


def attach_agent_to_knowledgebase(
    agent_name: str,
    openai_config: Dict[str, Any],
    index_name: str,
    search_endpoint: str,
    search_credential: Any,
    reranker_threshold: Optional[float] = 1,
) -> None:
    """
    Attach an agent to a knowledge base using Azure AI Search.

    Args:
        agent_name: Name of the agent to create or update
        openai_config: Dictionary containing OpenAI configuration with keys:
            - 'endpoint': Azure OpenAI endpoint URL
            - 'deployment_name': Deployment name for the model
            - 'model_name': Model name (e.g., 'gpt-4')
            - 'api_key': API key for Azure OpenAI
        index_name: Target Azure AI Search index name
        search_credential: Credential object for Azure Search (e.g., AzureKeyCredential)
        reranker_threshold: Optional reranker threshold (default: 0.2)

    Example:
        openai_config = {
            'endpoint': 'https://your-resource.openai.azure.com/',
            'deployment_name': 'gpt-4-deployment',
            'model_name': 'gpt-4',
            'api_key': 'your-api-key'
        }

        from azure.core.credentials import AzureKeyCredential
        search_credential = AzureKeyCredential("your-search-api-key")

        attach_agent_to_knowledgebase(
            agent_name="my-knowledge-agent",
            openai_config=openai_config,
            index_name="my-search-index",
            search_credential=search_credential,
            reranker_threshold=0.5
        )
    """
    # Load environment config for search endpoint

    if not search_endpoint:
        raise ValueError("AZURE_AI_SEARCH_ENDPOINT environment variable is not set")

    # Validate OpenAI config
    required_keys = ["endpoint", "deployment_name", "model_name", "api_key"]
    for key in required_keys:
        if key not in openai_config:
            raise ValueError(f"Missing required key '{key}' in openai_config")

    # Create the knowledge agent
    agent = KnowledgeAgent(
        name=agent_name,
        models=[
            KnowledgeAgentAzureOpenAIModel(
                azure_open_ai_parameters=AzureOpenAIVectorizerParameters(
                    resource_url=openai_config["endpoint"],
                    deployment_name=openai_config["deployment_name"],
                    model_name=openai_config["model_name"],
                    api_key=openai_config["api_key"],
                )
            )
        ],
        target_indexes=[
            KnowledgeAgentTargetIndex(
                index_name=index_name,
                default_reranker_threshold=reranker_threshold,
                default_include_reference_source_data=True,
                default_max_docs_for_reranker=200,  # Adjust as needed
            )
        ],
        request_limits=KnowledgeAgentRequestLimits(),
    )

    # Create or update the agent
    index_client = SearchIndexClient(
        endpoint=search_endpoint,
        credential=search_credential,
        api_version="2025-05-01-preview",  # **mandatory for agents**
    )
    index_client.create_or_update_agent(agent)

    print(f"Knowledge agent '{agent_name}' created or updated successfully")
    print(f"  - Attached to index: {index_name}")
    print(f"  - Reranker threshold: {reranker_threshold}")
    print(f"  - Using model: {openai_config['model_name']}")


class SimpleAgenticRetrieval:
    """
    Simple Azure AI agentic retrieval implementation following main.py pattern.

    This class provides a clean interface for Azure AI Search with agentic
    capabilities while supporting Semantic Kernel integration.
    """

    def __init__(self):
        """Initialize the retrieval client following main.py pattern."""
        self.logger = logging.getLogger(__name__)

        # Load configuration
        self.config = load_environment_variables()
        check_environment(self.config)

        # Initialize credentials
        self.credential, self.search_credential = get_credentials()
        # Prepare OpenAI configuration
        self.openai_config = {
            "endpoint": self.config["openai_endpoint"],
            "deployment_name": self.config["azure_openai_gpt_deployment"],
            "model_name": self.config["azure_openai_gpt_model"],
            "api_key": self.config["azure_openai_key"],
        }

        # Initialize project client (following main.py)
        self.project_client = AIProjectClient(
            endpoint=self.config["project_endpoint"], credential=self.credential
        )

        # Create AI agent
        self.project_client, self.ai_agent = create_ai_agent(
            agent_name=self.config["agent_name"],
            agent_model=self.config["agent_model"],
            model_instructions=self.config["agent_instructions"],
            azure_foundry_project_endpoint=self.config["project_endpoint"],
            azure_foundry_project_credential=self.credential,
        )

        # Attach agent to knowledge base
        attach_agent_to_knowledgebase(
            agent_name=self.config.get("agent_name"),
            openai_config=self.openai_config,
            index_name=self.config["index_name"],
            search_endpoint=self.config["search_endpoint"],
            search_credential=self.search_credential,
            reranker_threshold=1,
        )

        # Initialize search/agent client (following main.py pattern)
        self.agent_client = KnowledgeAgentRetrievalClient(
            endpoint=self.config["search_endpoint"],
            agent_name=self.config.get("agent_name", "ai-search-agent"),
            credential=self.search_credential,
        )
        self.logger.info(
            f"Created search client with endpoint: {self.config['search_endpoint']}"
        )

        # Create conversation thread
        self.thread = self.project_client.agents.threads.create()
        self.logger.info(f"Created thread: {self.thread.id}")

        # Storage for retrieval results (following main.py)
        self.retrieval_results = {}

        # Current query context for Semantic Kernel
        self.current_query = None

    def set_current_query(self, query: str):
        """Set the current user query for context-aware retrieval."""
        self.current_query = query
        self.logger.info(f"[SimpleAgenticRetrieval] Set current query: {query}")

    def add_user_message(self, content: str):
        """Add a user message to the conversation thread (following main.py pattern)."""
        try:
            message = self.project_client.agents.messages.create(
                thread_id=self.thread.id,
                role="user",
                content=content,
            )
            self.logger.info(f"Created message, ID: {message.id}")
            return message
        except Exception as e:
            self.logger.error(f"Failed to add user message: {e}")
            return None

    def get_recent_messages(self, limit: int = 5):
        """Get recent messages from the conversation thread (following main.py pattern)."""
        try:
            messages = self.project_client.agents.messages.list(
                self.thread.id, limit=limit, order=ListSortOrder.DESCENDING
            )
            messages = list(messages)
            messages.reverse()
            self.logger.info(f"Retrieved {len(messages)} messages from thread")
            return messages
        except Exception as e:
            self.logger.error(f"Failed to get recent messages: {e}")
            return []

    def _perform_agentic_retrieval(self, messages: list) -> Any:
        """
        Perform the actual agentic retrieval following main.py pattern.

        Args:
            messages: List of messages from the thread

        Returns:
            Retrieval result object
        """
        # Format messages for agentic retrieval (following main.py)
        formatted_messages = [
            KnowledgeAgentMessage(
                role=msg["role"],
                content=[KnowledgeAgentMessageTextContent(text=msg.content[0].text)],
            )
            for msg in messages
            if msg["role"] != "system"
        ]
        max_subqueries = 10
        max_docs_for_reranker = max_subqueries * 50
        # Perform retrieval (following main.py)
        retrieval_result = self.agent_client.retrieve(
            retrieval_request=KnowledgeAgentRetrievalRequest(
                messages=formatted_messages,
                target_index_params=[
                    KnowledgeAgentIndexParams(
                        index_name=self.config["index_name"],
                        reranker_threshold=0.2,
                        include_reference_source_data=True,
                        max_docs_for_reranker=max_docs_for_reranker,
                    )
                ],
            )
        )

        return retrieval_result

    def search(self, query: str, max_results: int = 20) -> str:
        """
        Perform agentic search following main.py pattern.

        Args:
            query: Search query
            max_results: Maximum number of results (kept for compatibility)

        Returns:
            Search results as formatted text
        """
        try:
            self.logger.info(f"Starting agentic search for: {query}")

            # Add the query as a user message if needed
            if query and query != self.current_query:
                self.add_user_message(query)

            # Get recent messages (following main.py pattern)
            messages = self.get_recent_messages(limit=5)

            # If no messages, create one with the query
            if not messages and query:
                message = self.add_user_message(query)
                if message:
                    messages = [message]

            if not messages:
                return "No messages available for search"

            # Perform retrieval
            retrieval_result = self._perform_agentic_retrieval(messages)

            # Store results (following main.py pattern)
            last_message = messages[-1]
            self.retrieval_results[last_message.id] = retrieval_result
            self.logger.info(f"Stored results for message: {last_message.id}")

            # Return response (following main.py pattern)
            response = retrieval_result.response[0].content[0].text
            self.logger.info("Search completed successfully")

            return response

        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            import traceback

            traceback.print_exc()

            if "Forbidden" in str(e):
                return "Search failed due to authentication/authorization error. Please check Azure credentials and permissions."
            else:
                return f"Search failed: {str(e)}"

    # Semantic Kernel integration functions
    @kernel_function(
        name="agentic_retrieval",
        description="Perform intelligent search using Azure AI following main.py pattern",
    )
    def agentic_retrieval(self, query: Optional[str] = None) -> str:
        """
        Kernel function for agentic retrieval.

        This function follows the exact pattern from main.py's agentic_retrieval function.
        """
        if not query:
            # If no query provided, use the last 5 messages like main.py
            messages = self.get_recent_messages(limit=5)
            if not messages:
                return "No messages available for retrieval"

            # Perform retrieval
            retrieval_result = self._perform_agentic_retrieval(messages)

            # Store results
            last_message = messages[-1]
            self.retrieval_results[last_message.id] = retrieval_result

            # Return response
            return retrieval_result.response[0].content[0].text
        else:
            # If query provided, use the search method
            return self.search(query)
