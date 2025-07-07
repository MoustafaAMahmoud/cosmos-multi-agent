# Consolidated imports
import os
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
from azure.search.documents.agent.models import (
    KnowledgeAgentRetrievalRequest,
    KnowledgeAgentMessage,
    KnowledgeAgentMessageTextContent,
    KnowledgeAgentIndexParams,
)
from azure.search.documents.indexes import SearchIndexClient


def load_environment_variables():
    """Load and return environment variables as a configuration dictionary."""
    load_dotenv(override=True)

    return {
        "project_endpoint": os.environ["AZURE_AI_FOUNDRY_PROJECT"],
        "agent_model": os.getenv("AGENT_MODEL"),
        "search_endpoint": os.getenv("AZURE_AI_SEARCH_ENDPOINT"),
        "search_api_key": os.getenv("AZURE_AI_SEARCH_API_KEY"),
        "search_service_name": os.getenv("AZURE_AI_SEARCH_SERVICE_NAME"),
        "index_name": os.getenv("AZURE_SEARCH_INDEX_NAME"),
        "semantic_config": os.getenv("AZURE_SEARCH_SEMANTIC_CONFIG"),
        "openai_endpoint": os.environ["AZURE_OPENAI_ENDPOINT"],
        "azure_openai_key": os.getenv("AZURE_OPENAI_API_KEY"),
        "azure_openai_gpt_deployment": os.getenv("AZURE_OPENAI_GPT_DEPLOYMENT"),
        "azure_openai_gpt_model": os.getenv("AZURE_OPENAI_GPT_MODEL"),
        "embedding_deployment": os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
        "embedding_model": os.getenv("AZURE_OPENAI_EMBEDDING_MODEL"),
        "agent_name": os.getenv("AZURE_SEARCH_AGENT_NAME"),
        "agent_instructions": os.getenv("AGENT_INSTRUCTIONS"),
    }


def get_credentials():
    """Initialize and return Azure credentials."""
    credential = DefaultAzureCredential()
    search_credential = AzureKeyCredential(os.environ["AZURE_AI_SEARCH_API_KEY"])
    return credential, search_credential


def attach_agent_to_knowledgebase(
    agent_name: str,
    openai_config: Dict[str, Any],
    index_name: str,
    search_endpoint: str,
    search_credential: Any,
    reranker_threshold: Optional[float] = 2.5,
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
        reranker_threshold: Optional reranker threshold (default: 2.5)

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
            reranker_threshold=3.0
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
                index_name=index_name, default_reranker_threshold=reranker_threshold
            )
        ],
        request_limits=KnowledgeAgentRequestLimits(),
    )

    # Create or update the agent
    index_client = SearchIndexClient(
        endpoint=search_endpoint, credential=search_credential
    )
    index_client.create_or_update_agent(agent)

    print(f"Knowledge agent '{agent_name}' created or updated successfully")
    print(f"  - Attached to index: {index_name}")
    print(f"  - Reranker threshold: {reranker_threshold}")
    print(f"  - Using model: {openai_config['model_name']}")


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


# Main execution - Chat interactions
def main():
    """Main function - everything happens here."""

    # Load configuration
    config = load_environment_variables()
    credential, search_credential = get_credentials()

    # Prepare OpenAI configuration
    openai_config = {
        "endpoint": config["openai_endpoint"],
        "deployment_name": config["azure_openai_gpt_deployment"],
        "model_name": config["azure_openai_gpt_model"],
        "api_key": config["azure_openai_key"],
    }

    # Create AI agent
    project_client, ai_agent = create_ai_agent(
        agent_name=config["agent_name"],
        agent_model=config["agent_model"],
        model_instructions=config["agent_instructions"],
        azure_foundry_project_endpoint=config["project_endpoint"],
        azure_foundry_project_credential=credential,
    )

    # Attach agent to knowledge base
    attach_agent_to_knowledgebase(
        agent_name=config.get("agent_name"),
        openai_config=openai_config,
        index_name=config["index_name"],
        search_endpoint=config["search_endpoint"],
        search_credential=search_credential,
        reranker_threshold=2.5,
    )


    agent_client = KnowledgeAgentRetrievalClient(
        endpoint=config["search_endpoint"],
        agent_name=config["agent_name"],
        credential=search_credential,
    )
    print(f"Created search client with endpoint: {config['search_endpoint']}")

    # Create thread
    thread = project_client.agents.threads.create()
    print(f"Created thread: {thread.id}")

    # Storage for retrieval results
    retrieval_results = {}

    def agentic_retrieval() -> str:
        """
        Search function in Azure AI Agentic Search Index.
        The returned string is in a JSON format that contains the reference id.
        Be sure to use the same format in the agent's response
        You must refer to references by id number
        """
        # Take the last 5 messages in the conversation
        messages = project_client.agents.messages.list(
            thread.id, limit=5, order=ListSortOrder.DESCENDING
        )
        # Reverse the order so the most recent message is last
        messages = list(messages)
        messages.reverse()
        retrieval_result = agent_client.retrieve(
            retrieval_request=KnowledgeAgentRetrievalRequest(
                messages=[
                    KnowledgeAgentMessage(
                        role=msg["role"],
                        content=[
                            KnowledgeAgentMessageTextContent(text=msg.content[0].text)
                        ],
                    )
                    for msg in messages
                    if msg["role"] != "system"
                ],
                target_index_params=[
                    KnowledgeAgentIndexParams(
                        index_name=config["index_name"], reranker_threshold=0.2
                    )
                ],
            )
        )

        # Associate the retrieval results with the last message in the conversation
        last_message = messages[-1]
        retrieval_results[last_message.id] = retrieval_result

        # Return the grounding response to the agent
        return retrieval_result.response[0].content[0].text

    # Setup function tools
    functions = FunctionTool({agentic_retrieval})
    toolset = ToolSet()
    toolset.add(functions)
    project_client.agents.enable_auto_function_calls(toolset)

    # First conversation
    message = project_client.agents.messages.create(
        thread_id=thread.id,
        role="user",
        content=""" Describe how genetic algorithms are used in the optimization process            
        """,
    )

    run = project_client.agents.runs.create_and_process(
        thread_id=thread.id,
        agent_id=ai_agent.id,
        tool_choice=AgentsNamedToolChoice(
            type=AgentsNamedToolChoiceType.FUNCTION,
            function=FunctionName(name="agentic_retrieval"),
        ),
        toolset=toolset,
    )
    if run.status == "failed":
        raise RuntimeError(f"Run failed: {run.last_error}")
    output = project_client.agents.messages.get_last_message_text_by_role(
        thread_id=thread.id, role="assistant"
    ).text.value

    print("Agent response:", output.replace(".", "\n"))

    retrieval_result = retrieval_results.get(message.id)
    if retrieval_result is None:
        raise RuntimeError(f"No retrieval results found for message {message.id}")

    print("Retrieval activity")
    print(
        json.dumps(
            [activity.as_dict() for activity in retrieval_result.activity], indent=2
        )
    )
    print("Retrieval results")
    print(
        json.dumps(
            [reference.as_dict() for reference in retrieval_result.references], indent=2
        )
    )


if __name__ == "__main__":
    main()
