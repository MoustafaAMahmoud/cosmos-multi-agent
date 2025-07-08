# file: create_agent_demo.py
"""
Create (or update) a Knowledge Agent and attach it to an Azure AI Search index.

Prerequisites
-------------
pip install azure-search-documents==11.6.0b12 python-dotenv

Required environment variables
------------------------------
# Search service + admin key
AZURE_AI_SEARCH_ENDPOINT   = https://<service>.search.windows.net
AZURE_AI_SEARCH_API_KEY    = <ADMIN key – 52 chars>
AZURE_SEARCH_INDEX_NAME    = <existing index to attach the agent to>

# OpenAI deployment used by the agent (same region / subscription not required)
AZURE_OPENAI_ENDPOINT          = https://<openai-resource>.openai.azure.com
AZURE_OPENAI_API_KEY           = <OpenAI key>
AZURE_OPENAI_GPT_DEPLOYMENT    = <deployment-name e.g. gpt-4o>
AZURE_OPENAI_GPT_MODEL         = <model id     e.g. gpt-4o>

Run it
------
python create_agent_demo.py --name my-agent
"""

import argparse
import os
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    KnowledgeAgent,
    KnowledgeAgentTargetIndex,
    KnowledgeAgentRequestLimits,
    KnowledgeAgentAzureOpenAIModel,
    AzureOpenAIVectorizerParameters,
)

load_dotenv(override=True)
# --------------------------------------------------------------------------- #
# 1) read mandatory configuration once
# --------------------------------------------------------------------------- #
ENDPOINT = os.getenv("AZURE_AI_SEARCH_ENDPOINT")
API_KEY = os.getenv("AZURE_AI_SEARCH_API_KEY")
INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME")

OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
OPENAI_KEY = os.getenv("AZURE_OPENAI_API_KEY")
OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_GPT_DEPLOYMENT")
OPENAI_MODEL = os.getenv("AZURE_OPENAI_GPT_MODEL")

for var in (
    "ENDPOINT",
    "API_KEY",
    "INDEX_NAME",
    "OPENAI_ENDPOINT",
    "OPENAI_KEY",
    "OPENAI_DEPLOYMENT",
    "OPENAI_MODEL",
):
    if globals()[var] is None:
        raise EnvironmentError(
            f"set {var.replace('_', ' ')} before running this script"
        )

credential = AzureKeyCredential(API_KEY)


# --------------------------------------------------------------------------- #
# 2) main helper – create or update the agent
# --------------------------------------------------------------------------- #
def create_agent(name: str) -> None:
    """
    Create or update a Knowledge Agent (requires 2025-05-01-preview API).
    """
    client = SearchIndexClient(
        endpoint=ENDPOINT,
        credential=credential,
        api_version="2025-05-01-preview",  # **mandatory for agents**
    )

    agent = KnowledgeAgent(
        name=name,
        models=[
            KnowledgeAgentAzureOpenAIModel(
                azure_open_ai_parameters=AzureOpenAIVectorizerParameters(
                    resource_url=OPENAI_ENDPOINT,
                    deployment_name=OPENAI_DEPLOYMENT,
                    model_name=OPENAI_MODEL,
                    api_key=OPENAI_KEY,
                )
            )
        ],
        target_indexes=[
            KnowledgeAgentTargetIndex(
                index_name=INDEX_NAME,
                default_reranker_threshold=0.2,
            )
        ],
        request_limits=KnowledgeAgentRequestLimits(),  # default limits
    )

    # create_or_update is idempotent – run it as often as you like
    result = client.create_or_update_agent(agent)
    print(f"✓ Knowledge Agent '{result.name}' attached to '{INDEX_NAME}'")


# --------------------------------------------------------------------------- #
# 3) very small CLI wrapper
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create / update a Knowledge Agent")
    parser.add_argument(
        "--name", default="demo-agent", help="agent name (default: demo-agent)"
    )
    args = parser.parse_args()

    create_agent(args.name)
