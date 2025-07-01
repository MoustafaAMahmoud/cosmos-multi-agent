# File: src/backend/patterns/search_plugin.py

import os
import json
import logging
import requests
from typing import Dict, Any, Union

from dotenv import load_dotenv


# === UPDATED IMPORT ===
from semantic_kernel.functions import kernel_function


def azure_ai_search_plugin(
    query: str,
    select: str = "content_id,text_document_id,content_text,document_title,content_path",
    k: int = 10,
    semantic_configuration: str = "research-agent-index-semantic-configuration",
    vector_field: str = "content_embedding",
    query_type: str = "semantic",
    query_language: str = "en-GB",
    timeout: int = 30,
) -> Union[Dict[str, Any], None]:
    """
    Execute Azure AI Search with semantic + vector search, returning a Python dict
    with total_count, results, search_id, and semantic_answers.
    """
    load_dotenv()
    logger = logging.getLogger(__name__)
    logger.info(f"[azureSearchPlugin] Invoked with query: '{query}'")

    search_endpoint = os.getenv("AZURE_AI_SEARCH_ENDPOINT")
    search_api_key = os.getenv("AZURE_AI_SEARCH_API_KEY")
    service_name = os.getenv("AZURE_AI_SEARCH_SERVICE_NAME", "aifoundry")
    index_name = os.getenv("AZURE_SEARCH_INDEX_NAME", "research-agent-index")

    if not search_endpoint:
        logger.error("Azure AI Search endpoint must be set.")
        return None

    if not query or not query.strip():
        logger.error("Search query is required.")
        return None

    # Construct the proper AI Foundry search endpoint
    if "cognitiveservices.azure.com" in search_endpoint:
        endpoint = f"{search_endpoint.rstrip('/')}/searchservices/{service_name}/indexes/{index_name}/docs/search?api-version=2024-05-01-Preview"
    else:
        # Fallback to traditional search service format
        endpoint = f"{search_endpoint.rstrip('/')}/indexes/{index_name}/docs/search?api-version=2024-05-01-Preview"
    headers = {"Content-Type": "application/json", "api-key": search_api_key}
    payload = {
        "search": query,
        "select": select,
        "vectorQueries": [
            {
                "kind": "text",
                "text": query,
                "fields": vector_field,
                "k": k,
            }
        ],
        "queryType": query_type,
        "semanticConfiguration": semantic_configuration,
        "queryLanguage": query_language,
        "top": k,
    }

    try:
        logger.info(f"Running Azure AI Search for query: '{query}'")
        response = requests.post(
            endpoint, headers=headers, data=json.dumps(payload), timeout=timeout
        )

        if response.status_code != 200:
            logger.error(
                f"Search failed with status {response.status_code}: {response.text}"
            )
            return None

        data = response.json()
        return {
            "total_count": data.get("@odata.count", len(data.get("value", []))),
            "results": data.get("value", []),
            "search_id": data.get("@search.searchId"),
            "semantic_answers": data.get("@search.answers", []),
        }

    except Exception as e:
        logger.error(f"Exception during Azure AI Search: {str(e)}")
        return None


class AzureSearchPlugin:
    @kernel_function(
        name="search",
        description="Perform a semantic + vector search against the research agent index",
    )
    def search(self, query: str) -> str:
        """
        Calls azure_ai_search_plugin() and returns a JSON string.

        Agents can parse this JSON to pick out "results" or "semantic_answers".
        """
        logger = logging.getLogger(__name__)
        logger.info("[AzureSearchPlugin.search] About to call azure_ai_search_plugin()")
        results = azure_ai_search_plugin(query)
        if results is None:
            logger.warning(
                "[AzureSearchPlugin.search] azure_ai_search_plugin returned None"
            )
            return json.dumps({"error": "search_failed"})
        count = len(results.get("results", []))
        logger.info(f"[AzureSearchPlugin.search] Search returned {count} documents")
        return json.dumps(results)
