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
    select: str = "content_id,text_document_id,document_title,image_document_id,content_text,content_path,locationMetadata",
    k: int = 10,
    semantic_configuration: str = "research-agent-index-semantic-configuration",
    vector_field: str = "content_embedding",
    query_type: str = "semantic",
    query_language: str = "en-us",
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
    index_name = os.getenv("AZURE_SEARCH_INDEX_NAME", "research-agent-index")

    if not search_endpoint or not search_api_key:
        logger.error("Azure AI Search endpoint and API key must be set.")
        return None

    if not query or not query.strip():
        logger.error("Search query is required.")
        return None

    endpoint = f"{search_endpoint}/indexes/{index_name}/docs/search?api-version=2024-05-01-Preview"
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
        "count": True,  # Include total count
        "captions": "extractive",  # Include captions
        "answers": "extractive|count-3"  # Include semantic answers
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
        
        # Extract document titles from results
        results = data.get("value", [])
        document_titles = []
        
        for result in results:
            title = result.get("document_title")
            if title and title not in document_titles:  # Avoid duplicates
                document_titles.append(title)
        
        return {
            "total_count": data.get("@odata.count", len(results)),
            "results": results,
            "search_id": data.get("@search.searchId"),
            "semantic_answers": data.get("@search.answers", []),
            "document_titles": document_titles,  # New array of unique document titles
        }

    except Exception as e:
        logger.error(f"Exception during Azure AI Search: {str(e)}")
        return None


class AzureSearchPlugin:
    @kernel_function(
        name="search",
        description="Perform a semantic + vector search against the scientific research index",
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
