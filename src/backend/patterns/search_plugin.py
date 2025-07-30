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
    select: str = "chunk_id, parent_id, title, chunk, content_embedding",
    k: int = 10,
    semantic_configuration: str = "research-agent-index-semantic",
    vector_field: str = "content_embedding",
    query_type: str = "semantic",
    query_language: str = "en-us",
    timeout: int = 30,
    excluded_titles: list = None,
) -> Union[Dict[str, Any], None]:
    """
    Execute Azure AI Search with semantic + vector search, returning a Python dict
    with total_count, results, search_id, semantic_answers, and document_titles.

    Args:
        query: Search query string
        select: Fields to select from the search index
        k: Number of results to return
        semantic_configuration: Semantic search configuration name
        vector_field: Field name for vector search
        query_type: Type of query (semantic, simple, etc.)
        query_language: Language code for the query
        timeout: Request timeout in seconds
        excluded_titles: List of document titles to exclude from results (e.g., ["BR102022001563A2.pdf"])

    The search results will include fields like:
    - chunk_id: Unique identifier for the chunk
    - parent_id: Identifier for the parent document
    - title: Document title
    - chunk: Text content of the chunk
    - content_embedding: Vector embedding
    - @search.score: Search relevance score
    - @search.rerankerScore: Reranker score (if using semantic search)
    - @search.captions: Extractive captions (if using semantic search)
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
        "answers": "extractive|count-3",  # Include semantic answers
    }

    # Add filter to exclude specified titles
    if excluded_titles and len(excluded_titles) > 0:
        # Create OData filter expression to exclude titles
        filter_conditions = []
        for title in excluded_titles:
            # Escape single quotes in the title for OData
            escaped_title = title.replace("'", "''")
            filter_conditions.append(f"title ne '{escaped_title}'")
        
        filter_expression = " and ".join(filter_conditions)
        payload["filter"] = filter_expression
        logger.info(f"[azureSearchPlugin] Applied exclusion filter: {filter_expression}")

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
            title = result.get("title")  # Updated field name
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
