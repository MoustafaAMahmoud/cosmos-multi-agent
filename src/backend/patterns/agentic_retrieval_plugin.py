# File: src/backend/patterns/agentic_retrieval_plugin.py

import os
import json
import logging
import requests
import asyncio
from typing import Dict, Any, Union, List, Optional
from datetime import datetime, timedelta

from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.core.credentials import AzureKeyCredential
from semantic_kernel.functions import kernel_function
from semantic_kernel.kernel import Kernel

# Import for conversation history tracking
from semantic_kernel.contents.chat_history import ChatHistory

# Azure Storage imports for SAS token generation (optional)
try:
    from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
    AZURE_STORAGE_AVAILABLE = True
except ImportError:
    AZURE_STORAGE_AVAILABLE = False
    BlobServiceClient = None
    generate_blob_sas = None
    BlobSasPermissions = None


class AgenticRetrievalPlugin:
    """
    Implements agentic retrieval using Azure AI Search with semantic ranking.
    This plugin analyzes conversation context to perform intelligent, multi-query retrieval.
    """

    def __init__(self, kernel: Kernel = None):
        """
        Initialize the agentic retrieval plugin.

        Args:
            kernel: Optional Semantic Kernel instance for accessing conversation history
        """
        load_dotenv()
        self.logger = logging.getLogger(__name__)
        self.kernel = kernel

        # Azure Search configuration
        self.search_endpoint = os.getenv("AZURE_AI_SEARCH_ENDPOINT")
        self.search_api_key = os.getenv("AZURE_AI_SEARCH_API_KEY")
        # Force the service name to be aifoundry (override any environment variable)
        self.service_name = "aifoundry"
        env_service_name = os.getenv("AZURE_AI_SEARCH_SERVICE_NAME")
        if env_service_name:
            self.logger.warning(f"[AgenticRetrievalPlugin] Overriding env service name '{env_service_name}' with 'aifoundry'")
        else:
            self.logger.info(f"[AgenticRetrievalPlugin] Using default service name: aifoundry")
        self.index_name = os.getenv("AZURE_SEARCH_INDEX_NAME", "research-agent-index")
        self.semantic_configuration = os.getenv(
            "AZURE_SEARCH_SEMANTIC_CONFIG", "research-agent-index-semantic-configuration"
        )

        if not self.search_endpoint:
            raise ValueError("Azure AI Search endpoint must be set.")

        # Construct the proper endpoint URL
        self.logger.info(f"[AgenticRetrievalPlugin] Endpoint: {self.search_endpoint}")
        self.logger.info(f"[AgenticRetrievalPlugin] Service name from env: {env_service_name}")
        self.logger.info(f"[AgenticRetrievalPlugin] Index: {self.index_name}")
        
        if "cognitiveservices.azure.com" in self.search_endpoint:
            self.base_url = f"{self.search_endpoint.rstrip('/')}/searchservices/{self.service_name}/indexes/{self.index_name}/docs/search"
            self.logger.info(f"[AgenticRetrievalPlugin] Using AI Foundry URL: {self.base_url}")
            self.logger.info(f"[AgenticRetrievalPlugin] Service name: {self.service_name}")
            self.logger.info(f"[AgenticRetrievalPlugin] Index name: {self.index_name}")
        else:
            # Traditional search service format
            self.base_url = f"{self.search_endpoint.rstrip('/')}/indexes/{self.index_name}/docs/search"
            self.logger.info(f"[AgenticRetrievalPlugin] Using traditional search URL: {self.base_url}")
            self.logger.info(f"[AgenticRetrievalPlugin] Note: Traditional search services don't use service name in URL")

        # Agentic retrieval parameters with aggressive rate limiting
        self.reranker_threshold = 2.0  # Threshold for reranker scores (typically 1-4 range)
        self.search_threshold = 0.01   # Fallback threshold for basic search scores
        self.max_search_queries = 4    # Further reduced to manage rate limits
        self.max_search_results = 20   # Further reduced results per query
        self.max_total_sources = 15    # Further reduced maximum total sources to manage token limits
        self.exhaustive_search_rounds = 2  # Further reduced rounds to manage rate limits
        self.search_delay = 1.0  # Increased delay between searches in seconds
        self.request_timeout = 30  # Timeout for search requests in seconds

        # Cache for retrieval results keyed by message ID
        self.retrieval_results = {}
        
        # Current user query for context
        self.current_query = None
        
        # Azure Blob Storage configuration for SAS token generation
        self.blob_account_name = os.getenv("AZURE_STORAGE_ACCOUNT_NAME", "researchagents")
        self.blob_account_key = os.getenv("AZURE_STORAGE_ACCOUNT_KEY")
        self.blob_container_name = os.getenv("AZURE_STORAGE_CONTAINER_NAME", "datasets")
        self.blob_base_url = f"https://{self.blob_account_name}.blob.core.windows.net/{self.blob_container_name}/"
        
        self.logger.info(f"[AgenticRetrievalPlugin] Blob configuration:")
        self.logger.info(f"  - Account: {self.blob_account_name}")
        self.logger.info(f"  - Container: {self.blob_container_name}")
        self.logger.info(f"  - Base URL: {self.blob_base_url}")
        self.logger.info(f"  - Has Account Key: {bool(self.blob_account_key)}")
        
        # Initialize blob service client if account key is available and azure.storage.blob is installed
        self.blob_service_client = None
        if AZURE_STORAGE_AVAILABLE and self.blob_account_key:
            self.blob_service_client = BlobServiceClient(
                account_url=f"https://{self.blob_account_name}.blob.core.windows.net",
                credential=self.blob_account_key
            )
            self.logger.info(f"[AgenticRetrievalPlugin] Blob service client initialized for account: {self.blob_account_name}")
        elif not AZURE_STORAGE_AVAILABLE:
            self.logger.warning("[AgenticRetrievalPlugin] azure-storage-blob package not available, SAS URLs will not be generated")
        else:
            self.logger.warning("[AgenticRetrievalPlugin] No blob account key found, SAS URLs will not be generated")
        
        # Test the search service connectivity
        self._test_search_connectivity()
    
    def _generate_blob_sas_url(self, blob_name: str, expiry_hours: int = 24) -> str:
        """
        Generate a SAS URL for a blob in Azure Storage.
        
        Args:
            blob_name: Name of the blob file (e.g., 'AU2020402165B2.pdf')
            expiry_hours: Hours until the SAS token expires (default: 24)
            
        Returns:
            Full SAS URL for the blob, or the base URL if SAS generation fails
        """
        if not AZURE_STORAGE_AVAILABLE or not self.blob_service_client or not self.blob_account_key:
            # Return base URL without SAS if no credentials or package unavailable
            return f"{self.blob_base_url}{blob_name}"
        
        try:
            # Generate SAS token with read permissions
            sas_token = generate_blob_sas(
                account_name=self.blob_account_name,
                container_name=self.blob_container_name,
                blob_name=blob_name,
                account_key=self.blob_account_key,
                permission=BlobSasPermissions(read=True),
                expiry=datetime.utcnow() + timedelta(hours=expiry_hours)
            )
            
            # Construct full SAS URL
            sas_url = f"{self.blob_base_url}{blob_name}?{sas_token}"
            self.logger.info(f"[AgenticRetrievalPlugin] Generated SAS URL for blob: {blob_name}")
            return sas_url
            
        except Exception as e:
            self.logger.error(f"[AgenticRetrievalPlugin] Failed to generate SAS URL for {blob_name}: {e}")
            # Return base URL without SAS as fallback
            return f"{self.blob_base_url}{blob_name}"
    
    def _test_search_connectivity(self):
        """Test if the search service is reachable."""
        try:
            # Test with a simple GET to the index
            test_url = f"{self.search_endpoint.rstrip('/')}/indexes/{self.index_name}?api-version=2024-05-01-Preview"
            headers = {"Content-Type": "application/json"}
            if self.search_api_key:
                headers["api-key"] = self.search_api_key
                
            response = requests.get(test_url, headers=headers, timeout=10)
            self.logger.info(f"[CONNECTIVITY TEST] Status: {response.status_code}")
            if response.status_code == 200:
                self.logger.info(f"[CONNECTIVITY TEST] Search index is accessible")
            else:
                self.logger.warning(f"[CONNECTIVITY TEST] Index not accessible: {response.text}")
        except Exception as e:
            self.logger.error(f"[CONNECTIVITY TEST] Failed to connect to search service: {e}")

    def set_current_query(self, query: str):
        """Set the current user query for context-aware retrieval."""
        self.current_query = query
        self.logger.info(f"[AgenticRetrievalPlugin] Set current query: {query}")

    def _get_conversation_context(self, limit: int = 5) -> List[Dict[str, str]]:
        """
        Extract recent conversation history for context-aware retrieval.

        Args:
            limit: Number of recent messages to consider

        Returns:
            List of message dictionaries with role and content
        """
        messages = []

        # Try to get conversation history from the kernel if available
        if self.kernel and hasattr(self.kernel, "chat_history"):
            history = self.kernel.chat_history
            recent_messages = list(history.messages[-limit:])

            for msg in recent_messages:
                messages.append({"role": str(msg.role), "content": str(msg.content)})

        self.logger.info(f"[AgenticRetrievalPlugin] Found {len(messages)} conversation messages")
        return messages

    def _generate_exhaustive_search_queries(self, topic: str, round_number: int = 1) -> List[str]:
        """
        Generate comprehensive search queries for exhaustive topic research.
        
        Args:
            topic: The main topic to research
            round_number: Which round of searching this is (affects query variation)
            
        Returns:
            List of search queries designed to find all relevant sources
        """
        queries = []
        
        # Base query
        queries.append(topic)
        
        # Extract key terms from the topic
        topic_words = topic.lower().split()
        main_terms = [word for word in topic_words if len(word) > 3]
        
        # Round 1: Core topic variations
        if round_number == 1:
            queries.extend([
                f"{topic} definition explanation",
                f"{topic} research study",
                f"{topic} analysis review",
                f"{topic} effects impact",
                f"{topic} benefits risks",
                f"{topic} mechanism process",
                f"{topic} types classification",
                f"{topic} applications uses"
            ])
            
        # Round 2: Technical and scientific aspects
        elif round_number == 2:
            queries.extend([
                f"{topic} scientific evidence",
                f"{topic} clinical trials",
                f"{topic} methodology techniques",
                f"{topic} technology innovation",
                f"{topic} safety evaluation",
                f"{topic} regulatory guidelines",
                f"{topic} industry standards",
                f"{topic} best practices"
            ])
            
        # Round 3: Related concepts and variations
        elif round_number == 3:
            for term in main_terms:
                queries.extend([
                    f"{term} related concepts",
                    f"{term} alternatives comparison",
                    f"{term} development history",
                    f"{term} future trends"
                ])
                
        # Round 4: Broader context and implications
        elif round_number == 4:
            queries.extend([
                f"{topic} environmental impact",
                f"{topic} economic implications",
                f"{topic} social aspects",
                f"{topic} ethical considerations",
                f"{topic} policy regulations",
                f"{topic} market analysis"
            ])
            
        # Round 5: Edge cases and specific scenarios
        else:
            queries.extend([
                f"{topic} case studies examples",
                f"{topic} troubleshooting problems",
                f"{topic} optimization improvement",
                f"{topic} implementation challenges",
                f"{topic} maintenance requirements",
                f"{topic} quality control"
            ])
        
        return queries[:self.max_search_queries]

    def _generate_search_queries(
        self, conversation_context: List[Dict[str, str]], fallback_query: str = None
    ) -> List[str]:
        """
        Generate multiple search queries based on conversation context.

        Args:
            conversation_context: Recent conversation messages
            fallback_query: Default query to use if no context available

        Returns:
            List of search queries to execute
        """
        queries = []

        # Get the last user message as primary query
        last_user_message = None
        if conversation_context:
            for msg in reversed(conversation_context):
                if msg["role"] == "user":
                    last_user_message = msg["content"]
                    break

        # Use fallback query if no conversation context
        if not last_user_message and fallback_query:
            last_user_message = fallback_query
            self.logger.info(f"[AgenticRetrievalPlugin] Using fallback query: {fallback_query}")
        elif not last_user_message and self.current_query:
            last_user_message = self.current_query
            self.logger.info(f"[AgenticRetrievalPlugin] Using current query: {self.current_query}")
        elif not last_user_message:
            # Try to extract from conversation messages differently
            self.logger.info(f"[AgenticRetrievalPlugin] Conversation context: {conversation_context}")
            if conversation_context:
                # Look for any message with content
                for msg in reversed(conversation_context):
                    if msg.get("content") and len(msg.get("content", "").strip()) > 0:
                        last_user_message = msg["content"]
                        self.logger.info(f"[AgenticRetrievalPlugin] Found message content: {last_user_message}")
                        break

        if last_user_message:
            # For exhaustive search, use the new method
            return self._generate_exhaustive_search_queries(last_user_message, 1)

        return queries

    def _perform_semantic_search(
        self, query: str, top: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Execute a single semantic search query using REST API.

        Args:
            query: Search query string
            top: Number of results to return

        Returns:
            List of search results
        """
        try:
            # Prepare the search request
            url = f"{self.base_url}?api-version=2024-05-01-Preview"
            headers = {"Content-Type": "application/json"}
            
            # Add API key if available
            if self.search_api_key:
                headers["api-key"] = self.search_api_key
            
            # Use semantic search with vector queries for best results
            payload = {
                "search": query,
                "select": "content_id,text_document_id,content_text,document_title,content_path",
                "queryType": "semantic",
                "semanticConfiguration": self.semantic_configuration,
                "top": top,
                "vectorQueries": [
                    {
                        "kind": "text",
                        "text": query,
                        "fields": "content_embedding",
                        "k": top,
                    }
                ],
            }

            self.logger.info(f"[SEARCH DEBUG] Query: '{query}'")
            self.logger.info(f"[SEARCH DEBUG] URL: {url}")
            self.logger.info(f"[SEARCH DEBUG] Headers: {headers}")
            self.logger.info(f"[SEARCH DEBUG] Payload: {json.dumps(payload, indent=2)}")

            # Perform the search request
            response = requests.post(
                url, headers=headers, data=json.dumps(payload), timeout=30
            )

            if response.status_code != 200:
                self.logger.error(
                    f"[AGENTIC] Search failed with status {response.status_code}: {response.text}"
                )
                self.logger.error(
                    f"[AGENTIC] URL was: {url}"
                )
                self.logger.error(
                    f"[AGENTIC] Service name: {self.service_name}, Base URL: {self.base_url}"
                )
                return []

            data = response.json()
            results = data.get("value", [])
            
            self.logger.info(f"[SEARCH DEBUG] Response status: {response.status_code}")
            self.logger.info(f"[SEARCH DEBUG] Results count: {len(results)}")
            self.logger.info(f"[SEARCH DEBUG] Full response keys: {list(data.keys()) if data else 'None'}")
            if results:
                self.logger.info(f"[SEARCH DEBUG] First result keys: {list(results[0].keys()) if results[0] else 'None'}")

            # Convert results to standardized format
            search_results = []
            for result in results:
                search_results.append(
                    {
                        "content_id": result.get("content_id", ""),
                        "text_document_id": result.get("text_document_id", ""),
                        "content_text": result.get("content_text", ""),
                        "document_title": result.get("document_title", ""),
                        "content_path": result.get("content_path", ""),
                        "@search.score": result.get("@search.score", 0),
                        "@search.reranker_score": result.get(
                            "@search.reranker_score", 0
                        ),
                    }
                )

            return search_results

        except Exception as e:
            error_str = str(e)
            self.logger.error(f"Search error for query '{query}': {error_str}")
            
            # Check if it's a rate limit error and add specific handling
            if "429" in error_str or "rate limit" in error_str.lower():
                self.logger.warning(f"Rate limit detected for query '{query}'. Adding extended delay.")
                # Add a longer delay for rate limit errors
                import time
                time.sleep(5.0)
            
            return []

    def _deduplicate_results(
        self, all_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Remove duplicate results based on chunk_id, keeping highest scored version.

        Args:
            all_results: Combined results from multiple queries

        Returns:
            Deduplicated and sorted results
        """
        # Group by content_id, keeping highest score
        unique_results = {}
        for result in all_results:
            content_id = result.get("content_id", "")
            if not content_id:
                continue

            reranker_score = result.get("@search.reranker_score", 0)

            if content_id not in unique_results or reranker_score > unique_results[
                content_id
            ].get("@search.reranker_score", 0):
                unique_results[content_id] = result

        # Sort by reranker score descending
        sorted_results = sorted(
            unique_results.values(),
            key=lambda x: x.get("@search.reranker_score", 0),
            reverse=True,
        )

        return sorted_results

    def _format_results_with_citations(self, results: List[Dict[str, Any]]) -> str:
        """
        Format search results in Perplexity-style with numbered citations.

        Args:
            results: Search results to format

        Returns:
            JSON string with formatted results
        """
        formatted_results = []
        document_references = {}  # Track unique document titles for citation
        citation_counter = 1

        for idx, result in enumerate(results):
            # Use document_title as primary reference since content_path can be null/image
            document_title = result.get("document_title", "Untitled")
            content_path = result.get("content_path", "")
            
            # Use document title as the primary reference
            filename = document_title if document_title != "Untitled" else "Unknown Document"
            
            # Track document references for Perplexity-style numbering
            if document_title not in document_references:
                # Generate SAS URL for the document if it's a PDF
                document_url = ""
                if document_title.endswith('.pdf'):
                    document_url = self._generate_blob_sas_url(document_title)
                elif content_path and content_path.startswith("http"):
                    document_url = content_path
                
                document_references[document_title] = {
                    "citation_number": citation_counter,
                    "document": document_title,
                    "filename": filename,
                    "content_snippets": [],
                    "relevance_score": result.get("@search.reranker_score", result.get("@search.score", 0)),
                    "content_type": "image" if content_path and "images/" in content_path else "text",
                    "document_url": document_url,
                }
                citation_counter += 1
            
            # Add content snippet to the document reference
            content_snippet = result.get("content_text", "")[:200] + "..." if len(result.get("content_text", "")) > 200 else result.get("content_text", "")
            if content_snippet and content_snippet not in document_references[document_title]["content_snippets"]:
                document_references[document_title]["content_snippets"].append(content_snippet)
                
            formatted_results.append(
                {
                    "citation_number": document_references[document_title]["citation_number"],
                    "title": document_title,
                    "filename": filename,
                    "content": result.get("content_text", ""),
                    "content_snippet": content_snippet,
                    "content_path": content_path,
                    "relevance_score": result.get("@search.reranker_score", result.get("@search.score", 0)),
                    "document_url": document_references[document_title]["document_url"],
                    "content_type": "image" if content_path and "images/" in content_path else "text",
                }
            )

        # Create Perplexity-style sources list
        sources = []
        for doc_title, doc_info in document_references.items():
            sources.append({
                "citation_number": doc_info["citation_number"],
                "title": doc_info["document"],
                "filename": doc_info["filename"],
                "content_snippets": doc_info["content_snippets"],
                "relevance_score": doc_info["relevance_score"],
                "content_type": doc_info["content_type"],
                "document_url": doc_info["document_url"]
            })

        # Sort sources by citation number
        sources.sort(key=lambda x: x["citation_number"])

        response = {
            "status": "success",
            "query_count": len(
                self._generate_search_queries(self._get_conversation_context())
            ),
            "total_results": len(formatted_results),
            "unique_documents": len(document_references),
            "sources": sources,
            "results": formatted_results,
            "instructions": "Use ALL the retrieved information to provide a comprehensive answer. ALWAYS cite sources using numbered citations [1], [2], etc. based on the citation_number field. Example: 'Vaping is the act of inhaling aerosol produced by electronic devices [1]. According to research, it may have health implications [2].' Make sure to reference ALL relevant sources and provide a complete answer based on ALL available information.",
        }

        return json.dumps(response, indent=2)

    def _perform_exhaustive_search(self, topic: str) -> List[Dict[str, Any]]:
        """
        Perform exhaustive search with rate limiting across multiple rounds.
        
        Args:
            topic: The research topic
            
        Returns:
            List of unique sources with comprehensive coverage
        """
        all_sources = {}  # Use dict to avoid duplicates by content_id
        
        print(f"[DEBUG] === STARTING RATE-LIMITED EXHAUSTIVE SEARCH FOR: {topic} ===")
        self.logger.info(f"[AgenticRetrievalPlugin] Starting exhaustive search for topic: {topic}")
        
        for search_round in range(1, self.exhaustive_search_rounds + 1):
            print(f"[DEBUG] Search Round {search_round}/{self.exhaustive_search_rounds}")
            self.logger.info(f"[AgenticRetrievalPlugin] Search round {search_round}/{self.exhaustive_search_rounds}")
            
            # Generate queries for this round
            round_queries = self._generate_exhaustive_search_queries(topic, search_round)
            
            # Limit queries per round to aggressively manage rate limits
            round_queries = round_queries[:2]  # Reduced to max 2 queries per round
            
            for i, query in enumerate(round_queries):
                if len(all_sources) >= self.max_total_sources:
                    print(f"[DEBUG] Reached maximum sources limit ({self.max_total_sources})")
                    break
                    
                print(f"[DEBUG] Round {search_round}, Query {i+1}/{len(round_queries)}: '{query}'")
                self.logger.info(f"[AgenticRetrievalPlugin] Executing query: {query}")
                
                # Add progressive delay to manage rate limits with exponential backoff
                if i > 0 or search_round > 1:  # Delay after first query or first round
                    import time
                    delay = self.search_delay * (1.5 ** (search_round - 1)) * (i + 1)  # Exponential + progressive delay
                    time.sleep(delay)
                    self.logger.info(f"[AgenticRetrievalPlugin] Applied {delay:.1f}s delay for rate limiting")
                
                # Perform search with further reduced result count
                results = self._perform_semantic_search(query, top=10)  # Further reduced from 15
                new_sources_count = 0
                
                for result in results:
                    content_id = result.get("content_id", "")
                    if content_id and content_id not in all_sources:
                        all_sources[content_id] = result
                        new_sources_count += 1
                        
                print(f"[DEBUG] Found {new_sources_count} new sources (Total: {len(all_sources)})")
                self.logger.info(f"[AgenticRetrievalPlugin] Found {new_sources_count} new sources, total: {len(all_sources)}")
                
                # Early exit if we have enough high-quality sources
                if len(all_sources) >= 15 and search_round >= 2:
                    print(f"[DEBUG] Early exit with {len(all_sources)} sources after round {search_round}")
                    break
                
            if len(all_sources) >= self.max_total_sources:
                break
                
        print(f"[DEBUG] === EXHAUSTIVE SEARCH COMPLETE: {len(all_sources)} total sources ===")
        self.logger.info(f"[AgenticRetrievalPlugin] Exhaustive search complete. Found {len(all_sources)} unique sources")
        
        return list(all_sources.values())

    def _create_source_summary(self, source: Dict[str, Any], topic: str) -> str:
        """
        Create a concise summary of how this source relates to the research topic.
        
        Args:
            source: Source document information
            topic: The research topic
            
        Returns:
            Concise summary text explaining the source's relation to the topic
        """
        content = source.get("content_text", "")
        title = source.get("document_title", "Unknown Document")
        
        # Extract key sentences that mention the topic or related terms
        topic_keywords = [word for word in topic.lower().split() if len(word) > 3]
        sentences = content.split('. ')[:5]  # Check only first 5 sentences for efficiency
        
        relevant_sentence = None
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(keyword in sentence_lower for keyword in topic_keywords):
                relevant_sentence = sentence.strip()
                break
        
        if relevant_sentence:
            summary = f"Covers {topic}: {relevant_sentence}"
            return summary[:150] + "..." if len(summary) > 150 else summary
        else:
            # Fallback to brief description
            return f"Contains technical information related to {topic}."

    def _format_exhaustive_results_with_citations(self, sources: List[Dict[str, Any]], topic: str) -> str:
        """
        Format exhaustive search results with individual source summaries.
        
        Args:
            sources: All found sources
            topic: The research topic
            
        Returns:
            JSON string with formatted results including source summaries
        """
        formatted_sources = []
        
        for idx, source in enumerate(sources[:self.max_total_sources]):
            citation_number = idx + 1
            document_title = source.get("document_title", "Untitled")
            
            # Generate SAS URL for PDF documents
            document_url = ""
            if document_title.endswith('.pdf'):
                document_url = self._generate_blob_sas_url(document_title)
            
            # Create source summary
            source_summary = self._create_source_summary(source, topic)
            
            # Trim content to reduce token usage
            content_text = source.get("content_text", "")
            content_snippet = content_text[:100] + "..." if len(content_text) > 100 else content_text
            
            formatted_sources.append({
                "citation_number": citation_number,
                "title": document_title,
                "filename": document_title,
                "content_snippet": content_snippet,
                "topic_relation_summary": source_summary,
                "relevance_score": source.get("@search.reranker_score", source.get("@search.score", 0)),
                "content_type": "image" if source.get("content_path", "") and "images/" in source.get("content_path", "") else "text",
                "document_url": document_url
            })
        
        # Create overall research summary
        total_sources = len(formatted_sources)
        document_types = {}
        for source in formatted_sources:
            doc_type = source["content_type"]
            document_types[doc_type] = document_types.get(doc_type, 0) + 1
        
        research_summary = (
            f"Comprehensive research on '{topic}' identified {total_sources} relevant sources. "
            f"The research covers multiple aspects including definitions, mechanisms, applications, "
            f"safety considerations, and regulatory aspects. "
            f"Sources include {document_types.get('text', 0)} text documents"
        )
        
        if document_types.get('image', 0) > 0:
            research_summary += f" and {document_types.get('image', 0)} image documents"
        research_summary += "."
        
        # Chunk sources if too many to manage token limits
        max_sources_per_response = 15  # Limit sources per response
        if len(formatted_sources) > max_sources_per_response:
            formatted_sources = formatted_sources[:max_sources_per_response]
            self.logger.info(f"[AgenticRetrievalPlugin] Chunked sources to {max_sources_per_response} to manage token limits")
        
        response = {
            "status": "success",
            "research_topic": topic,
            "total_sources_found": len(formatted_sources),
            "max_sources_limit": self.max_total_sources,
            "research_summary": research_summary,
            "sources": formatted_sources,
            "instructions": f"This focused research on '{topic}' has identified {len(formatted_sources)} high-quality sources. Each source includes a summary of its relation to the topic. Use numbered citations [1], [2], etc. when referencing sources. Focus on creating a comprehensive but concise response."
        }
        
        return json.dumps(response, indent=2)

    @kernel_function(
        name="exhaustive_research",
        description="Perform comprehensive exhaustive research to find ALL relevant sources on a topic, up to 50 sources with individual summaries"
    )
    def exhaustive_research(self, topic: str) -> str:
        """
        Perform exhaustive research on a topic to find all relevant sources.
        
        Args:
            topic: The research topic to investigate comprehensively
            
        Returns:
            JSON string with all found sources and their relation summaries
        """
        try:
            print(f"[DEBUG] === EXHAUSTIVE RESEARCH FUNCTION CALLED ===")
            print(f"[DEBUG] Topic: {topic}")
            
            self.logger.info(f"[AgenticRetrievalPlugin] Starting exhaustive research for: {topic}")
            
            # Perform exhaustive search
            all_sources = self._perform_exhaustive_search(topic)
            
            # Format results with source summaries
            result = self._format_exhaustive_results_with_citations(all_sources, topic)
            
            print(f"[DEBUG] Exhaustive research complete. Result length: {len(result)}")
            return result
            
        except Exception as e:
            self.logger.error(f"[AgenticRetrievalPlugin] Error during exhaustive research: {str(e)}")
            return json.dumps({
                "status": "error", 
                "message": f"Exhaustive research failed: {str(e)}"
            })

    @kernel_function(
        name="agentic_retrieval",
        description="Perform intelligent, context-aware retrieval using Azure AI Search with semantic ranking and multiple queries",
    )
    def agentic_retrieval(self, query: Optional[str] = None, conversation_id: Optional[str] = None) -> str:
        """
        Performs agentic retrieval by analyzing conversation context and executing multiple intelligent queries.

        Args:
            query: Optional explicit query to search for
            conversation_id: Optional conversation identifier for caching results

        Returns:
            JSON string containing search results with reference IDs for citation
        """
        try:
            print("[DEBUG] === AGENTIC RETRIEVAL START ===")
            print(f"[DEBUG] Query parameter: {query}")
            print(f"[DEBUG] Current query: {self.current_query}")
            print(f"[DEBUG] Endpoint: {self.search_endpoint}")
            
            self.logger.info("[AgenticRetrievalPlugin] *** FUNCTION CALLED *** Starting agentic retrieval")
            
            # Debug configuration
            self.logger.info(f"[DEBUG] Search endpoint: {self.search_endpoint}")
            self.logger.info(f"[DEBUG] Base URL: {self.base_url}")
            self.logger.info(f"[DEBUG] Index name: {self.index_name}")
            self.logger.info(f"[DEBUG] Has API key: {bool(self.search_api_key)}")
            
            print("[DEBUG] About to get conversation context...")
        except Exception as init_error:
            print(f"[DEBUG] Error in function init: {init_error}")
            return json.dumps({"status": "error", "message": f"Initialization failed: {str(init_error)}"})
        
        try:
            # Get conversation context
            conversation_context = self._get_conversation_context()

            # Generate multiple search queries based on context
            search_queries = self._generate_search_queries(conversation_context, fallback_query=query)
            
            print(f"[DEBUG] Generated queries: {search_queries}")
            self.logger.info(f"[DEBUG] Generated queries: {search_queries}")

            if not search_queries:
                # If no queries generated, try some default research terms
                default_queries = ["research", "analysis", "study", "findings", "data"]
                print(f"[DEBUG] No queries from context, using defaults: {default_queries}")
                self.logger.warning("[AgenticRetrievalPlugin] No queries from context, using defaults")
                search_queries = default_queries[:1]  # Just use one default query

            print(f"[DEBUG] Final search queries: {search_queries}")
            self.logger.info(
                f"[AgenticRetrievalPlugin] Generated {len(search_queries)} search queries: {search_queries}"
            )

            # Execute all queries and collect results
            all_results = []
            for i, search_query in enumerate(search_queries):
                print(f"[DEBUG] Executing query {i+1}/{len(search_queries)}: '{search_query}'")
                self.logger.info(f"[DEBUG] Executing query {i+1}/{len(search_queries)}: '{search_query}'")
                results = self._perform_semantic_search(
                    search_query, top=self.max_search_results
                )
                print(f"[DEBUG] Query {i+1} returned {len(results)} results")
                self.logger.info(f"[DEBUG] Query {i+1} returned {len(results)} results")
                all_results.extend(results)

            # Deduplicate and filter by reranker threshold
            unique_results = self._deduplicate_results(all_results)

            # Filter by appropriate threshold based on score type
            filtered_results = []
            score_debug_info = []
            
            for r in unique_results:
                reranker_score = r.get("@search.reranker_score", 0)
                search_score = r.get("@search.score", 0)
                
                # Use different thresholds for reranker vs search scores
                if reranker_score > 0:
                    # Use reranker threshold for reranker scores
                    threshold = self.reranker_threshold
                    score_to_check = reranker_score
                    score_type = "reranker"
                else:
                    # Use search threshold for basic search scores
                    threshold = self.search_threshold
                    score_to_check = search_score
                    score_type = "search"
                
                score_debug_info.append(f"Reranker: {reranker_score}, Search: {search_score}, Using: {score_to_check} ({score_type}, threshold: {threshold})")
                
                if score_to_check >= threshold:
                    filtered_results.append(r)
                    
            # If no results pass the threshold, lower it and try again
            if len(filtered_results) == 0 and len(unique_results) > 0:
                print(f"[DEBUG] No results passed their respective thresholds. Score details:")
                for i, score_info in enumerate(score_debug_info[:5]):  # Show first 5
                    print(f"  Result {i+1}: {score_info}")
                    
                # Lower thresholds to get some results
                print(f"[DEBUG] Lowering thresholds to get results...")
                for r in unique_results:
                    reranker_score = r.get("@search.reranker_score", 0)
                    search_score = r.get("@search.score", 0)
                    
                    # Use much lower thresholds as fallback
                    if reranker_score > 0 and reranker_score >= 1.0:  # Lower reranker threshold
                        filtered_results.append(r)
                    elif search_score >= 0.005:  # Lower search threshold
                        filtered_results.append(r)
                        
            # Limit results
            filtered_results = filtered_results[:self.max_search_results]
            
            print(f"[DEBUG] Before filtering: {len(unique_results)} results")
            print(f"[DEBUG] After filtering: {len(filtered_results)} results")
            self.logger.info(f"[DEBUG] Before filtering: {len(unique_results)} results") 
            self.logger.info(f"[DEBUG] After filtering: {len(filtered_results)} results")

            print(f"[DEBUG] Returning {len(filtered_results)} high-quality results")
            self.logger.info(
                f"[AgenticRetrievalPlugin] Returning {len(filtered_results)} high-quality results"
            )

            # Cache results if conversation_id provided
            if conversation_id:
                self.retrieval_results[conversation_id] = filtered_results

            # Format and return results
            result = self._format_results_with_citations(filtered_results)
            print(f"[DEBUG] Final result length: {len(result)}")
            return result

        except Exception as e:
            self.logger.error(
                f"[AgenticRetrievalPlugin] Error during agentic retrieval: {str(e)}"
            )
            return json.dumps(
                {"status": "error", "message": f"Agentic retrieval failed: {str(e)}"}
            )

    @kernel_function(
        name="search",
        description="Perform a semantic + vector search against the research agent index (backward compatibility)",
    )
    def search(self, query: str) -> str:
        """
        Backward compatible search function that performs a single query search.

        Args:
            query: Search query string

        Returns:
            JSON string with search results
        """
        self.logger.info(f"[AgenticRetrievalPlugin.search] Query: {query}")

        try:
            results = self._perform_semantic_search(query, top=10)

            # Format in the expected format
            response = {
                "total_count": len(results),
                "results": results,
                "search_id": f"search_{datetime.now().isoformat()}",
                "semantic_answers": [],  # Could be populated if semantic answers are available
            }

            return json.dumps(response)

        except Exception as e:
            self.logger.error(f"[AgenticRetrievalPlugin.search] Error: {str(e)}")
            return json.dumps({"error": "search_failed"})
