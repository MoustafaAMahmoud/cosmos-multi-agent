"""
Schema definition for the research-agent-index in Azure AI Search.

This module defines the structure and configuration for the scientific research
index used by the deep research agent system.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any


@dataclass
class SearchResult:
    """Represents a search result from the research index."""
    chunk_id: str
    parent_id: Optional[str] = None
    title: Optional[str] = None
    chunk: Optional[str] = None
    content_embedding: Optional[List[float]] = None
    search_score: Optional[float] = None
    search_reranker_score: Optional[float] = None
    search_captions: Optional[List[Dict[str, Any]]] = None


@dataclass
class ResearchIndexConfig:
    """Configuration for the research-agent-index."""
    
    # Index basic information
    index_name: str = "research-agent-index"
    
    # Field names for easy reference
    CHUNK_ID = "chunk_id"
    PARENT_ID = "parent_id"
    TITLE = "title"
    CHUNK = "chunk"
    CONTENT_EMBEDDING = "content_embedding"
    
    # Semantic search configuration
    semantic_config_name: str = "research-agent-index-semantic-configuration"
    
    # Vector search configuration
    vector_profile_name: str = "research-agent-index-azureOpenAi-text-profile"
    vector_algorithm_name: str = "research-agent-index-algorithm"
    vector_vectorizer_name: str = "research-agent-index-azureOpenAi-text-vectorizer"
    
    # Embedding configuration
    embedding_model: str = "text-embedding-3-large"
    embedding_dimensions: int = 3072
    
    # Search parameters
    top_k: int = 50  # Maximum number of results to return
    
    @classmethod
    def get_searchable_fields(cls) -> List[str]:
        """Get list of searchable fields."""
        return [
            cls.CHUNK_ID,
            cls.TITLE,
            cls.CHUNK
        ]
    
    @classmethod
    def get_retrievable_fields(cls) -> List[str]:
        """Get list of retrievable fields."""
        return [
            cls.CHUNK_ID,
            cls.PARENT_ID,
            cls.TITLE,
            cls.CHUNK,
            cls.CONTENT_EMBEDDING
        ]
    
    @classmethod
    def get_semantic_fields(cls) -> Dict[str, Any]:
        """Get semantic search field configuration."""
        return {
            "title_field": cls.TITLE,
            "content_fields": [cls.CHUNK],
            "keyword_fields": []
        }


# Index schema definition (for reference)
RESEARCH_INDEX_SCHEMA = {
    "name": "research-agent-index",
    "fields": [
        {
            "name": "chunk_id",
            "type": "Edm.String",
            "key": True,
            "searchable": True,
            "retrievable": True,
            "analyzer": "keyword"
        },
        {
            "name": "parent_id",
            "type": "Edm.String",
            "filterable": True,
            "retrievable": True
        },
        {
            "name": "title",
            "type": "Edm.String",
            "searchable": True,
            "retrievable": True
        },
        {
            "name": "chunk",
            "type": "Edm.String",
            "searchable": True,
            "retrievable": True
        },
        {
            "name": "content_embedding",
            "type": "Collection(Edm.Single)",
            "searchable": True,
            "retrievable": True,
            "dimensions": 3072,
            "vectorSearchProfile": "research-agent-index-azureOpenAi-text-profile"
        }
    ],
    "semantic": {
        "defaultConfiguration": "research-agent-index-semantic-configuration",
        "configurations": [
            {
                "name": "research-agent-index-semantic-configuration",
                "prioritizedFields": {
                    "titleField": {"fieldName": "title"},
                    "prioritizedContentFields": [{"fieldName": "chunk"}],
                    "prioritizedKeywordsFields": []
                }
            }
        ]
    },
    "vectorSearch": {
        "algorithms": [
            {
                "name": "research-agent-index-algorithm",
                "kind": "hnsw",
                "hnswParameters": {
                    "metric": "cosine",
                    "m": 4,
                    "efConstruction": 400,
                    "efSearch": 500
                }
            }
        ],
        "profiles": [
            {
                "name": "research-agent-index-azureOpenAi-text-profile",
                "algorithm": "research-agent-index-algorithm",
                "vectorizer": "research-agent-index-azureOpenAi-text-vectorizer"
            }
        ],
        "vectorizers": [
            {
                "name": "research-agent-index-azureOpenAi-text-vectorizer",
                "kind": "azureOpenAI",
                "azureOpenAIParameters": {
                    "deploymentId": "text-embedding-3-large",
                    "modelName": "text-embedding-3-large"
                }
            }
        ]
    }
}