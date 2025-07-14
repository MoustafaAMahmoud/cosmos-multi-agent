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
    content_id: str
    text_document_id: Optional[str] = None
    document_title: Optional[str] = None
    image_document_id: Optional[str] = None
    content_text: Optional[str] = None
    content_path: Optional[str] = None
    page_number: Optional[int] = None
    bounding_polygons: Optional[str] = None
    score: Optional[float] = None


@dataclass
class ResearchIndexConfig:
    """Configuration for the research-agent-index."""
    
    # Index basic information
    index_name: str = "research-agent-index"
    
    # Field names for easy reference
    CONTENT_ID = "content_id"
    TEXT_DOCUMENT_ID = "text_document_id"
    DOCUMENT_TITLE = "document_title"
    IMAGE_DOCUMENT_ID = "image_document_id"
    CONTENT_TEXT = "content_text"
    CONTENT_EMBEDDING = "content_embedding"
    CONTENT_PATH = "content_path"
    LOCATION_METADATA = "locationMetadata"
    
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
            cls.CONTENT_ID,
            cls.DOCUMENT_TITLE,
            cls.CONTENT_TEXT,
            cls.CONTENT_PATH
        ]
    
    @classmethod
    def get_retrievable_fields(cls) -> List[str]:
        """Get list of retrievable fields."""
        return [
            cls.CONTENT_ID,
            cls.TEXT_DOCUMENT_ID,
            cls.DOCUMENT_TITLE,
            cls.IMAGE_DOCUMENT_ID,
            cls.CONTENT_TEXT,
            cls.CONTENT_PATH,
            cls.LOCATION_METADATA
        ]
    
    @classmethod
    def get_semantic_fields(cls) -> Dict[str, Any]:
        """Get semantic search field configuration."""
        return {
            "title_field": cls.DOCUMENT_TITLE,
            "content_fields": [cls.CONTENT_TEXT],
            "keyword_fields": []
        }


# Index schema definition (for reference)
RESEARCH_INDEX_SCHEMA = {
    "name": "research-agent-index",
    "fields": [
        {
            "name": "content_id",
            "type": "Edm.String",
            "key": True,
            "searchable": True,
            "retrievable": True,
            "analyzer": "keyword"
        },
        {
            "name": "text_document_id",
            "type": "Edm.String",
            "filterable": True,
            "retrievable": True
        },
        {
            "name": "document_title",
            "type": "Edm.String",
            "searchable": True,
            "retrievable": True
        },
        {
            "name": "image_document_id",
            "type": "Edm.String",
            "filterable": True,
            "retrievable": True
        },
        {
            "name": "content_text",
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
        },
        {
            "name": "content_path",
            "type": "Edm.String",
            "searchable": True,
            "retrievable": True
        },
        {
            "name": "locationMetadata",
            "type": "Edm.ComplexType",
            "fields": [
                {
                    "name": "pageNumber",
                    "type": "Edm.Int32",
                    "filterable": True,
                    "retrievable": True
                },
                {
                    "name": "boundingPolygons",
                    "type": "Edm.String",
                    "retrievable": True
                }
            ]
        }
    ],
    "semantic": {
        "defaultConfiguration": "research-agent-index-semantic-configuration",
        "configurations": [
            {
                "name": "research-agent-index-semantic-configuration",
                "prioritizedFields": {
                    "titleField": {"fieldName": "document_title"},
                    "prioritizedContentFields": [{"fieldName": "content_text"}],
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