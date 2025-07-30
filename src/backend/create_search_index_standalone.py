"""
Azure AI Search Complete Setup Script
=====================================

Sets up Azure AI Search index, skillset, data source, and indexer with vector support
using index projections (one-to-many) for document chunking and embedding generation.

Requirements:
- Python SDK: azure-search-documents >= 11.5.0
- Azure OpenAI resource with text-embedding deployment
- Azure Blob Storage with documents to index

Environment Variables Required:
- AZURE_AI_SEARCH_ENDPOINT: Your search service endpoint
- AZURE_AI_SEARCH_API_KEY: Your search service admin key
- AZURE_SEARCH_INDEX_NAME: Name for your index (optional)
- AZURE_BLOB_CONNECTION_STRING: Blob storage connection string
- AZURE_BLOB_CONTAINER_NAME: Container name (default: "documents")
- AZURE_OPENAI_ENDPOINT: OpenAI resource endpoint
- AZURE_OPENAI_API_KEY: OpenAI API key
- AZURE_EMBEDDING_DEPLOYMENT_NAME: Embedding model deployment name
- EMBEDDING_VECTOR_SIZE: Vector dimensions (default: 3072 for text-embedding-3-large)
"""

import os
import time
from dotenv import load_dotenv

from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import ResourceNotFoundError

from azure.search.documents.indexes import (
    SearchIndexClient,
    SearchIndexerClient,
)
from azure.search.documents.indexes.models import (
    # Index models
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    HnswParameters,
    VectorSearchProfile,
    AzureOpenAIVectorizer,
    AzureOpenAIVectorizerParameters,
    SemanticConfiguration,
    SemanticPrioritizedFields,
    SemanticField,
    SemanticSearch,
    BM25SimilarityAlgorithm,
    LexicalAnalyzerName,
    VectorSearchAlgorithmMetric,
    # Projection classes (SDK >= 11.5)
    SearchIndexerIndexProjection,
    SearchIndexerIndexProjectionSelector,
    SearchIndexerIndexProjectionsParameters,
    # Indexer models
    SearchIndexer,
    SearchIndexerDataSourceConnection,
    SearchIndexerDataContainer,
    SearchIndexerDataSourceType,
    FieldMapping,
    SearchIndexerSkillset,
    InputFieldMappingEntry,
    OutputFieldMappingEntry,
    AzureOpenAIEmbeddingSkill,
    IndexingParameters,
    BlobIndexerParsingMode,
    SplitSkill,
    SplitSkillLanguage,
)

# ---------------------------------------------------------------------------
# Environment Setup
# ---------------------------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------


def _exists(client_method, name: str) -> bool:
    """Check if a resource exists."""
    try:
        client_method(name)
        return True
    except ResourceNotFoundError:
        return False


def _delete_if_exists(delete_method, name: str, resource_type: str = "resource"):
    """Delete a resource if it exists."""
    try:
        delete_method(name)
        print(f"  ✓ Deleted {resource_type}: {name}")
    except ResourceNotFoundError:
        print(f"  - {resource_type.capitalize()} not found: {name}")
    except Exception as e:
        print(f"  ✗ Error deleting {resource_type} {name}: {e}")


# ---------------------------------------------------------------------------
# Resource Cleanup
# ---------------------------------------------------------------------------


def delete_all_resources(index_name: str = None):
    """Delete all resources in the correct order."""
    if index_name is None:
        index_name = os.getenv("AZURE_SEARCH_INDEX_NAME", "research-agent-idx")

    endpoint = os.getenv("AZURE_AI_SEARCH_ENDPOINT")
    key = os.getenv("AZURE_AI_SEARCH_API_KEY")

    if not endpoint or not key:
        raise ValueError("Set AZURE_AI_SEARCH_ENDPOINT and AZURE_AI_SEARCH_API_KEY")

    # Define resource names
    ds_name = f"{index_name}-datasource"
    ss_name = f"{index_name}-skillset"
    ixr_name = f"{index_name}-indexer"

    # Initialize clients
    index_client = SearchIndexClient(endpoint, AzureKeyCredential(key))
    indexer_client = SearchIndexerClient(endpoint, AzureKeyCredential(key))

    print(f"\n🗑️  Deleting all resources for index '{index_name}'...\n")

    # Delete in reverse order of creation
    _delete_if_exists(indexer_client.delete_indexer, ixr_name, "indexer")
    time.sleep(1)

    _delete_if_exists(indexer_client.delete_skillset, ss_name, "skillset")
    time.sleep(1)

    _delete_if_exists(
        indexer_client.delete_data_source_connection, ds_name, "data source"
    )
    time.sleep(1)

    _delete_if_exists(index_client.delete_index, index_name, "index")
    time.sleep(2)

    print("\n✅ All resources deleted successfully!\n")


# ---------------------------------------------------------------------------
# Index Creation
# ---------------------------------------------------------------------------


def create_search_index(
    index_client: SearchIndexClient, index_name: str, recreate: bool = False
):
    """Create the search index with vector search capabilities."""
    if _exists(index_client.get_index, index_name):
        if recreate:
            print(f"  → Recreating index {index_name}...")
            _delete_if_exists(index_client.delete_index, index_name, "index")
            time.sleep(2)
        else:
            print(f"  → Index '{index_name}' already exists. Using existing index.")
            return index_client.get_index(index_name)

    # Define fields
    fields = [
        SearchField(
            name="chunk_id",
            type=SearchFieldDataType.String,
            key=True,
            searchable=True,
            filterable=False,
            sortable=True,
            facetable=False,
            analyzer_name=LexicalAnalyzerName.KEYWORD,
        ),
        SimpleField(
            name="parent_id",
            type=SearchFieldDataType.String,
            filterable=True,
            sortable=False,
            facetable=False,
        ),
        SearchableField(
            name="chunk",
            type=SearchFieldDataType.String,
            filterable=False,
            sortable=False,
            facetable=False,
        ),
        SearchableField(
            name="title",
            type=SearchFieldDataType.String,
            filterable=True,
            sortable=False,
            facetable=False,
        ),
        SearchField(
            name="content_embedding",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            hidden=False,
            vector_search_dimensions=3072,
            vector_search_profile_name=f"{index_name}-vec-profile",
        ),
    ]

    # Configure vector search
    vec_profile = VectorSearchProfile(
        name=f"{index_name}-vec-profile",
        algorithm_configuration_name=f"{index_name}-hnsw",
        vectorizer_name=f"{index_name}-oai-vec",
    )

    # Create index
    index = SearchIndex(
        name=index_name,
        fields=fields,
        vector_search=VectorSearch(
            algorithms=[
                HnswAlgorithmConfiguration(
                    name=f"{index_name}-hnsw",
                    parameters=HnswParameters(
                        metric=VectorSearchAlgorithmMetric.COSINE,
                        m=4,
                        ef_construction=400,
                        ef_search=500,
                    ),
                )
            ],
            profiles=[vec_profile],
            vectorizers=[
                AzureOpenAIVectorizer(
                    vectorizer_name=f"{index_name}-oai-vec",
                    parameters=AzureOpenAIVectorizerParameters(
                        resource_url=os.getenv("AZURE_OPENAI_ENDPOINT"),
                        deployment_name=os.getenv("AZURE_EMBEDDING_DEPLOYMENT_NAME"),
                        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                        model_name=os.getenv(
                            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
                            "text-embedding-3-large",
                        ),
                    ),
                )
            ],
        ),
        semantic_search=SemanticSearch(
            default_configuration_name=f"{index_name}-semantic",
            configurations=[
                SemanticConfiguration(
                    name=f"{index_name}-semantic",
                    prioritized_fields=SemanticPrioritizedFields(
                        title_field=SemanticField(field_name="title"),
                        content_fields=[SemanticField(field_name="chunk")],
                        keywords_fields=[],
                    ),
                )
            ],
        ),
        similarity=BM25SimilarityAlgorithm(),
    )

    result = index_client.create_index(index)
    print(f"  ✓ Index '{result.name}' created successfully")
    return result


# ---------------------------------------------------------------------------
# Data Source Creation
# ---------------------------------------------------------------------------


def create_blob_datasource(indexer_client: SearchIndexerClient, data_source_name: str):
    """Create a data source connection to Azure Blob Storage."""
    conn_str = os.getenv("AZURE_BLOB_CONNECTION_STRING")
    container = os.getenv("AZURE_BLOB_CONTAINER_NAME", "documents")

    if not conn_str:
        raise ValueError("AZURE_BLOB_CONNECTION_STRING not set")

    ds = SearchIndexerDataSourceConnection(
        name=data_source_name,
        type=SearchIndexerDataSourceType.AZURE_BLOB,
        connection_string=conn_str,
        container=SearchIndexerDataContainer(name=container),
    )

    result = indexer_client.create_or_update_data_source_connection(ds)
    print(f"  ✓ Data source '{result.name}' created successfully")
    return result


# ---------------------------------------------------------------------------
# Skillset Creation with Index Projections
# ---------------------------------------------------------------------------


def create_skillset(
    indexer_client: SearchIndexerClient, skillset_name: str, index_name: str
):
    """Create a skillset with text splitting and embedding generation."""
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    deployment = os.getenv("AZURE_EMBEDDING_DEPLOYMENT_NAME")

    if not all([endpoint, api_key, deployment]):
        raise ValueError("Azure OpenAI environment variables missing")

    # Text splitting skill
    split_skill = SplitSkill(
        name="split_skill",
        description="Split documents into chunks",
        context="/document",
        text_split_mode="pages",
        maximum_page_length=5000,
        page_overlap_length=500,
        default_language_code=SplitSkillLanguage.EN,
        inputs=[InputFieldMappingEntry(name="text", source="/document/content")],
        outputs=[OutputFieldMappingEntry(name="textItems", target_name="pages")],
    )

    # Embedding generation skill
    embed_skill = AzureOpenAIEmbeddingSkill(
        name="embed_skill",
        description="Generate embeddings using Azure OpenAI",
        context="/document/pages/*",
        resource_url=endpoint,
        api_key=api_key,
        deployment_name=deployment,
        model_name=os.getenv(
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large"
        ),
        inputs=[InputFieldMappingEntry(name="text", source="/document/pages/*")],
        outputs=[
            OutputFieldMappingEntry(name="embedding", target_name="content_embedding")
        ],
    )

    # Index projections for one-to-many document splitting
    projection = SearchIndexerIndexProjection(
        selectors=[
            SearchIndexerIndexProjectionSelector(
                target_index_name=index_name,
                parent_key_field_name="parent_id",
                source_context="/document/pages/*",
                mappings=[
                    InputFieldMappingEntry(name="chunk", source="/document/pages/*"),
                    InputFieldMappingEntry(
                        name="content_embedding",
                        source="/document/pages/*/content_embedding",
                    ),
                    InputFieldMappingEntry(
                        name="title", source="/document/metadata_storage_name"
                    ),
                ],
            )
        ],
        parameters=SearchIndexerIndexProjectionsParameters(
            projection_mode="skipIndexingParentDocuments",
        ),
    )

    # Create skillset
    skillset = SearchIndexerSkillset(
        name=skillset_name,
        description="Skillset for chunking documents and generating embeddings",
        skills=[split_skill, embed_skill],
        index_projection=projection,
    )

    result = indexer_client.create_or_update_skillset(skillset)
    print(f"  ✓ Skillset '{result.name}' created successfully")
    return result


# ---------------------------------------------------------------------------
# Indexer Creation
# ---------------------------------------------------------------------------


def create_indexer(
    indexer_client: SearchIndexerClient,
    indexer_name: str,
    data_source_name: str,
    skillset_name: str,
    index_name: str,
):
    """Create an indexer to process documents from blob storage."""
    params = IndexingParameters(
        batch_size=200,
        max_failed_items=20,
        max_failed_items_per_batch=20,
        configuration={
            "parsingMode": BlobIndexerParsingMode.DEFAULT,
            "indexStorageMode": "skipIndexingParentDocuments",
            "allowSkillsetToReadFileData": True,
            "dataToExtract": "contentAndMetadata",
        },
    )

    indexer = SearchIndexer(
        name=indexer_name,
        description="Indexer for processing documents from blob storage",
        data_source_name=data_source_name,
        target_index_name=index_name,
        skillset_name=skillset_name,
        field_mappings=[
            FieldMapping(
                source_field_name="metadata_storage_path", target_field_name="parent_id"
            ),
            FieldMapping(
                source_field_name="metadata_storage_name", target_field_name="title"
            ),
        ],
        parameters=params,
    )

    result = indexer_client.create_or_update_indexer(indexer)
    print(f"  ✓ Indexer '{result.name}' created successfully")
    return result


# ---------------------------------------------------------------------------
# Status Monitoring
# ---------------------------------------------------------------------------


def check_indexer_status(indexer_client: SearchIndexerClient, indexer_name: str):
    """Check and display detailed indexer status."""
    status = indexer_client.get_indexer_status(indexer_name)
    print(f"\n📊 Indexer Status: {status.status}")

    if status.last_result:
        print(f"   Last Run: {status.last_result.status}")
        print(
            f"   Documents: {status.last_result.item_count} processed, {status.last_result.failed_item_count} failed"
        )

        if status.last_result.errors:
            print("\n   ❌ Errors:")
            for i, error in enumerate(status.last_result.errors[:5], 1):
                print(f"      {i}. {error.error_message}")
            if len(status.last_result.errors) > 5:
                print(f"      ... and {len(status.last_result.errors) - 5} more errors")

        if status.last_result.warnings:
            print("\n   ⚠️  Warnings:")
            for i, warning in enumerate(status.last_result.warnings[:5], 1):
                print(f"      {i}. {warning.message}")
            if len(status.last_result.warnings) > 5:
                print(
                    f"      ... and {len(status.last_result.warnings) - 5} more warnings"
                )
    else:
        print("   No runs yet")


# ---------------------------------------------------------------------------
# Main Orchestration
# ---------------------------------------------------------------------------


def main(delete_first=True, recreate_index=False, reset_and_run=True):
    """
    Main function to orchestrate the creation of all Azure AI Search resources.

    Args:
        delete_first: Delete all existing resources before creating new ones
        recreate_index: Force recreation of the index even if it exists
        reset_and_run: Reset and run the indexer after creation
    """
    # Validate environment
    endpoint = os.getenv("AZURE_AI_SEARCH_ENDPOINT")
    key = os.getenv("AZURE_AI_SEARCH_API_KEY")
    if not endpoint or not key:
        raise ValueError("Set AZURE_AI_SEARCH_ENDPOINT and AZURE_AI_SEARCH_API_KEY")

    # Define resource names
    index_name = os.getenv("AZURE_SEARCH_INDEX_NAME", "research-agent-idx")
    ds_name = f"{index_name}-datasource"
    ss_name = f"{index_name}-skillset"
    ixr_name = f"{index_name}-indexer"

    # Delete existing resources if requested
    if delete_first:
        delete_all_resources(index_name)

    # Initialize clients
    index_client = SearchIndexClient(endpoint, AzureKeyCredential(key))
    indexer_client = SearchIndexerClient(endpoint, AzureKeyCredential(key))

    print("\n🚀 Creating Azure AI Search resources...\n")

    try:
        # Create resources in order
        print("1️⃣  Creating search index...")
        create_search_index(index_client, index_name, recreate_index)

        print("\n2️⃣  Creating blob data source...")
        create_blob_datasource(indexer_client, ds_name)

        print("\ 3️⃣  Creating skillset with index projections...")
        create_skillset(indexer_client, ss_name, index_name)

        print("\n4️⃣  Creating indexer...")
        create_indexer(indexer_client, ixr_name, ds_name, ss_name, index_name)

        if reset_and_run:
            print("\n5️⃣  Starting indexer...")
            indexer_client.reset_indexer(ixr_name)
            time.sleep(2)
            indexer_client.run_indexer(ixr_name)
            print("  ✓ Indexer started")

            print("\n⏳ Waiting for initial processing...")
            time.sleep(15)

            check_indexer_status(indexer_client, ixr_name)

        print("\n✅ Setup completed successfully!")
        print(f"\n💡 Tips:")
        print(f"   - Monitor progress in Azure Portal")
        print(f"   - Check status: check_indexer_status(indexer_client, '{ixr_name}')")
        print(f"   - View documents in index '{index_name}'")

    except Exception as e:
        print(f"\n❌ Error during setup: {e}")
        raise


# ---------------------------------------------------------------------------
# Script Execution
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    # Default: Delete all resources first, then create everything fresh
    main(delete_first=True, recreate_index=False, reset_and_run=True)

    # Alternative options:
    # - Keep existing resources: main(delete_first=False)
    # - Force recreate index only: main(delete_first=False, recreate_index=True)
    # - Create without running: main(reset_and_run=False)
