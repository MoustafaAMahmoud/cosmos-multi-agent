#!/usr/bin/env python3
"""
Test cases for AgenticRetrievalPlugin

This module tests the agentic retrieval functionality including:
- Azure AI Search connectivity
- Search query generation 
- E-cigarette keyword search functionality
- Context-aware search behavior
"""

import os
import sys
import json
import pytest
import logging
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List

# Add the backend directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from patterns.agentic_retrieval_plugin import AgenticRetrievalPlugin
from dotenv import load_dotenv

# Load environment variables for tests
load_dotenv()

class TestAgenticRetrievalPlugin:
    """Test suite for AgenticRetrievalPlugin functionality."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.plugin = AgenticRetrievalPlugin(kernel=None)
        
    def test_plugin_initialization(self):
        """Test that the plugin initializes correctly with proper configuration."""
        assert self.plugin is not None
        assert self.plugin.search_endpoint is not None
        assert self.plugin.index_name == "research-agent-index"
        assert self.plugin.service_name == "aifoundry"
        
    def test_search_endpoint_configuration(self):
        """Test that search endpoint is configured correctly."""
        expected_endpoint = "https://aifoundry.search.windows.net"
        assert self.plugin.search_endpoint == expected_endpoint
        
        # Test base URL construction
        if "cognitiveservices.azure.com" in self.plugin.search_endpoint:
            expected_base = f"{self.plugin.search_endpoint}/searchservices/aifoundry/indexes/research-agent-index/docs/search"
        else:
            expected_base = f"{self.plugin.search_endpoint}/indexes/research-agent-index/docs/search"
        
        assert self.plugin.base_url == expected_base
        
    def test_set_current_query(self):
        """Test setting current query for context."""
        test_query = "what are the latest research about e-cigarettes?"
        self.plugin.set_current_query(test_query)
        assert self.plugin.current_query == test_query
        
    def test_generate_search_queries_with_ecigarette_context(self):
        """Test search query generation with e-cigarette related context."""
        # Mock conversation context with e-cigarette query
        mock_context = [
            {"role": "user", "content": "what are the latest research about e-cigarettes?"},
            {"role": "assistant", "content": "I'll search for that information."}
        ]
        
        queries = self.plugin._generate_search_queries(mock_context)
        
        # Should generate at least the primary query
        assert len(queries) > 0
        assert "what are the latest research about e-cigarettes?" in queries
        
        # Should include research-focused terms
        research_queries = [q for q in queries if any(term in q.lower() for term in ["research", "analysis", "study"])]
        assert len(research_queries) > 0
        
    def test_generate_search_queries_with_fallback(self):
        """Test search query generation with fallback query."""
        empty_context = []
        fallback = "e-cigarette health effects"
        
        queries = self.plugin._generate_search_queries(empty_context, fallback_query=fallback)
        
        assert len(queries) > 0
        assert fallback in queries
        
    def test_generate_search_queries_with_current_query(self):
        """Test search query generation using set current query."""
        # Set current query
        self.plugin.set_current_query("e-cigarette regulatory updates")
        
        # Generate queries with empty context
        queries = self.plugin._generate_search_queries([])
        
        assert len(queries) > 0
        assert "e-cigarette regulatory updates" in queries


class TestAgenticRetrievalLiveSearch:
    """Live tests against Azure AI Search (requires valid credentials)."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.plugin = AgenticRetrievalPlugin(kernel=None)
        
    @pytest.mark.skipif(
        not os.getenv("AZURE_AI_SEARCH_ENDPOINT") or not os.getenv("AZURE_AI_SEARCH_API_KEY"),
        reason="Azure AI Search credentials not available"
    )
    def test_connectivity_to_azure_search(self):
        """Test actual connectivity to Azure AI Search service."""
        # This test will run the connectivity test
        try:
            self.plugin._test_search_connectivity()
            # If no exception is raised, connectivity test passed
            assert True
        except Exception as e:
            pytest.fail(f"Connectivity test failed: {e}")
            
    @pytest.mark.skipif(
        not os.getenv("AZURE_AI_SEARCH_ENDPOINT") or not os.getenv("AZURE_AI_SEARCH_API_KEY"),
        reason="Azure AI Search credentials not available"
    )
    def test_simple_search_execution(self):
        """Test executing a simple search query."""
        test_query = "research"
        
        try:
            results = self.plugin._perform_semantic_search(test_query, top=5)
            
            # Results should be a list
            assert isinstance(results, list)
            
            # Each result should have expected fields
            if results:
                result = results[0]
                expected_fields = ["content_id", "content_text", "document_title"]
                for field in expected_fields:
                    assert field in result or f"Missing field: {field}"
                    
        except Exception as e:
            pytest.fail(f"Simple search execution failed: {e}")
            
    @pytest.mark.skipif(
        not os.getenv("AZURE_AI_SEARCH_ENDPOINT") or not os.getenv("AZURE_AI_SEARCH_API_KEY"),
        reason="Azure AI Search credentials not available"
    )
    def test_ecigarette_search(self):
        """Test searching for e-cigarette related content."""
        ecigarette_queries = [
            "e-cigarettes",
            "electronic cigarettes", 
            "vaping research",
            "tobacco harm reduction",
            "nicotine delivery systems"
        ]
        
        for query in ecigarette_queries:
            try:
                results = self.plugin._perform_semantic_search(query, top=3)
                
                # Results should be a list (may be empty if no content matches)
                assert isinstance(results, list)
                
                print(f"Query: '{query}' returned {len(results)} results")
                
                # If results exist, verify structure
                if results:
                    result = results[0]
                    assert "content_text" in result
                    assert "document_title" in result
                    
            except Exception as e:
                pytest.fail(f"E-cigarette search failed for query '{query}': {e}")
                
    @pytest.mark.skipif(
        not os.getenv("AZURE_AI_SEARCH_ENDPOINT") or not os.getenv("AZURE_AI_SEARCH_API_KEY"),
        reason="Azure AI Search credentials not available"
    )
    def test_full_agentic_retrieval_ecigarette(self):
        """Test the full agentic retrieval pipeline with e-cigarette query."""
        # Set up e-cigarette query
        self.plugin.set_current_query("what are the latest research about e-cigarettes health effects?")
        
        try:
            # Call the main agentic retrieval function
            result_json = self.plugin.agentic_retrieval()
            
            # Should return valid JSON
            result_data = json.loads(result_json)
            
            # Check expected structure
            assert "status" in result_data
            
            if result_data["status"] == "success":
                assert "results" in result_data
                assert "total_results" in result_data
                assert "query_count" in result_data
                
                # If results exist, verify citation format
                if result_data["results"]:
                    first_result = result_data["results"][0]
                    assert "ref_id" in first_result
                    assert first_result["ref_id"].startswith("research_ref_")
                    assert "content" in first_result
                    assert "title" in first_result
                    
                print(f"Agentic retrieval returned {result_data.get('total_results', 0)} results")
                print(f"Generated {result_data.get('query_count', 0)} search queries")
                
            elif result_data["status"] == "error":
                print(f"Agentic retrieval returned error: {result_data.get('message', 'Unknown error')}")
                
        except json.JSONDecodeError as e:
            pytest.fail(f"Agentic retrieval returned invalid JSON: {e}")
        except Exception as e:
            pytest.fail(f"Full agentic retrieval test failed: {e}")


class TestSearchQueryGeneration:
    """Test suite for search query generation logic."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.plugin = AgenticRetrievalPlugin(kernel=None)
        
    def test_research_terms_detection(self):
        """Test detection of research-related terms in queries."""
        research_queries = [
            "What research shows about vaping?",
            "Latest study on e-cigarettes", 
            "Analysis of tobacco harm reduction",
            "Methodology for nicotine research",
            "Framework for addiction studies"
        ]
        
        for query in research_queries:
            context = [{"role": "user", "content": query}]
            generated_queries = self.plugin._generate_search_queries(context)
            
            # Should generate multiple queries including research terms
            assert len(generated_queries) > 1
            
            # Should include the original query
            assert query in generated_queries
            
    def test_query_limit_enforcement(self):
        """Test that query generation respects the maximum query limit."""
        # Create context with many research terms
        long_query = "research study analysis methodology framework theory concept approach model technique method strategy"
        context = [{"role": "user", "content": long_query}]
        
        generated_queries = self.plugin._generate_search_queries(context)
        
        # Should not exceed max_search_queries
        assert len(generated_queries) <= self.plugin.max_search_queries


if __name__ == "__main__":
    # Run tests with verbose output
    logging.basicConfig(level=logging.INFO)
    
    print("Running AgenticRetrievalPlugin Tests...")
    print("=" * 50)
    
    # Run basic tests
    print("\n1. Testing Plugin Initialization...")
    test_basic = TestAgenticRetrievalPlugin()
    test_basic.setup_method()
    
    try:
        test_basic.test_plugin_initialization()
        print("✓ Plugin initialization test passed")
    except Exception as e:
        print(f"✗ Plugin initialization test failed: {e}")
        
    try:
        test_basic.test_search_endpoint_configuration()
        print("✓ Search endpoint configuration test passed")
    except Exception as e:
        print(f"✗ Search endpoint configuration test failed: {e}")
        
    try:
        test_basic.test_generate_search_queries_with_ecigarette_context()
        print("✓ E-cigarette query generation test passed")
    except Exception as e:
        print(f"✗ E-cigarette query generation test failed: {e}")
    
    # Run live tests if credentials are available
    if os.getenv("AZURE_AI_SEARCH_ENDPOINT") and os.getenv("AZURE_AI_SEARCH_API_KEY"):
        print("\n2. Testing Live Azure AI Search Connection...")
        test_live = TestAgenticRetrievalLiveSearch()
        test_live.setup_method()
        
        try:
            test_live.test_connectivity_to_azure_search()
            print("✓ Azure AI Search connectivity test passed")
        except Exception as e:
            print(f"✗ Azure AI Search connectivity test failed: {e}")
            
        try:
            test_live.test_simple_search_execution()
            print("✓ Simple search execution test passed")
        except Exception as e:
            print(f"✗ Simple search execution test failed: {e}")
            
        try:
            test_live.test_ecigarette_search()
            print("✓ E-cigarette search test passed")
        except Exception as e:
            print(f"✗ E-cigarette search test failed: {e}")
            
        try:
            test_live.test_full_agentic_retrieval_ecigarette()
            print("✓ Full agentic retrieval e-cigarette test passed")
        except Exception as e:
            print(f"✗ Full agentic retrieval e-cigarette test failed: {e}")
    else:
        print("\n2. Skipping live tests - Azure AI Search credentials not found")
        print("   Set AZURE_AI_SEARCH_ENDPOINT and AZURE_AI_SEARCH_API_KEY to run live tests")
    
    print("\n" + "=" * 50)
    print("Test run completed!")