#!/usr/bin/env python3
"""
Standalone test runner for AgenticRetrievalPlugin tests

This script runs comprehensive tests on the agentic retrieval functionality
including connectivity tests and e-cigarette keyword searches.

Usage:
    python run_retrieval_tests.py
    
    Or from backend directory:
    python tests/run_retrieval_tests.py
"""

import os
import sys
import json
import logging
from datetime import datetime

# Add the backend directory to Python path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, backend_dir)

from patterns.agentic_retrieval_plugin import AgenticRetrievalPlugin
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_basic_functionality():
    """Test basic plugin functionality."""
    print("\n" + "="*60)
    print("TESTING BASIC FUNCTIONALITY")
    print("="*60)
    
    try:
        # Initialize plugin
        print("1. Initializing AgenticRetrievalPlugin...")
        plugin = AgenticRetrievalPlugin(kernel=None)
        print(f"✓ Plugin initialized successfully")
        print(f"  - Endpoint: {plugin.search_endpoint}")
        print(f"  - Service: {plugin.service_name}")
        print(f"  - Index: {plugin.index_name}")
        print(f"  - Base URL: {plugin.base_url}")
        
        # Test query setting
        print("\n2. Testing query context setting...")
        test_query = "what are the latest research about e-cigarettes?"
        plugin.set_current_query(test_query)
        print(f"✓ Current query set: {plugin.current_query}")
        
        # Test query generation
        print("\n3. Testing search query generation...")
        mock_context = [
            {"role": "user", "content": "what are the latest research about e-cigarettes health effects?"},
            {"role": "assistant", "content": "I'll search for that information."}
        ]
        
        queries = plugin._generate_search_queries(mock_context)
        print(f"✓ Generated {len(queries)} search queries:")
        for i, q in enumerate(queries, 1):
            print(f"    {i}. {q}")
            
        return plugin
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return None

def test_connectivity(plugin):
    """Test connectivity to Azure AI Search."""
    print("\n" + "="*60)
    print("TESTING AZURE AI SEARCH CONNECTIVITY")
    print("="*60)
    
    if not plugin:
        print("✗ Skipping connectivity tests - plugin not initialized")
        return False
        
    try:
        # Test search service connectivity
        print("1. Testing search service connectivity...")
        plugin._test_search_connectivity()
        print("✓ Connectivity test completed (check logs for details)")
        return True
        
    except Exception as e:
        print(f"✗ Connectivity test failed: {e}")
        return False

def test_simple_search(plugin):
    """Test simple search execution."""
    print("\n" + "="*60)
    print("TESTING SIMPLE SEARCH EXECUTION")
    print("="*60)
    
    if not plugin:
        print("✗ Skipping search tests - plugin not initialized")
        return False
        
    test_queries = [
        "research",
        "health",
        "study",
        "analysis"
    ]
    
    for query in test_queries:
        try:
            print(f"\n  Testing query: '{query}'")
            results = plugin._perform_semantic_search(query, top=3)
            
            print(f"    ✓ Query executed successfully")
            print(f"    ✓ Returned {len(results)} results")
            
            if results:
                first_result = results[0]
                print(f"    ✓ First result has keys: {list(first_result.keys())}")
                if 'content_text' in first_result:
                    content_preview = first_result['content_text'][:100] + "..." if len(first_result['content_text']) > 100 else first_result['content_text']
                    print(f"    ✓ Content preview: {content_preview}")
            else:
                print(f"    ! No results found for query '{query}'")
                
        except Exception as e:
            print(f"    ✗ Query '{query}' failed: {e}")
            
    return True

def test_ecigarette_searches(plugin):
    """Test e-cigarette specific searches."""
    print("\n" + "="*60)
    print("TESTING E-CIGARETTE KEYWORD SEARCHES")
    print("="*60)
    
    if not plugin:
        print("✗ Skipping e-cigarette tests - plugin not initialized")
        return False
        
    ecigarette_queries = [
        "e-cigarettes",
        "electronic cigarettes",
        "vaping",
        "vaping research",
        "tobacco harm reduction",
        "nicotine delivery",
        "smoking cessation",
        "vape health effects",
        "e-cigarette regulation",
        "vaping studies"
    ]
    
    results_summary = []
    
    for query in ecigarette_queries:
        try:
            print(f"\n  Testing e-cigarette query: '{query}'")
            results = plugin._perform_semantic_search(query, top=5)
            
            result_count = len(results)
            results_summary.append((query, result_count))
            
            print(f"    ✓ Query executed successfully")
            print(f"    ✓ Returned {result_count} results")
            
            if results:
                # Show relevance scores if available
                for i, result in enumerate(results[:2], 1):  # Show top 2 results
                    score = result.get('@search.reranker_score', result.get('@search.score', 'N/A'))
                    title = result.get('document_title', 'No title')
                    print(f"      {i}. Score: {score}, Title: {title}")
            else:
                print(f"    ! No results found for '{query}'")
                
        except Exception as e:
            print(f"    ✗ E-cigarette query '{query}' failed: {e}")
            results_summary.append((query, f"ERROR: {e}"))
    
    # Summary
    print(f"\n  E-CIGARETTE SEARCH SUMMARY:")
    print(f"  {'-'*40}")
    for query, count in results_summary:
        if isinstance(count, int):
            print(f"  {query:<25} : {count} results")
        else:
            print(f"  {query:<25} : {count}")
            
    return True

def test_full_agentic_retrieval(plugin):
    """Test the full agentic retrieval pipeline."""
    print("\n" + "="*60)
    print("TESTING FULL AGENTIC RETRIEVAL PIPELINE")
    print("="*60)
    
    if not plugin:
        print("✗ Skipping agentic retrieval tests - plugin not initialized")
        return False
        
    test_scenarios = [
        "what are the latest research about e-cigarettes?",
        "health effects of vaping",
        "e-cigarette regulation studies",
        "tobacco harm reduction research",
        "smoking cessation with e-cigarettes"
    ]
    
    for scenario in test_scenarios:
        try:
            print(f"\n  Testing scenario: '{scenario}'")
            
            # Set the query context
            plugin.set_current_query(scenario)
            
            # Call agentic retrieval
            result_json = plugin.agentic_retrieval()
            
            # Parse result
            result_data = json.loads(result_json)
            
            print(f"    ✓ Agentic retrieval completed")
            print(f"    ✓ Status: {result_data.get('status', 'unknown')}")
            
            if result_data.get('status') == 'success':
                total_results = result_data.get('total_results', 0)
                query_count = result_data.get('query_count', 0)
                
                print(f"    ✓ Generated {query_count} search queries")
                print(f"    ✓ Returned {total_results} results")
                
                if result_data.get('results'):
                    first_result = result_data['results'][0]
                    print(f"    ✓ First result ref_id: {first_result.get('ref_id', 'N/A')}")
                    print(f"    ✓ First result relevance: {first_result.get('relevance_score', 'N/A')}")
                    
            elif result_data.get('status') == 'error':
                print(f"    ! Error: {result_data.get('message', 'Unknown error')}")
            
        except json.JSONDecodeError as e:
            print(f"    ✗ Invalid JSON response: {e}")
        except Exception as e:
            print(f"    ✗ Agentic retrieval failed: {e}")
            
    return True

def main():
    """Main test runner."""
    print("AGENTIC RETRIEVAL PLUGIN - COMPREHENSIVE TEST SUITE")
    print("=" * 60)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check environment
    endpoint = os.getenv('AZURE_AI_SEARCH_ENDPOINT')
    api_key = os.getenv('AZURE_AI_SEARCH_API_KEY')
    
    print(f"\nEnvironment Check:")
    print(f"  AZURE_AI_SEARCH_ENDPOINT: {'✓ Set' if endpoint else '✗ Missing'}")
    print(f"  AZURE_AI_SEARCH_API_KEY: {'✓ Set' if api_key else '✗ Missing'}")
    
    if not endpoint or not api_key:
        print("\n⚠️  WARNING: Missing Azure AI Search credentials")
        print("   Some tests may fail or be skipped")
    
    # Run tests
    plugin = test_basic_functionality()
    
    if plugin:
        test_connectivity(plugin)
        test_simple_search(plugin)
        test_ecigarette_searches(plugin)
        test_full_agentic_retrieval(plugin)
    
    print("\n" + "="*60)
    print("TEST SUITE COMPLETED")
    print("="*60)
    print(f"Test finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if endpoint and api_key:
        print("\n✓ All tests executed with live Azure AI Search connection")
    else:
        print("\n! Some tests were skipped due to missing credentials")
        print("  To run full tests, set AZURE_AI_SEARCH_ENDPOINT and AZURE_AI_SEARCH_API_KEY")

if __name__ == "__main__":
    main()