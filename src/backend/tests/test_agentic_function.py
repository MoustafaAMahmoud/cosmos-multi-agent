#!/usr/bin/env python3
"""
Test the exact agentic retrieval function that agents call

This test bypasses semantic kernel setup and directly tests 
the agentic_retrieval function as it would be called by agents.
"""

import os
import sys
import json
import logging
from dotenv import load_dotenv

# Add the backend directory to Python path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, backend_dir)

# Import the simplified components we need
from patterns.agentic_retrieval_plugin import AgenticRetrievalPlugin

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_agentic_retrieval_function():
    """Test the agentic_retrieval function directly."""
    print("TESTING AGENTIC RETRIEVAL FUNCTION")
    print("="*50)
    
    try:
        # Create plugin instance (without semantic kernel)
        print("1. Creating AgenticRetrievalPlugin...")
        plugin = AgenticRetrievalPlugin(kernel=None)
        print("✓ Plugin created successfully")
        
        # Test scenarios that match what agents would call
        test_scenarios = [
            {
                "name": "E-cigarette health research",
                "query": "what are the latest research about e-cigarettes health effects?",
                "setup": lambda p: p.set_current_query("what are the latest research about e-cigarettes health effects?")
            },
            {
                "name": "Vaping safety studies", 
                "query": "vaping safety studies",
                "setup": lambda p: p.set_current_query("vaping safety studies")
            },
            {
                "name": "No query context",
                "query": None,
                "setup": lambda p: None
            }
        ]
        
        for i, scenario in enumerate(test_scenarios, 1):
            print(f"\n{i}. Testing scenario: {scenario['name']}")
            print("-" * 30)
            
            # Setup scenario
            if scenario["setup"]:
                scenario["setup"](plugin)
                
            # Call the agentic_retrieval function
            try:
                result_json = plugin.agentic_retrieval(
                    query=scenario["query"],
                    conversation_id=f"test_{i}"
                )
                
                print(f"✓ Function call successful")
                print(f"✓ Returned JSON string of length: {len(result_json)}")
                
                # Parse and analyze result
                try:
                    result_data = json.loads(result_json)
                    print(f"✓ Valid JSON returned")
                    
                    # Show result structure
                    print(f"  Status: {result_data.get('status', 'unknown')}")
                    
                    if result_data.get('status') == 'success':
                        total_results = result_data.get('total_results', 0)
                        query_count = result_data.get('query_count', 0)
                        
                        print(f"  Generated {query_count} queries")
                        print(f"  Found {total_results} results")
                        
                        # Show first few results
                        results = result_data.get('results', [])
                        for j, res in enumerate(results[:3], 1):
                            ref_id = res.get('ref_id', 'N/A')
                            title = res.get('title', 'No title')
                            score = res.get('relevance_score', 'N/A')
                            content_preview = res.get('content', '')[:100] + '...' if len(res.get('content', '')) > 100 else res.get('content', '')
                            
                            print(f"    Result {j}: {ref_id}")
                            print(f"      Title: {title}")  
                            print(f"      Score: {score}")
                            print(f"      Content: {content_preview}")
                            
                    elif result_data.get('status') == 'error':
                        error_msg = result_data.get('message', 'Unknown error')
                        print(f"  ✗ Error: {error_msg}")
                        
                except json.JSONDecodeError as e:
                    print(f"  ✗ Invalid JSON: {e}")
                    print(f"  Raw response: {result_json[:200]}...")
                    
            except Exception as e:
                print(f"  ✗ Function call failed: {e}")
                
        return True
        
    except Exception as e:
        print(f"✗ Test setup failed: {e}")
        return False


def test_query_generation():
    """Test the query generation logic specifically."""
    print("\n" + "="*50)
    print("TESTING QUERY GENERATION")
    print("="*50)
    
    try:
        plugin = AgenticRetrievalPlugin(kernel=None)
        
        # Test different conversation contexts
        test_contexts = [
            {
                "name": "E-cigarette user question",
                "context": [
                    {"role": "user", "content": "what are the latest research about e-cigarettes?"}
                ]
            },
            {
                "name": "Health effects question",
                "context": [
                    {"role": "user", "content": "Are e-cigarettes safer than traditional cigarettes?"},
                    {"role": "assistant", "content": "Let me search for that information."}
                ]
            },
            {
                "name": "Empty context",
                "context": []
            },
            {
                "name": "Non-user messages",
                "context": [
                    {"role": "assistant", "content": "I can help with research questions."},
                    {"role": "system", "content": "You are a research assistant."}
                ]
            }
        ]
        
        for i, test_case in enumerate(test_contexts, 1):
            print(f"\n{i}. Testing: {test_case['name']}")
            
            # Test query generation
            queries = plugin._generate_search_queries(test_case['context'])
            
            print(f"   Generated {len(queries)} queries:")
            for j, query in enumerate(queries, 1):
                print(f"     {j}. {query}")
                
        return True
        
    except Exception as e:
        print(f"✗ Query generation test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("AGENTIC RETRIEVAL FUNCTION TESTING")
    print("="*60)
    
    # Check environment
    endpoint = os.getenv('AZURE_AI_SEARCH_ENDPOINT')
    api_key = os.getenv('AZURE_AI_SEARCH_API_KEY')
    
    if not endpoint or not api_key:
        print("✗ Missing Azure AI Search credentials")
        print("  Set AZURE_AI_SEARCH_ENDPOINT and AZURE_AI_SEARCH_API_KEY")
        return
        
    print(f"✓ Azure AI Search endpoint: {endpoint}")
    print(f"✓ API key: {'Set' if api_key else 'Missing'}")
    
    # Run tests
    success1 = test_query_generation()
    success2 = test_agentic_retrieval_function()
    
    print("\n" + "="*60)
    if success1 and success2:
        print("✓ ALL TESTS COMPLETED SUCCESSFULLY")
    else:
        print("✗ SOME TESTS FAILED")
    print("="*60)


if __name__ == "__main__":
    main()