#!/usr/bin/env python3
"""
Simple Azure AI Search connectivity test for e-cigarette research

This test bypasses the full semantic kernel setup and directly tests 
the Azure AI Search connectivity and search functionality.
"""

import os
import sys
import json
import requests
import logging
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimpleSearchTester:
    """Simple Azure AI Search connectivity tester."""
    
    def __init__(self):
        """Initialize with environment configuration."""
        self.search_endpoint = os.getenv("AZURE_AI_SEARCH_ENDPOINT")
        self.search_api_key = os.getenv("AZURE_AI_SEARCH_API_KEY")
        self.service_name = os.getenv("AZURE_AI_SEARCH_SERVICE_NAME", "aifoundry")
        self.index_name = os.getenv("AZURE_SEARCH_INDEX_NAME", "research-agent-index")
        self.semantic_configuration = os.getenv(
            "AZURE_SEARCH_SEMANTIC_CONFIG", "research-agent-index-semantic-configuration"
        )
        
        # Construct base URL
        if "cognitiveservices.azure.com" in (self.search_endpoint or ""):
            self.base_url = f"{self.search_endpoint.rstrip('/')}/searchservices/{self.service_name}/indexes/{self.index_name}/docs/search"
        else:
            self.base_url = f"{self.search_endpoint.rstrip('/')}/indexes/{self.index_name}/docs/search"
            
        print(f"Configuration:")
        print(f"  Endpoint: {self.search_endpoint}")
        print(f"  Service: {self.service_name}")
        print(f"  Index: {self.index_name}")
        print(f"  Base URL: {self.base_url}")
        print(f"  Has API Key: {bool(self.search_api_key)}")
        
    def test_index_exists(self):
        """Test if the search index exists."""
        print(f"\n{'='*50}")
        print("TESTING INDEX EXISTENCE")
        print('='*50)
        
        try:
            # Test with a simple GET to the index
            test_url = f"{self.search_endpoint.rstrip('/')}/indexes/{self.index_name}?api-version=2024-05-01-Preview"
            headers = {"Content-Type": "application/json"}
            if self.search_api_key:
                headers["api-key"] = self.search_api_key
                
            print(f"Testing URL: {test_url}")
            response = requests.get(test_url, headers=headers, timeout=10)
            
            print(f"Response Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"✓ Index '{self.index_name}' exists")
                print(f"  - Field count: {len(data.get('fields', []))}")
                print(f"  - Has semantic config: {'semanticSearch' in data}")
                return True
            else:
                print(f"✗ Index access failed: {response.text}")
                return False
                
        except Exception as e:
            print(f"✗ Index existence test failed: {e}")
            return False
    
    def test_simple_search(self, query, top=5):
        """Test a simple search query."""
        try:
            url = f"{self.base_url}?api-version=2024-05-01-Preview"
            headers = {"Content-Type": "application/json"}
            
            if self.search_api_key:
                headers["api-key"] = self.search_api_key
            
            # Simple search payload
            payload = {
                "search": query,
                "select": "*",
                "queryType": "simple",
                "top": top,
            }
            
            print(f"  Query: '{query}'")
            print(f"  URL: {url}")
            print(f"  Payload: {json.dumps(payload, indent=2)}")
            
            response = requests.post(
                url, headers=headers, data=json.dumps(payload), timeout=30
            )
            
            print(f"  Response Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                results = data.get("value", [])
                
                print(f"  ✓ Query successful")
                print(f"  ✓ Found {len(results)} results")
                
                if results:
                    print(f"  ✓ Response keys: {list(data.keys())}")
                    print(f"  ✓ First result keys: {list(results[0].keys())}")
                    
                    # Show first result details
                    first_result = results[0]
                    for key, value in first_result.items():
                        if isinstance(value, str) and len(value) > 100:
                            print(f"    {key}: {value[:100]}...")
                        else:
                            print(f"    {key}: {value}")
                else:
                    print(f"  ! No results found for '{query}'")
                    
                return results
            else:
                print(f"  ✗ Search failed: {response.text}")
                return []
                
        except Exception as e:
            print(f"  ✗ Search error: {e}")
            return []
    
    def test_semantic_search(self, query, top=5):
        """Test semantic search query."""
        try:
            url = f"{self.base_url}?api-version=2024-05-01-Preview"
            headers = {"Content-Type": "application/json"}
            
            if self.search_api_key:
                headers["api-key"] = self.search_api_key
            
            # Semantic search payload
            payload = {
                "search": query,
                "select": "*",
                "queryType": "semantic",
                "semanticConfiguration": self.semantic_configuration,
                "top": top,
            }
            
            print(f"  Semantic Query: '{query}'")
            
            response = requests.post(
                url, headers=headers, data=json.dumps(payload), timeout=30
            )
            
            print(f"  Response Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                results = data.get("value", [])
                
                print(f"  ✓ Semantic query successful")
                print(f"  ✓ Found {len(results)} results")
                
                if results:
                    # Show semantic scores
                    for i, result in enumerate(results[:3], 1):
                        reranker_score = result.get('@search.reranker_score', 'N/A')
                        search_score = result.get('@search.score', 'N/A')
                        title = result.get('document_title', result.get('title', 'No title'))
                        print(f"    {i}. Reranker: {reranker_score}, Score: {search_score}")
                        print(f"       Title: {title}")
                        
                return results
            else:
                print(f"  ✗ Semantic search failed: {response.text}")
                return []
                
        except Exception as e:
            print(f"  ✗ Semantic search error: {e}")
            return []

def main():
    """Run the search connectivity tests."""
    print("AZURE AI SEARCH CONNECTIVITY TEST FOR E-CIGARETTE RESEARCH")
    print("=" * 70)
    print(f"Test started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize tester
    tester = SimpleSearchTester()
    
    # Check prerequisites
    if not tester.search_endpoint:
        print("✗ AZURE_AI_SEARCH_ENDPOINT not set")
        return
    
    if not tester.search_api_key:
        print("✗ AZURE_AI_SEARCH_API_KEY not set")
        return
    
    # Test index existence
    index_exists = tester.test_index_exists()
    
    if not index_exists:
        print("\n⚠️  Index doesn't exist or isn't accessible. Stopping tests.")
        return
    
    # Test simple searches
    print(f"\n{'='*50}")
    print("TESTING SIMPLE SEARCHES")
    print('='*50)
    
    simple_queries = [
        "research",
        "health",
        "study",
        "analysis",
        "data"
    ]
    
    for query in simple_queries:
        print(f"\nTesting simple search:")
        results = tester.test_simple_search(query, top=3)
    
    # Test e-cigarette specific searches
    print(f"\n{'='*50}")
    print("TESTING E-CIGARETTE SEARCHES")
    print('='*50)
    
    ecigarette_queries = [
        "e-cigarettes",
        "electronic cigarettes",
        "vaping",
        "tobacco harm reduction",
        "nicotine delivery",
        "smoking cessation",
        "vape health effects"
    ]
    
    for query in ecigarette_queries:
        print(f"\nTesting e-cigarette search:")
        results = tester.test_simple_search(query, top=5)
    
    # Test semantic searches
    print(f"\n{'='*50}")
    print("TESTING SEMANTIC SEARCHES")
    print('='*50)
    
    semantic_queries = [
        "latest research about e-cigarettes health effects",
        "vaping safety studies",
        "electronic cigarette regulation"
    ]
    
    for query in semantic_queries:
        print(f"\nTesting semantic search:")
        results = tester.test_semantic_search(query, top=3)
    
    print(f"\n{'='*70}")
    print("TEST COMPLETED")
    print("="*70)
    print(f"Test finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()