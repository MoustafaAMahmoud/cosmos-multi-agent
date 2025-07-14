#!/bin/bash

# Test the Scientific Research API
echo "Testing Scientific Research API..."

curl -X POST http://localhost:8000/api/v1/research-support \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the latest developments in renewable energy storage technologies?",
    "user_id": "test-user-001",
    "include_debate_details": false,
    "maximum_iterations": 5
  }'

# Note: This is a streaming API that returns multiple JSON responses
# Each response contains research findings from the multi-agent system:
# - ScientificResearchAgent (Literature search and analysis)
# - ScientificAnalyticsAgent (Statistical analysis and metrics)
# - ScientificResearchCritic (Quality evaluation and feedback)