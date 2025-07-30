import json
import sys
import os

# Add the parent directory to the path so we can import from patterns
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from patterns.search_plugin import azure_ai_search_plugin


def main():
    # 1) Invoke without exclusions
    # print("== Search without exclusions ==")
    # result1 = azure_ai_search_plugin(query="vaping")
    # print(json.dumps(result1, indent=2))

    # 2) Invoke with a list of document titles to exclude
    print("\n== Search excluding specific titles ==")
    excluded = ["BR102022001563A2.pdf", "BR102022001563A2.pdf"]
    result2 = azure_ai_search_plugin(query="vaping", excluded_titles=excluded)
    print(json.dumps(result2, indent=2))


if __name__ == "__main__":
    main()
