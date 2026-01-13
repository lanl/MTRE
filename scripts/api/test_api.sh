#!/bin/bash
# Test API connectivity
#
# Usage:
#   export API_BASE='https://your-api-endpoint.com'
#   export API_KEY='your-api-key-here'
#   bash test_api.sh

if [ -z "$API_BASE" ] || [ -z "$API_KEY" ]; then
    echo "Error: Set API_BASE and API_KEY environment variables first"
    echo "Example:"
    echo "  export API_BASE='https://api.example.com'"
    echo "  export API_KEY='sk-your-key-here'"
    exit 1
fi

curl -sS "$API_BASE/models" \
  -H "Authorization: Bearer $API_KEY"
