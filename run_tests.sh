#!/bin/bash
# Test runner script for keyframe-ui

set -e

echo "=========================================="
echo "Keyframe Query Compiler - Test Suite"
echo "=========================================="
echo ""

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo "Error: pytest not found. Please install it with:"
    echo "  pip install pytest"
    exit 1
fi

# Run tests based on arguments
if [ "$1" = "unit" ]; then
    echo "Running unit tests..."
    pytest tests/test_udf_scoring.py -v
elif [ "$1" = "integration" ]; then
    echo "Running integration tests..."
    pytest tests/test_predicate_evaluation.py -v
elif [ "$1" = "e2e" ]; then
    echo "Running end-to-end tests..."
    pytest tests/test_end_to_end.py -v
elif [ "$1" = "all" ] || [ -z "$1" ]; then
    echo "Running all tests..."
    pytest tests/ -v
else
    echo "Usage: $0 [unit|integration|e2e|all]"
    exit 1
fi

echo ""
echo "=========================================="
echo "Tests completed!"
echo "=========================================="

