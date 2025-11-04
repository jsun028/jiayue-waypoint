#!/bin/bash

# Unified script for query generation and query execution
# Usage:
#   ./run.sh query "natural language query" [--dump-pickle path.pkl] [other options...]
#   ./run.sh execute --spec spec.pkl --data dataset.csv [other options...]

set -e

MODE="$1"
shift

case "$MODE" in
    query)
        # Query generation mode: NL → QuerySpec
        # Supports all NL_dspy options including:
        #   --dump-pickle, --dump-json, --generate-stats, --stats-sample, --model, --temperature, --api-key
        NL_QUERY="$1"
        shift
        
        if [ -z "$NL_QUERY" ]; then
            echo "Error: Natural language query is required"
            echo "Usage: ./run.sh query \"your query text\" [--dump-pickle path.pkl] [other options...]"
            exit 1
        fi
        
        # Build command arguments
        ARGS=("--nl" "$NL_QUERY")
        
        # Pass through all remaining arguments (including --generate-stats, --stats-sample, etc.)
        while [[ $# -gt 0 ]]; do
            ARGS+=("$1")
            shift
        done
        
        # Run query generation
        python -m NL_dspy "${ARGS[@]}"
        ;;
    
    execute)
        # Query execution mode: QuerySpec → Results
        ARGS=()
        
        # Parse arguments and pass through to main.py
        while [[ $# -gt 0 ]]; do
            case "$1" in
                --animate-gif)
                    # Map --animate-gif to --viz (which generates GIFs)
                    ARGS+=("--viz")
                    shift
                    ;;
                *)
                    ARGS+=("$1")
                    shift
                    ;;
            esac
        done
        
        # Run query execution
        python main.py "${ARGS[@]}"
        ;;
    
    *)
        echo "Error: Invalid mode '$MODE'"
        echo ""
        echo "Usage:"
        echo "  Query generation:"
        echo "    ./run.sh query \"natural language query\" [--dump-pickle path.pkl] [options...]"
        echo ""
        echo "  Query execution:"
        echo "    ./run.sh execute --spec spec.pkl --data dataset.csv [options...]"
        echo ""
        echo "Examples:"
        echo "  # Query generation (without ego):"
        echo "  ./run.sh query \"two cars approach head-on\" --dump-pickle spec.pkl"
        echo ""
        echo "  # Query generation (with ego object):"
        echo "  ./run.sh query \"ego vehicle approaches a pedestrian and stops\" --dump-pickle spec_ego.pkl"
        echo "  ./run.sh query \"find cases where ego nearly hits a car\" --dump-pickle spec_ego.pkl"
        echo ""
        echo "  # Query generation with dataset statistics:"
        echo "  ./run.sh query \"ego approaches pedestrian\" --generate-stats --dataset-csv dataset/scene_scene-0225.csv --dump-pickle spec_ego.pkl"
        echo "  ./run.sh query \"two cars approach\" --generate-stats --dataset-csv dataset/scene_scene-0225.csv --stats-sample 0.5 --dump-pickle spec.pkl"
        echo "  # Or use environment variable:"
        echo "  KEYFRAME_DATASET_CSV=dataset/scene_scene-0225.csv ./run.sh query \"two cars approach\" --generate-stats --dump-pickle spec.pkl"
        echo ""
        echo "  # Query execution:"
        echo "  ./run.sh execute --spec spec.pkl --data dataset/scene_scene-0225.csv --metadata-path NL/metadata/scene_scene-0225_stats.json --animate-gif"
        echo "  ./run.sh execute --spec spec.pkl --data dataset/scene_scene-0225.csv --metadata-path NL/metadata/scene_scene-0225_stats.json --estimation-mode"
        exit 1
        ;;
esac

