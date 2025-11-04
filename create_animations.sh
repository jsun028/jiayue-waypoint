#!/bin/bash

# Batch generate animations for all CSV files in NL/dataset
# Usage: ./create_animations.sh [--fps FPS] [--force]

set -e

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default values
FPS=10
DATASET_DIR="NL/dataset"
OUTPUT_DIR="NL/dataset/animations"
FORCE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --fps)
            FPS="$2"
            shift 2
            ;;
        --force)
            FORCE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: ./create_animations.sh [--fps FPS] [--force]"
            exit 1
            ;;
    esac
done

echo "=============================================="
echo "Batch Animation Generator"
echo "=============================================="
echo "Dataset directory: $DATASET_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "FPS: $FPS"
echo "Force recreate: $FORCE"
echo "=============================================="

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Find all CSV files
CSV_FILES=($(find "$DATASET_DIR" -name "*.csv" -type f | sort))

echo "Found ${#CSV_FILES[@]} CSV files"
echo ""

# Counters
SUCCESSFUL=0
FAILED=0
SKIPPED=0

# Process each CSV file
for csv_file in "${CSV_FILES[@]}"; do
    # Get filename without extension
    filename=$(basename "$csv_file" .csv)
    
    # Generate output path
    output_file="$OUTPUT_DIR/${filename}.gif"
    
    # Skip if exists (unless force)
    if [ -f "$output_file" ] && [ "$FORCE" = false ]; then
        echo "⏭️  Skipping $filename.gif (already exists)"
        ((SKIPPED++))
        continue
    fi
    
    echo "🎬 Processing $filename..."
    
    # Run the animation generator
    if python -m NL.utils.nuscene_traj_viz \
        --csv_path "$csv_file" \
        --animate \
        --output "$output_file" \
        --fps "$FPS" \
        2>&1 | grep -v "^\[matplotlib.*\]$" | grep -v "^UserWarning"; then
        echo "✓ Created $filename.gif"
        ((SUCCESSFUL++))
    else
        echo "✗ Failed to create $filename.gif"
        ((FAILED++))
    fi
    
    echo ""
done

# Summary
echo "=============================================="
echo "Summary:"
echo "  Successful: $SUCCESSFUL"
echo "  Failed: $FAILED"
echo "  Skipped: $SKIPPED"
echo "  Total: ${#CSV_FILES[@]}"
echo "=============================================="

