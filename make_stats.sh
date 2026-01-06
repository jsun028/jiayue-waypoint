#!/bin/bash
# Usage: ./make_stats.sh dataset/nuscene/scene_scene-0300.csv
python -m keyframeql.optimizer.statistics_builder --dataset "$1" "${@:2}"
