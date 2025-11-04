#!/bin/bash
# Usage: ./make_stats.sh dataset/scene_scene-0300.csv
python -m NL.optimizer.statistics_builder --dataset "$1" "${@:2}"
