## Installation 
modAL
```
pip install modAL-python
```

## How to run 
Generate spec 
```
python3 NL_dspy/__main__.py  --model "openai/gpt-5" --dump-pickle ego_stop_for_ped.pkl --generate-stats --nl "ego vehicle is initially moving and stops. a pedestrian then walks in front of the stopped vehicle." --api-key 
```

Visualize a specific scene
```
python3 keyframeql/utils/nuscene_traj_viz.py --csv_path dataset/scene_scene-0301.csv --animate --output viz_out/scene-0301.gif
```

Run spec on a single setting
```
python3 main.py --spec ego_stop_for_ped.pkl --data dataset/scene_scene-0301.csv --slider-setting high --coverage 0.1 --track-stats --limit 5 --dedup-threshold 0.15 --viz-dir viz_out/
```

Return top-k from dataset
```
python3 topk_main.py --spec ego_stop_for_ped.pkl --dataset-dir dataset --slider-setting medium --coverage 0.1 --track-stats --limit 5 --dedup-threshold 0.15 --viz-dir viz_out/

```

Run spec across all data and slider settings
```
python3 run_batch.py --spec unseen_stop_ped_fixed_sliders_2.pkl --dataset-dir dataset --n-data 100 --output-dir ped_stop --max-workers 64 --coverage 0.1 --track-stats --viz --limit 5 --dedup-threshold 0.15 --viz --slider-setting low,medium,high --keep-empty
```
