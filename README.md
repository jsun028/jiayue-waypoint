## Installation 

modAL
```
pip install modAL-python
```

## How to run 
Generate spec 
```
python3 NL_dspy/__main__.py  --model "openai/gpt-5" --dump-pickle unseen_stop_ped_fixed_sliders_2.pkl --generate-stats --nl "car1 is moving and a pedestrian in front of it, though not visible to e car1, is also moving in somewhat perpindicular direction. car1 then sees pedestrian and is forced to a stop as pedestrian crosses." --api-key 
```
Run spec on a single setting
```
python3 main.py --spec unseen_stop_ped_fixed_sliders_2.pkl --data dataset/scene_scene-0301.csv --slider-setting medium --coverage 0.1 --track-stats --limit 5 --dedup-threshold 0.15 --viz-dir viz_out/
```

Return top-k from dataset
```
python3 topk_main.py --spec unseen_stop_ped_fixed_sliders_2.pkl --dataset-dir dataset --slider-setting medium --coverage 0.1 --track-stats --limit 5 --dedup-threshold 0.15 --viz --viz-dir viz_out/

```

Run spec across all data and slider settings
```
python3 run_batch.py --spec unseen_stop_ped_fixed_sliders_2.pkl --dataset-dir dataset --n-data 100 --output-dir ped_stop --max-workers 64 --coverage 0.1 --track-stats --viz --limit 5 --dedup-threshold 0.15 --viz --slider-setting low,medium,high --keep-empty
```
