## Installation 
### Create virtual environment
python3 -m venv venv

### Activate virtual environment
source venv/bin/activate

### Install dependencies
pip install -r requirements.txt


## How to run 
Generate keyframe spec (with stats building)
```
python3 NL_dspy/__main__.py  --model "openai/gpt-5" --dump-pickle ego_stop_for_ped.pkl --generate-stats --nl "a vehicle is initially moving. it then stops and waits for a pedestrian to walk in front of it (nothing else between vehicle and pedestrain)." --api-key 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
```

Visualize a specific scene
```
python3 dataset_specific/nuscene/nuscene_traj_viz.py --csv_path dataset/nuscene/scene_scene-0301.csv --animate --output viz_out/scene-0301.gif
```

Run spec on a single setting
```
python3 main.py --spec ego_stop_for_ped.pkl --data dataset/nuscene/scene_scene-0301.csv --slider-setting high --coverage 0.1 --track-stats --limit 5 --dedup-threshold 0.15 --viz-dir viz_out/
```

Return top-k from dataset
```
python3 topk_main.py --spec ego_stop_for_ped.pkl --dataset-dir dataset/nuscene --slider-setting medium --coverage 0.1 --track-stats --limit 5 --dedup-threshold 0.15 --viz-dir viz_out/

```

Run spec across all data and slider settings
```
python3 run_batch.py --spec unseen_stop_ped_fixed_sliders_2.pkl --dataset-dir dataset/nuscene --n-data 100 --output-dir ped_stop --max-workers 64 --coverage 0.1 --track-stats --viz --limit 5 --dedup-threshold 0.15 --viz --slider-setting low,medium,high --keep-empty
```

View search results
```
streamlit run ./feedback/app.py
```