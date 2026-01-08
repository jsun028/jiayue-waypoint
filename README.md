## Installation 
### Create virtual environment
python3 -m venv venv

### Activate virtual environment
source venv/bin/activate

### Install dependencies
pip install -r requirements.txt



## How to run 

### UI (Streamlit) 
Interactive query generation
```
streamlit run ./ui/query_app.py
```

Find top-k matches from dataset (Command line)
```
python3 topk_main.py --spec car_stop_for_ped.pkl --dataset-dir dataset/nuscene --dataset nuscene --slider-setting medium --coverage 0.1 --track-stats --limit 5 --dedup-threshold 0.15 --viz-dir viz_out/

python3 topk_main.py --spec virat_turning.pkl --dataset-dir dataset/virat --dataset virat --slider-setting low --coverage 0.1 --track-stats --limit 5 --dedup-threshold 0.15 --viz-dir viz_out/
```

View search results
```
streamlit run ./ui/feedback_app.py
```
### Command Line Workflow
Step 1: Generate keyframe spec (with stats building)
```
python3 NL_dspy/__main__.py  --model "openai/gpt-5" --dump-pickle car_stop_for_ped.pkl --generate-stats --dataset-dir dataset/nuscene --nl "a vehicle is initially moving. it then stops and waits for a pedestrian to walk in front of it (nothing else between vehicle and pedestrain)." --api-key [OPENAI_API_KEY]

python3 NL_dspy/__main__.py  --model "openai/gpt-5" --dump-pickle virat.pkl --generate-stats --dataset-dir dataset/virat --nl "a vehicle stops at a stop sign, and then continue moving." --api-key [OPENAI_API_KEY]

```

Step 2: Find top-k matches from dataset
```
python3 topk_main.py --spec ego_stop_for_ped.pkl --dataset-dir dataset/nuscene --slider-setting medium --coverage 0.1 --track-stats --limit 5 --dedup-threshold 0.15 --viz-dir viz_out/

python3 topk_main.py --spec virat_turning.pkl --dataset-dir dataset/virat --dataset virat --slider-setting low --coverage 0.1 --track-stats --limit 5 --dedup-threshold 0.15 --viz-dir viz_out/
```

### Utils 
Visualize a specific scene
```
python3 dataset_specific/nuscene/nuscene_traj_viz.py --csv_path dataset/nuscene/scene_scene-0301.csv --animate --output viz_out/scene-0301.gif
```

Run spec on a single setting
```
python3 main.py --spec ego_stop_for_ped.pkl --data dataset/nuscene/scene_scene-0301.csv --slider-setting high --coverage 0.1 --track-stats --limit 5 --dedup-threshold 0.15 --viz-dir viz_out/
python3 main.py --spec virat_turning.pkl --data dataset/virat/VIRAT_S_050300_01_000148_000396.csv --slider-setting low --dataset virat --coverage 0.1 --track-stats --limit 5 --dedup-threshold 0.15 --viz-dir viz_out/
```

Run spec across all data and slider settings
```
python3 run_batch.py --spec unseen_stop_ped_fixed_sliders_2.pkl --dataset-dir dataset/nuscene --n-data 100 --output-dir ped_stop --max-workers 64 --coverage 0.1 --track-stats --viz --limit 5 --dedup-threshold 0.15 --viz --slider-setting low,medium,high --keep-empty
```