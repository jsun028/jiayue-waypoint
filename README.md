## Installation 
### Create virtual environment
python3 -m venv venv

### Activate virtual environment
source venv/bin/activate

### Install dependencies
pip install -r requirements.txt



### 📥 Data Preparation
1. Download Dataset: Download the virat traffic dataset from [Google Drive](https://drive.google.com/file/d/1DIy0NOBPTnRaDsnqSl-o1e3vAeFfM3Kz/view?usp=sharing) and place it in the `videos/virat/` folder.


## How to run 

### UI (Streamlit) 
Interactive query generation
```
streamlit run ./ui/query_app.py
```

Find top-k matches from dataset (command line)
```
python3 topk_main.py --spec nuscene_turn.pkl --dataset-dir dataset/nuscene --dataset nuscene --slider-setting medium --coverage 0.1 --track-stats --limit 5 --dedup-threshold 0.15 --viz-dir viz_out/
```

View search results
```
streamlit run ./ui/feedback_app.py
```

### Command Line Workflow
Step 1: Generate keyframe spec (with stats building)
```
python3 NL_dspy/__main__.py  --model "openai/gpt-5" --dump-pickle nuscene_turn.pkl --generate-stats --dataset-dir dataset/nuscene --nl "A car makes a turn. The turn should be pretty noticable." --api-key [OPENAI_API_KEY]
```

Step 2: Find top-k matches from dataset
```
python3 topk_main.py --spec nuscene_turn.pkl --dataset-dir dataset/nuscene --dataset nuscene --slider-setting medium --coverage 0.1 --track-stats --limit 5 --dedup-threshold 0.15 --viz-dir viz_out/
```

### Utils 
Visualize a specific scene
```
python3 dataset_specific/nuscene/nuscene_traj_viz.py --csv_path dataset/nuscene/scene_scene-0301.csv --animate --output viz_out/scene-0301.gif
```

Run spec on a single setting
```
python3 main.py --spec nuscene_turn.pkl --data dataset/nuscene/scene_scene-0234.csv --dataset nuscene --slider-setting medium --coverage 0.1 --track-stats --limit 5 --dedup-threshold 0.15 --viz-dir viz_out/
```

Run spec across all data and slider settings
```
python3 run_batch.py --spec nuscene_turn.pkl --dataset-dir dataset/nuscene --n-data 100 --output-dir ped_stop --max-workers 64 --coverage 0.1 --track-stats --viz --limit 5 --dedup-threshold 0.15 --viz --slider-setting low,medium,high --keep-empty
```