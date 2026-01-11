SAM dependencies
```
python3 -m venv sam
source sam/bin/activate
pip install ultralytics opencv-python pandas numpy
pip install git+https://github.com/facebookresearch/segment-anything-2.git

mkdir -p checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt -P checkpoints/
```

Run object detecion on raw videos (10 FPS output)
```
python3 detection_sam.py --video_dir samples/ --output_dir .
```

Augment object detection results with velocity (vel_x, vel_y), acceleration and yaw
```
python3 augment_motion.py --input ../../dataset/virat/raw --output ../../dataset/virat/
```

Visualize
```
python3 viz.py --csv ../../dataset/virat/VIRAT_S_000206_07_001501_001600.csv --frame 1000 
python3 viz.py --csv ../../dataset/virat/VIRAT_S_000206_07_001501_001600.csv --img-width 1280 --img-height 720 --track-ids 738,779,847
python3 viz.py --csv ../../dataset/virat/VIRAT_S_000206_07_001501_001600.csv --img-width 1280 --img-height 720 --animate --output VIRAT_S_000206_07_001501_001600.gif --fps 50
python3 viz.py --csv ../../dataset/virat/VIRAT_S_050300_01_000148_000396.csv --img-width 1920 --img-height 1080 --animate --output VIRAT_S_050300_01_000148_000396.gif --fps 50
```
