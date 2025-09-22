## Overview of workflow
0. We assume that preprocessed data from videos are stored in csv format (e.g., examples under `dataset/`) 
1. `test_fixed_system.py` outputs an LLM-generated keyframe spec into a pickle file: `spec.pkl`. 
2. `main.py` reads `spec.pkl`, and tries to compile and execute the query on preprocessed csv files. 

### TODOs
Here are a list of things that are hardcoded, and might need to get fixed laters:
- spec.py: the list of available UDFs are not tied to those defined in registry.py 
- schema assumption
    - df_utils: assumes csv has 'track_id' and 'class_name'
    - registry: assumes csv has 'frame_index', 'track_id', 'x1', 'y1' etc. 