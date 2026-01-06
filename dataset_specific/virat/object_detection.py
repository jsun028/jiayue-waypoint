from ultralytics import YOLO
from pathlib import Path
import pandas as pd
import numpy as np

def main():

    path = Path("/data/VIRAT_ground_videos/videos-split/videos")
    output_dir = "/data/VIRAT_ground_videos/object-detection/output-csvs-with-split-videos"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model = YOLO("yolo11x.pt")
    video_files = path.glob("*.mp4")
    # Predefine CSV columns so we can write headers even if no detections
    columns = [
        "video_id", "frame_id", "object_id", "class_id", "class_name",
        "bbox_x1_abs", "bbox_y1_abs", "bbox_x2_abs", "bbox_y2_abs",
        "bbox_x1_rel", "bbox_y1_rel", "bbox_x2_rel", "bbox_y2_rel",
        "confidence", "image_width", "image_height",
    ]

    for vp in video_files:
        video_id = vp.stem
        frame_id = 0
        rows = []

        # Loop through each frame in the video
        for r in model.track(source=str(vp), stream=True, device=0, tracker="bytetrack.yaml", persist=True, half=True,
                             conf=0.6, iou=0.5):
            frame_id += 1
            if not r.boxes or len(r.boxes) == 0:
                continue

            h, w = r.orig_shape
            xyxy_abs = r.boxes.xyxy.cpu().numpy()
            xyxy_rel = r.boxes.xyxyn.cpu().numpy()
            cls  = r.boxes.cls.cpu().numpy().astype(int)
            conf = r.boxes.conf.cpu().numpy()
            ids  = r.boxes.id
            ids  = np.full(len(cls), -1, dtype=int) if ids is None else ids.cpu().numpy().astype(int) # if no ids, fill with -1
            names = r.names

            # Loop through each detected object in the frame
            for (x1a, y1a, x2a, y2a), (x1r, y1r, x2r, y2r), c, s, tid in zip(xyxy_abs, xyxy_rel, cls, conf, ids):
                rows.append({
                    "video_id": video_id, # id for each video
                    "frame_id": frame_id, # id for each frame in the video
                    "object_id": int(tid), # id for each object in each frame of the video
                    "class_id": int(c), # the class id of the object
                    "class_name": names[int(c)], # the class name of the object
                    "bbox_x1_abs": float(x1a), "bbox_y1_abs": float(y1a), # absolute bbox coordinates
                    "bbox_x2_abs": float(x2a), "bbox_y2_abs": float(y2a),
                    "bbox_x1_rel": float(x1r), "bbox_y1_rel": float(y1r), # relative bbox coordinates to image size
                    "bbox_x2_rel": float(x2r), "bbox_y2_rel": float(y2r),
                    "confidence": float(s), # confidence score of the detection
                    "image_width": int(w), "image_height": int(h), # original frame image size
                })
                
        out_csv = Path(output_dir) / f"{video_id}.csv"
        df = pd.DataFrame(rows, columns=columns)
        df.to_csv(out_csv, index=False)
        print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()