import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime


class KeyframeQLStatisticsBuilder:
    """
    Build per-attribute statistics metadata for KeyframeQL datasets.
    """

    def __init__(self, dataset_path, bins=20, sample_ratio=1.0):
        self.dataset_path = Path(dataset_path)
        self.bins = bins
        self.sample_ratio = sample_ratio
        self.df = None
        self.metadata = {}

    # ------------------------------
    # Step 1. Load & Preprocess
    # ------------------------------
    def load_dataset(self):
        df = pd.read_csv(self.dataset_path)
        if self.sample_ratio < 1.0:
            df = df.sample(frac=self.sample_ratio, random_state=42)

        # derived attributes
        df["velocity_mag"] = np.sqrt(df["vel_x"] ** 2 + df["vel_y"] ** 2)
        df["bbox_area"] = (df["x2"] - df["x1"]) * (df["y2"] - df["y1"])
        df["heading_angle"] = np.arctan2(df["heading_y"], df["heading_x"])

        self.df = df
        return self

    # ------------------------------
    # Step 2. Aggregations
    # ------------------------------
    # def _histogram(self, series):
    #     """
    #     Build equal-width (linear-spaced) histogram.
    #     """
    #     counts, edges = np.histogram(series, bins=self.bins)
    #     return {
    #         "bins": edges.round(4).tolist(),
    #         "counts": counts.tolist(),
    #     }
    def _histogram(self, series):
        """
        Build equal-frequency (quantile-based) histogram.

        Each bin contains approximately the same number of samples,
        ensuring non-empty selectivity buckets for interactive queries.
        """
        # Drop NaN or invalid values
        series = series.dropna().to_numpy()
        n = len(series)
        if n == 0:
            return {"bins": [], "counts": []}

        # compute quantile breakpoints
        quantiles = np.linspace(0, 1, self.bins + 1)
        edges = np.quantile(series, quantiles)

        # count how many values fall into each quantile bin
        counts = np.histogram(series, bins=edges)[0]

        # small safeguard: avoid zeros by enforcing at least 1 sample per bin
        counts = np.maximum(counts, 1)

        return {
            "bins": np.round(edges, 5).tolist(),
            "counts": counts.tolist(),
        }

    def compute_statistics(self):
        df = self.df
        n_frames = df["frame_index"].nunique()
        n_objects = len(df)

        # 1. class-level summary
        class_stats = (
            df.groupby("class_name")
            .agg(
                total_objects=("track_id", "count"),
                avg_velocity=("velocity_mag", "mean"),
                avg_area=("bbox_area", "mean"),
            )
            .reset_index()
        )
        class_stats["ratio"] = (
            class_stats["total_objects"] / class_stats["total_objects"].sum()
        )

        # 2. attribute histograms
        attr_hist = {
            "velocity_mag": self._histogram(df["velocity_mag"]),
            "bbox_area": self._histogram(df["bbox_area"]),
            "heading_angle": self._histogram(df["heading_angle"]),
            "x1": self._histogram(df["x1"]),
            "y1": self._histogram(df["y1"]),
        }

        # 3. frame-level aggregation
        frame_density = (
            df.groupby("frame_index")
            .agg(objects_per_frame=("track_id", "count"))
            .objects_per_frame
            .describe()
            .to_dict()
        )

        # 4. pack metadata
        self.metadata = {
            "dataset_name": self.dataset_path.stem,
            "n_frames": int(n_frames),
            "n_objects": int(n_objects),
            "class_distribution": class_stats.set_index("class_name")
            .round(4)
            .to_dict(orient="index"),
            "attribute_histograms": attr_hist,
            "frame_density": {k: float(v) for k, v in frame_density.items()},
            "last_updated": datetime.utcnow().isoformat(),
        }
        return self

    # ------------------------------
    # Step 3. Persist metadata
    # ------------------------------
    def save_metadata(self, out_dir="metadata"):
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        out_file = out_path / f"{self.dataset_path.stem}_stats.json"
        with open(out_file, "w") as f:
            json.dump(self.metadata, f, indent=2)
        print(f"Statistics saved to {out_file}")
        return out_file


# ------------------------------------------------------
# CLI Usage Example
# ------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build KeyframeQL Statistics")
    parser.add_argument("--dataset", required=True, help="path to CSV dataset")
    parser.add_argument("--bins", type=int, default=20)
    parser.add_argument("--sample_ratio", type=float, default=1.0)
    args = parser.parse_args()

    builder = (
        KeyframeQLStatisticsBuilder(args.dataset, bins=args.bins, sample_ratio=args.sample_ratio)
        .load_dataset()
        .compute_statistics()
    )
    builder.save_metadata()