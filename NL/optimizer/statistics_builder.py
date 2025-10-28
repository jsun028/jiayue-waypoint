import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime


class KeyframeQLStatisticsBuilder:
    """
    Build KeyframeQL statistics.
    - Per-agent numeric attrs (incl. yaw) -> attribute_histograms
    - Ego pose (from rows where class_name=='ego' or track_id==0) -> metadata["ego"] with compact histograms (no raw trajectory)
    """

    def __init__(self, dataset_path, bins=20, sample_ratio=1.0, ego_bins=10):
        self.dataset_path = Path(dataset_path)
        self.bins = bins
        self.sample_ratio = sample_ratio
        self.ego_bins = ego_bins
        self.df = None
        self.metadata = {}

    # ------------------------------
    # Load & preprocess
    # ------------------------------
    def load_dataset(self):
        df = pd.read_csv(self.dataset_path)
        if self.sample_ratio < 1.0:
            df = df.sample(frac=self.sample_ratio, random_state=42)

        # velocity magnitude (if available)
        if {"vel_x", "vel_y"}.issubset(df.columns):
            df["velocity_mag"] = np.sqrt(df["vel_x"] ** 2 + df["vel_y"] ** 2)
        else:
            df["velocity_mag"] = np.nan

        # bbox area (if available)
        if {"x1", "y1", "x2", "y2"}.issubset(df.columns):
            df["bbox_area"] = (df["x2"] - df["x1"]) * (df["y2"] - df["y1"])
        else:
            df["bbox_area"] = np.nan

        # yaw: use agent_yaw exclusively; normalize to [-pi, pi]
        if "agent_yaw" in df.columns:
            yaw = ((df["agent_yaw"] + np.pi) % (2 * np.pi)) - np.pi
            df["yaw"] = yaw.astype(float)
        else:
            df["yaw"] = np.nan

        self.df = df
        return self

    # ------------------------------
    # Quantile histogram (general)
    # ------------------------------
    def _histogram(self, series, bins):
        series = pd.Series(series).dropna().to_numpy()
        n = len(series)
        if n == 0:
            return {"bins": [], "counts": []}

        # all-equal guard
        if np.all(series == series[0]):
            return {"bins": [float(series[0]), float(series[0])], "counts": [n]}

        quantiles = np.linspace(0, 1, bins + 1)
        edges = np.quantile(series, quantiles)
        edges = np.unique(edges)
        if len(edges) < 2:
            edges = np.array([series.min(), series.max()])

        counts = np.histogram(series, bins=edges)[0]
        counts = np.maximum(counts, 1)

        return {"bins": np.round(edges, 5).tolist(), "counts": counts.tolist()}

    # ------------------------------
    # Compute stats
    # ------------------------------
    def compute_statistics(self):
        df = self.df
        n_frames = df["frame_index"].nunique() if "frame_index" in df.columns else None
        n_objects = len(df)

        # class-level summary
        if "class_name" in df.columns and "track_id" in df.columns:
            class_stats = (
                df.groupby("class_name")
                .agg(
                    total_objects=("track_id", "count"),
                    avg_velocity=("velocity_mag", "mean"),
                    avg_area=("bbox_area", "mean"),
                )
                .reset_index()
            )
            class_stats["ratio"] = class_stats["total_objects"] / class_stats["total_objects"].sum()
            class_dist = class_stats.set_index("class_name").round(4).to_dict(orient="index")
        else:
            class_dist = {}

        # per-class attribute histograms (for numeric attrs)
        class_attr_hists = {}
        if "class_name" in df.columns:
            for cname, g in df.groupby("class_name"):
                hists = {}
                for col in ["velocity_mag", "bbox_area", "yaw", "x1", "y1"]:
                    if col in g.columns:
                        h = self._histogram(g[col], bins=self.bins)
                        # only keep if we have at least 2 edges
                        if h.get("bins"):
                            hists[col] = h
                if hists:
                    class_attr_hists[str(cname)] = hists

        # per-agent attribute histograms
        attr_hist = {}
        for col in ["velocity_mag", "bbox_area", "yaw", "x1", "y1"]:
            if col in df.columns:
                attr_hist[col] = self._histogram(df[col], bins=self.bins)

        # frame density
        if "frame_index" in df.columns and "track_id" in df.columns:
            frame_density = (
                df.groupby("frame_index")
                .agg(objects_per_frame=("track_id", "count"))
                .objects_per_frame
                .describe()
                .to_dict()
            )
            frame_density = {k: float(v) for k, v in frame_density.items()}
        else:
            frame_density = {}

        # Pairwise distances within each frame (sampled)
        pairwise_hist = {"bins": [], "counts": []}
        # Ego-to-agent distances within each frame (sampled)
        ego_to_agent_hist = {"bins": [], "counts": []}
        if "frame_index" in df.columns:
            # choose coordinates: bbox center if available, else (x1, y1)
            if {"x1", "y1", "x2", "y2"}.issubset(df.columns):
                df_coords = df.assign(
                    _cx=(df["x1"] + df["x2"]) / 2.0,
                    _cy=(df["y1"] + df["y2"]) / 2.0,
                )
                coord_cols = ["_cx", "_cy"]
            elif {"x1", "y1"}.issubset(df.columns):
                df_coords = df.rename(columns={"x1": "_cx", "y1": "_cy"})
                coord_cols = ["_cx", "_cy"]
            else:
                df_coords = None
                coord_cols = []

            if df_coords is not None:
                rng = np.random.default_rng(42)
                all_dists = []
                ego_dists = []
                for _, g in df_coords.groupby("frame_index"):
                    pts = g[coord_cols].to_numpy(dtype=float, copy=False)
                    n = len(pts)
                    if n < 2:
                        continue
                    # sample up to max_pairs distances per frame to cap cost
                    max_pairs = 200
                    # number of unique pairs
                    total_pairs = n * (n - 1) // 2
                    if total_pairs <= max_pairs:
                        # compute full upper-tri distances
                        diffs = pts[:, None, :] - pts[None, :, :]
                        dists = np.sqrt((diffs ** 2).sum(axis=2))
                        iu = np.triu_indices(n, 1)
                        all_dists.extend(dists[iu].tolist())
                    else:
                        # random sample of unique pairs
                        # generate random indices (i<j)
                        idx_i = rng.integers(0, n - 1, size=max_pairs)
                        idx_j = rng.integers(0, n - 1, size=max_pairs)
                        # enforce i<j and non-equal; fix by swapping or incrementing
                        for i in range(max_pairs):
                            a = int(idx_i[i])
                            b = int(idx_j[i])
                            if a == b:
                                b = (b + 1) % n
                            if a > b:
                                a, b = b, a
                            dx = pts[a, 0] - pts[b, 0]
                            dy = pts[a, 1] - pts[b, 1]
                            all_dists.append(float(np.sqrt(dx * dx + dy * dy)))
                    # Ego-to-agent distances for this frame, if ego present
                    has_class = "class_name" in g.columns
                    has_track = "track_id" in g.columns
                    if has_class or has_track:
                        mask_ego = pd.Series(False, index=g.index)
                        if has_class:
                            try:
                                mask_ego = mask_ego | (g["class_name"] == "ego")
                            except Exception:
                                pass
                        if has_track:
                            try:
                                mask_ego = mask_ego | (g["track_id"] == 0)
                            except Exception:
                                pass
                        if mask_ego.any():
                            ego_pts = g.loc[mask_ego, coord_cols]
                            # choose the first ego row for this frame
                            ex = float(ego_pts.iloc[0, 0])
                            ey = float(ego_pts.iloc[0, 1])
                            agents = g.loc[~mask_ego, coord_cols].to_numpy(dtype=float, copy=False)
                            m = len(agents)
                            if m > 0:
                                max_pairs = 200
                                if m <= max_pairs:
                                    dx = agents[:, 0] - ex
                                    dy = agents[:, 1] - ey
                                    ego_dists.extend(np.sqrt(dx * dx + dy * dy).tolist())
                                else:
                                    sample_idx = rng.integers(0, m, size=max_pairs)
                                    samp = agents[sample_idx]
                                    dx = samp[:, 0] - ex
                                    dy = samp[:, 1] - ey
                                    ego_dists.extend(np.sqrt(dx * dx + dy * dy).tolist())
                if all_dists:
                    pairwise_hist = self._histogram(all_dists, bins=self.bins)
                if ego_dists:
                    ego_to_agent_hist = self._histogram(ego_dists, bins=self.bins)

        # EGO metadata as histograms (no raw per-frame data)
        ego_meta = {}
        if {"frame_index", "class_name", "x1", "y1", "x2", "y2", "agent_yaw"}.issubset(df.columns):
            ego_rows = df[(df["class_name"] == "ego") | (("track_id" in df.columns) & (df["track_id"] == 0))]
            if not ego_rows.empty:
                # compute centers from bbox and normalize yaw
                ego_center_x = (ego_rows["x1"].to_numpy() + ego_rows["x2"].to_numpy()) / 2.0
                ego_center_y = (ego_rows["y1"].to_numpy() + ego_rows["y2"].to_numpy()) / 2.0
                ego_yaw = ((ego_rows["agent_yaw"].to_numpy() + np.pi) % (2 * np.pi)) - np.pi
                ego_df = pd.DataFrame({
                    "frame_index": ego_rows["frame_index"].to_numpy(),
                    "ego_x": ego_center_x,
                    "ego_y": ego_center_y,
                    "ego_yaw": ego_yaw,
                }).drop_duplicates(subset=["frame_index"]).sort_values("frame_index")
                ego_hist = {
                    "ego_x": self._histogram(ego_df["ego_x"], bins=self.ego_bins),
                    "ego_y": self._histogram(ego_df["ego_y"], bins=self.ego_bins),
                    "ego_yaw": self._histogram(ego_df["ego_yaw"], bins=self.ego_bins),
                }
                ego_summary = {
                    "n_frames": int(len(ego_df)),
                    "means": {
                        "ego_x": float(np.nanmean(ego_df["ego_x"])),
                        "ego_y": float(np.nanmean(ego_df["ego_y"])),
                        "ego_yaw": float(np.nanmean(ego_df["ego_yaw"])),
                    },
                    "mins": {
                        "ego_x": float(np.nanmin(ego_df["ego_x"])),
                        "ego_y": float(np.nanmin(ego_df["ego_y"])),
                        "ego_yaw": float(np.nanmin(ego_df["ego_yaw"])),
                    },
                    "maxs": {
                        "ego_x": float(np.nanmax(ego_df["ego_x"])),
                        "ego_y": float(np.nanmax(ego_df["ego_y"])),
                        "ego_yaw": float(np.nanmax(ego_df["ego_yaw"])),
                    },
                }
                ego_meta = {"histograms": ego_hist, "summary": ego_summary}

        self.metadata = {
            "dataset_name": self.dataset_path.stem,
            "n_frames": int(n_frames) if n_frames is not None else None,
            "n_objects": int(n_objects),
            "class_distribution": class_dist,
            "class_attribute_histograms": class_attr_hists,
            "attribute_histograms": attr_hist,  # includes 'yaw'
            "frame_density": frame_density,
            "pairwise_distance_histogram": pairwise_hist,
            "ego_to_agent_distance_histogram": ego_to_agent_hist,
            "ego": ego_meta,                     # histograms + summary only
            "last_updated": datetime.utcnow().isoformat(),
        }
        return self

    # ------------------------------
    # Persist
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
# CLI
# ------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build KeyframeQL Statistics (yaw-only, ego hists)")
    parser.add_argument("--dataset", required=True, help="path to CSV dataset")
    parser.add_argument("--bins", type=int, default=20, help="per-agent histogram bins")
    parser.add_argument("--ego_bins", type=int, default=8, help="ego histogram bins")
    parser.add_argument("--sample_ratio", type=float, default=1.0)
    args = parser.parse_args()

    builder = (
        KeyframeQLStatisticsBuilder(
            args.dataset, bins=args.bins, sample_ratio=args.sample_ratio, ego_bins=args.ego_bins
        )
        .load_dataset()
        .compute_statistics()
    )
    builder.save_metadata()