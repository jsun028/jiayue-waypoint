import json
import numpy as np
from pathlib import Path


class SelectivityEstimator:
    """
    Load KeyframeQL statistics metadata (JSON)
    and estimate selectivity and cost for query predicates.
    """

    def __init__(self, metadata_path=None):
        """Initialize with optional metadata path.

        If no path is provided, fall back to empty stats so callers can still
        estimate with neutral defaults.
        """
        self.stats = {"attribute_histograms": {}, "class_distribution": {}, "n_objects": 0}
        self.metadata_path = None
        if metadata_path:
            self.metadata_path = Path(metadata_path)
            with open(self.metadata_path, "r") as f:
                self.stats = json.load(f)

    # ---------------------------------
    # Helper: histogram fraction
    # ---------------------------------
    def _hist_fraction(self, hist, threshold, op=">"):
        bins = np.array(hist["bins"])
        counts = np.array(hist["counts"])
        total = counts.sum()

        if total == 0 or len(bins) == 0 or len(counts) == 0:
            return 0.0

        if threshold <= bins[0]:
            return 1.0 if op == ">" else 0.0
        if threshold >= bins[-1]:
            return 0.0 if op == ">" else 1.0

        idx = np.searchsorted(bins, threshold) - 1
        idx = np.clip(idx, 0, len(counts) - 1)

        cumsum = np.cumsum(counts) / total
        if op == ">":
            return 1.0 - cumsum[idx]
        elif op == "<":
            return cumsum[idx]
        else:
            raise ValueError(f"Unsupported operator {op}")

    # ---------------------------------
    # Main: estimate selectivity
    # ---------------------------------
    def estimate_selectivity(self, predicates):
        """
        predicates example:
          {
            "class_name": "car",
            "velocity_mag": (">", 2.0),
            "yaw": [(">", -0.5), ("<", 0.5)]
          }
        """
        ratio = 1.0

        # class filter
        if "class_name" in predicates:
            cname = predicates["class_name"]
            class_dist = self.stats.get("class_distribution", {})
            if cname in class_dist:
                ratio *= class_dist[cname]["ratio"]
            else:
                return 0.0

        # numeric attributes present in metadata
        attr_hists = self.stats.get("attribute_histograms", {})
        for key, cond in predicates.items():
            if key == "class_name":
                continue
            if key not in attr_hists:
                continue  # unknown attr; ignore

            hist = attr_hists[key]
            if isinstance(cond, list):
                for op, val in cond:
                    ratio *= self._hist_fraction(hist, threshold=val, op=op)
            else:
                op, val = cond
                ratio *= self._hist_fraction(hist, threshold=val, op=op)

        return ratio

    # ---------------------------------
    # Simple cost model
    # ---------------------------------
    def estimate_cost(self, predicates, alpha=1.0, beta=10.0):
        total_objects = self.stats.get("n_objects", 0)
        selectivity = self.estimate_selectivity(predicates)
        est_objects = total_objects * selectivity
        return {
            "selectivity": round(float(selectivity), 5),
            "estimated_objects": int(est_objects),
        }


# ----------------------------------------------------------
# CLI
# ----------------------------------------------------------
if __name__ == "__main__":
    import argparse, json as _json

    parser = argparse.ArgumentParser(description="Estimate selectivity using KeyframeQL metadata")
    parser.add_argument("--metadata", required=True, help="path to metadata JSON file")
    parser.add_argument("--class_name", type=str, default=None)
    parser.add_argument("--velocity", type=float, nargs=2, metavar=("GT", "LT"), default=(None, None),
                        help="velocity range (greater than, less than)")
    parser.add_argument("--yaw", type=float, nargs=2, metavar=("GT", "LT"), default=(None, None),
                        help="yaw range (greater than, less than)")
    args = parser.parse_args()

    est = SelectivityEstimator(args.metadata)

    predicates = {}
    if args.class_name:
        predicates["class_name"] = args.class_name

    # velocity range
    if args.velocity[0] is not None and args.velocity[1] is not None:
        predicates["velocity_mag"] = [(">", args.velocity[0]), ("<", args.velocity[1])]
    elif args.velocity[0] is not None:
        predicates["velocity_mag"] = (">", args.velocity[0])
    elif args.velocity[1] is not None:
        predicates["velocity_mag"] = ("<", args.velocity[1])

    # yaw range
    if args.yaw[0] is not None and args.yaw[1] is not None:
        predicates["yaw"] = [(">", args.yaw[0]), ("<", args.yaw[1])]
    elif args.yaw[0] is not None:
        predicates["yaw"] = (">", args.yaw[0])
    elif args.yaw[1] is not None:
        predicates["yaw"] = ("<", args.yaw[1])

    result = est.estimate_cost(predicates)
    print(_json.dumps(result, indent=2))