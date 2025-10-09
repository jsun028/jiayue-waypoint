import json
import numpy as np
from pathlib import Path


class SelectivityEstimator:
    """
    Load precomputed KeyframeQL statistics metadata (JSON)
    and estimate selectivity and cost for query predicates.
    """

    def __init__(self, metadata_path):
        self.metadata_path = Path(metadata_path)
        with open(self.metadata_path, "r") as f:
            self.stats = json.load(f)

    # ---------------------------------
    # Helper: find proportion above threshold using histogram
    # ---------------------------------
    def _hist_fraction(self, hist, threshold, op=">"):
        bins = np.array(hist["bins"])
        counts = np.array(hist["counts"])
        total = counts.sum()

        # If threshold is below min or above max
        if threshold <= bins[0]:
            return 1.0 if op == ">" else 0.0
        if threshold >= bins[-1]:
            return 0.0 if op == ">" else 1.0

        # Find which bin the threshold falls into
        idx = np.searchsorted(bins, threshold) - 1
        idx = np.clip(idx, 0, len(counts) - 1)

        # compute cumulative fraction
        cumsum = np.cumsum(counts) / total
        if op == ">":
            # fraction above threshold
            return 1.0 - cumsum[idx]
        elif op == "<":
            # fraction below threshold
            return cumsum[idx]
        else:
            raise ValueError(f"Unsupported operator {op}")

    # ---------------------------------
    # Main: estimate selectivity for predicate dict
    # ---------------------------------
    def estimate_selectivity(self, predicates):
        """
        predicates: dict like
            {"class_name": "car", "velocity_mag": (">", 2.0)}
        """
        ratio = 1.0
        # 1. class filter
        if "class_name" in predicates:
            cname = predicates["class_name"]
            class_dist = self.stats["class_distribution"]
            if cname in class_dist:
                ratio *= class_dist[cname]["ratio"]
            else:
                ratio *= 0.0

        # 2. numeric predicate
        for key, cond in predicates.items():
            if key in ["velocity_mag", "bbox_area", "heading_angle"]:
                # Handle case where condition is a list of predicates (e.g. range query)
                if isinstance(cond, list):
                    for op, val in cond:
                        hist = self.stats["attribute_histograms"][key]
                        frac = self._hist_fraction(hist, threshold=val, op=op)
                        ratio *= frac
                # Handle single predicate case
                else:
                    op, val = cond
                    hist = self.stats["attribute_histograms"][key]
                    frac = self._hist_fraction(hist, threshold=val, op=op)
                    ratio *= frac

        return ratio

    # ---------------------------------
    # Optional: simple cost model
    # ---------------------------------
    def estimate_cost(self, predicates, alpha=1.0, beta=10.0):
        """
        Simple cost model:
        total_cost = α * scan_cost + β * filtered_object_cost
        """
        total_objects = self.stats["n_objects"]
        selectivity = self.estimate_selectivity(predicates)
        est_objects = total_objects * selectivity

        # simplistic linear model
        # cost = alpha * total_objects + beta * est_objects
        return {
            "selectivity": round(selectivity, 5),
            "estimated_objects": int(est_objects),
            # "estimated_cost": round(cost, 2),
        }


# ----------------------------------------------------------
# CLI usage example
# ----------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Estimate selectivity using KeyframeQL metadata")
    parser.add_argument("--metadata", required=True, help="path to metadata JSON file")
    parser.add_argument("--class_name", type=str, default=None)
    parser.add_argument("--velocity", type=float, nargs=2, metavar=('GT', 'LT'), default=(None, None), help="velocity range (greater than, less than)")
    args = parser.parse_args()

    est = SelectivityEstimator(args.metadata)

    predicates = {}
    if args.class_name:
        predicates["class_name"] = args.class_name
    
    # Handle velocity range predicates
    if args.velocity[0] is not None and args.velocity[1] is not None:
        # If both bounds specified, use a list of predicates
        predicates["velocity_mag"] = [(">", args.velocity[0]), ("<", args.velocity[1])]
    elif args.velocity[0] is not None:
        predicates["velocity_mag"] = (">", args.velocity[0])
    elif args.velocity[1] is not None:
        predicates["velocity_mag"] = ("<", args.velocity[1])

    result = est.estimate_cost(predicates)
    print(json.dumps(result, indent=2))