from NL.optimizer.selectivity_estimator import SelectivityEstimator
import numpy as np

class SelectivityIntegration:
    def __init__(self, metadata_path, df):
        # Allow None for metadata_path; estimator will fall back to empty stats
        self.est = SelectivityEstimator(metadata_path)
        self.df = df  # for sampling-based estimation (dist_within_two_obj)

    def estimate_keyframe_selectivity(self, keyframe_spec):
        preds = keyframe_spec.where
        est_result = {}

        def handle_atom(atom):
            t = atom.type
            if t == "velocity_above":
                hist = (self.est.stats.get("attribute_histograms") or {}).get("velocity_mag")
                if not hist:
                    return 1.0
                return self.est._hist_fraction(hist, threshold=atom.value, op=">")
            elif t == "velocity_below":
                hist = (self.est.stats.get("attribute_histograms") or {}).get("velocity_mag")
                if not hist:
                    return 1.0
                return self.est._hist_fraction(hist, threshold=atom.value, op="<")
            elif t == "dist_within_two_obj":
                return self._estimate_dist_within(atom.obj, atom.other_obj, atom.value)
            else:
                return 1.0  # unknown predicates -> neutral selectivity (no reduction)

        # traverse PredicateExpr recursively
        def traverse(expr):
            if expr.op == "AND":
                vals = [traverse(arg) for arg in expr.args]
                return np.prod(vals)
            elif expr.op == "OR":
                vals = [traverse(arg) for arg in expr.args]
                return 1 - np.prod([1 - v for v in vals])
            elif expr.op == "ATOM":
                return handle_atom(expr.atom)
            else:
                return 1.0

        selectivity = traverse(preds)
        est_result["selectivity"] = selectivity
        return est_result

    def _estimate_dist_within(self, obj_a, obj_b, threshold):
        """Fast approximate estimation by random sampling from dataset"""
        df = self.df.sample(frac=0.20, random_state=42)  # 20% sample
        veh = df[df["class_name"] == "vehicle"][["x1", "y1"]].to_numpy()
        ped = df[df["class_name"] == "pedestrian"][["x1", "y1"]].to_numpy()

        if len(veh) == 0 or len(ped) == 0:
            return 0.0

        # Compute pairwise distances for small sample
        dists = np.sqrt(
            (veh[:, None, 0] - ped[None, :, 0]) ** 2
            + (veh[:, None, 1] - ped[None, :, 1]) ** 2
        )
        within = (dists < threshold).mean()
        return float(within)