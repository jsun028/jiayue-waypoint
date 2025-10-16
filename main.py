import pickle
import argparse
import json
from pathlib import Path
import pandas as pd
from loguru import logger

from NL.registry import UDFRegistry
from NL.compiler import QueryCompiler
from NL.utils.nuscene_traj_viz import plot_bev_snapshot


logger.add("runs.log", rotation="1 week")

def _coerce_jsonable(obj):
    if isinstance(obj, (int, float, str)) or obj is None:
        return obj
    if isinstance(obj, (list, tuple)):
        return [_coerce_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _coerce_jsonable(v) for k, v in obj.items()}
    try:
        # numpy, pandas scalars
        import numpy as np  # type: ignore
        if isinstance(obj, (np.generic,)):
            return obj.item()
    except Exception:
        pass
    try:
        return json.loads(json.dumps(obj))
    except Exception:
        return str(obj)

def _write_results_json(results, out_path: str) -> None:
    safe = [_coerce_jsonable(r) for r in results]
    with open(out_path, "w") as f:
        json.dump(safe, f, indent=2, sort_keys=True)
    logger.info(f"Wrote results JSON → {out_path}")

def _generate_visualizations(df: pd.DataFrame, results: list[dict], out_dir: Path, fps: int = 10) -> None:
    # Lazy import to avoid backend issues when not requested
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import patches
    from matplotlib.animation import FuncAnimation, PillowWriter

    def _parse_time_range(tr: str) -> tuple[int, int]:
        try:
            s = tr.strip().lstrip("(").rstrip(")")
            a, b = s.split(",")
            return int(a.strip()), int(b.strip())
        except Exception:
            # fallback to df bounds
            return int(df['frame_index'].min()), int(df['frame_index'].max())

    for idx, res in enumerate(results):
        assignment = res.get('object_assignment', {})
        highlight_ids = set(int(v) for v in assignment.values()) if assignment else set()
        time_range = res.get('time_range')
        start_f, end_f = _parse_time_range(time_range) if isinstance(time_range, str) else (int(df['frame_index'].min()), int(df['frame_index'].max()))
        frames = list(range(start_f, end_f + 1))

        fig, ax = plt.subplots(figsize=(12, 12))

        # Precompute fixed axis limits over the animation window to avoid jank
        df_range = df[df['frame_index'].between(start_f, end_f)]
        try:
            all_x = pd.concat([df_range['x1'], df_range['x2'], df_range['ego_x']])
            all_y = pd.concat([df_range['y1'], df_range['y2'], df_range['ego_y']])
            margin = 20
            xlim = (float(all_x.min()) - margin, float(all_x.max()) + margin)
            ylim = (float(all_y.min()) - margin, float(all_y.max()) + margin)
        except Exception:
            xlim = None
            ylim = None

        def update(frame_idx: int):
            ax.clear()
            plot_bev_snapshot(df, frame_idx, ax)
            # Overlay highlights for assigned agents
            frame_data = df[df['frame_index'] == frame_idx]
            for _, row in frame_data.iterrows():
                tid = int(row['track_id'])
                if tid not in highlight_ids:
                    continue
                x1, y1, x2, y2 = row['x1'], row['y1'], row['x2'], row['y2']
                w = x2 - x1
                h = y2 - y1
                rect = patches.Rectangle((x1, y1), w, h, linewidth=3.5, edgecolor='yellow', facecolor='none', alpha=0.9)
                ax.add_patch(rect)
                ax.text((x1 + x2) / 2, (y1 + y2) / 2, str(tid), ha='center', va='center', fontsize=10, color='yellow', weight='bold')
            ax.set_title(f"Result {idx+1} – frame {frame_idx}")
            # Lock axis limits and aspect
            if xlim and ylim:
                ax.set_xlim(*xlim)
                ax.set_ylim(*ylim)
            ax.set_aspect('equal')
            ax.set_autoscale_on(False)

        anim = FuncAnimation(fig, lambda i: update(frames[i]), frames=len(frames), interval=1000 / max(1, fps), repeat=True)
        out_path = out_dir / f"result_{idx+1}.gif"
        writer = PillowWriter(fps=fps)
        anim.save(out_path, writer=writer)
        plt.close(fig)

    logger.info(f"Wrote visualization GIFs → {out_dir}")

# Example usage
def example_usage(spec_path, data_path, coverage: float | None = None, track_stats: bool = True, 
out_path: str | None = None, do_viz: bool = False, viz_dir: str | None = None, limit: int | None = None, dedup_threshold: float | None = None):
    # Sample data
    df = pd.read_csv(data_path)
    
    # UDF registry
    registry = UDFRegistry(df)

    # Load a sample spec
    spec = pickle.load(open(spec_path, "rb"))
    print("spec: ", spec)

    print("[INFO] QueryCompiler initialized successfully with two-stage search implementation")
    print(f"Available UDFs: {list(registry.get_all_udfs().keys())}")
    
    compiler = QueryCompiler(registry, df, logger, coverage=coverage, track_stats=track_stats, dedup_threshold=dedup_threshold, limit=limit)
    # Execute query with two-stage search
    results = compiler.execute_query(spec)
    for result in results:
        print(result)

    # Optionally write results to JSON
    if out_path:
        _write_results_json(results, out_path)

    # Optionally generate visualizations per result and keyframe
    if do_viz:
        out_dir = Path(viz_dir) if viz_dir else Path(out_path).with_suffix("").with_name(Path(out_path).stem + "_viz") if out_path else Path("viz_out")
        out_dir.mkdir(parents=True, exist_ok=True)
        _generate_visualizations(df, results, out_dir)
    

if __name__ == "__main__":
    # example_usage()
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--coverage", type=float, default=1.0, help="Fraction of frames to scan (0-1]")
    parser.add_argument("--track-stats", action="store_true", help="Enable predicate selectivity stats")
    parser.add_argument("--out", type=str, default=None, help="Path to write results JSON")
    parser.add_argument("--viz", action="store_true", help="Generate visualization images per result keyframe")
    parser.add_argument("--viz-dir", type=str, default=None, help="Directory for visualization images (default derived from --out)")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of results to return")
    parser.add_argument("--dedup-threshold", type=float, default=0.25, help="Deduplication threshold for overlapping time windows")
    args = parser.parse_args()

    logger.info(f"Running example usage with spec: {args.spec} and data: {args.data}, coverage={args.coverage}, track_stats={args.track_stats}, out={args.out}, viz={args.viz}")
    example_usage(
        args.spec,
        args.data,
        coverage=args.coverage,
        track_stats=args.track_stats,
        out_path=args.out,
        do_viz=args.viz,
        viz_dir=args.viz_dir,
        limit=args.limit,
        dedup_threshold=args.dedup_threshold,
    )

