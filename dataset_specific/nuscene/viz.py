from .nuscene_traj_viz import plot_bev_snapshot
from pathlib import Path
import json
import pandas as pd
from loguru import logger

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

def _generate_visualizations(df: pd.DataFrame, results: list[dict], out_dir: Path, 
                             fps: int = 10, top_k: int | None = None, fname: str | None = None) -> None:
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

    # Sort by score (descending) and take top-k if specified
    sorted_results = sorted(results, key=lambda x: x.get('aggregate_score', 0.0), reverse=True)
    if top_k is not None and top_k > 0:
        sorted_results = sorted_results[:top_k]
        logger.info(f"Visualizing top {len(sorted_results)} results (top_k={top_k})")
    
    for idx, res in enumerate(sorted_results):
        assignment = res.get('object_assignment', {})
        highlight_ids = set(int(v) for v in assignment.values()) if assignment else set()
        time_range = res.get('time_range')
        start_f, end_f = _parse_time_range(time_range) if isinstance(time_range, str) else (int(df['frame_index'].min()), int(df['frame_index'].max()))
        frames = list(range(start_f, end_f + 1))
        score = res.get('aggregate_score', 0.0)

        fig, ax = plt.subplots(figsize=(12, 12))

        # Precompute fixed axis limits over the animation window to avoid jank
        df_range = df[df['frame_index'].between(start_f, end_f)]
        try:
            all_x = pd.concat([df_range['x1'], df_range['x2']])
            all_y = pd.concat([df_range['y1'], df_range['y2']])
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
            ax.set_title(f"Result {idx+1} (score: {score:.4f}) – frame {frame_idx}")
            # Lock axis limits and aspect
            if xlim and ylim:
                ax.set_xlim(*xlim)
                ax.set_ylim(*ylim)
            ax.set_aspect('equal')
            ax.set_autoscale_on(False)

        anim = FuncAnimation(fig, lambda i: update(frames[i]), frames=len(frames), interval=1000 / max(1, fps), repeat=True)
        if fname is not None:
            out_path = out_dir / f"{fname}.gif"
        else:
            out_path = out_dir / f"result_{idx+1}_score_{score:.4f}.gif"
        writer = PillowWriter(fps=fps)
        anim.save(out_path, writer=writer)
        plt.close(fig)

    logger.info(f"Wrote visualization GIFs → {out_dir}")