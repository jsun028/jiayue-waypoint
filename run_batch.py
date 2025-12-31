import argparse
import json
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple


#CONDA_PYTHON = "/nethome/jxu680/miniconda3/envs/kql/bin/python"
CONDA_PYTHON = "/Users/kexinrong/Documents/research/d2i/keyframe-ui/.venv/bin/python"
MAIN_ENTRY = Path(__file__).parent / "main.py"


def find_data_files(dataset_dir: Path, pattern: str, recursive: bool, limit: Optional[int]) -> List[Path]:
    if recursive:
        files = sorted(dataset_dir.rglob(pattern))
    else:
        files = sorted(dataset_dir.glob(pattern))
    if limit is not None:
        files = files[: max(0, limit)]
    return [f for f in files if f.is_file()]


def build_command(
    spec_path: Path,
    data_path: Path,
    out_json_path: Path,
    viz: bool,
    viz_dir: Optional[Path],
    coverage: Optional[float],
    track_stats: bool,
    limit: Optional[int],
    dedup_threshold: Optional[float],
    slider_setting: str,
) -> List[str]:
    cmd: List[str] = [
        CONDA_PYTHON,
        str(MAIN_ENTRY),
        "--spec",
        str(spec_path),
        "--data",
        str(data_path),
        "--out",
        str(out_json_path),
        "--slider-setting",
        slider_setting,
    ]
    if coverage is not None:
        cmd += ["--coverage", str(coverage)]
    if track_stats:
        cmd += ["--track-stats"]
    if viz:
        cmd += ["--viz"]
        if viz_dir is not None:
            cmd += ["--viz-dir", str(viz_dir)]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    if dedup_threshold is not None:
        cmd += ["--dedup-threshold", str(dedup_threshold)]
    return cmd


def has_nonempty_results(results_json_path: Path) -> bool:
    if not results_json_path.exists():
        return False
    try:
        with results_json_path.open("r") as f:
            data = json.load(f)
        if isinstance(data, list) and len(data) > 0:
            return True
        # Some runs may produce a dict with a key like "results"
        if isinstance(data, dict):
            for key in ("results", "items"):
                if key in data and isinstance(data[key], list) and len(data[key]) > 0:
                    return True
    except Exception:
        # If results file is corrupt, treat as empty
        return False
    return False


def run_single(
    spec_path: Path,
    data_path: Path,
    sample_out_dir: Path,
    viz: bool,
    coverage: Optional[float],
    track_stats: bool,
    limit: Optional[int],
    dedup_threshold: Optional[float],
    overwrite: bool,
    env: Optional[dict],
    slider_setting: str,
) -> Tuple[Path, bool, int]:
    sample_out_dir.mkdir(parents=True, exist_ok=True)
    results_json_path = sample_out_dir / "results.json"
    viz_dir = sample_out_dir / "viz"

    if results_json_path.exists() and not overwrite:
        nonempty = has_nonempty_results(results_json_path)
        return (sample_out_dir, nonempty, 0)

    log_path = sample_out_dir / "run.log"
    cmd = build_command(
        spec_path=spec_path,
        data_path=data_path,
        out_json_path=results_json_path,
        viz=viz,
        viz_dir=viz_dir if viz else None,
        coverage=coverage,
        track_stats=track_stats,
        limit=limit,
        dedup_threshold=dedup_threshold,
        slider_setting=slider_setting,
    )

    # Ensure environment inherits current plus any overrides
    proc_env = os.environ.copy()
    if env:
        proc_env.update(env)

    with log_path.open("wb") as log_file:
        proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=str(MAIN_ENTRY.parent),
            env=proc_env,
        )
        return_code = proc.wait()

    nonempty = has_nonempty_results(results_json_path)
    return (sample_out_dir, nonempty, return_code)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch runner for main.py over a dataset folder.")
    parser.add_argument("--spec", type=str, required=True, help="Path to a pickled spec file (.pkl)")
    parser.add_argument("--dataset-dir", type=str, required=True, help="Folder containing data files (e.g., CSVs)")
    parser.add_argument("--pattern", type=str, default="*.csv", help="Glob pattern to match data files")
    parser.add_argument("--recursive", action="store_true", help="Search recursively for data files")
    parser.add_argument("--n-data", type=int, default=None, help="Limit number of samples to process")
    parser.add_argument("--output-dir", type=str, required=True, help="Master output folder")
    parser.add_argument("--max-workers", type=int, default=os.cpu_count() or 4, help="Max concurrent workers")
    parser.add_argument("--overwrite", action="store_true", help="Re-run even if results.json exists")
    parser.add_argument("--keep-empty", action="store_true", help="Keep (rename) empty result dirs instead of deleting them")

    # Pass-through options to main.py
    parser.add_argument("--coverage", type=float, default=None)
    parser.add_argument("--track-stats", action="store_true")
    parser.add_argument("--viz", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dedup-threshold", type=float, default=None)
    
    # Slider settings configuration
    parser.add_argument("--slider-settings", type=str, default="low,medium,high",
                       help="Comma-delimited slider settings to run (e.g., 'low,medium,high' or 'medium' or 'low,high')")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    spec_path = Path(args.spec).resolve()
    dataset_dir = Path(args.dataset_dir).resolve()
    master_out_dir = Path(args.output_dir).resolve()
    master_out_dir.mkdir(parents=True, exist_ok=True)

    data_files = find_data_files(dataset_dir, args.pattern, args.recursive, args.n_data)
    if not data_files:
        print("No data files found. Nothing to do.")
        return

    # Parse slider settings from comma-delimited string
    slider_settings = [s.strip() for s in args.slider_settings.split(",") if s.strip()]
    if not slider_settings:
        print("Error: No valid slider settings provided.")
        return
    
    # Validate slider settings
    valid_settings = {"low", "medium", "high"}
    invalid = [s for s in slider_settings if s not in valid_settings]
    if invalid:
        print(f"Error: Invalid slider settings: {invalid}. Valid options are: low, medium, high")
        return
    
    total_runs = len(data_files) * len(slider_settings)
    print(f"Discovered {len(data_files)} data files × {len(slider_settings)} slider settings {slider_settings} = {total_runs} total runs.")
    print(f"Running up to {args.max_workers} in parallel.")

    futures = []
    results: List[Tuple[Path, bool, int]] = []

    def submit(executor: ThreadPoolExecutor, idx: int, data_path: Path, slider_setting: str):
        # Create directory structure: low/00001_scene-0225/, medium/00001_scene-0225/, high/00001_scene-0225/
        base_subdir_name = f"{idx:05d}_" + data_path.stem
        sample_out_dir = master_out_dir / slider_setting / base_subdir_name
        return executor.submit(
            run_single,
            spec_path,
            data_path,
            sample_out_dir,
            args.viz,
            args.coverage,
            args.track_stats,
            args.limit,
            args.dedup_threshold,
            args.overwrite,
            None,
            slider_setting,
        )

    with ThreadPoolExecutor(max_workers=max(1, int(args.max_workers))) as executor:
        for idx, data_path in enumerate(data_files, start=1):
            # Submit jobs for all three slider settings
            for slider_setting in slider_settings:
                futures.append(submit(executor, idx, data_path, slider_setting))

        for fut in as_completed(futures):
            try:
                res = fut.result()
                results.append(res)
                out_dir, nonempty, return_code = res
                status = "OK" if return_code == 0 else f"RC={return_code}"
                # Show relative path from master_out_dir for clarity
                rel_path = out_dir.relative_to(master_out_dir)
                print(f"Completed: {rel_path} → results={'nonempty' if nonempty else 'empty'} ({status})")
            except Exception as e:
                print(f"Worker failed with exception: {e}")

    kept = 0
    deleted = 0
    renamed = 0
    failed = 0

    for out_dir, nonempty, return_code in results:
        if nonempty:
            kept += 1
            continue
        # Treat nonempty False as zero results; delete or rename the folder
        try:
            if args.keep_empty:
                # Rename with _empty suffix
                renamed_dir = out_dir.with_name(out_dir.name + "_empty")
                # Handle case where renamed dir already exists
                counter = 1
                while renamed_dir.exists():
                    renamed_dir = out_dir.with_name(out_dir.name + f"_empty_{counter}")
                    counter += 1
                out_dir.rename(renamed_dir)
                renamed += 1
            else:
                shutil.rmtree(out_dir, ignore_errors=True)
                deleted += 1
        except Exception:
            failed += 1

    total = len(results)
    if args.keep_empty:
        print(
            f"Done. Total: {total}, kept (with results): {kept}, renamed (no results): {renamed}, failures: {failed}"
        )
    else:
        print(
            f"Done. Total: {total}, kept (with results): {kept}, deleted (no results): {deleted}, delete-failures: {failed}"
        )


if __name__ == "__main__":
    main()


