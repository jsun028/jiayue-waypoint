from pathlib import Path
import json
from typing import Optional, List

def find_data_files(dataset_dir: Path, pattern: str, recursive: bool, 
                    limit: Optional[int]) -> List[Path]:
    if recursive:
        files = sorted(dataset_dir.rglob(pattern))
    else:
        files = sorted(dataset_dir.glob(pattern))
    if limit is not None:
        files = files[: max(0, limit)]
    return [f for f in files if f.is_file()]

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