from pathlib import Path
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