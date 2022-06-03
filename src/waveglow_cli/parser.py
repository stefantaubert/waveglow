from pathlib import Path
from typing import Generator, List

from waveglow.utils import get_all_files_in_all_subfolders


def load_dataset(path: Path) -> List[Path]:
  files = get_all_files_in_all_subfolders(path)
  result_wavs = (file for file in files if file.suffix.lower() == ".wav")
  return list(result_wavs)
