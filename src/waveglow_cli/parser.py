from pathlib import Path

from waveglow.typing import Entries, Entry
from waveglow.utils import get_all_files_in_all_subfolders


def load_dataset(path: Path) -> Entries:
  files = get_all_files_in_all_subfolders(path)
  result_wavs = (file for file in files if file.suffix.lower() == ".wav")
  result = []
  for wav in result_wavs:
    entry = Entry(
      stem=wav.relative_to(path).parent / wav.stem,
      basename=wav.stem,
      wav_absolute_path=wav
    )
    result.append(entry)
  return result
