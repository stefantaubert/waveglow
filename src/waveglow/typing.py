

from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass()
class Entry():
  stem: str
  basename: str
  wav_absolute_path: Path
  #wav_duration: float
  #wav_sampling_rate: int
  #mel_absolute_path: Path
  #mel_n_channels: int


Entries = List[Entry]

