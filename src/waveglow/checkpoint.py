from dataclasses import asdict, dataclass
from logging import getLogger
from pathlib import Path
from typing import Optional, Type, TypeVar

import torch

from waveglow.utils import get_dataclass_from_dict

_HParamsType = TypeVar("_HParamsType")


@dataclass
class Checkpoint():
  # Renaming of any of these fields will destroy previous models!
  state_dict: dict
  optimizer: dict
  learning_rate: float
  iteration: int
  hparams: dict

  def get_hparams(self, hparam_type: Type[_HParamsType]) -> _HParamsType:
    logger = getLogger(__name__)
    res, ignored = get_dataclass_from_dict(self.hparams, hparam_type)
    if len(ignored) > 0:
      logger.warning(
        f"Ignored these hparams from checkpoint because they did not exist in the current HParams: {ignored}.")
    return res

  def save(self, checkpoint_path: Path):
    logger = getLogger(__name__)
    logger.info(f"Saving model at iteration {self.iteration}...")
    checkpoint_dict = asdict(self)
    torch.save(checkpoint_dict, checkpoint_path)
    logger.info(f"Saved model to '{checkpoint_path}'.")

  @classmethod
  def load(cls, checkpoint_path: Path, device: torch.device):
    logger = getLogger(__name__)
    assert checkpoint_path.is_file()
    logger.info(f"Loading model '{checkpoint_path}'...")
    checkpoint_dict = torch.load(checkpoint_path, map_location=device)
    result = cls(**checkpoint_dict)
    logger.info(f"Loaded model at iteration {result.iteration}.")
    return result


def get_iteration(checkpoint: Optional[Checkpoint]) -> int:
  return checkpoint.iteration if checkpoint is not None else 0
