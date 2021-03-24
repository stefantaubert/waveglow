import os
from typing import Optional

from waveglow.app.defaults import DEFAULT_WAVEGLOW
from waveglow.app.io import (get_checkpoints_dir, get_train_dir,
                             save_prep_settings)
from waveglow.core.converter.convert import convert_glow
from waveglow.core.dl_pretrained import dl_wg
from waveglow.utils import get_pytorch_filename


def get_checkpoint_pretrained(checkpoints_dir: str):
  return os.path.join(checkpoints_dir, get_pytorch_filename("1"))


def dl_pretrained(base_dir: str, train_name: str = DEFAULT_WAVEGLOW, merge_name: Optional[str] = None, prep_name: Optional[str] = None, version: int = 3):
  train_dir = get_train_dir(base_dir, train_name, create=True)
  assert os.path.isdir(train_dir)
  checkpoints_dir = get_checkpoints_dir(train_dir)
  dest_path = get_checkpoint_pretrained(checkpoints_dir)

  print("Downloading pretrained waveglow model from Nvida...")
  dl_wg(
    destination=dest_path,
    version=version
  )

  print("Pretrained model is now beeing converted to be able to use it...")
  convert_glow(
    origin=dest_path,
    destination=dest_path,
    keep_orig=False
  )

  # if prep_name is not None:
  # merge_dir = get_merged_dir(base_dir, merge_name, create=False)
  # prep_dir = get_prep_dir(merge_dir, prep_name, create=False)
  # wholeset = load_merged_data(merge_dir)
  # save_testset(prep_dir, wholeset)
  # # can be removed
  # save_valset(prep_dir, wholeset)

  save_prep_settings(train_dir, merge_name=merge_name, prep_name=prep_name)


if __name__ == "__main__":
  dl_pretrained(
    version=3,
    train_name="pretrained_v3",
    base_dir="/datasets/models/taco2pt_v5",
    prep_name="arctic_ipa",
  )
