import shutil
from logging import getLogger
from pathlib import Path

from waveglow.app.defaults import DEFAULT_WAVEGLOW, DEFAULT_WAVEGLOW_VERSION
from waveglow.app.io import get_checkpoints_dir, get_train_dir
from waveglow.core import convert_glow, dl_wg
from waveglow.utils import get_pytorch_filename


def dl_pretrained(base_dir: Path, train_name: str = DEFAULT_WAVEGLOW, version: int = DEFAULT_WAVEGLOW_VERSION) -> None:
  train_dir = get_train_dir(base_dir, train_name)
  assert train_dir.is_dir()
  checkpoints_dir = get_checkpoints_dir(train_dir)
  tmp_dest_path = checkpoints_dir / get_pytorch_filename("1")

  dl_wg(
    destination=tmp_dest_path,
    version=version
  )

  checkpoint = convert_glow(
    origin=tmp_dest_path,
    destination=tmp_dest_path,
    keep_orig=False
  )

  final_dest_path = checkpoints_dir / get_pytorch_filename(checkpoint.iteration)
  if tmp_dest_path != final_dest_path:
    shutil.move(tmp_dest_path, final_dest_path)

  logger = getLogger(__name__)
  logger.info(f"Completed. Downloaded to: {final_dest_path}")

  # if prep_name is not None:
  # merge_dir = get_merged_dir(base_dir, merge_name, create=False)
  # prep_dir = get_prep_dir(merge_dir, prep_name, create=False)
  # wholeset = load_merged_data(merge_dir)
  # save_testset(prep_dir, wholeset)
  # # can be removed
  # save_valset(prep_dir, wholeset)

  # save_prep_settings(train_dir, ttsp_dir=None, merge_name=merge_name, prep_name=prep_name)
