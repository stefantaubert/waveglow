import os
from logging import Logger
from typing import Dict, Optional

from tts_preparation import (get_merged_dir, get_prep_dir, load_trainset,
                             load_valset)
from waveglow.app.io import (get_checkpoints_dir, get_train_dir,
                             get_train_log_file, get_train_logs_dir,
                             load_prep_settings, save_prep_settings)
from waveglow.core import CheckpointWaveglow
from waveglow.core import continue_train as continue_train_core
from waveglow.core import train as train_core
from waveglow.utils import get_custom_or_last_checkpoint, prepare_logger


def try_load_checkpoint(base_dir: str, train_name: Optional[str], checkpoint: Optional[int], logger: Logger) -> Optional[CheckpointWaveglow]:
  result = None
  if train_name:
    train_dir = get_train_dir(base_dir, train_name, False)
    checkpoint_path, _ = get_custom_or_last_checkpoint(
      get_checkpoints_dir(train_dir), checkpoint)
    result = CheckpointWaveglow.load(checkpoint_path, logger)
  return result


def train(base_dir: str, ttsp_dir: str, train_name: str, merge_name: str, prep_name: str, custom_hparams: Optional[Dict[str, str]] = None, warm_start_train_name: Optional[str] = None, warm_start_checkpoint: Optional[int] = None):
  merge_dir = get_merged_dir(ttsp_dir, merge_name, create=False)
  prep_dir = get_prep_dir(merge_dir, prep_name, create=False)
  train_dir = get_train_dir(base_dir, train_name, create=True)
  trainset = load_trainset(prep_dir)
  valset = load_valset(prep_dir)

  # wholeset = load_filelist(prep_dir)
  # trainset, testset, valset = split_prepared_data_train_test_val(
  #   wholeset, test_size=test_size, validation_size=validation_size, seed=split_seed, shuffle=True)
  # save_trainset(train_dir, trainset)
  # save_testset(train_dir, testset)
  # save_valset(train_dir, valset)

  logs_dir = get_train_logs_dir(train_dir)
  logger = prepare_logger(get_train_log_file(logs_dir), reset=True)

  warm_model = try_load_checkpoint(
    base_dir=base_dir,
    train_name=warm_start_train_name,
    checkpoint=warm_start_checkpoint,
    logger=logger
  )

  save_prep_settings(train_dir, ttsp_dir, merge_name, prep_name)

  train_core(
    custom_hparams=custom_hparams,
    logdir=logs_dir,
    trainset=trainset,
    valset=valset,
    save_checkpoint_dir=get_checkpoints_dir(train_dir),
    debug_logger=logger,
    warm_model=warm_model,
  )


def continue_train(base_dir: str, train_name: str, custom_hparams: Optional[Dict[str, str]] = None):
  train_dir = get_train_dir(base_dir, train_name, create=False)
  assert train_dir.is_dir()

  logs_dir = get_train_logs_dir(train_dir)
  logger = prepare_logger(get_train_log_file(logs_dir))

  ttsp_dir, merge_name, prep_name = load_prep_settings(train_dir)
  merge_dir = get_merged_dir(ttsp_dir, merge_name, create=False)
  prep_dir = get_prep_dir(merge_dir, prep_name, create=False)
  trainset = load_trainset(prep_dir)
  valset = load_valset(prep_dir)

  continue_train_core(
    custom_hparams=custom_hparams,
    logdir=logs_dir,
    trainset=trainset,
    valset=valset,
    save_checkpoint_dir=get_checkpoints_dir(train_dir),
    debug_logger=logger
  )
