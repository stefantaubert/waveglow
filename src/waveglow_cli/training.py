from argparse import ArgumentParser, Namespace
from logging import Logger
from pathlib import Path
from tempfile import gettempdir
from typing import Optional

from waveglow.model_checkpoint import CheckpointWaveglow
from waveglow.train import train
from waveglow.utils import (get_custom_or_last_checkpoint, get_last_checkpoint,
                            prepare_logger, split_hparams_string)

from waveglow_cli.argparse_helper import (get_optional,
                                          parse_existing_directory,
                                          parse_existing_file, parse_non_empty,
                                          parse_path)
from waveglow_cli.io import get_checkpoints_dir, get_train_dir
from waveglow_cli.parser import load_dataset


def try_load_checkpoint(base_dir: Path, train_name: Optional[str], checkpoint: Optional[int], logger: Logger) -> Optional[CheckpointWaveglow]:
  result = None
  if train_name:
    train_dir = get_train_dir(base_dir, train_name)
    checkpoint_path, _ = get_custom_or_last_checkpoint(
      get_checkpoints_dir(train_dir), checkpoint)
    result = CheckpointWaveglow.load(checkpoint_path, logger)
  return result


def init_train_parser(parser: ArgumentParser) -> None:
  default_log_path = Path(gettempdir()) / "waveglow_logs"
  parser.add_argument('train_folder', metavar="TRAIN-FOLDER-PATH", type=parse_existing_directory)
  parser.add_argument('val_folder', metavar="VAL-FOLDER-PATH", type=parse_existing_directory)
  parser.add_argument('checkpoints_dir',
                      metavar="CHECKPOINTS-FOLDER-PATH", type=parse_path)
  parser.add_argument('--custom-hparams', type=get_optional(parse_non_empty),
                      default=None, help="custom hparams comma separated")
  # Pretrained model
  parser.add_argument('--pretrained-model', type=get_optional(parse_existing_file), default=None)
  # Warm start
  parser.add_argument('--warm-start', action='store_true')
  parser.add_argument('--tl-dir', type=parse_path, default=default_log_path)
  parser.add_argument('--log-path', type=parse_path,
                      default=default_log_path / "log.txt")
  return train_ns


def train_ns(ns: Namespace) -> bool:
  logger = prepare_logger(ns.log_path, reset=True)

  warm_model = None
  if ns.pretrained_model is not None and ns.warm_start:
    warm_model = CheckpointWaveglow.load(ns.pretrained_model, logger)

  trainset = load_dataset(ns.train_folder)
  valset = load_dataset(ns.val_folder)

  custom_hparams = split_hparams_string(ns.custom_hparams)

  train(
    custom_hparams=custom_hparams,
    logdir=ns.tl_dir,
    trainset=trainset,
    valset=valset,
    save_checkpoint_dir=ns.checkpoints_dir,
    checkpoint=None,
    logger=logger,
    warm_model=warm_model,
  )

  return True


def init_continue_train_parser(parser: ArgumentParser) -> None:
  default_log_path = Path(gettempdir()) / "waveglow_logs"
  parser.add_argument('train_folder', metavar="TRAIN-FOLDER-PATH", type=parse_existing_directory)
  parser.add_argument('val_folder', metavar="VAL-FOLDER-PATH", type=parse_existing_directory)
  parser.add_argument('checkpoints_dir',
                      metavar="CHECKPOINTS-FOLDER-PATH", type=parse_path)
  parser.add_argument('--custom-hparams', type=get_optional(parse_non_empty),
                      default=None, help="custom hparams comma separated")
  parser.add_argument('--tl-dir', type=parse_path, default=default_log_path)
  parser.add_argument('--log-path', type=parse_path,
                      default=default_log_path / "log.txt")
  return continue_train_ns


def continue_train_ns(ns: Namespace) -> bool:
  logger = prepare_logger(ns.log_path, reset=True)

  trainset = load_dataset(ns.train_folder)
  valset = load_dataset(ns.val_folder)

  custom_hparams = split_hparams_string(ns.custom_hparams)

  last_checkpoint_path, _ = get_last_checkpoint(ns.checkpoints_dir)
  checkpoint = CheckpointWaveglow.load(last_checkpoint_path, logger)

  train(
    custom_hparams=custom_hparams,
    logdir=ns.tl_dir,
    trainset=trainset,
    valset=valset,
    save_checkpoint_dir=ns.checkpoints_dir,
    checkpoint=checkpoint,
    logger=logger,
    warm_model=None,
  )

  return True
