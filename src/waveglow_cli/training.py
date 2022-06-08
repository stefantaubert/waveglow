from argparse import ArgumentParser, Namespace
from pathlib import Path
from tempfile import gettempdir

from waveglow.model_checkpoint import CheckpointWaveglow
from waveglow.train import train
from waveglow.utils import (get_last_checkpoint, prepare_logger, set_torch_thread_to_max,
                            split_hparams_string)
from waveglow_cli.argparse_helper import (get_optional, parse_existing_directory,
                                          parse_existing_file, parse_path)
from waveglow_cli.helper import add_device_argument, add_hparams_argument
from waveglow_cli.parser import load_dataset

# def try_load_checkpoint(base_dir: Path, train_name: Optional[str], checkpoint: Optional[int], logger: Logger) -> Optional[CheckpointWaveglow]:
#   result = None
#   if train_name:
#     train_dir = get_train_dir(base_dir, train_name)
#     checkpoint_path, _ = get_custom_or_last_checkpoint(
#       get_checkpoints_dir(train_dir), checkpoint)
#     result = CheckpointWaveglow.load(checkpoint_path, device, logger)
#   return result


def init_training_parser(parser: ArgumentParser) -> None:
  default_log_path = Path(gettempdir()) / "waveglow_logs"
  parser.description = "Start training of a new model."
  parser.add_argument('train_folder', metavar="TRAIN-FOLDER",
                      type=parse_existing_directory, help="path to folder containing training data (i.e., .wav <=> .TextGrid pairs)")
  parser.add_argument('val_folder', metavar="VAL-FOLDER",
                      type=parse_existing_directory, help="path to folder containing validation data (i.e., .wav <=> .TextGrid pairs)")
  parser.add_argument('checkpoints_dir',
                      metavar="CHECKPOINTS-FOLDER", type=parse_path, help="path to folder to write checkpoints")
  add_device_argument(parser)
  add_hparams_argument(parser)
  # Pretrained model
  parser.add_argument('--pre-trained-model', type=get_optional(parse_existing_file), metavar="PRE-TRAINED-MODEL",
                      default=None, help="path to checkpoint that will be used for warm-start")
  # Warm start
  parser.add_argument('--warm-start', action='store_true',
                      help="warm start using PRE-TRAINED-MODEL")
  parser.add_argument('--tl-dir', type=parse_path, metavar="TENSORBOARD-LOG", default=default_log_path,
                      help="path to folder for outputting tensorboard logs (currently not available)")
  parser.add_argument('--log-path', type=parse_path, metavar="LOG",
                      default=default_log_path / "log.txt", help="path to file for outputting training logs")
  return train_ns


def train_ns(ns: Namespace) -> bool:
  set_torch_thread_to_max()
  ns.log_path.parent.mkdir(parents=True, exist_ok=True)
  logger = prepare_logger(ns.log_path, reset=True)

  warm_model = None
  if ns.pre_trained_model is not None and ns.warm_start:
    warm_model = CheckpointWaveglow.load(ns.pre_trained_model, ns.device, logger)

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
    device=ns.device,
  )

  return True


def init_continue_training_parser(parser: ArgumentParser) -> None:
  default_log_path = Path(gettempdir()) / "waveglow_logs"
  parser.add_argument('train_folder', metavar="TRAIN-FOLDER",
                      type=parse_existing_directory, help="path to folder containing training data (i.e., .wav <=> .TextGrid pairs)")
  parser.add_argument('val_folder', metavar="VAL-FOLDER",
                      type=parse_existing_directory, help="path to folder containing validation data (i.e., .wav <=> .TextGrid pairs)")
  parser.add_argument('checkpoints_dir',
                      metavar="CHECKPOINTS-FOLDER", type=parse_existing_directory, help="path to folder to write checkpoints")
  add_device_argument(parser)
  add_hparams_argument(parser)
  parser.add_argument('--tl-dir', type=parse_path, metavar="TENSORBOARD-LOG", default=default_log_path,
                      help="path to folder for outputting tensorboard logs (currently not available)")
  parser.add_argument('--log-path', type=parse_path, metavar="LOG",
                      default=default_log_path / "log.txt", help="path to file for outputting training logs")
  return continue_train_ns


def continue_train_ns(ns: Namespace) -> bool:
  set_torch_thread_to_max()

  logger = prepare_logger(ns.log_path, reset=False)

  trainset = load_dataset(ns.train_folder)
  valset = load_dataset(ns.val_folder)

  custom_hparams = split_hparams_string(ns.custom_hparams)

  last_checkpoint_path, _ = get_last_checkpoint(ns.checkpoints_dir)
  checkpoint = CheckpointWaveglow.load(last_checkpoint_path, ns.device, logger)

  train(
    custom_hparams=custom_hparams,
    logdir=ns.tl_dir,
    trainset=trainset,
    valset=valset,
    save_checkpoint_dir=ns.checkpoints_dir,
    checkpoint=checkpoint,
    logger=logger,
    warm_model=None,
    device=ns.device,
  )

  return True
