from argparse import ArgumentParser, Namespace
from functools import partial
from pathlib import Path
from typing import Optional, Set

import imageio
import numpy as np
from ordered_set import OrderedSet
from tqdm import tqdm

from waveglow.audio_utils import float_to_wav
from waveglow.image_utils import stack_images_vertically
from waveglow.model_checkpoint import CheckpointWaveglow
from waveglow.typing import Entry
from waveglow.utils import (get_checkpoint, get_last_checkpoint, prepare_logger,
                            set_torch_thread_to_max, split_hparams_string)
from waveglow.validation import ValidationEntries, ValidationEntryOutput, get_df, validate
from waveglow_cli.argparse_helper import (ConvertToOrderedSetAction, ConvertToSetAction,
                                          get_optional, parse_existing_directory, parse_non_empty,
                                          parse_non_negative_integer, parse_path,
                                          parse_positive_integer)
from waveglow_cli.helper import (add_denoiser_and_sigma_arguments, add_device_argument,
                                 add_hparams_argument)
from waveglow_cli.parser import load_dataset


def get_repr_entries(entry_ids: Optional[Set[int]]) -> str:
  if entry_ids is None:
    return "none"
  if len(entry_ids) == 0:
    return "empty"
  return ",".join(list(sorted(map(str, entry_ids))))


def get_repr_speaker(speaker: Optional[str]) -> str:
  if speaker is None:
    return "none"
  return speaker


# def get_val_dir(train_dir: Path, ds: str, iterations: Set[int], full_run: bool, entry_ids: Optional[Set[int]]) -> Path:
#   if len(iterations) > 3:
#     its = ",".join(str(x) for x in sorted(iterations)[:3]) + ",..."
#   else:
#     its = ",".join(str(x) for x in sorted(iterations))

#   subdir_name = f"{datetime.datetime.now():%d.%m.%Y__%H-%M-%S}__ds={ds}__entries={get_repr_entries(entry_ids)}__its={its}__full={full_run}"
#   return _get_validation_root_dir(train_dir) / subdir_name


def get_val_entry_dir(val_dir: Path, entry: Entry, iteration: int) -> Path:
  return val_dir / f"it={iteration}_name={entry.basename}"


def save_stats(val_dir: Path, validation_entries: ValidationEntries) -> None:
  path = val_dir / "total.csv"
  df = get_df(validation_entries)
  df.to_csv(path, sep="\t", header=True, index=False)


def save_results(entry: Entry, output: ValidationEntryOutput, val_dir: Path, iteration: int) -> None:
  dest_dir = get_val_entry_dir(val_dir, entry, iteration)
  dest_dir.mkdir(parents=True, exist_ok=True)
  imageio.imsave(dest_dir / "original.png", output.mel_orig_img)
  imageio.imsave(dest_dir / "inferred_denoised.png", output.mel_inferred_denoised_img)
  imageio.imsave(dest_dir / "diff.png", output.mel_denoised_diff_img)
  np.save(dest_dir / "original.mel.npy", output.mel_orig)
  np.save(dest_dir / "inferred_denoised.mel.npy", output.mel_inferred_denoised)
  float_to_wav(output.wav_orig, dest_dir / "original.wav", sample_rate=output.orig_sr)

  float_to_wav(output.wav_inferred_denoised, dest_dir /
               "inferred_denoised.wav", sample_rate=output.inferred_sr)

  float_to_wav(output.wav_inferred, dest_dir / "inferred.wav", sample_rate=output.inferred_sr)

  stack_images_vertically(
    list_im=[
      dest_dir / "original.png",
      dest_dir / "inferred_denoised.png",
      dest_dir / "diff.png",
    ],
    out_path=dest_dir / "comparison.png"
  )


def init_validation_parser(parser: ArgumentParser) -> None:
  parser.description = "Validate checkpoint(s) using the validation set or any other dataset."
  parser.add_argument('checkpoints_dir',
                      metavar="CHECKPOINTS-FOLDER", type=parse_existing_directory, help="path to folder containing the checkpoints that should be validated")
  parser.add_argument('output_dir',
                      metavar="OUTPUT-FOLDER", type=parse_path, help="path to folder in which the resulting files should be written")
  parser.add_argument('dataset_dir', metavar="DATA-FOLDER",
                      type=parse_existing_directory, help="path to validation set folder or any other dataset")
  add_denoiser_and_sigma_arguments(parser)
  add_device_argument(parser)
  add_hparams_argument(parser)
  parser.add_argument('--full-run', action='store_true', help="validate all files in DATA-FOLDER")
  parser.add_argument('--files', type=parse_non_empty, nargs="*", metavar="UTTERANCE",
                      help="names of utterances in DATA-FOLDER that should be validated; if left unset a random utterance will be chosen", default=OrderedSet(), action=ConvertToSetAction)
  parser.add_argument('--custom-checkpoints',
                      type=parse_positive_integer, nargs="*", default=OrderedSet(), action=ConvertToOrderedSetAction, help="validate checkpoints with these iterations; if left unset the last iteration is chosen")
  parser.add_argument('--custom-seed', type=get_optional(parse_non_negative_integer),
                      default=None, help="custom seed used for synthesis; if left unset a random seed will be chosen")
  return validate_ns


def validate_ns(ns: Namespace) -> bool:
  set_torch_thread_to_max()
  data = load_dataset(ns.dataset_dir)

  iterations: OrderedSet[int]

  if len(ns.custom_checkpoints) == 0:
    _, last_it = get_last_checkpoint(ns.checkpoints_dir)
    iterations = OrderedSet((last_it,))
  else:
    iterations = ns.custom_checkpoints

  ns.output_dir.mkdir(parents=True, exist_ok=True)
  val_log_path = ns.output_dir / "log.txt"
  logger = prepare_logger(val_log_path)
  logger.info("Validating...")
  logger.info(f"Checkpoints: {','.join(str(x) for x in sorted(iterations))}")

  result = ValidationEntries()
  custom_hparams = split_hparams_string(ns.custom_hparams)

  for iteration in tqdm(sorted(iterations)):
    logger.info(f"Current checkpoint: {iteration}")
    checkpoint_path = get_checkpoint(ns.checkpoints_dir, iteration)
    taco_checkpoint = CheckpointWaveglow.load(checkpoint_path, ns.device, logger)
    save_callback = partial(save_results, val_dir=ns.output_dir, iteration=iteration)

    validation_entries = validate(
      checkpoint=taco_checkpoint,
      data=data,
      custom_hparams=custom_hparams,
      entry_names=ns.files,
      full_run=ns.full_run,
      save_callback=save_callback,
      logger=logger,
      sigma=ns.sigma,
      denoiser_strength=ns.denoiser_strength,
      seed=ns.custom_seed,
      device=ns.device,
    )

    result.extend(validation_entries)

  if len(result) > 0:
    save_stats(ns.output_dir, result)
    logger.info(f"Saved output to: {ns.output_dir.absolute()}")

  return True
