import datetime
import os
from functools import partial
from typing import Dict, Optional, Set

import imageio
import numpy as np
from audio_utils import float_to_wav
from image_utils import stack_images_vertically
from tqdm import tqdm
from tts_preparation import (PreparedData, PreparedDataList, get_merged_dir,
                             get_prep_dir, load_testset, load_valset)
from tts_preparation.app.prepare2 import load_totalset
from waveglow.app.io import (_get_validation_root_dir, get_checkpoints_dir,
                             get_train_dir, load_prep_settings)
from waveglow.core.model_checkpoint import CheckpointWaveglow
from waveglow.core.validation import (ValidationEntries, ValidationEntryOutput,
                                      validate)
from waveglow.utils import (get_all_checkpoint_iterations, get_checkpoint,
                            get_last_checkpoint, get_subdir, prepare_logger)


def get_repr_entries(entry_ids: Optional[Set[int]]):
  if entry_ids is None:
    return "none"
  if len(entry_ids) == 0:
    return "empty"
  return ",".join(list(sorted(map(str, entry_ids))))


def get_repr_speaker(speaker: Optional[str]):
  if speaker is None:
    return "none"
  return speaker


def get_val_dir(train_dir: str, ds: str, iterations: Set[int], full_run: bool, entry_ids: Optional[Set[int]]):
  if len(iterations) > 3:
    its = ",".join(str(x) for x in sorted(iterations)[:3]) + ",..."
  else:
    its = ",".join(str(x) for x in sorted(iterations))

  subdir_name = f"{datetime.datetime.now():%d.%m.%Y__%H-%M-%S}__ds={ds}__entries={get_repr_entries(entry_ids)}__its={its}__full={full_run}"
  return get_subdir(_get_validation_root_dir(train_dir), subdir_name, create=True)


def get_val_entry_dir(val_dir: str, entry: PreparedData, iteration: int) -> None:
  return get_subdir(val_dir, f"it={iteration}_id={entry.entry_id}", create=True)


def save_stats(val_dir: str, validation_entries: ValidationEntries) -> None:
  path = os.path.join(val_dir, "total.csv")
  validation_entries.save(path, header=True)


def save_results(entry: PreparedData, output: ValidationEntryOutput, val_dir: str, iteration: int):
  dest_dir = get_val_entry_dir(val_dir, entry, iteration)
  imageio.imsave(os.path.join(dest_dir, "original.png"), output.mel_orig_img)
  imageio.imsave(os.path.join(dest_dir, "inferred.png"), output.mel_inferred_img)
  imageio.imsave(os.path.join(dest_dir, "diff.png"), output.mel_diff_img)
  np.save(os.path.join(dest_dir, "original.mel.npy"), output.mel_orig)
  np.save(os.path.join(dest_dir, "inferred.mel.npy"), output.mel_inferred)
  float_to_wav(output.wav_orig, os.path.join(
    dest_dir, "original.wav"), sample_rate=output.orig_sr)

  float_to_wav(output.wav_inferred, os.path.join(
    dest_dir, "inferred.wav"), sample_rate=output.inferred_sr)

  float_to_wav(output.wav_inferred_no_denoise, os.path.join(
    dest_dir, "inferred_no_denoise.wav"), sample_rate=output.inferred_sr)

  stack_images_vertically(
    list_im=[
      os.path.join(dest_dir, "original.png"),
      os.path.join(dest_dir, "inferred.png"),
      os.path.join(dest_dir, "diff.png"),
    ],
    out_path=os.path.join(dest_dir, "comparison.png")
  )


def app_validate_generic(base_dir: str, ttsp_dir: str, merge_name: str, prep_name: str, train_name: str, ds: str = "val", entry_ids: Optional[Set[int]] = None, custom_checkpoints: Optional[Set[int]] = None, sigma: float = 0.666, denoiser_strength: float = 0.00, custom_hparams: Optional[Dict[str, str]] = None, full_run: bool = False):
  train_dir = get_train_dir(base_dir, train_name, create=False)
  assert os.path.isdir(train_dir)

  merge_dir = get_merged_dir(ttsp_dir, merge_name, create=False)
  prep_dir = get_prep_dir(merge_dir, prep_name, create=False)

  validate_core(
    train_dir=train_dir,
    train_name=train_name,
    prep_dir=prep_dir,
    entry_ids=entry_ids,
    custom_checkpoints=custom_checkpoints,
    sigma=sigma,
    denoiser_strength=denoiser_strength,
    custom_hparams=custom_hparams,
    full_run=full_run,
    ds=ds,
  )


def app_validate(base_dir: str, train_name: str, entry_ids: Optional[Set[int]] = None, ds: str = "val", custom_checkpoints: Optional[Set[int]] = None, sigma: float = 0.666, denoiser_strength: float = 0.00, custom_hparams: Optional[Dict[str, str]] = None, full_run: bool = False):
  train_dir = get_train_dir(base_dir, train_name, create=False)
  assert os.path.isdir(train_dir)

  ttsp_dir, merge_name, prep_name = load_prep_settings(train_dir)
  merge_dir = get_merged_dir(ttsp_dir, merge_name, create=False)
  prep_dir = get_prep_dir(merge_dir, prep_name, create=False)

  validate_core(
    train_dir=train_dir,
    train_name=train_name,
    prep_dir=prep_dir,
    entry_ids=entry_ids,
    custom_checkpoints=custom_checkpoints,
    sigma=sigma,
    denoiser_strength=denoiser_strength,
    custom_hparams=custom_hparams,
    full_run=full_run,
    ds=ds,
  )


def validate_core(train_dir, train_name: str, prep_dir: str, entry_ids: Optional[Set[int]] = None, custom_checkpoints: Optional[Set[int]] = None, sigma: float = 0.666, denoiser_strength: float = 0.00, custom_hparams: Optional[Dict[str, str]] = None, full_run: bool = False, ds: str = "val"):
  if ds == "val":
    data = load_valset(prep_dir)
  elif ds == "test":
    data = load_testset(prep_dir)
  elif ds == "total":
    data = load_totalset(prep_dir)
  else:
    raise Exception()

  iterations: Set[int] = set()
  checkpoint_dir = get_checkpoints_dir(train_dir)

  if custom_checkpoints is None:
    _, last_it = get_last_checkpoint(checkpoint_dir)
    iterations.add(last_it)
  else:
    if len(custom_checkpoints) == 0:
      iterations = set(get_all_checkpoint_iterations(checkpoint_dir))
    else:
      iterations = custom_checkpoints

  val_dir = get_val_dir(
    train_dir=train_dir,
    ds=ds,
    entry_ids=entry_ids,
    full_run=full_run,
    iterations=iterations,
  )

  val_log_path = os.path.join(val_dir, "log.txt")
  logger = prepare_logger(val_log_path)
  logger.info("Validating...")
  logger.info(f"Checkpoints: {','.join(str(x) for x in sorted(iterations))}")

  result = ValidationEntries()

  for iteration in tqdm(sorted(iterations)):
    logger.info(f"Current checkpoint: {iteration}")
    checkpoint_path = get_checkpoint(checkpoint_dir, iteration)
    taco_checkpoint = CheckpointWaveglow.load(checkpoint_path, logger)
    save_callback = partial(save_results, val_dir=val_dir, iteration=iteration)

    validation_entries = validate(
      checkpoint=taco_checkpoint,
      data=data,
      custom_hparams=custom_hparams,
      entry_ids=entry_ids,
      full_run=full_run,
      train_name=train_name,
      save_callback=save_callback,
      logger=logger,
      sigma=sigma,
      denoiser_strength=denoiser_strength,
    )

    result.extend(validation_entries)

  if len(result) > 0:
    save_stats(val_dir, result)
    logger.info(f"Saved output to: {val_dir}")
