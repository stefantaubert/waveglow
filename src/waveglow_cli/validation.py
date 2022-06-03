import datetime
import os
from functools import partial
from pathlib import Path
from typing import Dict, Optional, Set

import imageio
import numpy as np
from image_utils import stack_images_vertically
from tqdm import tqdm
from tts_preparation import (PreparedData, get_merged_dir, get_prep_dir,
                             load_testset, load_valset)
from tts_preparation.app.prepare import load_totalset
from waveglow import (CheckpointWaveglow, ValidationEntries,
                      ValidationEntryOutput)
from waveglow import validate as validate_core
from waveglow.audio_utils import float_to_wav
from waveglow.utils import (get_all_checkpoint_iterations, get_checkpoint,
                            get_last_checkpoint, prepare_logger)
from waveglow.validation import get_df

from waveglow_cli.defaults import (DEFAULT_DENOISER_STRENGTH, DEFAULT_SEED,
                                   DEFAULT_SIGMA)
from waveglow_cli.io import (_get_validation_root_dir, get_checkpoints_dir,
                             get_train_dir, load_prep_settings)


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


def get_val_dir(train_dir: Path, ds: str, iterations: Set[int], full_run: bool, entry_ids: Optional[Set[int]]) -> Path:
  if len(iterations) > 3:
    its = ",".join(str(x) for x in sorted(iterations)[:3]) + ",..."
  else:
    its = ",".join(str(x) for x in sorted(iterations))

  subdir_name = f"{datetime.datetime.now():%d.%m.%Y__%H-%M-%S}__ds={ds}__entries={get_repr_entries(entry_ids)}__its={its}__full={full_run}"
  return _get_validation_root_dir(train_dir) / subdir_name


def get_val_entry_dir(val_dir: Path, entry: PreparedData, iteration: int) -> Path:
  return val_dir / f"it={iteration}_id={entry.entry_id}"


def save_stats(val_dir: Path, validation_entries: ValidationEntries) -> None:
  path = val_dir / "total.csv"
  df = get_df(validation_entries)
  df.to_csv(path, sep="\t", header=True, index=False)


def save_results(entry: PreparedData, output: ValidationEntryOutput, val_dir: Path, iteration: int) -> None:
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


def validate_generic(base_dir: Path, ttsp_dir: Path, merge_name: str, prep_name: str, train_name: str, ds: str = "val", entry_ids: Optional[Set[int]] = None, custom_checkpoints: Optional[Set[int]] = None, sigma: float = DEFAULT_SIGMA, denoiser_strength: float = DEFAULT_DENOISER_STRENGTH, custom_hparams: Optional[Dict[str, str]] = None, full_run: bool = False, seed: int = DEFAULT_SEED) -> None:
  train_dir = get_train_dir(base_dir, train_name)
  assert train_dir.is_dir()

  merge_dir = get_merged_dir(ttsp_dir, merge_name)
  prep_dir = get_prep_dir(merge_dir, prep_name)

  _validate(
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
    seed=seed,
  )


def validate(base_dir: Path, train_name: str, entry_ids: Optional[Set[int]] = None, ds: str = "val", custom_checkpoints: Optional[Set[int]] = None, sigma: float = DEFAULT_SIGMA, denoiser_strength: float = DEFAULT_DENOISER_STRENGTH, custom_hparams: Optional[Dict[str, str]] = None, full_run: bool = False, seed: int = DEFAULT_SEED) -> None:
  train_dir = get_train_dir(base_dir, train_name)
  assert train_dir.is_dir()

  ttsp_dir, merge_name, prep_name = load_prep_settings(train_dir)
  merge_dir = get_merged_dir(ttsp_dir, merge_name)
  prep_dir = get_prep_dir(merge_dir, prep_name)

  _validate(
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
    seed=seed,
  )


def _validate(train_dir, train_name: str, prep_dir: Path, entry_ids: Optional[Set[int]], custom_checkpoints: Optional[Set[int]], sigma: float, denoiser_strength: float, custom_hparams: Optional[Dict[str, str]], full_run: bool, ds: str, seed: int) -> None:
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

  val_dir.mkdir(parents=True, exist_ok=True)
  val_log_path = val_dir / "log.txt"
  logger = prepare_logger(val_log_path)
  logger.info("Validating...")
  logger.info(f"Checkpoints: {','.join(str(x) for x in sorted(iterations))}")

  result = ValidationEntries()

  for iteration in tqdm(sorted(iterations)):
    logger.info(f"Current checkpoint: {iteration}")
    checkpoint_path = get_checkpoint(checkpoint_dir, iteration)
    taco_checkpoint = CheckpointWaveglow.load(checkpoint_path, logger)
    save_callback = partial(save_results, val_dir=val_dir, iteration=iteration)

    validation_entries = validate_core(
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
      seed=seed,
    )

    result.extend(validation_entries)

  if len(result) > 0:
    save_stats(val_dir, result)
    logger.info(f"Saved output to: {val_dir}")
