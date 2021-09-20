import datetime
import os
from functools import partial
from logging import getLogger
from pathlib import Path
from shutil import copyfile
from typing import Any, Dict, List, Optional

import imageio
import numpy as np
import regex as re
from audio_utils import float_to_wav
from general_utils import parse_json, pass_lines_list, save_json
from image_utils import stack_images_vertically
from image_utils.main import stack_images_horizontally
from waveglow.app.defaults import (DEFAULT_DENOISER_STRENGTH,
                                   DEFAULT_READ_MEL_INFO_PATH,
                                   DEFAULT_SAVE_WAV_INFO_COPY_PATH,
                                   DEFAULT_SEED, DEFAULT_SENTENCE_PAUSE_S,
                                   DEFAULT_SIGMA)
from waveglow.app.io import (get_checkpoints_dir, get_inference_root_dir,
                             get_train_dir, get_wav_info_dict,
                             get_wav_out_dict)
from waveglow.core import (CheckpointWaveglow, InferenceEntries,
                           InferenceEntryOutput)
from waveglow.core import infer as infer_core
from waveglow.core.inference import InferMelEntry
from waveglow.utils import get_custom_or_last_checkpoint, prepare_logger

OUTPUT_INFO_FILE_NAME = "wav_out.json"


def get_infer_dir(train_dir: Path, run_name: str) -> Path:
  #input_name = get_basename(wav_path)
  return get_inference_root_dir(train_dir) / run_name


def get_inferred_mel_dir(infer_dir: int, nr: int) -> Path:
  dest_dir = infer_dir / f"{nr}"
  return dest_dir


def save_results(output: InferenceEntryOutput, infer_dir: str, denoised_audio_wav_paths: List[Dict[str, Any]]):
  dest_dir = get_inferred_mel_dir(infer_dir, output.identifier)
  os.makedirs(dest_dir, exist_ok=True)
  imageio.imsave(dest_dir / "original.png", output.mel_orig_img)
  imageio.imsave(dest_dir / "inferred_denoised.png", output.mel_inferred_denoised_img)
  imageio.imsave(dest_dir / "diff.png", output.mel_denoised_diff_img)
  np.save(dest_dir / "original.mel.npy", output.mel_orig)
  np.save(dest_dir / "inferred_denoised.mel.npy", output.mel_inferred_denoised)

  inferred_denoised_path = dest_dir / "inferred_denoised.wav"
  float_to_wav(output.wav_inferred_denoised, inferred_denoised_path, sample_rate=output.inferred_sr)

  pat = re.compile("id=([0-9]*)_")
  entry_id = re.findall(pat, output.identifier)
  if len(entry_id) == 1:
    inferred_denoised_path_copy = infer_dir / f"{entry_id[0]}.wav"
    copyfile(inferred_denoised_path, inferred_denoised_path_copy)

  float_to_wav(output.wav_inferred, os.path.join(
    dest_dir, "inferred.wav"), sample_rate=output.inferred_sr)

  stack_images_vertically(
    list_im=[
      dest_dir / "original.png",
      dest_dir / "inferred_denoised.png",
      dest_dir / "diff.png",
    ],
    out_path=dest_dir / "comparison.png"
  )

  wav_info = get_wav_info_dict(
    identifier=output.identifier,
    path=inferred_denoised_path,
    sr=output.inferred_sr,
  )

  denoised_audio_wav_paths.append(wav_info)


def save_stats(infer_dir: str, stats: InferenceEntries) -> None:
  path = infer_dir / "total.csv"
  stats.save(path, header=True)


def mel_inferred_denoised_v_plot(infer_dir: str, sentences: InferenceEntries):
  paths = [get_inferred_mel_dir(infer_dir, x.identifier) / "inferred_denoised.png"
           for x in sentences.items()]
  path = infer_dir / "inferred_denoised_v.png"
  stack_images_vertically(paths, path)


def mel_inferred_denoised_h_plot(infer_dir: str, sentences: InferenceEntries):
  paths = [get_inferred_mel_dir(infer_dir, x.identifier) / "inferred_denoised.png"
           for x in sentences.items()]
  path = infer_dir / "inferred_denoised_h.png"
  stack_images_horizontally(paths, path)


def infer_parse_json(base_dir: str, train_name: str, json_path: str = DEFAULT_READ_MEL_INFO_PATH, custom_checkpoint: Optional[int] = None, sigma: float = DEFAULT_SIGMA, denoiser_strength: float = DEFAULT_DENOISER_STRENGTH, sentence_pause_s: Optional[float] = DEFAULT_SENTENCE_PAUSE_S, custom_hparams: Optional[Dict[str, str]] = None, no_concatenation: bool = False, seed: int = DEFAULT_SEED, copy_wav_info_to: Optional[str] = DEFAULT_SAVE_WAV_INFO_COPY_PATH):
  logger = getLogger(__name__)
  if not json_path.is_file():
    logger.info("Json file not found.")
    return

  json_content = parse_json(json_path)
  if len(json_content) == 0:
    logger.info("No mels found in this file.")
    return

  logger.info("Inferring these mels:")
  pass_lines_list(logger.info, [x["path"] for x in json_content["mels"]])

  name = json_content["name"]

  train_dir = get_train_dir(base_dir, train_name, create=False)
  assert train_dir.is_dir()

  checkpoint_path, iteration = get_custom_or_last_checkpoint(
    get_checkpoints_dir(train_dir), custom_checkpoint)

  mel_entries = []
  for mel_data in json_content["mels"]:
    mel_entry = InferMelEntry(
      identifier=mel_data["id"],
      mel=np.load(mel_data["path"]),
      sr=mel_data["sr"],
    )
    mel_entries.append(mel_entry)

  run_name = f"{datetime.datetime.now():%Y-%m-%d,%H-%M-%S}__mels={len(mel_entries)}__it={iteration}____{name}"

  infer_dir = get_infer_dir(train_dir, run_name)

  _infer(
    infer_dir=infer_dir,
    run_name=run_name,
    mel_entries=mel_entries,
    checkpoint_path=checkpoint_path,
    custom_hparams=custom_hparams,
    denoiser_strength=denoiser_strength,
    no_concatenation=no_concatenation,
    sentence_pause_s=sentence_pause_s,
    sigma=sigma,
    seed=seed,
    copy_wav_info_to=copy_wav_info_to,
  )


def infer(base_dir: str, train_name: str, mel_paths: List[str], sampling_rate: int, custom_checkpoint: Optional[int] = None, sigma: float = DEFAULT_SIGMA, denoiser_strength: float = DEFAULT_DENOISER_STRENGTH, sentence_pause_s: Optional[float] = DEFAULT_SENTENCE_PAUSE_S, custom_hparams: Optional[Dict[str, str]] = None, no_concatenation: bool = False, seed: int = DEFAULT_SEED, copy_wav_info_to: Optional[str] = DEFAULT_SAVE_WAV_INFO_COPY_PATH):
  train_dir = get_train_dir(base_dir, train_name, create=False)
  assert train_dir.is_dir()

  checkpoint_path, iteration = get_custom_or_last_checkpoint(
    get_checkpoints_dir(train_dir), custom_checkpoint)

  run_name = f"{datetime.datetime.now():%Y-%m-%d,%H-%M-%S}__mels={len(mel_paths)}__it={iteration}"

  infer_dir = get_infer_dir(train_dir, run_name)

  mel_entries = []
  for identifier, mel_path in zip(range(1, len(mel_paths) + 1), mel_paths):
    mel_entry = InferMelEntry(
      identifier=identifier,
      mel=np.load(mel_path),
      sr=sampling_rate,
    )
    mel_entries.append(mel_entry)

  _infer(
    infer_dir=infer_dir,
    run_name=run_name,
    mel_entries=mel_entries,
    checkpoint_path=checkpoint_path,
    custom_hparams=custom_hparams,
    denoiser_strength=denoiser_strength,
    no_concatenation=no_concatenation,
    sentence_pause_s=sentence_pause_s,
    sigma=sigma,
    seed=seed,
    copy_wav_info_to=copy_wav_info_to,
  )


def _infer(infer_dir: str, run_name: str, checkpoint_path: str, mel_entries: List[InferMelEntry], sigma: float, denoiser_strength: float, sentence_pause_s: Optional[float], custom_hparams: Optional[Dict[str, str]], no_concatenation: bool, seed: int, copy_wav_info_to: Optional[str]):
  logger = prepare_logger(infer_dir / "log.txt")

  checkpoint = CheckpointWaveglow.load(checkpoint_path, logger)
  concatenate = not no_concatenation

  denoised_audio_wav_paths: List[Dict[str, Any]] = []
  save_callback = partial(
    save_results,
    infer_dir=infer_dir,
    denoised_audio_wav_paths=denoised_audio_wav_paths
  )

  inference_results, complete = infer_core(
    mel_entries=mel_entries,
    checkpoint=checkpoint,
    custom_hparams=custom_hparams,
    denoiser_strength=denoiser_strength,
    sigma=sigma,
    sentence_pause_s=sentence_pause_s,
    logger=logger,
    save_callback=save_callback,
    concatenate=concatenate,
    seed=seed,
  )

  if concatenate:
    complete_wav_denoised, complete_wav_denoised_sr = complete
    assert complete_wav_denoised is not None
    assert complete_wav_denoised_sr is not None
    float_to_wav(complete_wav_denoised, os.path.join(
      infer_dir, "complete_denoised.wav"), sample_rate=complete_wav_denoised_sr)

  logger.info("Creating mel_inferred_denoised_v.png")
  mel_inferred_denoised_v_plot(infer_dir, inference_results)

  logger.info("Creating mel_inferred_denoised_h.png")
  mel_inferred_denoised_h_plot(infer_dir, inference_results)

  logger.info("Creating total.csv")
  save_stats(infer_dir, inference_results)

  wav_paths_json = save_denoised_audio_wav_paths(
    infer_dir=infer_dir,
    name=run_name,
    denoised_audio_wav_paths=denoised_audio_wav_paths,
  )

  logger.info("Wrote all inferred mel paths including sampling rate into these file(s):")
  logger.info(wav_paths_json)

  if copy_wav_info_to is not None:
    create_parent_folder(copy_wav_info_to)
    copyfile(wav_paths_json, copy_wav_info_to)
    logger.info(copy_wav_info_to)

  logger.info(f"Saved output to: {infer_dir}")

  # save_infer_wav(infer_dir, wav_sr, wav)
  # save_infer_plot(infer_dir, wav_mel)
  # save_infer_orig_wav(infer_dir, wav_path)
  # save_infer_orig_plot(infer_dir, orig_mel)
  # score = save_diff_plot(infer_dir)
  # save_v(infer_dir)

  # logger.info(f"Imagescore: {score*100}%")
  # logger.info(f"Saved output to: {infer_dir}")


def save_denoised_audio_wav_paths(infer_dir: str, name: str, denoised_audio_wav_paths: List[Dict[str, Any]]) -> str:
  info_json = get_wav_out_dict(
    name=name,
    root_dir=infer_dir,
    wav_info_dict=denoised_audio_wav_paths,
  )

  path = infer_dir / OUTPUT_INFO_FILE_NAME
  save_json(path, info_json)
  #text = '\n'.join(mel_postnet_npy_paths)
  #save_txt(path, text)
  return path
