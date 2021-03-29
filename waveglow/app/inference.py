import datetime
import os
from functools import partial
from typing import Dict, List, Optional

import imageio
import numpy as np
from audio_utils import float_to_wav
from image_utils import stack_images_vertically
from image_utils.main import stack_images_horizontally
from waveglow.app.defaults import DEFAULT_SENTENCE_PAUSE_S
from waveglow.app.io import (get_checkpoints_dir, get_infer_log,
                             get_inference_root_dir, get_train_dir)
from waveglow.core import (CheckpointWaveglow, InferenceEntries,
                           InferenceEntryOutput, infer as infer_core)
from waveglow.utils import (get_custom_or_last_checkpoint, get_subdir,
                            prepare_logger)


def get_infer_dir(train_dir: str, mel_paths: List[str], iteration: int):
  #input_name = get_basename(wav_path)
  subdir_name = f"{datetime.datetime.now():%Y-%m-%d,%H-%M-%S},mels={len(mel_paths)},it={iteration}"
  return get_subdir(get_inference_root_dir(train_dir), subdir_name, create=True)


def get_inferred_mel_dir(infer_dir: int, nr: int):
  dest_dir = os.path.join(infer_dir, f"{nr}")
  return dest_dir


def save_results(output: InferenceEntryOutput, infer_dir: str):
  dest_dir = get_inferred_mel_dir(infer_dir, output.nr)
  os.makedirs(dest_dir, exist_ok=True)
  imageio.imsave(os.path.join(dest_dir, "original.png"), output.mel_orig_img)
  imageio.imsave(os.path.join(dest_dir, "inferred_denoised.png"), output.mel_inferred_denoised_img)
  imageio.imsave(os.path.join(dest_dir, "diff.png"), output.mel_denoised_diff_img)
  np.save(os.path.join(dest_dir, "original.mel.npy"), output.mel_orig)
  np.save(os.path.join(dest_dir, "inferred_denoised.mel.npy"), output.mel_inferred_denoised)

  float_to_wav(output.wav_inferred_denoised, os.path.join(
    dest_dir, "inferred_denoised.wav"), sample_rate=output.inferred_sr)

  float_to_wav(output.wav_inferred, os.path.join(
    dest_dir, "inferred.wav"), sample_rate=output.inferred_sr)

  stack_images_vertically(
    list_im=[
      os.path.join(dest_dir, "original.png"),
      os.path.join(dest_dir, "inferred_denoised.png"),
      os.path.join(dest_dir, "diff.png"),
    ],
    out_path=os.path.join(dest_dir, "comparison.png")
  )


def save_stats(infer_dir: str, stats: InferenceEntries) -> None:
  path = os.path.join(infer_dir, "total.csv")
  stats.save(path, header=True)


def mel_inferred_denoised_v_plot(infer_dir: str, sentences: InferenceEntries):
  paths = [os.path.join(get_inferred_mel_dir(infer_dir, x.nr), "inferred_denoised.png")
           for x in sentences.items()]
  path = os.path.join(infer_dir, "inferred_denoised_v.png")
  stack_images_vertically(paths, path)


def mel_inferred_denoised_h_plot(infer_dir: str, sentences: InferenceEntries):
  paths = [os.path.join(get_inferred_mel_dir(infer_dir, x.nr), "inferred_denoised.png")
           for x in sentences.items()]
  path = os.path.join(infer_dir, "inferred_denoised_h.png")
  stack_images_horizontally(paths, path)


def infer(base_dir: str, train_name: str, mel_paths: List[str], sampling_rate: int, custom_checkpoint: Optional[int] = None, sigma: float = 0.666, denoiser_strength: float = 0.00, sentence_pause_s: float = DEFAULT_SENTENCE_PAUSE_S, custom_hparams: Optional[Dict[str, str]] = None):
  train_dir = get_train_dir(base_dir, train_name, create=False)
  assert os.path.isdir(train_dir)

  checkpoint_path, iteration = get_custom_or_last_checkpoint(
    get_checkpoints_dir(train_dir), custom_checkpoint)

  infer_dir = get_infer_dir(train_dir, mel_paths, iteration)

  logger = prepare_logger(get_infer_log(infer_dir))
  logger.info(f"Inferring...")

  checkpoint = CheckpointWaveglow.load(checkpoint_path, logger)
  mels = [np.load(mel_path)for mel_path in mel_paths]
  save_callback = partial(save_results, infer_dir=infer_dir)

  complete_wav_denoised, inference_results = infer_core(
    mels=mels,
    sampling_rate=sampling_rate,
    checkpoint=checkpoint,
    custom_hparams=custom_hparams,
    denoiser_strength=denoiser_strength,
    sigma=sigma,
    sentence_pause_s=sentence_pause_s,
    logger=logger,
    save_callback=save_callback,
  )

  float_to_wav(complete_wav_denoised, os.path.join(
    # TODO: the sampling_rate variable is here not correct
    infer_dir, "complete_denoised.wav"), sample_rate=sampling_rate)

  logger.info("Creating mel_inferred_denoised_v.png")
  mel_inferred_denoised_v_plot(infer_dir, inference_results)

  logger.info("Creating mel_inferred_denoised_h.png")
  mel_inferred_denoised_h_plot(infer_dir, inference_results)

  logger.info("Creating total.csv")
  save_stats(infer_dir, inference_results)

  logger.info(f"Saved output to: {infer_dir}")

  # save_infer_wav(infer_dir, wav_sr, wav)
  # save_infer_plot(infer_dir, wav_mel)
  # save_infer_orig_wav(infer_dir, wav_path)
  # save_infer_orig_plot(infer_dir, orig_mel)
  # score = save_diff_plot(infer_dir)
  # save_v(infer_dir)

  # logger.info(f"Imagescore: {score*100}%")
  # logger.info(f"Saved output to: {infer_dir}")
