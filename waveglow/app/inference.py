import datetime
import os
from shutil import copyfile
from typing import Dict, Optional

import matplotlib.pylab as plt
import numpy as np
from audio_utils.mel import plot_melspec
from waveglow.app.io import (get_checkpoints_dir, get_infer_log,
                             get_inference_root_dir, get_train_dir,
                             save_diff_plot, save_infer_plot, save_infer_wav,
                             save_v)
from waveglow.core.inference import infer
from waveglow.core.model_checkpoint import CheckpointWaveglow
from waveglow.utils import (get_basename, get_custom_or_last_checkpoint,
                            get_parent_dirname, get_subdir, prepare_logger)


def get_infer_dir(train_dir: str, wav_path: str, iteration: int):
  input_name = get_basename(wav_path)
  subdir_name = f"{datetime.datetime.now():%Y-%m-%d,%H-%M-%S},wav={input_name},it={iteration}"
  return get_subdir(get_inference_root_dir(train_dir), subdir_name, create=True)


def save_infer_orig_plot(infer_dir: str, mel: np.ndarray):
  plot_melspec(mel, title="Original")
  path = os.path.join(infer_dir, f"{get_parent_dirname(infer_dir)}_orig.png")
  plt.savefig(path, bbox_inches='tight')
  plt.close()
  return path


def save_infer_orig_wav(infer_dir: str, wav_path_orig: str):
  path = os.path.join(infer_dir, f"{get_parent_dirname(infer_dir)}_orig.wav")
  copyfile(wav_path_orig, path)


def app_infer(base_dir: str, train_name: str, wav_path: str, custom_checkpoint: Optional[int] = None, sigma: float = 0.666, denoiser_strength: float = 0.00, custom_hparams: Optional[Dict[str, str]] = None):
  train_dir = get_train_dir(base_dir, train_name, create=False)
  assert os.path.isdir(train_dir)

  checkpoint_path, iteration = get_custom_or_last_checkpoint(
    get_checkpoints_dir(train_dir), custom_checkpoint)
  infer_dir = get_infer_dir(train_dir, wav_path, iteration)

  logger = prepare_logger(get_infer_log(infer_dir))
  logger.info(f"Inferring {wav_path}...")

  checkpoint = CheckpointWaveglow.load(checkpoint_path, logger)

  wav, wav_sr, wav_mel, orig_mel = infer(
    wav_path=wav_path,
    denoiser_strength=denoiser_strength,
    sigma=sigma,
    checkpoint=checkpoint,
    custom_hparams=custom_hparams,
    logger=logger
  )

  save_infer_wav(infer_dir, wav_sr, wav)
  save_infer_plot(infer_dir, wav_mel)
  save_infer_orig_wav(infer_dir, wav_path)
  save_infer_orig_plot(infer_dir, orig_mel)
  score = save_diff_plot(infer_dir)
  save_v(infer_dir)

  logger.info(f"Imagescore: {score*100}%")
  logger.info(f"Saved output to: {infer_dir}")
