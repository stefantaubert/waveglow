# For copyright see LICENCE

import logging
import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from audio_utils import is_overamp
from src.core.common.train import overwrite_custom_hparams
from src.core.waveglow.denoiser import Denoiser
from src.core.waveglow.train import CheckpointWaveglow, load_model


class Synthesizer():
  def __init__(self, checkpoint: CheckpointWaveglow, custom_hparams: Optional[Dict[str, str]], logger: logging.Logger):
    super().__init__()
    self._logger = logger

    hparams = checkpoint.get_hparams(logger)
    hparams = overwrite_custom_hparams(hparams, custom_hparams)

    model = load_model(hparams, checkpoint.state_dict)
    model = model.remove_weightnorm(model)
    model = model.eval()

    denoiser = Denoiser(
      waveglow=model,
      hparams=hparams,
      mode="zeros",
      logger=logger,
    ).cuda()

    self.hparams = hparams
    self.model = model
    self.denoiser = denoiser

  def infer(self, mel, sigma: float, denoiser_strength: float) -> Tuple[np.ndarray, float]:
    start = time.perf_counter()
    with torch.no_grad():
      audio = self.model.infer(mel, sigma=sigma)
      end = time.perf_counter()
      if denoiser_strength > 0:
        audio = self.denoiser(audio, strength=denoiser_strength)
    inference_duration_s = end - start
    audio = audio.squeeze()
    audio = audio.cpu()
    audio_np: np.ndarray = audio.numpy()

    if is_overamp(audio_np):
      self._logger.warn("Waveglow output was overamplified.")

    return audio_np, inference_duration_s
