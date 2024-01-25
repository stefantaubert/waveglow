# For copyright see LICENCE

import datetime
import time
from dataclasses import dataclass
from logging import getLogger
from typing import Dict, Optional

import numpy as np
import torch

from waveglow.audio_utils import is_overamp
from waveglow.denoiser import Denoiser
from waveglow.model_checkpoint import CheckpointWaveglow
from waveglow.train import load_model
from waveglow.utils import (get_default_device, init_global_seeds, overwrite_custom_hparams,
                            try_copy_to)


@dataclass
class InferenceResult():
  wav: np.ndarray
  wav_denoised: np.ndarray
  sampling_rate: int
  inference_duration_s: float
  denoising_duration_s: float
  was_overamplified: bool
  timepoint: datetime.datetime


class Synthesizer():
  def __init__(self, checkpoint: CheckpointWaveglow, *, custom_hparams: Optional[Dict[str, str]] = None, device: torch.device = get_default_device()):
    hparams = checkpoint.get_hparams()
    hparams = overwrite_custom_hparams(hparams, custom_hparams)

    model = load_model(hparams, checkpoint.state_dict, device)
    model = model.remove_weightnorm(model)
    model = model.eval()

    denoiser = Denoiser(
      waveglow=model,
      hparams=hparams,
      mode="zeros",
      device=device,
    )

    denoiser = try_copy_to(denoiser, device)

    self.device = device
    self.hparams = hparams
    self.model = model
    self.denoiser = denoiser

  def infer(self, mel: torch.FloatTensor, *, sigma: float = 1.0, denoiser_strength: float = 0.0005, seed: int = 0) -> InferenceResult:
    timepoint = datetime.datetime.now()
    init_global_seeds(seed)
    denoising_duration = 0
    start = time.perf_counter()
    with torch.no_grad():
      audio = self.model.infer(mel, sigma=sigma)
      end = time.perf_counter()
      audio_denoised = audio
      if denoiser_strength > 0:
        start_denoising = time.perf_counter()
        audio_denoised = self.denoiser(audio, strength=denoiser_strength)
        end_denoising = time.perf_counter()
        denoising_duration = end_denoising - start_denoising
    inference_duration_s = end - start
    audio = audio.squeeze()
    audio = audio.cpu()
    audio_np: np.ndarray = audio.numpy()

    audio_denoised = audio_denoised.squeeze()
    audio_denoised = audio_denoised.cpu()
    audio_denoised_np: np.ndarray = audio_denoised.numpy()

    was_overamplified = False

    if is_overamp(audio_np):
      was_overamplified = True
      logger = getLogger(__name__)
      logger.debug("Waveglow output was overamplified.")

    res = InferenceResult(
      sampling_rate=self.hparams.sampling_rate,
      inference_duration_s=inference_duration_s,
      wav=audio_np,
      was_overamplified=was_overamplified,
      wav_denoised=audio_denoised_np,
      denoising_duration_s=denoising_duration,
      timepoint=timepoint,
    )

    return res
