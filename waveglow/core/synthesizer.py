# For copyright see LICENCE

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from audio_utils import is_overamp
from tqdm import tqdm
from waveglow.core.denoiser import Denoiser
from waveglow.core.model_checkpoint import CheckpointWaveglow
from waveglow.core.train import load_model
from waveglow.utils import init_global_seeds, overwrite_custom_hparams


@dataclass
class InferenceResult():
  wav: np.ndarray
  wav_denoised: np.ndarray
  sampling_rate: int
  inference_duration_s: float
  denoising_duration_s: float
  was_overamplified: bool


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

  def infer(self, mel: torch.FloatTensor, sigma: float, denoiser_strength: float, seed: int) -> InferenceResult:
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
      self._logger.warn("Waveglow output was overamplified.")

    res = InferenceResult(
      sampling_rate=self.hparams.sampling_rate,
      inference_duration_s=inference_duration_s,
      wav=audio_np,
      was_overamplified=was_overamplified,
      wav_denoised=audio_denoised_np,
      denoising_duration_s=denoising_duration,
    )

    return res

  def infer_all(self, mels: List[torch.FloatTensor], sigma: float, denoiser_strength: float, seed: int) -> List[InferenceResult]:
    result: List[InferenceResult] = []

    for mel in tqdm(mels):
      infer_res = self.infer(mel, sigma, denoiser_strength, seed)
      result.append(infer_res)

    return result
