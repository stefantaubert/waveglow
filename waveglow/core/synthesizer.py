# For copyright see LICENCE

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from audio_utils import concatenate_audios, is_overamp, normalize_wav
from tqdm import tqdm
from waveglow.core.denoiser import Denoiser
from waveglow.core.model_checkpoint import CheckpointWaveglow
from waveglow.core.train import load_model
from waveglow.utils import overwrite_custom_hparams


@dataclass
class InferenceResult():
  mel: np.ndarray
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

  def _concatenate_wavs(self, result: List[InferenceResult], sentence_pause_s: float):
    wavs = [res.wav for res in result]
    if len(wavs) > 1:
      self._logger.info("Concatening audios...")
    output = concatenate_audios(wavs, sentence_pause_s, self.hparams.sampling_rate)

    return output

  def infer(self, mel: torch.FloatTensor, sigma: float, denoiser_strength: float) -> InferenceResult:
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

    was_overamplified = False

    if is_overamp(audio_np):
      was_overamplified = True
      self._logger.warn("Waveglow output was overamplified.")

    res = InferenceResult(
      mel=mel,
      sampling_rate=self.hparams.sampling_rate,
      inference_duration_s=inference_duration_s,
      wav=audio_np,
      was_overamplified=was_overamplified,
      wav_denoised=audio_denoised,
      denoising_duration_s=denoising_duration,
    )

    return res

  def infer_all(self, mels: List[torch.FloatTensor], sigma: float, denoiser_strength: float, sentence_pause_s: float) -> Tuple[np.ndarray, List[InferenceResult]]:
    result: List[InferenceResult] = []

    # Speed is: 1min inference for 3min wav result
    for mel in tqdm(mels):
      infer_res = self.infer(mel, sigma, denoiser_strength)
      result.append(infer_res)

    output = self._concatenate_wavs(result, sentence_pause_s)
    output = normalize_wav(output)

    for infer_res in result:
      infer_res.wav = normalize_wav(infer_res.wav)

    return output, result
