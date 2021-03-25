from dataclasses import dataclass
from logging import Logger
from typing import Dict, Optional

import numpy as np
import torch
from audio_utils import normalize_wav
from audio_utils.mel import TacotronSTFT
from waveglow.core.model_checkpoint import CheckpointWaveglow
from waveglow.core.synthesizer import Synthesizer
from waveglow.utils import GenericList, cosine_dist_mels


@dataclass
class InferenceEntry():
  entry_id: int = None
  original_duration_s: float = None
  inferred_duration_s: float = None
  diff_duration_s: float = None
  iteration: int = None
  inference_duration_s: float = None
  timepoint: str = None
  train_name: str = None
  sampling_rate: int = None
  mcd_dtw: float = None
  mcd_dtw_frames: int = None
  structural_similarity: float = None
  cosine_similarity: float = None


class InferenceEntries(GenericList[InferenceEntry]):
  pass


@dataclass
class InferenceEntryOutput():
  sampling_rate: int = None
  mel_img: np.ndarray = None
  postnet_img: np.ndarray = None
  postnet_mel: np.ndarray = None
  alignments_img: np.ndarray = None
  # gate_out_img: np.ndarray = None


def infer(wav_path: str, checkpoint: CheckpointWaveglow, custom_hparams: Optional[Dict[str, str]], denoiser_strength: float, sigma: float, logger: Logger):
  synth = Synthesizer(
    checkpoint=checkpoint,
    custom_hparams=custom_hparams,
    logger=logger
  )

  taco_stft = TacotronSTFT(synth.hparams, logger=logger)

  mel = taco_stft.get_mel_tensor_from_file(wav_path)
  mel_var = torch.autograd.Variable(mel)
  mel_var = mel_var.cuda()
  mel_var = mel_var.unsqueeze(0)

  audio, _ = synth.infer(mel_var, sigma, denoiser_strength)
  audio = normalize_wav(audio)

  audio_tensor = torch.FloatTensor(audio)
  mel_pred = taco_stft.get_mel_tensor(audio_tensor)
  orig_np = mel.cpu().numpy()
  pred_np = mel_pred.numpy()

  score = cosine_dist_mels(orig_np, pred_np)
  logger.info(f"Cosine similarity is: {score*100}%")

  #score, diff_img = compare_mels(a, b)
  return audio, synth.hparams.sampling_rate, pred_np, orig_np
