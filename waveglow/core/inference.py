import datetime
from dataclasses import dataclass
from logging import Logger
from typing import Callable, Dict, List, Optional, Tuple

import imageio
import numpy as np
import torch
from audio_utils import get_duration_s, normalize_wav
from audio_utils.mel import TacotronSTFT, plot_melspec_np
from image_utils import (calculate_structual_similarity_np,
                         make_same_width_by_filling_white)
from mcd import get_mcd_between_mel_spectograms
from tqdm import tqdm
from waveglow.core.model_checkpoint import CheckpointWaveglow
from waveglow.core.synthesizer import InferenceResult, Synthesizer
from waveglow.globals import MCD_NO_OF_COEFFS_PER_FRAME
from waveglow.utils import GenericList, cosine_dist_mels


@dataclass
class InferenceEntry():
  nr: int = None
  input_path: str = None
  denoising_duration_s: float = None
  was_overamplified: bool = None
  inferred_duration_s: float = None
  iteration: int = None
  inference_duration_s: float = None
  timepoint: str = None
  sampling_rate: int = None
  mcd_dtw: float = None
  mcd_dtw_penalty: int = None
  mcd_dtw_frames: int = None
  mcd: float = None
  mcd_penalty: int = None
  mcd_frames: int = None
  structural_similarity: float = None
  cosine_similarity: float = None
  denoiser_strength: float = None
  sigma: float = None


class InferenceEntries(GenericList[InferenceEntry]):
  pass


@dataclass
class InferenceEntryOutput():
  nr: int = None
  mel_orig: np.ndarray = None
  mel_orig_img: np.ndarray = None
  orig_sr: int = None
  inferred_sr: int = None
  mel_inferred_denoised: np.ndarray = None
  mel_inferred_denoised_img: np.ndarray = None
  wav_inferred_denoised: np.ndarray = None
  mel_denoised_diff_img: np.ndarray = None
  wav_inferred: np.ndarray = None


def mel_to_torch(mel: np.ndarray) -> np.ndarray:
  res = torch.FloatTensor(mel)
  res = res.cuda()
  return res


def infer(mels: List[np.ndarray], sampling_rate: int, checkpoint: CheckpointWaveglow, custom_hparams: Optional[Dict[str, str]], denoiser_strength: float, sigma: float, sentence_pause_s: float, save_callback: Callable[[InferenceEntryOutput], None], logger: Logger) -> Tuple[np.ndarray, InferenceEntries]:
  inference_entries = InferenceEntries()

  if len(mels) == 0:
    logger.info("Nothing to synthesize!")
    return inference_entries

  synth = Synthesizer(
    checkpoint=checkpoint,
    custom_hparams=custom_hparams,
    logger=logger
  )

  taco_stft = TacotronSTFT(synth.hparams, logger=logger)
  mels_torch = []
  mels_torch_prepared = []
  for mel in mels:
    mel = mel_to_torch(mel)
    mels_torch.append(mel)
    mel_var = torch.autograd.Variable(mel)
    mel_var = mel_var.cuda()
    mel_var = mel_var.unsqueeze(0)
    mels_torch_prepared.append(mel_var)

  complete_wav_denoised, inference_results = synth.infer_all(
    mels_torch_prepared, sigma, denoiser_strength, sentence_pause_s)
  complete_wav_denoised = normalize_wav(complete_wav_denoised)

  inference_result: InferenceResult
  for nr, mel_torch, inference_result in tqdm(zip(range(len(mels_torch)), mels_torch, inference_results)):
    wav_inferred_denoised = normalize_wav(inference_result.wav_denoised)
    timepoint = f"{datetime.datetime.now():%Y/%m/%d %H:%M:%S}"

    val_entry = InferenceEntry(
      nr=nr,
      iteration=checkpoint.iteration,
      timepoint=timepoint,
      sampling_rate=inference_result.sampling_rate,
      inference_duration_s=inference_result.inference_duration_s,
      was_overamplified=inference_result.was_overamplified,
      denoising_duration_s=inference_result.denoising_duration_s,
      inferred_duration_s=get_duration_s(
        inference_result.wav_denoised, inference_result.sampling_rate),
      denoiser_strength=denoiser_strength,
      sigma=sigma,
    )

    mel_orig = mel_torch.cpu().numpy()

    mel_inferred_denoised_tensor = torch.FloatTensor(inference_result.wav_denoised)
    mel_inferred_denoised = taco_stft.get_mel_tensor(mel_inferred_denoised_tensor)
    mel_inferred_denoised = mel_inferred_denoised.numpy()

    validation_entry_output = InferenceEntryOutput(
      nr=nr,
      mel_orig=mel_orig,
      inferred_sr=inference_result.sampling_rate,
      mel_inferred_denoised=mel_inferred_denoised,
      wav_inferred_denoised=wav_inferred_denoised,
      orig_sr=sampling_rate,
      wav_inferred=normalize_wav(inference_result.wav),
      mel_denoised_diff_img=None,
      mel_inferred_denoised_img=None,
      mel_orig_img=None,
    )

    mcd_dtw, penalty_dtw, final_frame_number_dtw = get_mcd_between_mel_spectograms(
      mel_1=mel_orig,
      mel_2=mel_inferred_denoised,
      n_mfcc=MCD_NO_OF_COEFFS_PER_FRAME,
      take_log=False,
      use_dtw=True,
    )

    val_entry.mcd_dtw = mcd_dtw
    val_entry.mcd_dtw_penalty = penalty_dtw
    val_entry.mcd_dtw_frames = final_frame_number_dtw

    mcd, penalty, final_frame_number = get_mcd_between_mel_spectograms(
      mel_1=mel_orig,
      mel_2=mel_inferred_denoised,
      n_mfcc=MCD_NO_OF_COEFFS_PER_FRAME,
      take_log=False,
      use_dtw=False,
    )

    val_entry.mcd = mcd
    val_entry.mcd_penalty = penalty
    val_entry.mcd_frames = final_frame_number

    cosine_similarity = cosine_dist_mels(mel_orig, mel_inferred_denoised)
    val_entry.cosine_similarity = cosine_similarity

    mel_original_img_raw, mel_original_img = plot_melspec_np(mel_orig)
    mel_inferred_denoised_img_raw, mel_inferred_denoised_img = plot_melspec_np(
      mel_inferred_denoised)

    validation_entry_output.mel_orig_img = mel_original_img
    validation_entry_output.mel_inferred_denoised_img = mel_inferred_denoised_img

    mel_original_img_raw_same_dim, mel_inferred_denoised_img_raw_same_dim = make_same_width_by_filling_white(
      img_a=mel_original_img_raw,
      img_b=mel_inferred_denoised_img_raw,
    )

    mel_original_img_same_dim, mel_inferred_denoised_img_same_dim = make_same_width_by_filling_white(
      img_a=mel_original_img,
      img_b=mel_inferred_denoised_img,
    )

    structural_similarity_raw, mel_difference_denoised_img_raw = calculate_structual_similarity_np(
        img_a=mel_original_img_raw_same_dim,
        img_b=mel_inferred_denoised_img_raw_same_dim,
    )
    val_entry.structural_similarity = structural_similarity_raw

    structural_similarity, mel_denoised_diff_img = calculate_structual_similarity_np(
        img_a=mel_original_img_same_dim,
        img_b=mel_inferred_denoised_img_same_dim,
    )
    validation_entry_output.mel_denoised_diff_img = mel_denoised_diff_img

    imageio.imsave("/tmp/mel_original_img_raw.png", mel_original_img_raw)
    imageio.imsave("/tmp/mel_inferred_img_raw.png", mel_inferred_denoised_img_raw)
    imageio.imsave("/tmp/mel_difference_denoised_img_raw.png", mel_difference_denoised_img_raw)

    # logger.info(val_entry)
    logger.info(f"MCD DTW: {val_entry.mcd_dtw}")
    logger.info(f"MCD DTW penalty: {val_entry.mcd_dtw_penalty}")
    logger.info(f"MCD DTW frames: {val_entry.mcd_dtw_frames}")

    logger.info(f"MCD: {val_entry.mcd}")
    logger.info(f"MCD penalty: {val_entry.mcd_penalty}")
    logger.info(f"MCD frames: {val_entry.mcd_frames}")

    # logger.info(f"MCD DTW V2: {val_entry.mcd_dtw_v2}")
    logger.info(f"Structural Similarity: {val_entry.structural_similarity}")
    logger.info(f"Cosine Similarity: {val_entry.cosine_similarity}")
    save_callback(validation_entry_output)
    inference_entries.append(val_entry)
    #score, diff_img = compare_mels(a, b)

  return complete_wav_denoised, inference_entries
