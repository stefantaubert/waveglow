import datetime
from dataclasses import dataclass
from logging import Logger
from typing import Callable, Dict, Optional, Set

import imageio
import numpy as np
import torch
from audio_utils import get_duration_s, normalize_wav, wav_to_float32
from audio_utils.mel import TacotronSTFT, plot_melspec_np
from image_utils import (calculate_structual_similarity_np,
                         make_same_width_by_filling_white)
from mcd import get_mcd_between_mel_spectograms
from text_utils import deserialize_list
from tts_preparation import PreparedData, PreparedDataList
from waveglow.core.model_checkpoint import CheckpointWaveglow
from waveglow.core.synthesizer import Synthesizer
from waveglow.globals import MCD_NO_OF_COEFFS_PER_FRAME
from waveglow.utils import GenericList, cosine_dist_mels


@dataclass
class ValidationEntry():
  entry_id: int = None
  ds_entry_id: int = None
  text_original: str = None
  text: str = None
  wav_path: str = None
  denoising_duration_s: float = None
  was_overamplified: bool = None
  original_duration_s: float = None
  inferred_duration_s: float = None
  diff_duration_s: float = None
  unique_symbols_count: int = None
  speaker_id: int = None
  iteration: int = None
  inference_duration_s: float = None
  symbol_count: int = None
  timepoint: str = None
  train_name: str = None
  sampling_rate: int = None
  diff_frames: int = None
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


class ValidationEntries(GenericList[ValidationEntry]):
  pass


@dataclass
class ValidationEntryOutput():
  mel_orig: np.ndarray = None
  mel_orig_img: np.ndarray = None
  orig_sr: int = None
  wav_orig: np.ndarray = None
  inferred_sr: int = None
  mel_inferred_denoised: np.ndarray = None
  mel_inferred_denoised_img: np.ndarray = None
  wav_inferred_denoised: np.ndarray = None
  mel_denoised_diff_img: np.ndarray = None
  wav_inferred: np.ndarray = None


def validate(checkpoint: CheckpointWaveglow, data: PreparedDataList, custom_hparams: Optional[Dict[str, str]], denoiser_strength: float, sigma: float, entry_ids: Optional[Set[int]], train_name: str, full_run: bool, save_callback: Callable[[PreparedData, ValidationEntryOutput], None], logger: Logger):
  validation_entries = ValidationEntries()

  if full_run:
    entries = data
  else:
    speaker_id: Optional[int] = None
    entries = PreparedDataList(data.get_for_validation(entry_ids, speaker_id))

  if len(entries) == 0:
    logger.info("Nothing to synthesize!")
    return validation_entries

  synth = Synthesizer(
    checkpoint=checkpoint,
    custom_hparams=custom_hparams,
    logger=logger
  )

  taco_stft = TacotronSTFT(synth.hparams, logger=logger)

  for entry in entries.items(True):
    mel = taco_stft.get_mel_tensor_from_file(entry.wav_path)
    mel_var = torch.autograd.Variable(mel)
    mel_var = mel_var.cuda()
    mel_var = mel_var.unsqueeze(0)

    inference_result = synth.infer(mel_var, sigma, denoiser_strength)
    wav_inferred_denoised = normalize_wav(inference_result.wav_denoised)

    symbol_count = len(deserialize_list(entry.serialized_symbol_ids))
    unique_symbols_count = len(set(deserialize_list(entry.serialized_symbol_ids)))
    timepoint = f"{datetime.datetime.now():%Y/%m/%d %H:%M:%S}"

    val_entry = ValidationEntry(
      entry_id=entry.entry_id,
      ds_entry_id=entry.ds_entry_id,
      text_original=entry.text_original,
      text=entry.text,
      wav_path=entry.wav_path,
      original_duration_s=entry.duration,
      speaker_id=entry.speaker_id,
      iteration=checkpoint.iteration,
      unique_symbols_count=unique_symbols_count,
      symbol_count=symbol_count,
      timepoint=timepoint,
      train_name=train_name,
      sampling_rate=inference_result.sampling_rate,
      inference_duration_s=inference_result.inference_duration_s,
      was_overamplified=inference_result.was_overamplified,
      denoising_duration_s=inference_result.denoising_duration_s,
      inferred_duration_s=get_duration_s(
        inference_result.wav_denoised, inference_result.sampling_rate),
      denoiser_strength=denoiser_strength,
      sigma=sigma,
    )

    val_entry.diff_duration_s = val_entry.inferred_duration_s - val_entry.original_duration_s

    mel_orig = mel.cpu().numpy()

    mel_inferred_denoised_tensor = torch.FloatTensor(inference_result.wav_denoised)
    mel_inferred_denoised = taco_stft.get_mel_tensor(mel_inferred_denoised_tensor)
    mel_inferred_denoised = mel_inferred_denoised.numpy()

    wav_orig, orig_sr = wav_to_float32(entry.wav_path)

    validation_entry_output = ValidationEntryOutput(
      mel_orig=mel_orig,
      inferred_sr=inference_result.sampling_rate,
      mel_inferred_denoised=mel_inferred_denoised,
      wav_inferred_denoised=wav_inferred_denoised,
      wav_orig=wav_orig,
      orig_sr=orig_sr,
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

    val_entry.diff_frames = mel_inferred_denoised.shape[1] - mel_orig.shape[1]
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
    save_callback(entry, validation_entry_output)
    validation_entries.append(val_entry)
    #score, diff_img = compare_mels(a, b)

  return validation_entries