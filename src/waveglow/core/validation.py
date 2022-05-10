import datetime
import random
from dataclasses import dataclass
from logging import Logger
from pathlib import Path
from typing import Callable, Dict, Optional, Set

import imageio
import numpy as np
import torch
from audio_utils import get_duration_s, normalize_wav, wav_to_float32
from audio_utils.mel import TacotronSTFT, plot_melspec_np
from general_utils import GenericList
from image_utils import (calculate_structual_similarity_np,
                         make_same_width_by_filling_white)
from mel_cepstral_distance import get_metrics_mels
from pandas import DataFrame
from tts_preparation import PreparedData, PreparedDataList
from waveglow.core.model_checkpoint import CheckpointWaveglow
from waveglow.core.synthesizer import InferenceResult, Synthesizer
from waveglow.globals import MCD_NO_OF_COEFFS_PER_FRAME
from waveglow.utils import cosine_dist_mels


@dataclass
class ValidationEntry():
  utterance: PreparedData = None
  inference_result: InferenceResult = None
  seed: int = None
  #original_duration_s: float = None
  diff_duration_s: float = None
  iteration: int = None
  inferred_duration_s: float = None
  timepoint: datetime.datetime = None
  train_name: str = None
  diff_frames: int = None
  mfcc_no_coeffs: int = None
  mfcc_dtw_mcd: float = None
  mfcc_dtw_penalty: float = None
  mfcc_dtw_frames: int = None
  mcd: float = None
  mcd_penalty: int = None
  mcd_frames: int = None
  structural_similarity: float = None
  cosine_similarity: float = None
  denoiser_strength: float = None
  sigma: float = None


class ValidationEntries(GenericList[ValidationEntry]):
  pass


def get_df(entries: ValidationEntries) -> DataFrame:
  if len(entries) == 0:
    return DataFrame()

  data = [
    {
      "Id": entry.utterance.identifier,
      "Timepoint": f"{entry.timepoint:%Y/%m/%d %H:%M:%S}",
      "Iteration": entry.iteration,
      "Seed": entry.seed,
      "Language": repr(entry.utterance.symbols_language),
      "Symbols": ''.join(entry.utterance.symbols),
      "Symbols format": repr(entry.utterance.symbols_format),
      "Speaker": entry.utterance.speaker_name,
      "Speaker Id": entry.utterance.speaker_id,
      "Sigma": entry.sigma,
      "Denoiser strength": entry.denoiser_strength,
      "Inference duration (s)": entry.inference_result.inference_duration_s,
      "Denoising duration (s)": entry.inference_result.denoising_duration_s,
      "Overamplified?": entry.inference_result.was_overamplified,
      "Inferred wav duration (s)": entry.inferred_duration_s,
      "Original wav duration (s)": entry.utterance.wav_duration,
      "Difference wav duration (s)": entry.inferred_duration_s - entry.utterance.wav_duration,
      "# Difference frames": entry.diff_frames,
      "Sampling rate (Hz)": entry.inference_result.sampling_rate,
      "# MFCC Coefficients": entry.mfcc_no_coeffs,
      "MFCC DTW MCD": entry.mfcc_dtw_mcd,
      "MFCC DTW PEN": entry.mfcc_dtw_penalty,
      "# MFCC DTW frames": entry.mfcc_dtw_frames,
      "MCD": entry.mcd,
      "PEN": entry.mcd_penalty,
      "# Frames": entry.mcd_frames,
      "Cosine Similarity (Padded)": entry.cosine_similarity,
      "Structual Similarity (Padded)": entry.structural_similarity,
      "1-gram rarity (total set)": entry.utterance.one_gram_rarity,
      "2-gram rarity (total set)": entry.utterance.two_gram_rarity,
      "3-gram rarity (total set)": entry.utterance.three_gram_rarity,
      "Combined rarity (total set)": entry.utterance.one_gram_rarity + entry.utterance.two_gram_rarity + entry.utterance.three_gram_rarity,
      "# Symbols": len(entry.utterance.symbols),
      "Unique symbols": ' '.join(sorted(set(entry.utterance.symbols))),
      "# Unique symbols": len(set(entry.utterance.symbols)),
      "Train name": entry.train_name,
      "Ds-Id": entry.utterance.ds_entry_id,
      "Wav path original": str(entry.utterance.wav_original_absolute_path),
      "Wav path": str(entry.utterance.wav_absolute_path),
    }
    for entry in entries.items()
  ]

  df = DataFrame(
    data=[x.values() for x in data],
    columns=data[0].keys(),
  )

  return df


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


def validate(checkpoint: CheckpointWaveglow, data: PreparedDataList, custom_hparams: Optional[Dict[str, str]], denoiser_strength: float, sigma: float, entry_ids: Optional[Set[int]], train_name: str, full_run: bool, save_callback: Callable[[PreparedData, ValidationEntryOutput], None], seed: int, logger: Logger) -> None:
  validation_entries = ValidationEntries()

  if full_run:
    entries = data
  elif entry_ids is not None:
    entries = PreparedDataList([x for x in data.items() if x.entry_id in entry_ids])
    if len(entries) != len(entry_ids):
      logger.error("Not all entry_id's were found!")
      assert False
  else:
    assert seed is not None
    assert len(data) > 0
    random.seed(seed)
    entry = random.choice(data)
    entries = PreparedDataList([entry])

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
    mel = taco_stft.get_mel_tensor_from_file(entry.wav_absolute_path)
    mel_var = torch.autograd.Variable(mel)
    mel_var = mel_var.cuda()
    mel_var = mel_var.unsqueeze(0)

    timepoint = datetime.datetime.now()
    inference_result = synth.infer(
      mel=mel_var,
      sigma=sigma,
      denoiser_strength=denoiser_strength,
      seed=seed
    )

    wav_inferred_denoised_normalized = normalize_wav(inference_result.wav_denoised)

    val_entry = ValidationEntry(
      utterance=entry,
      inference_result=inference_result,
      seed=seed,
      iteration=checkpoint.iteration,
      timepoint=timepoint,
      train_name=train_name,
      inferred_duration_s=get_duration_s(
        inference_result.wav_denoised, inference_result.sampling_rate),
      denoiser_strength=denoiser_strength,
      sigma=sigma,
      mfcc_no_coeffs=MCD_NO_OF_COEFFS_PER_FRAME,
    )

    val_entry.diff_duration_s = val_entry.inferred_duration_s - val_entry.utterance.wav_duration

    mel_orig = mel.cpu().numpy()

    mel_inferred_denoised_tensor = torch.FloatTensor(wav_inferred_denoised_normalized)
    mel_inferred_denoised = taco_stft.get_mel_tensor(mel_inferred_denoised_tensor)
    mel_inferred_denoised = mel_inferred_denoised.numpy()

    wav_orig, orig_sr = wav_to_float32(entry.wav_absolute_path)

    validation_entry_output = ValidationEntryOutput(
      mel_orig=mel_orig,
      inferred_sr=inference_result.sampling_rate,
      mel_inferred_denoised=mel_inferred_denoised,
      wav_inferred_denoised=wav_inferred_denoised_normalized,
      wav_orig=wav_orig,
      orig_sr=orig_sr,
      wav_inferred=normalize_wav(inference_result.wav),
      mel_denoised_diff_img=None,
      mel_inferred_denoised_img=None,
      mel_orig_img=None,
    )

    mcd_dtw, penalty_dtw, final_frame_number_dtw = get_metrics_mels(
      mel_orig, mel_inferred_denoised,
      n_mfcc=MCD_NO_OF_COEFFS_PER_FRAME,
      take_log=False,
      use_dtw=True,
    )

    val_entry.diff_frames = mel_inferred_denoised.shape[1] - mel_orig.shape[1]
    val_entry.mfcc_dtw_mcd = mcd_dtw
    val_entry.mfcc_dtw_penalty = penalty_dtw
    val_entry.mfcc_dtw_frames = final_frame_number_dtw

    mcd, penalty, final_frame_number = get_metrics_mels(
      mel_orig, mel_inferred_denoised,
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

    imageio.imsave(Path("/tmp/mel_original_img_raw.png"), mel_original_img_raw)
    imageio.imsave(Path("/tmp/mel_inferred_img_raw.png"), mel_inferred_denoised_img_raw)
    imageio.imsave(Path("/tmp/mel_difference_denoised_img_raw.png"),
                   mel_difference_denoised_img_raw)

    # logger.info(val_entry)
    logger.info(f"Current: {val_entry.utterance.entry_id}")
    logger.info(f"MCD DTW: {val_entry.mfcc_dtw_mcd}")
    logger.info(f"MCD DTW penalty: {val_entry.mfcc_dtw_penalty}")
    logger.info(f"MCD DTW frames: {val_entry.mfcc_dtw_frames}")

    logger.info(f"MCD: {val_entry.mcd}")
    logger.info(f"MCD penalty: {val_entry.mcd_penalty}")
    logger.info(f"MCD frames: {val_entry.mcd_frames}")

    logger.info(f"Structural Similarity: {val_entry.structural_similarity}")
    logger.info(f"Cosine Similarity: {val_entry.cosine_similarity}")
    save_callback(entry, validation_entry_output)
    validation_entries.append(val_entry)
    #score, diff_img = compare_mels(a, b)

  return validation_entries
