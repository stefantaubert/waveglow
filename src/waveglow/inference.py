# from dataclasses import dataclass
# from logging import Logger
# from pathlib import Path
# from typing import Callable, Dict, List, Optional, Tuple

# import imageio
# import numpy as np
# import torch
# from waveglow.utils.generic_list import GenericList
# from mel_cepstral_distance import get_metrics_mels
# from pandas import DataFrame
# from tqdm import tqdm

# from waveglow.audio_utils import (concatenate_audios, get_duration_s,
#                                   normalize_wav, plot_melspec_np)
# from waveglow.globals import MCD_NO_OF_COEFFS_PER_FRAME
# from waveglow.image_utils import (calculate_structual_similarity_np,
#                                   make_same_width_by_filling_white)
# from waveglow.model_checkpoint import CheckpointWaveglow
# from waveglow.synthesizer import InferenceResult, Synthesizer
# from waveglow.taco_stft import TacotronSTFT
# from waveglow.utils import cosine_dist_mels


# @dataclass
# class InferMelEntry():
#   identifier: str
#   mel: np.ndarray
#   mel_path: Path
#   sr: int


# @dataclass
# class InferenceEntry():
#   entry: InferMelEntry = None
#   inference_result: InferenceResult = None
#   seed: int = None
#   inferred_duration_s: float = None
#   iteration: int = None
#   mel_original_frames: int = None
#   mel_inferred_frames: int = None
#   mcd_dtw: float = None
#   mcd_dtw_penalty: int = None
#   mcd_dtw_frames: int = None
#   mcd: float = None
#   mcd_penalty: int = None
#   mcd_frames: int = None
#   structural_similarity: float = None
#   cosine_similarity: float = None
#   denoiser_strength: float = None
#   sigma: float = None
#   train_name: str = None
#   mel_path: Path = None


# class InferenceEntries(GenericList[InferenceEntry]):
#   pass


# def get_df(entries: InferenceEntries) -> DataFrame:
#   if len(entries) == 0:
#     return DataFrame()

#   data = [
#     {
#       "Id": entry.entry.identifier,
#       "Timepoint": f"{entry.inference_result.timepoint:%Y/%m/%d %H:%M:%S}",
#       "Iteration": entry.iteration,
#       "Seed": entry.seed,
#       "Sigma": entry.sigma,
#       "Denoiser strength": entry.denoiser_strength,
#       "Inference duration (s)": entry.inference_result.inference_duration_s,
#       "Denoising duration (s)": entry.inference_result.denoising_duration_s,
#       "Overamplified?": entry.inference_result.was_overamplified,
#       "Inferred wav duration (s)": entry.inferred_duration_s,
#       "# Frames original mel": entry.mel_original_frames,
#       "# Frames inferred mel": entry.mel_inferred_frames,
#       "# Difference frames": entry.mel_inferred_frames - entry.mel_original_frames,
#       "Sampling rate (Hz)": entry.inference_result.sampling_rate,
#       "Train name": entry.train_name,
#       "Mel path": str(entry.entry.mel_path),
#       "Mel sampling rate": str(entry.entry.sr),
#     }
#     for entry in entries.items()
#   ]

#   df = DataFrame(
#     data=[x.values() for x in data],
#     columns=data[0].keys(),
#   )

#   return df


# @dataclass
# class InferenceEntryOutput():
#   identifier: int = None
#   mel_orig: np.ndarray = None
#   mel_orig_img: np.ndarray = None
#   orig_sr: int = None
#   inferred_sr: int = None
#   mel_inferred_denoised: np.ndarray = None
#   mel_inferred_denoised_img: np.ndarray = None
#   wav_inferred_denoised: np.ndarray = None
#   mel_denoised_diff_img: np.ndarray = None
#   wav_inferred: np.ndarray = None
#   was_fast: bool = None


# def mel_to_torch(mel: np.ndarray) -> np.ndarray:
#   res = torch.FloatTensor(mel)
#   res = res.cuda()
#   return res


# def infer(mel_entries: List[InferMelEntry], checkpoint: CheckpointWaveglow, custom_hparams: Optional[Dict[str, str]], denoiser_strength: float, sigma: float, sentence_pause_s: float, save_callback: Callable[[InferenceEntryOutput], None], concatenate: bool, seed: int, train_name: str, logger: Logger, fast: bool) -> Tuple[InferenceEntries, Tuple[Optional[np.ndarray], int]]:
#   inference_entries = InferenceEntries()

#   if len(mel_entries) == 0:
#     logger.info("Nothing to synthesize!")
#     return inference_entries

#   synth = Synthesizer(
#     checkpoint=checkpoint,
#     custom_hparams=custom_hparams,
#     logger=logger
#   )

#   # Check mels have the same sampling rate as trained waveglow model
#   for mel_entry in mel_entries:
#     assert mel_entry.sr == synth.hparams.sampling_rate

#   taco_stft = TacotronSTFT(synth.hparams, logger=logger)
#   mels_torch = []
#   mels_torch_prepared = []
#   for mel_entry in mel_entries:
#     mel_torch = mel_to_torch(mel_entry.mel)
#     mels_torch.append(mel_torch)
#     mel_var = torch.autograd.Variable(mel_torch)
#     mel_var = mel_var.cuda()
#     mel_var = mel_var.unsqueeze(0)
#     mels_torch_prepared.append(mel_var)

#   inference_results = synth.infer_all(
#     mels_torch_prepared, sigma, denoiser_strength, seed=seed)

#   complete_wav_denoised: Optional[np.ndarray] = None
#   if concatenate:
#     if len(inference_results) >= 1:
#       logger.info("Concatening audios...")
#     complete_wav_denoised = concatenate_audios(
#       [x.wav_denoised for x in inference_results], sentence_pause_s, synth.hparams.sampling_rate)
#     complete_wav_denoised = normalize_wav(complete_wav_denoised)
#     if len(inference_results) >= 1:
#       logger.info("Done.")

#   inference_result: InferenceResult
#   mel_entry: InferMelEntry
#   for mel_entry, inference_result in tqdm(zip(mel_entries, inference_results)):
#     wav_inferred_denoised_normalized = normalize_wav(inference_result.wav_denoised)

#     val_entry = InferenceEntry(
#       entry=mel_entry,
#       inference_result=inference_result,
#       iteration=checkpoint.iteration,
#       inferred_duration_s=get_duration_s(
#         inference_result.wav_denoised, inference_result.sampling_rate),
#       denoiser_strength=denoiser_strength,
#       sigma=sigma,
#       seed=seed,
#       train_name=train_name,
#     )

#     mel_orig = mel_entry.mel

#     wav_inferred_denoised_normalized_tensor = torch.FloatTensor(wav_inferred_denoised_normalized)
#     mel_inferred_denoised = taco_stft.get_mel_tensor(wav_inferred_denoised_normalized_tensor)
#     mel_inferred_denoised = mel_inferred_denoised.numpy()

#     validation_entry_output = InferenceEntryOutput(
#       identifier=mel_entry.identifier,
#       mel_orig=mel_orig,
#       inferred_sr=inference_result.sampling_rate,
#       mel_inferred_denoised=mel_inferred_denoised,
#       wav_inferred_denoised=wav_inferred_denoised_normalized,
#       orig_sr=mel_entry.sr,
#       wav_inferred=normalize_wav(inference_result.wav),
#       mel_denoised_diff_img=None,
#       mel_inferred_denoised_img=None,
#       mel_orig_img=None,
#       was_fast=fast,
#     )

#     if not fast:
#       mcd_dtw, penalty_dtw, final_frame_number_dtw = get_metrics_mels(
#         mel_orig, mel_inferred_denoised,
#         n_mfcc=MCD_NO_OF_COEFFS_PER_FRAME,
#         take_log=False,
#         use_dtw=True,
#       )

#       val_entry.mel_original_frames = mel_orig.shape[1]
#       val_entry.mel_inferred_frames = mel_inferred_denoised.shape[1]
#       val_entry.mcd_dtw = mcd_dtw
#       val_entry.mcd_dtw_penalty = penalty_dtw
#       val_entry.mcd_dtw_frames = final_frame_number_dtw

#       mcd, penalty, final_frame_number = get_metrics_mels(
#         mel_orig, mel_inferred_denoised,
#         n_mfcc=MCD_NO_OF_COEFFS_PER_FRAME,
#         take_log=False,
#         use_dtw=False,
#       )

#       val_entry.mcd = mcd
#       val_entry.mcd_penalty = penalty
#       val_entry.mcd_frames = final_frame_number

#       cosine_similarity = cosine_dist_mels(mel_orig, mel_inferred_denoised)
#       val_entry.cosine_similarity = cosine_similarity

#       mel_original_img_raw, mel_original_img = plot_melspec_np(mel_orig)
#       mel_inferred_denoised_img_raw, mel_inferred_denoised_img = plot_melspec_np(
#         mel_inferred_denoised)

#       validation_entry_output.mel_orig_img = mel_original_img
#       validation_entry_output.mel_inferred_denoised_img = mel_inferred_denoised_img

#       mel_original_img_raw_same_dim, mel_inferred_denoised_img_raw_same_dim = make_same_width_by_filling_white(
#         img_a=mel_original_img_raw,
#         img_b=mel_inferred_denoised_img_raw,
#       )

#       mel_original_img_same_dim, mel_inferred_denoised_img_same_dim = make_same_width_by_filling_white(
#         img_a=mel_original_img,
#         img_b=mel_inferred_denoised_img,
#       )

#       structural_similarity_raw, mel_difference_denoised_img_raw = calculate_structual_similarity_np(
#           img_a=mel_original_img_raw_same_dim,
#           img_b=mel_inferred_denoised_img_raw_same_dim,
#       )
#       val_entry.structural_similarity = structural_similarity_raw

#       structural_similarity, mel_denoised_diff_img = calculate_structual_similarity_np(
#           img_a=mel_original_img_same_dim,
#           img_b=mel_inferred_denoised_img_same_dim,
#       )
#       validation_entry_output.mel_denoised_diff_img = mel_denoised_diff_img

#       imageio.imsave("/tmp/mel_original_img_raw.png", mel_original_img_raw)
#       imageio.imsave("/tmp/mel_inferred_img_raw.png", mel_inferred_denoised_img_raw)
#       imageio.imsave("/tmp/mel_difference_denoised_img_raw.png", mel_difference_denoised_img_raw)

#       # logger.info(val_entry)
#       logger.info(f"Current: {val_entry.entry.identifier}")
#       logger.info(f"MCD DTW: {val_entry.mcd_dtw}")
#       logger.info(f"MCD DTW penalty: {val_entry.mcd_dtw_penalty}")
#       logger.info(f"MCD DTW frames: {val_entry.mcd_dtw_frames}")

#       logger.info(f"MCD: {val_entry.mcd}")
#       logger.info(f"MCD penalty: {val_entry.mcd_penalty}")
#       logger.info(f"MCD frames: {val_entry.mcd_frames}")

#       # logger.info(f"MCD DTW V2: {val_entry.mcd_dtw_v2}")
#       logger.info(f"Structural Similarity: {val_entry.structural_similarity}")
#       logger.info(f"Cosine Similarity: {val_entry.cosine_similarity}")
#     save_callback(validation_entry_output)
#     inference_entries.append(val_entry)

#   return inference_entries, (complete_wav_denoised, synth.hparams.sampling_rate)
