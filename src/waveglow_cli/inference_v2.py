import random
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from typing import List

import imageio
import numpy as np
import torch
from mel_cepstral_distance import get_metrics_mels
from pandas import DataFrame
from tqdm import tqdm

from waveglow.audio_utils import float_to_wav, get_duration_s, normalize_wav, plot_melspec_np
from waveglow.globals import MCD_NO_OF_COEFFS_PER_FRAME
from waveglow.image_utils import (calculate_structual_similarity_np,
                                  make_same_width_by_filling_white, stack_images_vertically)
from waveglow.model_checkpoint import CheckpointWaveglow
from waveglow.synthesizer import InferenceResult, Synthesizer
from waveglow.taco_stft import TacotronSTFT
from waveglow.utils import (cosine_dist_mels, get_all_files_in_all_subfolders,
                            set_torch_thread_to_max, split_hparams_string, try_copy_to)
from waveglow_cli.argparse_helper import (get_optional, parse_existing_directory,
                                          parse_existing_file, parse_non_negative_integer,
                                          parse_path)
from waveglow_cli.helper import (add_denoiser_and_sigma_arguments, add_device_argument,
                                 add_hparams_argument)


@dataclass
class InferenceEntry():
  # entry: InferMelEntry = None
  inference_result: InferenceResult = None
  seed: int = None
  inferred_duration_s: float = None
  iteration: int = None
  mel_original_frames: int = None
  mel_inferred_frames: int = None
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
  mel_path: Path = None


def init_synthesis_parser(parser: ArgumentParser) -> None:
  parser.description = "Synthesize mel-spectrograms to audio files (.wav)."
  parser.add_argument('checkpoint', type=parse_existing_file,
                      metavar="CHECKPOINT", help="path to checkpoint which should be used for synthesis")
  parser.add_argument('folder', type=parse_existing_directory, metavar="FOLDER",
                      help="path to folder which contains the mel-spectrograms that should be synthesized")
  add_denoiser_and_sigma_arguments(parser)
  add_device_argument(parser)
  add_hparams_argument(parser)
  parser.add_argument('--custom-seed', type=get_optional(parse_non_negative_integer),
                      default=None, help="custom seed used for synthesis; if left unset a random seed will be chosen")
  # parser.add_argument('--batch-size', type=parse_positive_integer, default=64)
  parser.add_argument('--include-stats', action='store_true',
                      help="include logging of statistics (increases synthesis duration)")
  parser.add_argument('-out', '--output-directory', type=parse_path, default=None,
                      help="custom output directory if FOLDER should not be used")
  parser.add_argument('-o', '--overwrite', action='store_true',
                      help="overwrite already synthesized lines")
  return infer_mels


def infer_mels(ns: Namespace) -> bool:
  logger = getLogger(__name__)
  set_torch_thread_to_max()

  output_directory = ns.output_directory
  if output_directory is None:
    output_directory = ns.folder
  else:
    if output_directory.is_file():
      logger.error("Output directory is a file!")
      return False

  if ns.custom_seed is not None:
    seed = ns.custom_seed
  else:
    seed = random.randint(1, 9999)
    logger.info(f"Using random seed: {seed}.")

  try:
    checkpoint_inst = CheckpointWaveglow.load(ns.checkpoint, ns.device, logger)
  except Exception as ex:
    logger.error("Checkpoint couldn't be loaded!")
    return False

  all_files = get_all_files_in_all_subfolders(ns.folder)
  all_mel_files = list(file for file in all_files if file.suffix.lower() == ".npy")

  custom_hparams = split_hparams_string(ns.custom_hparams)

  synth = Synthesizer(
    checkpoint=checkpoint_inst,
    custom_hparams=custom_hparams,
    device=ns.device,
    logger=logger,
  )

  taco_stft = TacotronSTFT(synth.hparams, ns.device, logger=logger)

  all_mel_files = tqdm(all_mel_files, unit=" mel(s)", ncols=100, desc="Inferring")
  for mel_path in all_mel_files:
    out_stem = f"{mel_path.name}"

    logger.debug(f"Loading mel from {mel_path} ...")
    mel = np.load(mel_path)
    mel_var = torch.FloatTensor(mel)
    mel_var = try_copy_to(mel_var, ns.device)
    # del mel
    # mel_var = torch.autograd.Variable(mel_torch)
    mel_var = mel_var.unsqueeze(0)
    logger.debug("Inferring mel...")
    inference_result = synth.infer(mel_var, ns.sigma, ns.denoiser_strength, seed)
    # del mel_var
    wav_inferred_denoised_normalized = normalize_wav(inference_result.wav_denoised)

    # out.npy.wav
    wav_path = output_directory / mel_path.relative_to(ns.folder).parent / f"{out_stem}.wav"
    logger.debug(f"Saving {wav_path.absolute()} ...")
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    float_to_wav(wav_inferred_denoised_normalized, wav_path)

    if ns.include_stats:
      # wav_path = output_directory / mel_path.relative_to(ns.folder).parent / f"{out_stem}.noise.wav"
      # wav_inferred_normalized = normalize_wav(inference_result.wav)
      # float_to_wav(wav_inferred_normalized, wav_path)

      val_entry = InferenceEntry(
        # entry=mel_entry,
        inference_result=inference_result,
        iteration=checkpoint_inst.iteration,
        inferred_duration_s=get_duration_s(
          inference_result.wav_denoised, inference_result.sampling_rate),
        denoiser_strength=ns.denoiser_strength,
        sigma=ns.sigma,
        seed=seed,
      )

      mel_orig = mel

      wav_inferred_denoised_normalized_tensor = torch.FloatTensor(wav_inferred_denoised_normalized)
      mel_inferred_denoised = taco_stft.get_mel_tensor(wav_inferred_denoised_normalized_tensor)
      mel_inferred_denoised = mel_inferred_denoised.numpy()

      validation_entry_output = InferenceEntryOutput(
        # identifier=mel_entry.identifier,
        mel_orig=mel_orig,
        inferred_sr=inference_result.sampling_rate,
        mel_inferred_denoised=mel_inferred_denoised,
        wav_inferred_denoised=wav_inferred_denoised_normalized,
        # orig_sr=mel_entry.sr,
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

      val_entry.mel_original_frames = mel_orig.shape[1]
      val_entry.mel_inferred_frames = mel_inferred_denoised.shape[1]
      val_entry.mcd_dtw = mcd_dtw
      val_entry.mcd_dtw_penalty = penalty_dtw
      val_entry.mcd_dtw_frames = final_frame_number_dtw

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

      # imageio.imsave("/tmp/mel_original_img_raw.png", mel_original_img_raw)
      # imageio.imsave("/tmp/mel_inferred_img_raw.png", mel_inferred_denoised_img_raw)
      # imageio.imsave("/tmp/mel_difference_denoised_img_raw.png", mel_difference_denoised_img_raw)

      # logger.info(val_entry)
      # logger.info(f"Current: {val_entry.entry.identifier}")
      logger.info(f"MCD DTW: {val_entry.mcd_dtw}")
      logger.info(f"MCD DTW penalty: {val_entry.mcd_dtw_penalty}")
      logger.info(f"MCD DTW frames: {val_entry.mcd_dtw_frames}")

      logger.info(f"MCD: {val_entry.mcd}")
      logger.info(f"MCD penalty: {val_entry.mcd_penalty}")
      logger.info(f"MCD frames: {val_entry.mcd_frames}")

      # logger.info(f"MCD DTW V2: {val_entry.mcd_dtw_v2}")
      logger.info(f"Structural Similarity: {val_entry.structural_similarity}")
      logger.info(f"Cosine Similarity: {val_entry.cosine_similarity}")
      save_results(validation_entry_output, output_directory / mel_path.stem)

  logger.info(f"Done. Written output to: {output_directory.absolute()}")
  return True


@dataclass
class InferenceEntryOutput():
  # identifier: int = None
  mel_orig: np.ndarray = None
  mel_orig_img: np.ndarray = None
  orig_sr: int = None
  inferred_sr: int = None
  mel_inferred_denoised: np.ndarray = None
  mel_inferred_denoised_img: np.ndarray = None
  wav_inferred_denoised: np.ndarray = None
  mel_denoised_diff_img: np.ndarray = None
  wav_inferred: np.ndarray = None


class InferenceEntries(List[InferenceEntry]):
  pass


def get_df(entries: InferenceEntries) -> DataFrame:
  if len(entries) == 0:
    return DataFrame()

  data = [
    {
      "Id": entry.entry.identifier,
      "Timepoint": f"{entry.inference_result.timepoint:%Y/%m/%d %H:%M:%S}",
      "Iteration": entry.iteration,
      "Seed": entry.seed,
      "Sigma": entry.sigma,
      "Denoiser strength": entry.denoiser_strength,
      "Inference duration (s)": entry.inference_result.inference_duration_s,
      "Denoising duration (s)": entry.inference_result.denoising_duration_s,
      "Overamplified?": entry.inference_result.was_overamplified,
      "Inferred wav duration (s)": entry.inferred_duration_s,
      "# Frames original mel": entry.mel_original_frames,
      "# Frames inferred mel": entry.mel_inferred_frames,
      "# Difference frames": entry.mel_inferred_frames - entry.mel_original_frames,
      "Sampling rate (Hz)": entry.inference_result.sampling_rate,
      "Train name": entry.train_name,
      "Mel path": str(entry.entry.mel_path),
      "Mel sampling rate": str(entry.entry.sr),
    }
    for entry in entries
  ]

  df = DataFrame(
    data=[x.values() for x in data],
    columns=data[0].keys(),
  )

  return df


def save_results(output: InferenceEntryOutput, dest_dir: Path) -> None:
  # dest_dir = get_inferred_mel_dir(infer_dir, output.identifier)
  dest_dir.mkdir(parents=True, exist_ok=True)
  imageio.imsave(dest_dir / "original.png", output.mel_orig_img)
  imageio.imsave(dest_dir / "inferred_denoised.png", output.mel_inferred_denoised_img)
  imageio.imsave(dest_dir / "diff.png", output.mel_denoised_diff_img)

  stack_images_vertically(
    list_im=[
      dest_dir / "original.png",
      dest_dir / "inferred_denoised.png",
      dest_dir / "diff.png",
    ],
    out_path=dest_dir / "comparison.png"
  )

  # np.save(dest_dir / "original.mel.npy", output.mel_orig)
  # np.save(dest_dir / "inferred_denoised.mel.npy", output.mel_inferred_denoised)

  # inferred_denoised_path = dest_dir / "inferred_denoised.wav"
  # float_to_wav(output.wav_inferred_denoised, inferred_denoised_path, sample_rate=output.inferred_sr)

  # pat = re.compile("id=([0-9]*)_")
  # entry_id = re.findall(pat, output.identifier)
  # if len(entry_id) == 1:
  #   inferred_denoised_path_copy = infer_dir / f"{entry_id[0]}.wav"
  #   copyfile(inferred_denoised_path, inferred_denoised_path_copy)

  # float_to_wav(output.wav_inferred, dest_dir / "inferred.wav", sample_rate=output.inferred_sr)

  # wav_info = get_wav_info_dict(
  #   identifier=output.identifier,
  #   path=inferred_denoised_path,
  #   sr=output.inferred_sr,
  # )

  # denoised_audio_wav_paths.append(wav_info)
