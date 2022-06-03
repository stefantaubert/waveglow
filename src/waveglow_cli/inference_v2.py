import random
from argparse import ArgumentParser
from logging import getLogger
from pathlib import Path
from typing import Optional

import numpy as np
from general_utils import get_all_files_in_all_subfolders
from tqdm import tqdm
from waveglow import CheckpointWaveglow
from waveglow.audio_utils import float_to_wav, normalize_wav
from waveglow.inference import mel_to_torch
from waveglow.model_checkpoint import CheckpointWaveglow
from waveglow.synthesizer import Synthesizer

from waveglow_cli.argparse_helper import (parse_existing_directory,
                                          parse_existing_file, parse_path)


def init_inference_v2_parser(parser: ArgumentParser) -> None:
  parser.add_argument('checkpoint', type=parse_existing_file)
  parser.add_argument('directory', type=parse_existing_directory)
  parser.add_argument('--sigma', type=float, default=1.0)
  parser.add_argument('--denoiser-strength', type=float, default=0.0005)
  parser.add_argument('--custom-seed', type=int, default=None)
  parser.add_argument('--batch-size', type=int, default=64)
  parser.add_argument('--include-stats', action="store_true")
  parser.add_argument('-out', '--output-directory', type=Path)
  parser.add_argument('-o', '--overwrite', action="store_true")
  return infer_mels


def infer_mels(base_dir: Path, checkpoint: Path, directory: Path, sigma: float, denoiser_strength: float, custom_seed: Optional[int], include_stats: bool, batch_size: int, output_directory: Optional[Path], overwrite: bool) -> bool:
  logger = getLogger(__name__)

  if not checkpoint.is_file():
    logger.error("Checkpoint was not found!")
    return False

  if not directory.is_dir():
    logger.error("Directory was not found!")
    return False

  if not 0 <= sigma <= 1:
    logger.error("Sigma needs to be in interval [0, 1]!")
    return False

  if not 0 <= denoiser_strength <= 1:
    logger.error("Denoiser strength needs to be in interval [0, 1]!")
    return False

  if not batch_size > 0:
    logger.error("Batch size need to be greater than zero!")
    return False

  if output_directory is None:
    output_directory = directory
  else:
    if output_directory.is_file():
      logger.error("Output directory is a file!")
      return False

  if custom_seed is not None and not custom_seed >= 0:
    logger.error("Custom seed needs to be greater than or equal to zero!")
    return False

  if custom_seed is not None:
    seed = custom_seed
  else:
    seed = random.randint(1, 9999)
    logger.info(f"Using random seed: {seed}.")

  try:
    checkpoint_inst = CheckpointWaveglow.load(checkpoint, logger)
  except Exception as ex:
    logger.error("Checkpoint couldn't be loaded!")
    return False

  all_files = get_all_files_in_all_subfolders(directory)
  all_mel_files = list(file for file in all_files if file.suffix.lower() == ".npy")

  synth = Synthesizer(
    checkpoint=checkpoint_inst,
    custom_hparams=None,
    logger=logger,
  )

  # taco_stft = TacotronSTFT(synth.hparams, logger=logger)

  with tqdm(total=len(all_mel_files), unit=" mels", ncols=100, desc="Inference") as progress_bar:
    for mel_path in all_mel_files:
      out_stem = f"{mel_path.name}"
      # out.npy.wav
      wav_path = output_directory / mel_path.relative_to(directory).parent / f"{out_stem}.wav"
      if wav_path.exists() and not overwrite:
        logger.info(f"{mel_path.relative_to(directory)}: Skipped because it is already inferred!")
        continue

      logger.debug(f"Loading mel from {mel_path} ...")
      mel = np.load(mel_path)
      mel_var = mel_to_torch(mel)
      del mel
      #mel_var = torch.autograd.Variable(mel_torch)
      mel_var = mel_var.unsqueeze(0)
      logger.debug("Inferring mel...")
      inference_result = synth.infer(mel_var, sigma, denoiser_strength, seed)
      del mel_var
      wav_inferred_denoised_normalized = normalize_wav(inference_result.wav_denoised)

      logger.debug(f"Saving {wav_path.absolute()} ...")
      wav_path.parent.mkdir(parents=True, exist_ok=True)
      float_to_wav(wav_inferred_denoised_normalized, wav_path)

      progress_bar.update()

  logger.info(f"Done. Written output to: {output_directory.absolute()}")
  return True
