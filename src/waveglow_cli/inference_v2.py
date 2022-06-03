import random
from argparse import ArgumentParser, Namespace
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
                                          parse_existing_file,
                                          parse_float_between_zero_and_one,
                                          parse_non_negative_float,
                                          parse_non_negative_integer,
                                          parse_path, parse_positive_integer)
from waveglow_cli.defaults import (DEFAULT_DENOISER_STRENGTH, DEFAULT_SEED,
                                   DEFAULT_SIGMA)


def init_inference_v2_parser(parser: ArgumentParser) -> None:
  parser.add_argument('checkpoint', type=parse_existing_file)
  parser.add_argument('directory', type=parse_existing_directory)
  parser.add_argument("--denoiser-strength", default=DEFAULT_DENOISER_STRENGTH,
                      type=parse_float_between_zero_and_one, help='Removes model bias.')
  parser.add_argument("--sigma", type=parse_float_between_zero_and_one, default=DEFAULT_SIGMA)
  parser.add_argument('--custom-seed', type=parse_non_negative_integer, default=None)
  #parser.add_argument('--batch-size', type=parse_positive_integer, default=64)
  parser.add_argument('--include-stats', action="store_true")
  parser.add_argument('-out', '--output-directory', type=parse_path)
  parser.add_argument('-o', '--overwrite', action="store_true")
  return infer_mels


def infer_mels(ns: Namespace) -> bool:
  logger = getLogger(__name__)

  output_directory = ns.output_directory
  if output_directory is None:
    output_directory = ns.directory
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
    checkpoint_inst = CheckpointWaveglow.load(ns.checkpoint, logger)
  except Exception as ex:
    logger.error("Checkpoint couldn't be loaded!")
    return False

  all_files = get_all_files_in_all_subfolders(ns.directory)
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
      wav_path = output_directory / mel_path.relative_to(ns.directory).parent / f"{out_stem}.wav"
      if wav_path.exists() and not ns.overwrite:
        logger.info(f"{mel_path.relative_to(ns.directory)}: Skipped because it is already inferred!")
        continue

      logger.debug(f"Loading mel from {mel_path} ...")
      mel = np.load(mel_path)
      mel_var = mel_to_torch(mel)
      del mel
      #mel_var = torch.autograd.Variable(mel_torch)
      mel_var = mel_var.unsqueeze(0)
      logger.debug("Inferring mel...")
      inference_result = synth.infer(mel_var, ns.sigma, ns.denoiser_strength, seed)
      del mel_var
      wav_inferred_denoised_normalized = normalize_wav(inference_result.wav_denoised)

      logger.debug(f"Saving {wav_path.absolute()} ...")
      wav_path.parent.mkdir(parents=True, exist_ok=True)
      float_to_wav(wav_inferred_denoised_normalized, wav_path)

      progress_bar.update()

  logger.info(f"Done. Written output to: {output_directory.absolute()}")
  return True
