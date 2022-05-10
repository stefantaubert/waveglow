import datetime
import random
from dataclasses import dataclass
from functools import partial
from logging import Logger, getLogger
from pathlib import Path
from shutil import copyfile
from typing import Any, Callable, Dict, List, Optional, Tuple

import imageio
import numpy as np
import regex as re
import torch
from audio_utils import float_to_wav, get_duration_s, normalize_wav
from audio_utils.audio import concatenate_audios
from general_utils import (get_all_files_in_all_subfolders, parse_json,
                           pass_lines_list, save_json)
from tqdm import tqdm
from waveglow.app.defaults import (DEFAULT_DENOISER_STRENGTH,
                                   DEFAULT_READ_MEL_INFO_PATH,
                                   DEFAULT_SAVE_WAV_INFO_COPY_PATH,
                                   DEFAULT_SEED, DEFAULT_SENTENCE_PAUSE_S,
                                   DEFAULT_SIGMA)
from waveglow.app.io import (get_checkpoints_dir, get_inference_root_dir,
                             get_train_dir, get_wav_info_dict,
                             get_wav_out_dict)
from waveglow.core import (CheckpointWaveglow, InferenceEntries,
                           InferenceEntryOutput)
from waveglow.core import infer as infer_core
from waveglow.core.inference import InferMelEntry, get_df, mel_to_torch
from waveglow.core.model_checkpoint import CheckpointWaveglow
from waveglow.core.synthesizer import InferenceResult, Synthesizer
from waveglow.globals import MCD_NO_OF_COEFFS_PER_FRAME
from waveglow.utils import (cosine_dist_mels, get_custom_or_last_checkpoint,
                            prepare_logger)


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
