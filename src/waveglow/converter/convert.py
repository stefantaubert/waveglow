import logging
import os
import pathlib
import shutil
import sys
import tempfile
from dataclasses import asdict

import torch

from waveglow.hparams import HParams
from waveglow.train import CheckpointWaveglow


def convert_glow(origin: str, destination: str, device: torch.device, keep_orig: bool = False) -> CheckpointWaveglow:
  logger = logging.getLogger(__name__)
  logger.info("Pretrained model is beeing converted...")
  tmp_out = tempfile.mktemp()
  res = _convert_core(origin, tmp_out, device)
  if keep_orig:
    if origin == destination:
      original_path = f"{origin}.orig"
      shutil.move(origin, original_path)
  else:
    os.remove(origin)
  shutil.move(tmp_out, destination)
  logger.info("Done.")
  return res


def _convert_core(source: str, destination: str, device: torch.device) -> CheckpointWaveglow:
  '''in version 3 there is only "model"'''
  assert source.is_file()
  logger = logging.getLogger(__name__)
  # torch.nn.Module.dump_patches = True
  rel_converter_location = str(pathlib.Path(__file__).parent.absolute())
  sys.path.append(rel_converter_location)
  checkpoint_dict = torch.load(source, map_location=device)

  hparams = HParams(
    # see WaveGlow paper
    # "We use a sampling rate of 22,050kHz"
    sampling_rate=22050,
    # "we use mel-spectrograms with 80 bins"
    n_mel_channels=80,
    # "The parameters of the mel-spectrograms are FFT size 1024, hop size 256, and window size 1024."
    filter_length=1024,
    hop_length=256,
    win_length=1024,
    # "with a batch size of 24 and a step size of 1×10−4"
    batch_size=24,
    learning_rate=1e-4,
    # "We also output 2 of the channels after every 4 coupling layers."
    n_early_every=4,
    n_early_size=2,
    # "The coupling layer networks (WN) each have 8 layers of dilated convolutions"
    n_layers=8,
    # "using randomly chosen clips of 16,000 samples"
    segment_length=16000
  )

  # "for 580,000 iterations"
  iteration = 580000
  # if "iteration" in checkpoint_dict.keys():
  #   iteration = checkpoint_dict["iteration"]

  optimizer = {}
  # if "optimizer" in checkpoint_dict.keys():
  #   optimizer = checkpoint_dict["optimizer"]

  learning_rate = hparams.learning_rate
  # if "learning_rate" in checkpoint_dict.keys():
  #   learning_rate = checkpoint_dict["learning_rate"]

  state_dict = {}
  if "model" in checkpoint_dict.keys():
    model = checkpoint_dict["model"]
    state_dict = model.state_dict()

  res = CheckpointWaveglow(
    hparams=asdict(hparams),
    iteration=iteration,
    learning_rate=learning_rate,
    optimizer=optimizer,
    state_dict=state_dict
  )

  res.save(destination, logger)

  logger.info("Successfully converted.")
  return res
