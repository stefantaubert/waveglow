from pathlib import Path

import torch

DEFAULT_SENTENCE_PAUSE_S = 0.2
#DEFAULT_WAVEGLOW = "pretrained_v5"
DEFAULT_WAVEGLOW_VERSION = 3
DEFAULT_SIGMA = 1.0
DEFAULT_DENOISER_STRENGTH = 0.0005
DEFAULT_SEED = 1111
DEFAULT_READ_MEL_INFO_PATH = Path("/tmp/mel_out.json")
DEFAULT_SAVE_WAV_INFO_COPY_PATH = Path("/tmp/wav_out.json")

if torch.cuda.is_available():
  __DEFAULT_DEVICE = "cuda:0"
else:
  __DEFAULT_DEVICE = "cpu"

DEFAULT_DEVICE = __DEFAULT_DEVICE
