from pathlib import Path

import numpy as np
import torch

from waveglow import CheckpointWaveglow, convert_glow_files
from waveglow.converter.convert import convert_glow_files
from waveglow.dl_pretrained import download_pretrained_model
from waveglow.synthesizer import Synthesizer
from waveglow.taco_stft import TacotronSTFT
from waveglow.utils import get_default_device, try_copy_to
from waveglow_tests.globals import *


def test_component():
  device = get_default_device()
  if not DL_PATH.is_file():
    download_pretrained_model(DL_PATH, version=3)
  if not WG_PATH.is_file():
    checkpoint = convert_glow_files(DL_PATH, WG_PATH, device=device, keep_orig=True)
  else:
    checkpoint = CheckpointWaveglow.load(WG_PATH, device=device)

  s = Synthesizer(checkpoint)

  audio = Path("res/audio.wav")

  taco_stft = TacotronSTFT(s.hparams, device)
  mel = taco_stft.get_mel_tensor_from_file(str(audio.absolute()))
  mel_var = torch.FloatTensor(mel)
  mel_var = try_copy_to(mel_var, device)
  # del mel
  # mel_var = torch.autograd.Variable(mel_torch)
  mel_var = mel_var.unsqueeze(0)

  result = s.infer(mel_var, seed=0)

  assert result.sampling_rate == 22050
  assert result.was_overamplified is False
  np.testing.assert_array_almost_equal(
    result.wav[:10],
    np.array([
        -0.00143214, -0.00149224, -0.00172377, -0.002468, -0.00232015,
        -0.00253711, -0.00260813, -0.00190917, -0.00178499, -0.00134584
      ],
      dtype=float
    )
  )

  np.testing.assert_array_almost_equal(
    result.wav_denoised[:10],
    np.array([
        -0.00130634, -0.00123645, -0.00103314, -0.00166089, -0.0022371,
        -0.00194235, -0.00182102, -0.00135233, -0.00156001, -0.0014026
      ],
      dtype=float
    )
  )
  assert result.denoising_duration_s > 0
  assert result.inference_duration_s > 0
