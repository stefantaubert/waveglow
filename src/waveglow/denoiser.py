from logging import Logger

import torch
from torch import Tensor

from waveglow.model import WaveGlow
from waveglow.stft import STFT
from waveglow.taco_stft import TSTFTHParams
from waveglow.utils import try_copy_to

BIAS_MEL_LENGTH = 88


class Denoiser(torch.nn.Module):
  """ Removes model bias from audio produced with waveglow """

  def __init__(self, waveglow: WaveGlow, hparams: TSTFTHParams, mode: str, device: torch.device, logger: Logger):
    super().__init__()
    self.device = device
    stft = STFT(
      filter_length=hparams.filter_length,
      hop_length=hparams.hop_length,
      win_length=hparams.win_length,
      device=device,
    )
    stft = try_copy_to(stft, device)
    self.stft = stft

    if mode == 'zeros':
      mel_input = torch.zeros(
        (1, hparams.n_mel_channels, BIAS_MEL_LENGTH),
        dtype=waveglow.upsample.weight.dtype,
        device=waveglow.upsample.weight.device)
    elif mode == 'normal':
      mel_input = torch.randn(
        (1, hparams.n_mel_channels, BIAS_MEL_LENGTH),
        dtype=waveglow.upsample.weight.dtype,
        device=waveglow.upsample.weight.device)
    else:
      msg = f"Mode {mode} if not supported"
      logger.exception(msg)
      raise Exception(msg)

    with torch.no_grad():
      bias_audio = waveglow.infer(mel_input, sigma=0.0).float()
      bias_spec, _ = self.stft.transform(bias_audio)

    self.register_buffer('bias_spec', bias_spec[:, :, 0][:, :, None])

  def forward(self, audio: Tensor, strength: float):
    audio = try_copy_to(audio, self.device)
    audio_spec, audio_angles = self.stft.transform(audio.float())
    audio_spec_denoised = audio_spec - self.bias_spec * strength
    audio_spec_denoised = torch.clamp(audio_spec_denoised, 0.0)
    audio_denoised = self.stft.inverse(audio_spec_denoised, audio_angles)
    return audio_denoised
