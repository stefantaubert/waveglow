import random
from typing import List, Optional, Tuple, TypeVar

import matplotlib.ticker as ticker
import numpy as np
import torch
from fastdtw.fastdtw import fastdtw
from matplotlib import pyplot as plt
from scipy.io.wavfile import read, write
from scipy.spatial.distance import euclidean

from waveglow.utils import figure_to_numpy_rgb

_T = TypeVar('_T')
PYTORCH_EXT = ".pt"


def get_duration_s(wav, sampling_rate) -> float:
  return get_duration_s_samples(len(wav), sampling_rate)


def get_sample_count(sampling_rate: int, duration_s: float):
  return int(round(sampling_rate * duration_s, 0))


def float_to_wav(wav, path, dtype=np.int16, sample_rate=22050):
  # denoiser_out is float64
  # waveglow_out is float32

  wav = convert_wav(wav, dtype)

  write(filename=path, rate=sample_rate, data=wav)


def plot_melspec(mel: np.ndarray, mel_dim_x=16, mel_dim_y=5, factor=1, title=None):
  height, width = mel.shape
  width_factor = width / 1000
  _, axes = plt.subplots(
    nrows=1,
    ncols=1,
    figsize=(mel_dim_x * factor * width_factor, mel_dim_y * factor),
  )

  axes.set_title(title)
  axes.set_yticks(np.arange(0, height, step=10))
  axes.set_xticks(np.arange(0, width, step=100))
  axes.set_xlabel("Samples")
  axes.set_ylabel("Freq. channel")
  axes.imshow(mel, aspect='auto', origin='lower', interpolation='none')
  return axes


def convert_wav(wav, to_dtype):
  '''
  if the wav is overamplified the result will also be overamplified.
  '''
  if wav.dtype != to_dtype:
    wav = wav / (-1 * get_min_value(wav.dtype)) * get_max_value(to_dtype)
    if to_dtype in (np.int16, np.int32):
      # the default seems to be np.fix instead of np.round on wav.astype()
      wav = np.round(wav, 0)
    wav = wav.astype(to_dtype)

  return wav


def normalize_wav(wav: np.ndarray):
  # Mono or stereo is supported.

  if wav.dtype == np.int16 and np.min(wav) == get_min_value(np.int16):
    return wav
  if wav.dtype == np.int32 and np.min(wav) == get_min_value(np.int32):
    return wav

  wav_abs = np.abs(wav)
  max_val = np.max(wav_abs)
  is_div_by_zero = max_val == 0
  max_possible_value = get_max_value(wav.dtype)
  is_already_normalized = max_val == max_possible_value
  # on int16 resulting min wav value would be max. -32767 not -32768 (which would be possible with wavfile.write) maybe later TODO

  if not is_already_normalized and not is_div_by_zero:
    orig_dtype = wav.dtype
    wav_float = wav.astype(np.float32)
    wav_float = wav_float * max_possible_value / max_val
    if orig_dtype == np.int16 or orig_dtype == np.int32:
      # the default seems to be np.fix instead of np.round on wav.astype()
      # 32766.998 gets 32767 because of float unaccuracy
      wav_float = np.round(wav_float, 0)
    wav = wav_float.astype(orig_dtype)

  assert np.max(np.abs(wav)) == max_possible_value or np.max(np.abs(wav)) == 0
  assert not is_overamp(wav)

  return wav


def concatenate_audios(audios: List[np.ndarray], sentence_pause_s: float, sampling_rate: int) -> np.array:
  sentence_pause_samples_count = get_sample_count(sampling_rate, sentence_pause_s)
  return concatenate_audios_core(audios, sentence_pause_samples_count)


def concatenate_audios_core(audios: List[np.ndarray], sentence_pause_samples_count: int = 0) -> np.ndarray:
  """Concatenates the np.ndarray list on the last axis."""
  if len(audios) == 1:
    cpy = np.array(audios[0])
    return cpy

  pause_shape = list(audios[0].shape)
  pause_shape[-1] = sentence_pause_samples_count
  sentence_pause_samples = np.zeros(tuple(pause_shape))
  conc = []
  for audio in audios[:-1]:
    conc.append(audio)
    conc.append(sentence_pause_samples)
  conc.append(audios[-1])
  output = np.concatenate(tuple(conc), axis=-1)
  return output


def get_duration_s_samples(samples: int, sampling_rate: int) -> float:
  duration = samples / sampling_rate
  return duration


def get_duration_s_file(wav_path) -> float:
  sampling_rate, wav = read(wav_path)
  return get_duration_s(wav, sampling_rate)


# TODO: does not really work that check
def is_overamp(wav: np.ndarray) -> bool:
  lowest_value = get_min_value(wav.dtype)
  highest_value = get_max_value(wav.dtype)
  wav_min = np.min(wav)
  wav_max = np.max(wav)
  is_overamplified = wav_min < lowest_value or wav_max > highest_value
  return is_overamplified


def get_wav_tensor_segment(wav_tensor: torch.Tensor, segment_length: int) -> torch.Tensor:
  if wav_tensor.size(0) >= segment_length:
    max_audio_start = wav_tensor.size(0) - segment_length
    audio_start = random.randint(0, max_audio_start)
    wav_tensor = wav_tensor[audio_start:audio_start + segment_length]
  else:
    fill_size = segment_length - wav_tensor.size(0)
    wav_tensor = torch.nn.functional.pad(wav_tensor, (0, fill_size), 'constant').data

  return wav_tensor


def align_mels_with_dtw(mel_spec_1: np.ndarray, mel_spec_2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, List[int], List[int]]:
  mel_spec_1, mel_spec_2 = mel_spec_1.T, mel_spec_2.T
  dist, path = fastdtw(mel_spec_1, mel_spec_2, dist=euclidean)
  path_for_mel_spec_1 = list(map(lambda l: l[0], path))
  path_for_mel_spec_2 = list(map(lambda l: l[1], path))
  aligned_mel_spec_1 = mel_spec_1[path_for_mel_spec_1].T
  aligned_mel_spec_2 = mel_spec_2[path_for_mel_spec_2].T
  return aligned_mel_spec_1, aligned_mel_spec_2, dist, path_for_mel_spec_1, path_for_mel_spec_2


def get_msd(dist: float, total_frame_number: int) -> float:
  msd = dist / total_frame_number
  return msd


def plot_melspec_np(mel: np.ndarray, mel_dim_x: int = 16, mel_dim_y: int = 5, factor: int = 1, title: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
  height, width = mel.shape
  width_factor = width / 1000
  fig, axes = plt.subplots(
      nrows=1,
      ncols=1,
      figsize=(mel_dim_x * factor * width_factor, mel_dim_y * factor),
  )

  img = axes.imshow(
      X=mel,
      aspect='auto',
      origin='lower',
      interpolation='none'
  )

  axes.set_yticks(np.arange(0, height, step=5))
  axes.set_xticks(np.arange(0, width, step=50))
  axes.xaxis.set_major_locator(ticker.NullLocator())
  axes.yaxis.set_major_locator(ticker.NullLocator())
  plt.tight_layout()  # font logging occurs here
  figa_core = figure_to_numpy_rgb(fig)

  fig.colorbar(img, ax=axes)
  axes.xaxis.set_major_locator(ticker.AutoLocator())
  axes.yaxis.set_major_locator(ticker.AutoLocator())

  if title is not None:
    axes.set_title(title)
  axes.set_xlabel("Frames")
  axes.set_ylabel("Freq. channel")
  plt.tight_layout()  # font logging occurs here
  figa_labeled = figure_to_numpy_rgb(fig)
  plt.close()

  return figa_core, figa_labeled


def wav_to_float32_tensor(path: str) -> Tuple[torch.Tensor, int]:
  wav, sampling_rate = wav_to_float32(path)
  wav_tensor = torch.FloatTensor(wav)

  return wav_tensor, sampling_rate


def wav_to_float32(path: str) -> Tuple[np.float, int]:
  sampling_rate, wav = read(path)
  wav = convert_wav(wav, np.float32)
  return wav, sampling_rate


def get_max_value(dtype):
  # see wavfile.write() max positive eg. on 16-bit PCM is 32767
  if dtype == np.int16:
    return INT16_MAX

  if dtype == np.int32:
    return INT32_MAX

  if dtype in (np.float32, np.float64):
    return FLOAT32_64_MAX_WAV

  assert False


def get_min_value(dtype):
  if dtype == np.int16:
    return INT16_MIN

  if dtype == np.int32:
    return INT32_MIN

  if dtype == np.float32 or dtype == np.float64:
    return FLOAT32_64_MIN_WAV

  assert False


FLOAT32_64_MIN_WAV = -1.0
FLOAT32_64_MAX_WAV = 1.0
INT16_MIN = np.iinfo(np.int16).min  # -32768 = -(2**15)
INT16_MAX = np.iinfo(np.int16).max  # 32767 = 2**15 - 1
INT32_MIN = np.iinfo(np.int32).min  # -2147483648 = -(2**31)
INT32_MAX = np.iinfo(np.int32).max  # 2147483647 = 2**31 - 1


def mel_to_numpy(mel: torch.Tensor) -> np.ndarray:
  mel = mel.squeeze(0)
  mel = mel.cpu()
  mel_np: np.ndarray = mel.numpy()
  return mel_np
