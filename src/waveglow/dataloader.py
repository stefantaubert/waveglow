import random
from logging import Logger
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from waveglow.audio_utils import get_wav_tensor_segment
from waveglow.hparams import HParams
from waveglow.taco_stft import TacotronSTFT
from waveglow.typing import Entries
from waveglow.utils import try_copy_to


class MelLoader(Dataset):
  """
  This is the main class that calculates the spectrogram and returns the
  spectrogram, audio pair.
  """

  def __init__(self, prepare_ds_data: Entries, hparams: HParams, device: torch.device, logger: Logger):
    self.device = device
    self.taco_stft = TacotronSTFT(hparams, device, logger=logger)
    self.hparams = hparams
    self._logger = logger

    data = prepare_ds_data.copy()
    random.seed(hparams.seed)
    random.shuffle(data)

    wav_paths = {}
    for i, values in enumerate(data):
      wav_paths[i] = values.wav_absolute_path
    self.wav_paths = wav_paths

    if hparams.cache_wavs:
      self._logger.info("Loading wavs into memory...")
      cache = {}
      for i, wav_path in tqdm(wav_paths.items()):
        cache[i] = self.taco_stft.get_wav_tensor_from_file(wav_path)
      self._logger.info("Done")
      self.cache = cache

  def __getitem__(self, index):
    if self.hparams.cache_wavs:
      wav_tensor = self.cache[index].clone().detach()
    else:
      wav_tensor = self.taco_stft.get_wav_tensor_from_file(self.wav_paths[index])
    wav_tensor = get_wav_tensor_segment(wav_tensor, self.hparams.segment_length)
    mel_tensor = self.taco_stft.get_mel_tensor(wav_tensor)
    mel_tensor = try_copy_to(mel_tensor, self.device)
    wav_tensor = try_copy_to(wav_tensor, self.device)
    return (mel_tensor, wav_tensor)

  def __len__(self):
    return len(self.wav_paths)


def parse_batch(batch) -> Tuple[torch.autograd.Variable, torch.autograd.Variable]:
  mel, audio = batch
  mel = torch.autograd.Variable(mel)
  audio = torch.autograd.Variable(audio)
  return (mel, audio), (mel, audio)


def prepare_trainloader(hparams: HParams, trainset: Entries, device: torch.device, logger: Logger) -> None:
  # logger.info(
  #   f"Duration trainset {trainset.total_duration_s / 60:.2f}m / {trainset.total_duration_s / 60 / 60:.2f}h")

  trn = MelLoader(trainset, hparams, device, logger)

  train_sampler = None
  shuffle = False  # maybe set to true bc taco is also true

  train_loader = DataLoader(
    dataset=trn,
    num_workers=0,
    shuffle=shuffle,
    sampler=train_sampler,
    batch_size=hparams.batch_size,
    pin_memory=False,
    drop_last=True
  )

  return train_loader


def prepare_valloader(hparams: HParams, valset: Entries, device: torch.device, logger: Logger) -> None:
  # logger.info(
  #   f"Duration valset {valset.total_duration_s / 60:.2f}m / {valset.total_duration_s / 60 / 60:.2f}h")

  val = MelLoader(valset, hparams, device, logger)
  val_sampler = None

  val_loader = DataLoader(
    dataset=val,
    sampler=val_sampler,
    num_workers=0,
    shuffle=False,
    batch_size=hparams.batch_size,
    pin_memory=False
  )

  return val_loader
