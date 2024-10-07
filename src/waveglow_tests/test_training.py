import os
import wave
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Union

import numpy as np

from waveglow.train import train
from waveglow_cli.defaults import DEFAULT_DEVICE
from waveglow_cli.parser import load_dataset


def test_component():
  with TemporaryDirectory(prefix="waveglow.test_training") as tmp_dir:
    base_dir = Path(tmp_dir)
    trn = base_dir / "trn"
    val = base_dir / "val"
    ckp = base_dir / "checkpoints"
    ckp.mkdir()

    generate_random_audio_dataset(trn)
    generate_random_audio_dataset(val)

    trainset = load_dataset(trn)
    valset = load_dataset(val)

    train(None, base_dir / "logs", trainset, valset, ckp, None, None, DEFAULT_DEVICE)


def generate_random_audio_dataset(output_folder: Union[str, os.PathLike]) -> None:
  """
  Generates a dataset of random audio files in WAV format.

  Parameters:
  output_folder (str or PathLike): The folder where the audio files will be saved.

  This function generates 50 random audio files, each 7 seconds long, 
  with a sampling rate of 22,050 Hz, and saves them in the specified folder.
  """

  num_files = 10
  duration_seconds = 7
  sampling_rate = 22050

  if not os.path.exists(output_folder):
    os.makedirs(output_folder)

  def create_random_wav(filename: str, duration: int, sampling_rate: int) -> None:
    """
    Creates a random audio file (white noise) and saves it as a WAV file.

    Parameters:
    filename (str): The path where the WAV file will be saved.
    duration (int): The duration of the audio in seconds.
    sampling_rate (int): The sampling rate of the audio.
    """
    num_samples = int(duration * sampling_rate)

    # Generate random audio data (white noise)
    audio_data = np.random.uniform(low=-1.0, high=1.0, size=num_samples)

    # Convert the audio data to int16 format
    audio_data = np.int16(audio_data * 32767)

    with wave.open(filename, 'w') as wav_file:
      wav_file.setnchannels(1)  # Mono audio
      wav_file.setsampwidth(2)  # 2 bytes per sample
      wav_file.setframerate(sampling_rate)
      wav_file.writeframes(audio_data.tobytes())

  # Generate the audio files
  for i in range(num_files):
    filename = os.path.join(output_folder, f'random_audio_{i+1}.wav')
    create_random_wav(filename, duration_seconds, sampling_rate)

  print(f'{num_files} random audio files created in "{output_folder}".')
