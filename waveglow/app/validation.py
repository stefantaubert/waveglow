import os
from typing import Dict, Optional

from tts_preparation import (get_merged_dir, get_prep_dir,
                             load_merged_speakers_json, load_testset,
                             load_valset)
from waveglow.app.io import (get_checkpoints_dir, get_train_dir, get_val_dir,
                             get_val_log, load_prep_settings, save_diff_plot,
                             save_v, save_val_orig_plot, save_val_orig_wav,
                             save_val_plot, save_val_wav)
from waveglow.core.inference import infer
from waveglow.core.model_checkpoint import CheckpointWaveglow
from waveglow.utils import get_custom_or_last_checkpoint, prepare_logger


def validate(base_dir: str, train_name: str, entry_id: Optional[int] = None, speaker: Optional[str] = None, ds: str = "val", custom_checkpoint: Optional[int] = None, sigma: float = 0.666, denoiser_strength: float = 0.00, custom_hparams: Optional[Dict[str, str]] = None):
  train_dir = get_train_dir(base_dir, train_name, create=False)
  assert os.path.isdir(train_dir)

  merge_name, prep_name = load_prep_settings(train_dir)
  merge_dir = get_merged_dir(base_dir, merge_name, create=False)
  prep_dir = get_prep_dir(merge_dir, prep_name, create=False)

  if ds == "val":
    data = load_valset(prep_dir)
  elif ds == "test":
    data = load_testset(prep_dir)
  else:
    raise Exception()

  speaker_id: Optional[int] = None
  if speaker is not None:
    speakers = load_merged_speakers_json(merge_dir)
    speaker_id = speakers.get_id(speaker)

  entry = data.get_for_validation(entry_id, speaker_id)

  checkpoint_path, iteration = get_custom_or_last_checkpoint(
    get_checkpoints_dir(train_dir), custom_checkpoint)
  val_dir = get_val_dir(train_dir, entry, iteration)

  logger = prepare_logger(get_val_log(val_dir))
  logger.info(f"Validating {entry.wav_path}...")

  checkpoint = CheckpointWaveglow.load(checkpoint_path, logger)

  wav, wav_sr, wav_mel, orig_mel = infer(
    wav_path=entry.wav_path,
    denoiser_strength=denoiser_strength,
    sigma=sigma,
    checkpoint=checkpoint,
    custom_hparams=custom_hparams,
    logger=logger
  )

  save_val_wav(val_dir, wav_sr, wav)
  save_val_plot(val_dir, wav_mel)
  save_val_orig_wav(val_dir, entry.wav_path)
  save_val_orig_plot(val_dir, orig_mel)
  score = save_diff_plot(val_dir)
  save_v(val_dir)

  logger.info(f"Imagescore: {score*100}%")
  logger.info(f"Saved output to: {val_dir}")


if __name__ == "__main__":

  # validate(
  #   base_dir="/datasets/models/taco2pt_v5",
  #   train_name="pretrained",
  #   entry_id=865
  # )

  # validate(
  #   base_dir="/datasets/models/taco2pt_v5",
  #   train_name="pretrained_v2",
  #   entry_id=865
  # )

  validate(
    base_dir="/datasets/models/taco2pt_v5",
    train_name="pretrained_v3",
    entry_id=865,
    custom_hparams={
      "sampling_rate": 44100
    }
  )
