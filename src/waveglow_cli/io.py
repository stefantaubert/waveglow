import logging
import os
from pathlib import Path
from shutil import copyfile
from typing import Any, Dict

import imageio
import matplotlib.pylab as plt
import numpy as np

from waveglow.audio_utils import float_to_wav, plot_melspec
from waveglow.image_utils import calculate_structual_similarity, stack_images_vertically


def get_train_dir(base_dir: Path, train_name: str) -> None:
  return base_dir / train_name


def save_diff_plot(infer_dir: Path) -> None:
  path1 = infer_dir / f"{infer_dir.parent.name}.png"
  path2 = infer_dir / f"{infer_dir.parent.name}_orig.png"

  old_level = logging.getLogger().level
  logging.getLogger().setLevel(logging.INFO)
  score, diff_img = calculate_structual_similarity(path1, path2)
  path = infer_dir / f"{infer_dir.parent.name}_diff.png"
  imageio.imsave(path, diff_img)
  logging.getLogger().setLevel(old_level)
  return score


def save_v(infer_dir: Path) -> None:
  path1 = infer_dir / f"{infer_dir.parent.name}_orig.png"
  path2 = infer_dir / f"{infer_dir.parent.name}.png"
  path3 = infer_dir / f"{infer_dir.parent.name}_diff.png"
  path = infer_dir / f"{infer_dir.parent.name}_v.png"
  stack_images_vertically([path1, path2, path3], path)


# region Training
# _train_csv = "train.csv"
# _test_csv = "test.csv"
# _val_csv = "validation.csv"
_settings_json = "settings.json"


def get_train_logs_dir(train_dir: Path) -> Path:
  return train_dir / "logs"


def get_train_log_file(logs_dir: Path) -> Path:
  return logs_dir / "log.txt"


def get_train_checkpoints_log_file(logs_dir: Path) -> None:
  return logs_dir / "log_checkpoints.txt"


def get_checkpoints_dir(train_dir: Path) -> Path:
  return train_dir / "checkpoints"


def get_wav_info_dict(identifier: int, path: Path, sr: int) -> Dict[str, Any]:
  mel_info = {
    "id": identifier,
    "path": str(path),
    "sr": sr,
  }

  return mel_info


def get_wav_out_dict(name: str, root_dir: Path, wav_info_dict: Dict[str, Any]) -> Dict[str, Any]:
  info_json = {
    "name": name,
    "root_dir": str(root_dir),
    "wavs": wav_info_dict,
  }

  return info_json

# def save_trainset(train_dir: Path, dataset: Entries):
#   path = train_dir / _train_csv
#   dataset.save(path)


# def load_trainset(train_dir: Path) -> Entries:
#   path = train_dir / _train_csv
#   return Entries.load(PreparedData, path)


# def save_testset(train_dir: Path, dataset: Entries):
#   path = train_dir / _test_csv
#   dataset.save(path)


# def load_testset(train_dir: Path) -> Entries:
#   path = train_dir / _test_csv
#   return Entries.load(PreparedData, path)


# def save_valset(train_dir: Path, dataset: Entries):
#   path = train_dir / _val_csv
#   dataset.save(path)


# def load_valset(train_dir: Path) -> Entries:
#   path = train_dir / _val_csv
#   return Entries.load(PreparedData, path)


# def load_prep_settings(train_dir: Path) -> Tuple[Path, str, str]:
#   path = train_dir / _settings_json
#   res = parse_json(path)
#   return Path(res["ttsp_dir"]), res["merge_name"], res["prep_name"]


# def save_prep_settings(train_dir: Path, ttsp_dir: Path, merge_name: Optional[str], prep_name: Optional[str]) -> None:
#   settings = {
#     "ttsp_dir": str(ttsp_dir),
#     "merge_name": merge_name,
#     "prep_name": prep_name,
#   }
#   path = train_dir / _settings_json
#   save_json(path, settings)


# def split_dataset(prep_dir: Path, train_dir: Path, test_size: float = 0.01, validation_size: float = 0.05, split_seed: int = 1234):
#   wholeset = load_filelist(prep_dir)
#   trainset, testset, valset = split_prepared_data_train_test_val(
#     wholeset, test_size=test_size, validation_size=validation_size, seed=split_seed, shuffle=True)
#   save_trainset(train_dir, trainset)
#   save_testset(train_dir, testset)
#   save_valset(train_dir, valset)
#   return trainset, valset

# endregion

# region Inference


def get_inference_root_dir(train_dir: Path) -> None:
  return train_dir / "inference"


def save_infer_wav(infer_dir: Path, sampling_rate: int, wav: np.ndarray) -> None:
  path = infer_dir / f"{infer_dir.parent.name}.wav"
  float_to_wav(wav, path, sample_rate=sampling_rate)


def save_infer_plot(infer_dir: Path, mel: np.ndarray) -> None:
  plot_melspec(mel, title=infer_dir.parent.name)
  path = infer_dir / f"{infer_dir.parent.name}.png"
  plt.savefig(path, bbox_inches='tight')
  plt.close()
  return path

# endregion

# region Validation


def _get_validation_root_dir(train_dir: Path) -> Path:
  return train_dir / "validation"


# def get_val_dir(train_dir: Path, entry: PreparedData, iteration: int) -> Path:
#   subdir_name = f"{datetime.datetime.now():%Y-%m-%d,%H-%M-%S},id={entry.entry_id},speaker={entry.speaker_id},it={iteration}"
#   return _get_validation_root_dir(train_dir) / subdir_name


def save_val_plot(val_dir: Path, mel) -> None:
  parent_dir = val_dir.parent.name
  plot_melspec(mel, title=parent_dir)
  path = val_dir / f"{parent_dir}.png"
  plt.savefig(path, bbox_inches='tight')
  plt.close()


def save_val_orig_plot(val_dir: Path, mel) -> None:
  parent_dir = val_dir.parent.name
  plot_melspec(mel, title=parent_dir)
  path = val_dir / f"{parent_dir}_orig.png"
  plt.savefig(path, bbox_inches='tight')
  plt.close()


def save_val_comparison(val_dir: Path) -> None:
  parent_dir = val_dir.parent.name
  path1 = val_dir / f"{parent_dir}_orig.png"
  path2 = val_dir / f"{parent_dir}.png"
  assert os.path.exists(path1)
  assert os.path.exists(path2)
  path = val_dir / f"{parent_dir}_comp.png"
  stack_images_vertically([path1, path2], path)


def get_val_wav_path(val_dir: Path) -> Path:
  path = val_dir / f"{val_dir.parent.name}.wav"
  return path


def save_val_wav(val_dir: Path, sampling_rate: int, wav) -> Path:
  path = get_val_wav_path(val_dir)
  float_to_wav(wav, path, sample_rate=sampling_rate)
  return path


def get_val_orig_wav_path(val_dir: Path) -> Path:
  path = val_dir / f"{val_dir.parent.name}_orig.wav"
  return path


def save_val_orig_wav(val_dir: Path, wav_path: Path) -> None:
  path = get_val_orig_wav_path(val_dir)
  copyfile(wav_path, path)


def get_val_log(val_dir: Path) -> Path:
  return val_dir / f"{val_dir.parent.name}.txt"

# endregion
