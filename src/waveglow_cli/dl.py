import shutil
from argparse import ArgumentParser, Namespace
from logging import getLogger
from pathlib import Path

from waveglow import convert_glow, dl_wg
from waveglow.utils import get_pytorch_filename

from waveglow_cli.argparse_helper import parse_path
from waveglow_cli.defaults import DEFAULT_WAVEGLOW, DEFAULT_WAVEGLOW_VERSION
from waveglow_cli.io import get_checkpoints_dir, get_train_dir


def init_download_parser(parser: ArgumentParser) -> None:
  parser.add_argument('checkpoint', type=parse_path)
  parser.add_argument('--version', type=int, default=DEFAULT_WAVEGLOW_VERSION)
  return dl_pretrained


def dl_pretrained(ns: Namespace) -> None:
  logger = getLogger(__name__)

  dl_wg(
    destination=ns.checkpoint,
    version=ns.version
  )

  checkpoint = convert_glow(
    origin=ns.checkpoint,
    destination=ns.checkpoint,
    keep_orig=False
  )

  assert checkpoint == ns.checkpoint

  logger.info(f"Completed. Downloaded to: {ns.checkpoint.absolute()}")

  # if prep_name is not None:
  # merge_dir = get_merged_dir(base_dir, merge_name, create=False)
  # prep_dir = get_prep_dir(merge_dir, prep_name, create=False)
  # wholeset = load_merged_data(merge_dir)
  # save_testset(prep_dir, wholeset)
  # # can be removed
  # save_valset(prep_dir, wholeset)

  # save_prep_settings(train_dir, ttsp_dir=None, merge_name=merge_name, prep_name=prep_name)
