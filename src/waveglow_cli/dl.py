from argparse import ArgumentParser, Namespace
from logging import getLogger

from waveglow.converter.convert import convert_glow
from waveglow.dl_pretrained import dl_wg
from waveglow.utils import set_torch_thread_to_max
from waveglow_cli.argparse_helper import parse_device, parse_path
from waveglow_cli.defaults import DEFAULT_DEVICE, DEFAULT_WAVEGLOW_VERSION


def init_download_parser(parser: ArgumentParser) -> None:
  parser.add_argument('checkpoint', type=parse_path)
  parser.add_argument(
    '--ver', type=int, choices=[1, 2, 3, 4, 5], default=DEFAULT_WAVEGLOW_VERSION, help="version")
  parser.add_argument("--device", type=parse_device, default=DEFAULT_DEVICE,
                      help="device used for conversion")
  return dl_pretrained


def dl_pretrained(ns: Namespace) -> None:
  logger = getLogger(__name__)
  set_torch_thread_to_max()

  dl_wg(
    destination=ns.checkpoint,
    version=ns.ver
  )

  checkpoint = convert_glow(
    origin=ns.checkpoint,
    destination=ns.checkpoint,
    device=ns.device,
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
