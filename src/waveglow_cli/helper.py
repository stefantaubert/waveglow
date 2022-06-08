from argparse import ArgumentParser

from waveglow_cli.argparse_helper import (get_optional, parse_device,
                                          parse_float_between_zero_and_one, parse_non_empty)
from waveglow_cli.defaults import DEFAULT_DENOISER_STRENGTH, DEFAULT_DEVICE, DEFAULT_SIGMA


def add_device_argument(parser: ArgumentParser) -> None:
  parser.add_argument("--device", type=parse_device, metavar="DEVICE", default=DEFAULT_DEVICE,
                      help="use this device, e.g., \"cpu\" or \"cuda:0\"")


def add_hparams_argument(parser: ArgumentParser) -> None:
  parser.add_argument('--custom-hparams', type=get_optional(parse_non_empty),
                      metavar="CUSTOM-HYPERPARAMETERS", default=None, help="custom hyperparameters comma separated")


def add_denoiser_and_sigma_arguments(parser: ArgumentParser) -> None:
  parser.add_argument("--sigma", type=parse_float_between_zero_and_one,
                      default=DEFAULT_SIGMA, help="sigma used for synthesis")
  parser.add_argument("--denoiser-strength", default=DEFAULT_DENOISER_STRENGTH,
                      type=parse_float_between_zero_and_one, help='strength of denoising to remove model bias after synthesis')
