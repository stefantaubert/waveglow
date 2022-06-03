import os
from argparse import ArgumentParser
from pathlib import Path


from waveglow_cli import (DEFAULT_WAVEGLOW, DEFAULT_WAVEGLOW_VERSION,
                          dl_pretrained)
from waveglow_cli.inference import (init_inference_parse_json_parser,
                                    init_inference_parser)
from waveglow_cli.inference_v2 import init_inference_v2_parser
from waveglow_cli.training import init_continue_train_parser, init_train_parser
from waveglow_cli.validation import (init_validate_generic_parser,
                                     init_validate_parser)

BASE_DIR_VAR = "base_dir"


def init_download_parser(parser: ArgumentParser) -> None:
  parser.add_argument('--train_name', type=str, default=DEFAULT_WAVEGLOW)
  parser.add_argument('--version', type=int, default=DEFAULT_WAVEGLOW_VERSION)
  return dl_pretrained


def add_base_dir(parser: ArgumentParser) -> None:
  if BASE_DIR_VAR in os.environ.keys():
    base_dir = Path(os.environ[BASE_DIR_VAR])
    parser.set_defaults(base_dir=base_dir)


def _add_parser_to(subparsers, name: str, init_method) -> None:
  parser = subparsers.add_parser(name, help=f"{name} help")
  invoke_method = init_method(parser)
  parser.set_defaults(invoke_handler=invoke_method)
  add_base_dir(parser)
  return parser


def _init_parser():
  result = ArgumentParser()
  subparsers = result.add_subparsers(help='sub-command help')

  _add_parser_to(subparsers, "download", init_download_parser)
  _add_parser_to(subparsers, "train", init_train_parser)
  _add_parser_to(subparsers, "continue-train", init_continue_train_parser)
  _add_parser_to(subparsers, "validate", init_validate_parser)
  _add_parser_to(subparsers, "validate-generic", init_validate_generic_parser)
  _add_parser_to(subparsers, "infer", init_inference_parser)
  _add_parser_to(subparsers, "infer-mels", init_inference_v2_parser)
  _add_parser_to(subparsers, "infer-json", init_inference_parse_json_parser)

  return result


def _process_args(args) -> None:
  params = vars(args)
  if "invoke_handler" in params:
    invoke_handler = params.pop("invoke_handler")
    invoke_handler(**params)


if __name__ == "__main__":
  main_parser = _init_parser()

  received_args = main_parser.parse_args()
  #args = main_parser.parse_args("ljs-text --base_dir=/datasets/models/taco2pt_v2 --mel_name=ljs --ds_name=test_ljs --convert_to_ipa".split())

  _process_args(received_args)
