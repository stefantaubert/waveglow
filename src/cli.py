import os
from argparse import ArgumentParser
from pathlib import Path

from general_utils import split_hparams_string, split_int_set_str, split_string

from waveglow.app import (DEFAULT_DENOISER_STRENGTH, DEFAULT_SENTENCE_PAUSE_S,
                          DEFAULT_SIGMA, DEFAULT_WAVEGLOW,
                          DEFAULT_WAVEGLOW_VERSION, continue_train,
                          dl_pretrained, infer, train, validate,
                          validate_generic)
from waveglow.app.defaults import DEFAULT_READ_MEL_INFO_PATH, DEFAULT_SEED
from waveglow.app.inference import infer_parse_json
from waveglow.app.inference_v2 import infer_mels

BASE_DIR_VAR = "base_dir"


def init_train_parser(parser: ArgumentParser) -> None:
  parser.add_argument('--ttsp_dir', type=Path, required=True)
  parser.add_argument('--train_name', type=str, required=True)
  parser.add_argument('--merge_name', type=str, required=True)
  parser.add_argument('--prep_name', type=str, required=True)
  parser.add_argument('--custom_hparams', type=str)
  parser.add_argument('--warm_start_train_name', type=str)
  parser.add_argument('--warm_start_checkpoint', type=int)
  return train_cli


def train_cli(**args) -> None:
  args["custom_hparams"] = split_hparams_string(args["custom_hparams"])
  train(**args)


def init_continue_train_parser(parser: ArgumentParser) -> None:
  parser.add_argument('--train_name', type=str, required=True)
  parser.add_argument('--custom_hparams', type=str)
  return continue_train_cli


def continue_train_cli(**args) -> None:
  args["custom_hparams"] = split_hparams_string(args["custom_hparams"])
  continue_train(**args)


def init_validate_generic_parser(parser: ArgumentParser) -> None:
  parser.add_argument('--train_name', type=str, required=True)
  parser.add_argument('--ttsp_dir', type=Path, required=True)
  parser.add_argument('--merge_name', type=str, required=True)
  parser.add_argument('--prep_name', type=str, required=True)
  parser.add_argument('--entry_ids', type=str, help="Utterance id or nothing if random")
  parser.add_argument('--ds', type=str, help="Choose if validation- or testset should be taken.",
                      choices=["val", "test", "total"], default="val")
  parser.add_argument('--custom_checkpoints', type=str)
  parser.add_argument("--denoiser_strength", default=DEFAULT_DENOISER_STRENGTH,
                      type=float, help='Removes model bias.')
  parser.add_argument("--sigma", type=float, default=DEFAULT_SIGMA)
  parser.add_argument('--custom_hparams', type=str)
  parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
  parser.add_argument('--full_run', action="store_true")
  return validate_generic_cli


def validate_generic_cli(**args) -> None:
  args["entry_ids"] = split_int_set_str(args["entry_ids"])
  args["custom_checkpoints"] = split_int_set_str(args["custom_checkpoints"])
  args["custom_hparams"] = split_hparams_string(args["custom_hparams"])
  validate_generic(**args)


def init_validate_parser(parser: ArgumentParser) -> None:
  parser.add_argument('--train_name', type=str, required=True)
  parser.add_argument('--entry_ids', type=str, help="Utterance id or nothing if random")
  parser.add_argument('--ds', type=str, help="Choose if validation- or testset should be taken.",
                      choices=["val", "test"], default="val")
  parser.add_argument('--custom_checkpoints', type=str)
  parser.add_argument("--denoiser_strength", default=DEFAULT_DENOISER_STRENGTH,
                      type=float, help='Removes model bias.')
  parser.add_argument("--sigma", type=float, default=DEFAULT_SIGMA)
  parser.add_argument('--custom_hparams', type=str)
  parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
  parser.add_argument('--full_run', action="store_true")
  return validate_cli


def validate_cli(**args) -> None:
  args["entry_ids"] = split_int_set_str(args["entry_ids"])
  args["custom_checkpoints"] = split_int_set_str(args["custom_checkpoints"])
  args["custom_hparams"] = split_hparams_string(args["custom_hparams"])
  validate(**args)


def init_inference_parse_json_parser(parser: ArgumentParser) -> None:
  parser.add_argument('--train_name', type=str, required=True)
  parser.add_argument('--json_path', type=str, default=DEFAULT_READ_MEL_INFO_PATH)
  parser.add_argument('--custom_checkpoint', type=int)
  parser.add_argument('--sigma', type=float, default=DEFAULT_SIGMA)
  parser.add_argument('--denoiser_strength', type=float, default=DEFAULT_DENOISER_STRENGTH)
  parser.add_argument('--sentence_pause_s', type=float, default=DEFAULT_SENTENCE_PAUSE_S)
  parser.add_argument('--custom_hparams', type=str)
  parser.add_argument('--no_concatenation', action="store_true")
  parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
  parser.add_argument('--fast', action="store_true")
  return infer_parse_json_cli


def infer_parse_json_cli(**args) -> None:
  args["custom_hparams"] = split_hparams_string(args["custom_hparams"])
  infer_parse_json(**args)


def init_inference_parser(parser: ArgumentParser) -> None:
  parser.add_argument('--train_name', type=str, required=True)
  parser.add_argument('--mel_paths', type=str, required=True)
  parser.add_argument('--sampling_rate', type=int, default=22050)
  parser.add_argument('--custom_checkpoint', type=int)
  parser.add_argument('--sigma', type=float, default=DEFAULT_SIGMA)
  parser.add_argument('--denoiser_strength', type=float, default=DEFAULT_DENOISER_STRENGTH)
  parser.add_argument('--sentence_pause_s', type=float, default=DEFAULT_SENTENCE_PAUSE_S)
  parser.add_argument('--custom_hparams', type=str)
  parser.add_argument('--no_concatenation', action="store_true")
  parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
  parser.add_argument('--fast', action="store_true")
  return infer_cli


def infer_cli(**args) -> None:
  args["custom_hparams"] = split_hparams_string(args["custom_hparams"])
  args["mel_paths"] = split_string(args["mel_paths"])
  infer(**args)


def init_download_parser(parser: ArgumentParser) -> None:
  parser.add_argument('--train_name', type=str, default=DEFAULT_WAVEGLOW)
  parser.add_argument('--version', type=int, default=DEFAULT_WAVEGLOW_VERSION)
  return dl_pretrained


def init_inference_v2_parser(parser: ArgumentParser) -> None:
  parser.add_argument('checkpoint', type=Path)
  parser.add_argument('directory', type=Path)
  parser.add_argument('--sigma', type=float, default=1.0)
  parser.add_argument('--denoiser-strength', type=float, default=0.0005)
  parser.add_argument('--custom-seed', type=int, default=None)
  parser.add_argument('--batch-size', type=int, default=64)
  parser.add_argument('--include-stats', action="store_true")
  parser.add_argument('-out', '--output-directory', type=Path)
  parser.add_argument('-o', '--overwrite', action="store_true")
  return infer_mels


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
