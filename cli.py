import os
from argparse import ArgumentParser

from waveglow.app.defaults import DEFAULT_WAVEGLOW
from waveglow.app.dl import dl_pretrained
from waveglow.app.inference import infer
from waveglow.app.training import continue_training, start_new_training
from waveglow.app.validation import validate
from waveglow.utils import split_hparams_string

BASE_DIR_VAR = "base_dir"


def init_train_parser(parser: ArgumentParser):
  parser.add_argument('--train_name', type=str, required=True)
  parser.add_argument('--merge_name', type=str, required=True)
  parser.add_argument('--prep_name', type=str, required=True)
  parser.add_argument('--custom_hparams', type=str)
  parser.add_argument('--warm_start_train_name', type=str)
  parser.add_argument('--warm_start_checkpoint', type=int)
  return train_cli


def train_cli(**args):
  args["custom_hparams"] = split_hparams_string(args["custom_hparams"])
  start_new_training(**args)


def init_continue_train_parser(parser: ArgumentParser):
  parser.add_argument('--train_name', type=str, required=True)
  parser.add_argument('--custom_hparams', type=str)
  return continue_train_cli


def continue_train_cli(**args):
  args["custom_hparams"] = split_hparams_string(args["custom_hparams"])
  continue_training(**args)


def init_validate_parser(parser: ArgumentParser):
  parser.add_argument('--train_name', type=str, required=True)
  parser.add_argument('--entry_id', type=int, help="Utterance id or nothing if random")
  parser.add_argument('--speaker', type=str, help="ds_name,speaker_name")
  parser.add_argument('--ds', type=str, help="Choose if validation- or testset should be taken.",
                      choices=["val", "test"], default="val")
  parser.add_argument('--custom_checkpoint', type=int)
  parser.add_argument("--denoiser_strength", default=0.00, type=float, help='Removes model bias.')
  parser.add_argument("--sigma", type=float, default=0.666)
  parser.add_argument('--custom_hparams', type=str)
  return validate_cli


def validate_cli(**args):
  args["custom_hparams"] = split_hparams_string(args["custom_hparams"])
  validate(**args)


def init_inference_parser(parser: ArgumentParser):
  parser.add_argument('--train_name', type=str, required=True)
  parser.add_argument('--wav_path', type=str, required=True)
  parser.add_argument('--custom_checkpoint', type=int)
  parser.add_argument('--sigma', type=float, default=0.666)
  parser.add_argument('--denoiser_strength', type=float, default=0.00)
  parser.add_argument('--custom_hparams', type=str)
  return infer_cli


def infer_cli(**args):
  args["custom_hparams"] = split_hparams_string(args["custom_hparams"])
  infer(**args)


def init_download_parser(parser: ArgumentParser):
  parser.add_argument('--train_name', type=str, default=DEFAULT_WAVEGLOW)
  parser.add_argument('--version', type=int, default=3)
  parser.add_argument('--merge_name', type=str)
  parser.add_argument('--prep_name', type=str)
  return dl_pretrained


def add_base_dir(parser: ArgumentParser):
  assert BASE_DIR_VAR in os.environ.keys()
  base_dir = os.environ[BASE_DIR_VAR]
  parser.set_defaults(base_dir=base_dir)


def _add_parser_to(subparsers, name: str, init_method):
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
  _add_parser_to(subparsers, "infer", init_inference_parser)

  return result


def _process_args(args):
  params = vars(args)
  invoke_handler = params.pop("invoke_handler")
  invoke_handler(**params)


if __name__ == "__main__":
  main_parser = _init_parser()

  received_args = main_parser.parse_args()
  #args = main_parser.parse_args("ljs-text --base_dir=/datasets/models/taco2pt_v2 --mel_name=ljs --ds_name=test_ljs --convert_to_ipa".split())

  _process_args(received_args)
