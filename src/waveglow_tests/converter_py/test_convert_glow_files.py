
from waveglow.converter.convert import convert_glow_files
from waveglow.dl_pretrained import download_pretrained_model
from waveglow.utils import get_default_device
from waveglow_tests.globals import *


def test_component():
  device = get_default_device()
  if not DL_PATH.is_file():
    download_pretrained_model(DL_PATH, version=3)
  checkpoint = convert_glow_files(DL_PATH, WG_PATH, device=device, keep_orig=True)
  assert checkpoint.learning_rate == 1e-4
