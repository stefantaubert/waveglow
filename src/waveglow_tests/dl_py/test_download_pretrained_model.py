from waveglow.dl_pretrained import download_pretrained_model
from waveglow_tests.globals import *


def test_component():
  if not DL_PATH.is_file():
    download_pretrained_model(DL_PATH, version=3)
  assert DL_PATH.is_file()
