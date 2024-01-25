from pathlib import Path

from waveglow.converter.convert import convert_glow_files
from waveglow.dl_pretrained import download_pretrained_model
from waveglow.utils import get_default_device


def test_component():
  device = get_default_device()
  target_path = Path("/tmp/waveglow-test.pt")
  target_path_conv = Path("/tmp/waveglow-test-conv.pt")
  if not target_path.is_file():
    download_pretrained_model(target_path, version=3)
  checkpoint = convert_glow_files(
    target_path, target_path_conv, device=device, keep_orig=True)
  assert checkpoint.learning_rate == 1e-4
