from pathlib import Path

from waveglow.dl_pretrained import download_pretrained_model


def test_component():
  target_path = Path("/tmp/waveglow-test.pt")
  if not target_path.is_file():
    download_pretrained_model(target_path, version=3)
  assert target_path.is_file()
