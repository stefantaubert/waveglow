from logging import getLogger
from pathlib import Path

import gdown
import wget

# src: https://ngc.nvidia.com/catalog/models/nvidia:waveglow_ljs_256channels


def dl_wg(destination: Path, version: int) -> None:
  logger = getLogger(__name__)
  logger.info(f"Downloading pretrained waveglow model v{version} from Nvida...")

  if version == 1:
    _dl_v1(destination)
  elif version == 2:
    _dl_v2(destination)
  elif version == 3:
    _dl_v3(destination)
  else:
    assert False
  logger.info("Done.")


def _dl_v3(destination: Path) -> None:
  download_url = "https://api.ngc.nvidia.com/v2/models/nvidia/waveglow_ljs_256channels/versions/3/files/waveglow_256channels_ljs_v3.pt"
  destination.parent.mkdir(parents=True, exist_ok=True)
  wget.download(download_url, destination)


def _dl_v2(destination: Path) -> None:
  download_url = "https://api.ngc.nvidia.com/v2/models/nvidia/waveglow_ljs_256channels/versions/2/files/waveglow_256channels_ljs_v2.pt"
  destination.parent.mkdir(parents=True, exist_ok=True)
  wget.download(download_url, destination)


def _dl_v1(destination: Path) -> None:
  # Download waveglow_universal_256channels_v5.pt (644M)
  download_url = "https://drive.google.com/uc?id=1rpK8CzAAirq9sWZhe9nlfvxMF1dRgFbF"
  destination.parent.mkdir(parents=True, exist_ok=True)
  gdown.download(download_url, destination)
