import tempfile
from pathlib import Path

DL_PATH = Path(tempfile.gettempdir()) / "waveglow-test.pt"
WG_PATH = Path(tempfile.gettempdir()) / "waveglow-test-conv.pt"
