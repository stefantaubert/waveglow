# waveglow-cli

[![PyPI](https://img.shields.io/pypi/v/waveglow-cli.svg)](https://pypi.python.org/pypi/waveglow-cli)
[![PyPI](https://img.shields.io/pypi/pyversions/waveglow-cli.svg)](https://pypi.python.org/pypi/waveglow-cli)
[![MIT](https://img.shields.io/github/license/stefantaubert/waveglow.svg)](https://github.com/stefantaubert/waveglow/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/wheel/waveglow-cli.svg)](https://pypi.python.org/pypi/waveglow-cli)
[![PyPI](https://img.shields.io/pypi/implementation/waveglow-cli.svg)](https://pypi.python.org/pypi/waveglow-cli)
[![PyPI](https://img.shields.io/github/commits-since/stefantaubert/waveglow/latest/master.svg)](https://pypi.python.org/pypi/waveglow-cli)

Command-line interface (CLI) to train WaveGlow using .wav files.

## Features

- train/synthesize on CPU or GPU
- download pre-trained models by Nvidia

## Installation

```sh
pip install waveglow-cli --user
```

## Usage

```txt
usage: waveglow-cli [-h] [-v] {download,train,continue-train,validate,synthesize} ...

This program trains WaveGlow.

positional arguments:
  {download,train,continue-train,validate,synthesize}
                     description
    download         download pre-trained checkpoint from Nvidia
    train            start training
    continue-train   continue training
    validate         validate checkpoint(s)
    synthesize       synthesize mel-spectrograms into an audio signal

optional arguments:
  -h, --help         show this help message and exit
  -v, --version      show program's version number and exit
```

## Pretrained Models

- [LJS-v3-580000](https://tuc.cloud/index.php/s/yBRaWz5oHrFwigf): Adapted model trained on LJ Speech dataset by [Nvidia](https://api.ngc.nvidia.com/v2/models/nvidia/waveglow_ljs_256channels/versions/3/files/waveglow_256channels_ljs_v3.pt).

## Audio Example

"The North Wind and the Sun were disputing which was the stronger, when a traveler came along wrapped in a warm cloak." [Listen here](https://tuc.cloud/index.php/s/gzaYDNKinHw6GCz) (headphones recommended)

## Dependencies

- `torch`
- `mel-cepstral-distance>=0.0.1`
- `pandas`
- `librosa`
- `plotly`
- `scikit-image`
- `matplotlib`
- `scikit-learn`
- `tqdm`
- `wget`
- `gdown`
- `Unidecode`
- `Pillow`
- `fastdtw`
- `numpy`
- `scipy`
- `ordered_set>=4.1.0`

## Roadmap

- Outsource method to convert audio files to mel-spectrograms before training
- Improve logging
- Add more audio examples
- Adding tests

## License

MIT License

## Acknowledgments

Model code adapted from [Nvidia](https://github.com/NVIDIA/waveglow).

Paper:

- [Waveglow: A Flow-based Generative Network for Speech Synthesis](https://ieeexplore.ieee.org/document/8683143)

Funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) – Project-ID 416228727 – CRC 1410

## Citation

If you want to cite this repo, you can use this BibTeX-entry:

```bibtex
@misc{tsw22,
  author = {Taubert, Stefan},
  title = {waveglow-cli},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/stefantaubert/waveglow}}
}
```
