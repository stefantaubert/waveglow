# waveglow-cli

[![PyPI](https://img.shields.io/pypi/v/waveglow-cli.svg)](https://pypi.python.org/pypi/waveglow-cli)
[![PyPI](https://img.shields.io/pypi/pyversions/waveglow-cli.svg)](https://pypi.python.org/pypi/waveglow-cli)
[![MIT](https://img.shields.io/github/license/stefantaubert/waveglow.svg)](https://github.com/stefantaubert/waveglow/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/wheel/waveglow-cli.svg)](https://pypi.python.org/pypi/waveglow-cli)
[![PyPI](https://img.shields.io/pypi/implementation/waveglow-cli.svg)](https://pypi.python.org/pypi/waveglow-cli)
[![PyPI](https://img.shields.io/github/commits-since/stefantaubert/waveglow/latest/master.svg)](https://github.com/stefantaubert/waveglow/compare/v0.0.2...master)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10569141.svg)](https://doi.org/10.5281/zenodo.10569141)

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
usage: waveglow-cli [-h] [-v] {download,train,continue-train,validate,synthesize,synthesize-wav} ...

This program trains WaveGlow.

positional arguments:
  {download,train,continue-train,validate,synthesize,synthesize-wav}
                              description
    download                  download pre-trained checkpoint from Nvidia
    train                     start training
    continue-train            continue training
    validate                  validate checkpoint(s)
    synthesize                synthesize mel-spectrograms into an audio signal
    synthesize-wav            synthesize audio file into an audio signal

options:
  -h, --help                  show this help message and exit
  -v, --version               show program's version number and exit
```

## Pretrained Models

- [LJS-v3-580000](https://tuc.cloud/index.php/s/yBRaWz5oHrFwigf): Adapted model trained on LJ Speech dataset by [Nvidia](https://api.ngc.nvidia.com/v2/models/nvidia/waveglow_ljs_256channels/versions/3/files/waveglow_256channels_ljs_v3.pt).

## Audio Example

"The North Wind and the Sun were disputing which was the stronger, when a traveler came along wrapped in a warm cloak." [Listen here](https://tuc.cloud/index.php/s/gzaYDNKinHw6GCz) (headphones recommended)

## Roadmap

- Outsource method to convert audio files to mel-spectrograms before training
- Improve logging
- Add more audio examples
- Adding tests

## Development setup

```sh
# update
sudo apt update
# install Python 3.8-3.11 for ensuring that tests can be run
sudo apt install python3-pip \
  python3.8 python3.8-dev python3.8-distutils python3.8-venv \
  python3.9 python3.9-dev python3.9-distutils python3.9-venv \
  python3.10 python3.10-dev python3.10-distutils python3.10-venv \
  python3.11 python3.11-dev python3.11-distutils python3.11-venv
# install pipenv for creation of virtual environments
python3.8 -m pip install pipenv --user

# check out repo
git clone https://github.com/stefantaubert/waveglow.git
cd waveglow
# create virtual environment
python3.8 -m pipenv install --dev
```

## Running the tests

```sh
# first install the tool like in "Development setup"
# then, navigate into the directory of the repo (if not already done)
cd waveglow
# activate environment
python3.8 -m pipenv shell
# run tests
tox
```

Final lines of test result output:

```log
py38: commands succeeded
py39: commands succeeded
py310: commands succeeded
py311: commands succeeded
congratulations :)
```

## License

MIT License

## Acknowledgments

Model code adapted from [Nvidia](https://github.com/NVIDIA/waveglow).

Paper:

- [Waveglow: A Flow-based Generative Network for Speech Synthesis](https://ieeexplore.ieee.org/document/8683143)

Funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) – Project-ID 416228727 – CRC 1410

## Citation

If you want to cite this repo, you can use the BibTeX-entry generated by GitHub (see *About => Cite this repository*).

```txt
Taubert, S. (2024). waveglow-cli (Version 0.0.2) [Computer software]. [https://doi.org/10.5281/zenodo.10569141](https://doi.org/10.5281/zenodo.10569141)
```
