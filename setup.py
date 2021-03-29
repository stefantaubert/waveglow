from setuptools import setup

setup(
    dependency_links=[
        "git+https://github.com/stefantaubert/audio-utils.git@e3c9398aeebe445a55e54d8cf55f286173d171c0#egg=audio-utils",
        "git+https://github.com/stefantaubert/cmudict-parser.git@5f7c38d98dcae0a462ec7dedb5f4a3b49310bfaf#egg=cmudict-parser",
        "git+https://github.com/stefantaubert/image-utils.git@b4d885a766edca806884d85ede49c594375c12d5#egg=image-utils",
        "git+https://github.com/jasminsternkopf/mel_cepstral_distance.git@fb4916d0b53fea9873b6bd96402db51ad3f614f8#egg=mcd",
        "git+https://github.com/stefantaubert/speech-dataset-parser.git@26aeb590c9e1653a38311bfe1a17c610e181cae3#egg=speech-dataset-parser",
        "git+https://github.com/stefantaubert/speech-dataset-preprocessing.git@37bd650c88ff8b545d3f771ee6cfa8c5eb2dfbd5#egg=speech-dataset-preprocessing",
        "git+https://github.com/stefantaubert/text-utils.git@7962aada2ab6fac7f74f747bbb9c9b31420abeeb#egg=text-utils",
        "git+https://github.com/stefantaubert/tts-preparation.git@1a78f902238bcf7fc7a186ad7047cf39e5d66e1b#egg=tts-preparation",
    ],
    name="waveglow",
    version="1.0.0",
    url="https://github.com/stefantaubert/waveglow.git",
    author="Stefan Taubert",
    author_email="stefan.taubert@posteo.de",
    description="waveglow",
    packages=["waveglow"],
    install_requires=[
        "pandas",
        "matplotlib",
        "tqdm",
        "numpy",
        "Unidecode",
        "torch<=1.7.1",
    ],
)
