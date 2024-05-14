# Introduction

TODO

# Installation instruction

## Installation

```
git clone https://github.com/arthurdehgan/meegnet.git
cd meegnet
pip install -r requirements.txt
pip install .
```

The package will be added to pipy in the future (removing the need to clone the repo)

## Install from scratch

We recommend creating a fresh python environment in order to use this packages scrips:
```
VENV_PATH="/path/to/environment/"
python -m venv $VENV_PATH\meegnet
source $VENV_PATH\meegnet/bin/activate
```

Once the environment is created and activated, it is possible to install as instructed in the previous section.

# Features

WIP

# How to use (WIP)

## scripts/prepare_data.py

This script is intended to work for the camcan MEG dataset with maxfilter
and transform to default common space (mf_transdef) data in BIDS format.

The Camcan dataset is not open access and you need to send a request on the
websitde in order to get access (https://camcan-archive.mrc-cbu.cam.ac.uk/dataaccess/).

This script assumes a copy of the cc700 and dataman folders to a data path parsed
through the argparser.

