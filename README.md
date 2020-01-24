# [GPCM](http://github.com/wesselb/gpcm)

[![Build](https://travis-ci.org/wesselb/gpcm.svg?branch=master)](https://travis-ci.org/wesselb/gpcm)
[![Coverage Status](https://coveralls.io/repos/github/wesselb/gpcm/badge.svg?branch=master&service=github)](https://coveralls.io/github/wesselb/gpcm?branch=master)
[![Latest Docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://wesselb.github.io/gpcm)

Implementation of the GPCM and variations

## Installation

Python 3.6 or higher is required.
To begin with, clone and enter the repo.

```bash
git clone https://github.com/wesselb/gpcm
cd gpcm
```

Then make a virtual environment and install the requirements.

```bash
virtualenv -p python3 venv
source venv/bin/activate
pip install -r requirements_dev.txt
```

## Sample From the Prior

```bash
PYTHONPATH=. python experiments/sample.py
```

## Learn a GP With a Known Kernel

```bash
PYTHONPATH=. python experiments/learn_eq.py
```

```bash
PYTHONPATH=. python experiments/learn_smk.py
```
