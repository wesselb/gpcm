# [GPCM](http://github.com/wesselb/gpcm)

[![CI](https://github.com/wesselb/gpcm/workflows/CI/badge.svg?branch=master)](https://github.com/wesselb/gpcm/actions?query=workflow%3ACI)
[![Coverage Status](https://coveralls.io/repos/github/wesselb/gpcm/badge.svg?branch=master&service=github)](https://coveralls.io/github/wesselb/gpcm?branch=master)
[![Latest Docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://wesselb.github.io/gpcm)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Implementation of the GPCM and variations

## Installation

See [the instructions here](https://gist.github.com/wesselb/4b44bf87f3789425f96e26c4308d0adc).

Then clone and enter the repo.

```bash
git clone https://github.com/wesselb/gpcm
cd gpcm
```

Finally, make a virtual environment and install the package.

```bash
virtualenv -p python3 venv
source venv/bin/activate
pip install -r requirements.txt -e .
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
