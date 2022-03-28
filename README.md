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

## Example

```python
import numpy as np

from gpcm import RGPCM

model = RGPCM(window=2, scale=1, n_u=30, t=(0, 10))

# Sample from the prior.
t = np.linspace(0, 10, 100)
k, y = model.sample(t)  # Sampled kernel matrix and sampled noisy function

# Fit model to the sample.
model.fit(t, y)

# Compute the ELBO.
print(model.elbo(t, y))

# Make predictions.
posterior = model.condition(t, y)
mean, var = posterior.predict(t)
```

## Available Models and Approximation Schemes

The following models are available:

| Model | Description |
| - | - |
| `GPCM` | White noise excitation with a smooth filter |
| `CGPCM` | White noise excitation with a smooth causal filter |
| `RGPCM` | Ornstein-Uhlenbeck excitation with a white noise filter |

The simplest way of constructing a model is to set the following keywords:

| Keyword | Description |
| - | - |
| `window` | Largest length scale of signal |
| `scale` | Smallest length scale of signal |
| `t` | Some iterable containing the limits of the inputs of interest |

Example:

```python
from gpcm import RGPCM

model = RGPCM(window=4, scale=0.5, t=(0, 10))
```

The constructors of these models also take in a keyword `scheme`, which can be
set  to one of the following values:

| `scheme` | Description |
| - | - |
| `"structured"` (default) | Structured approximation |
| `"mean-field-ca"` | Mean-field approximation learned by coordinate ascent. This does not learn hyperparameters. |
| `"mean-field-gradient"` | Mean-field approximation learned by gradient-based optimisation |
| `"mean-field-collapsed-gradient"` | Collapsed mean-field approximation learned by gradient-based optimisation |

Example:

```python
from gpcm import RGPCM

model = RGPCM(scheme="mean-field-ca", window=4, scale=0.5, t=(0, 10))
```

## Experiments

### Sample From the Prior

```bash
PYTHONPATH=. python experiments/sample.py
```

### Learn a GP With a Known Kernel

```bash
PYTHONPATH=. python experiments/eq.py
```

```bash
PYTHONPATH=. python experiments/smk.py
```
