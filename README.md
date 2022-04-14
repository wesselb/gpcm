# [GPCM](http://github.com/wesselb/gpcm)

[![CI](https://github.com/wesselb/gpcm/workflows/CI/badge.svg?branch=master)](https://github.com/wesselb/gpcm/actions?query=workflow%3ACI)
[![Coverage Status](https://coveralls.io/repos/github/wesselb/gpcm/badge.svg?branch=master&service=github)](https://coveralls.io/github/wesselb/gpcm?branch=master)
[![Latest Docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://wesselb.github.io/gpcm)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Implementation of the GPCM and variations

Citation:

```
@inproceedings{Bruinsma:2022:Modelling_Non-Smooth_Signals_With_Complex,
    title = {Modelling Non-Smooth Signals With Complex Spectral Structure},
    year = {2022},
    author = {Wessel P. Bruinsma and Martin Tegn{\' e}r and Richard E. Turner},
    booktitle = {Proceedings of the 25th International Conference on Artificial Intelligence and Statistics},
    series = {Proceedings of Machine Learning Research},
    publisher = {PMLR},
    eprint = {https://arxiv.org/abs/2203.06997},
}
```

Contents:

- [Installation](#installation)
- [Example](#example)
- [Available Models and Inference Schemes](#available-models-and-approximation-schemes)
- [Making Predictions](#make-predictions)
- [Sample Experiments](#sample-experiments)
- [Reproduce Experiments From the Paper](#reproduce-experiments-from-the-paper)

## Installation

See [the instructions here](https://gist.github.com/wesselb/4b44bf87f3789425f96e26c4308d0adc).
Then simply

```
pip install gpcm
```

If you have a GPU available, it is recommended that you use a GPU-accelerated version 
of JAX.

## Example

```python
import numpy as np

from gpcm import RGPCM

model = RGPCM(window=2, scale=1, n_u=30, t=(0, 10))

# Sample from the prior.
t = np.linspace(0, 10, 100)
K, y = model.sample(t)  # Sampled kernel matrix and sampled noisy function

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

Please see the API for a detailed description of the keyword arguments which you can
set.
Amongst these keyword arguments, we highlight the following few which are important:

| Optional Keyword | Description |
| - | - |
| `n_u` | Number of inducing points for the filter |
| `n_z` (`GPCM` and `CGPCM`) | Number of inducing points for the excitation signal |
| `m_max` (`RGPCM`) | Half of the number of variational Fourier features. Set to `n_z // 2` for equal computational expense. |
| `t` | Some iterable containing the limits of the inputs of interest |

The constructors of these models also take in a keyword `scheme`, which can be
set  to one of the following values:

| `scheme` | Description |
| - | - |
| `"structured"` (default) | Structured approximation. Recommended. |
| `"mean-field-ca"` | Mean-field approximation learned by coordinate ascent. This does not learn hyperparameters. |
| `"mean-field-gradient"` | Mean-field approximation learned by gradient-based optimisation |
| `"mean-field-collapsed-gradient"` | Collapsed mean-field approximation learned by gradient-based optimisation |

Example:

```python
from gpcm import RGPCM

model = RGPCM(scheme="mean-field-ca", window=4, scale=0.5, t=(0, 10))
```

## Making Predictions With a Model

The implemented models follow the interface from
[ProbMods](https://github.com/wesselb/probmods).

To begin with, construct a model:

```python
from gpcm import GPCM

model = GPCM(window=4, scale=0.5, t=(0, 10))
```

### Sample From the Paper

Sampling gives back the sampled kernel matrix and the noisy outputs.

```python
K, y = model.sample(t)
```

### Fit the Model to Data

It is recommended that you normalise the data before fitting.

```python
model.fit(t, y)
```

The function `fit` takes in the keyword argument `iters`.
The rule of thumb which you can use is as follows:

| `iters` | Description |
| - | - |
| `5_000` (default) | Reasonable fit |
| `10_000` | Better fit |
| `20_000` | Good fit |
| `30_000` | Pretty good fit |

The function `fit` also takes in the keyword argument `rate`.
The rule of thumb which you can use here is as follows:

| `rate` | Description |
| - | - |
| `5e-2` (default) | Fast learning |
| `2e-2` | Slower, but more stable learning |
| `5e-3` | Slow learning |

### Compute the ELBO

It is recommended that you normalise the data before computing the ELBO.

```python
elbo = model.elbo(t, y)
```

### Condition the Model on Data

It is recommended that you normalise the data before conditioning and unnormalise
the predictions.

```python
posterior_model = model.condition(t, y)
```

### Make Predictions

Predictions for new inputs:
```python
mean, var = posterior_model.predict(t_new)
```

Predictions for the kernel:
```python
pred = posterior_model.predict_kernel()
x, mean, var = pred.x, pred.mean, pred.var
```

Predictions for the PSD:
```python
pred = posterior_model.predict_psd()
x, mean, var = pred.x, pred.mean, pred.var
```

## Sample Experiments

### Learn a GP With a Known Kernel

```bash
python experiments/eq.py
```

```bash
python experiments/smk.py
```

### Learn the Mauna Loa CO2 Data Set

```bash
python experiments/mauna_loa.py
```

## Reproduce Experiments From the Paper

See [here](https://github.com/wesselb/gpcm/tree/master/experiments/paper).
