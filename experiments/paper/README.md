# Experiments from Bruinsma, Tegnér, and Turner (2022)

This document explains how the experiments from the below citation can be reproduced and
which differences you can expect w.r.t. the precise numbers reported in the paper.

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

The experiments were run on an NVIDIA GeForce RTX 2080 Ti with 11 GB of memory.
On this GPU, the total runtime of running all experiments in sequence should be around
10 hours.
Note that the CGPCM also requires significant CPU compute, so the total runtime also
depends on the CPU.
Also note that the numbers may be slightly perturbed due to different random seeds.

To reproduce the experiments, clone this repository, make a virtual environment, and
install the versions of packages at the time the experiments were produced:

```bash
# Clone and enter the repository.
git clone https://github.com/wesselb/gpcm
cd gpcm

# Checkout to the right commit.
git checkout 292beacfd77eb1a53b4944da89be57b874f0d08f

# Setup a virtual environment.
virtualenv -p python3 venv
source venv/bin/activate

# Install the packages.
pip install -r experiments/paper/requirements_snapshot.txt
pip install -e .
```

Then run `experiments/paper/reproduce.sh` to run all experiments in sequence at once:

```bash
sh experiments/paper/reproduce.sh
```

Once the experiments have completed, you can find the results in the newly created
folder  `_experiments`.

In the remainder of this document, we will go through each figure and table and explain
how that particular figure or table can be reproduced and which differences you might
expect.

## Figure 1 and Figure 3
These figures are the same.

**Commands:**

```bash
python experiments/paper/priors.py --train
```

**Results:**

```
_experiments/priors/priors.pdf
```

**Notes:**

No expected differences or other notes.

## Figure 4

**Commands:**

```python
python experiments/paper/sample.py --train
```

**Results:**

```
_experiments/sample/sample.pdf
```

**Notes:**

No expected differences or other notes.

## Figure 5

**Commands:**

```bash
python experiments/paper/compare_inference.py
```

**Results:**

```
_experiments/compare_inference/elbos.pdf
```

**Notes:**

The runtimes depend on your machine and will look different.
The relative comparison of the learning curves, however, should be consistent with the
paper.

## Figure 6

**Commands:**

```bash
python experiments/paper/smk.py --train
```

**Results:**

```
_experiments/smk/smk.pdf
```

**Notes:**

No expected differences or other notes.

## Figure 7

**Commands:**

```bash
python experiments/paper/comparison.py
python experiments/paper/comparison_process.py
```

**Results:**

```
_experiments/comparison_process/comparison.pdf
_experiments/comparison_process/log.txt
```

**Notes:**

Unlike what the table in the paper says, the experiments reports the _standardised_ MLL
(SMLL) rather than the MLL, which is an error on our part.

## Table 1

**Commands:**

```bash
python experiments/paper/vix_forecast.py
python experiments/paper/vix_forecast_process.py
```

**Results:**

```
_experiments/vix_forecast_process/log.txt
```

**Notes:**

No expected differences or other notes.

## Figure 8

**Commands:**

```bash
for year in 2012 2013 2014 2015 2016 2017;
do
    python experiments/paper/crude_oil.py --year $year --train --predict
done
python experiments/paper/crude_oil_aggregate.py
```

**Results:**

```
_experiments/crude_oil/2013/crude_oil.pdf
_experiments/crude_oil_aggregate/log.txt
```

**Notes:**

No expected differences or other notes.

## Figure 9

**Commands:**

```bash
python experiments/paper/vix_analyse.py --train --predict
# Sometimes `vix_analyse.py` OOMs at prediction time, so rerun prediction again
# to be sure that prediction succeeded. Prediction is quick anyway.
python experiments/paper/vix_analyse.py --predict
```

**Results:**

```
_experiments/vix_analyse/psd.pdf
```

**Notes:**

No expected differences or other notes.

