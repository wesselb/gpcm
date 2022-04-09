#!/usr/bin/env bash

# Synthetic experiments:
python experiments/paper/priors.py --train
python experiments/paper/sample.py --train
python experiments/paper/sample_interpolation.py --train
python experiments/paper/compare_inference.py

python experiments/paper/comparison.py
python experiments/paper/smk.py --train

# Real-world data experiments:
for year in 2012 2013 2014 2015 2016 2017;
do
    python experiments/paper/crude_oil.py --year $year --train --predict
done
python experiments/paper/vix_forecast.py
python experiments/paper/vix_analyse.py --train --predict

# Run separate postprocessing scripts.
python experiments/paper/comparison_process.py
python experiments/paper/vix_forecast_process.py
