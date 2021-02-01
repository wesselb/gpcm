import lab as B
from gpcm.sample import ESS
from stheno import Normal
import numpy as np


def test_ess():
    # Construct a prior and a likelihood.
    prior = Normal(np.array([[0.6, 0.3], [0.3, 0.6]]))
    lik = Normal(
        np.array([[0.2], [0.3]]),
        np.array([[1, 0.2], [0.2, 1]]),
    )

    # Perform sampling.
    sampler = ESS(lik.logpdf, prior.sample)
    num_samples = 30_000
    samples = B.concat(*sampler.sample(num=num_samples), axis=1)

    samples_mean = B.mean(samples, axis=1)[:, None]
    samples_cov = (
        B.matmul(samples - samples_mean, samples - samples_mean, tr_b=True)
        / num_samples
    )

    # Compute posterior statistics.
    prec_prior = B.inv(prior.var)
    prec_lik = B.inv(lik.var)
    cov = B.inv(prec_prior + prec_lik)
    mean = cov @ (prec_prior @ prior.mean + prec_lik @ lik.mean)

    approx(samples_cov, cov, atol=5e-2)
    approx(samples_mean, mean, atol=5e-2)
