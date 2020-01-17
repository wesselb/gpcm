import lab as B
import matplotlib.pyplot as plt
import numpy as np
import wbml.out
import wbml.plot

from gpcm.gpcm import GPCM
from gpcm.gprv import GPRV
from gpcm.util import estimate_psd

B.epsilon = 1e-7


def sample(model, t, noise_f):
    """Sample from a model.

    Args:
        model (:class:`gpcm.model.Model`): Model to sample from.
        t (vector): Time points to sample at.
        noise_f (vector): Noise for the sample of the function. Should have the
            same size as `t`.

    Returns:
        tuple[vector]: Tuple containing kernel samples and function samples.
    """
    ks, fs = [], []

    with wbml.out.Progress(name='Sampling', total=5) as progress:
        for i in range(5):
            progress()
            f, K = None, None

            while f is None:
                try:
                    # Sample random u.
                    u = B.sample(model.compute_K_u())[:, 0]

                    # Construct the kernel matrix.
                    K = B.reg(model.kernel_approx(t, t, u))

                    # Set to unity variance.
                    wbml.out.kv('Sampled variance', K[0, 0])
                    K /= K[0, 0]

                    # Draw sample function.
                    f = B.matmul(B.cholesky(K), noise_f)[:, 0]
                except np.linalg.LinAlgError as e:
                    wbml.out.out(f'Sampling failed with exception "{e}". '
                                 f'Trying again...')
                    continue

            fs.append(f)
            ks.append(K[0, :])

    return ks, fs


t = np.linspace(0, 10, 200)
noise_f = np.random.randn(len(t), 1)

# Construct models.
models = [('GP-RV', GPRV(window=2, scale=0.5, n_u=30, t=t, gamma=1)),
          ('GPCM', GPCM(window=2, scale=1, n_u=30, t=t)),
          ('CGPCM', GPCM(window=2, scale=1, n_u=30, t=t, causal=True))]

# Perform sampling.
model_ks, model_fs = zip(*[sample(model, t, noise_f) for _, model in models])

# Plot.
plt.figure(figsize=(15, 9))

for i, ((name, model), ks, fs) in enumerate(zip(models, model_ks, model_fs)):
    plt.subplot(3, 2, 1 + 2*i)
    plt.plot(t, np.stack(fs).T, lw=1)
    if hasattr(model, 't_z'):
        plt.scatter(model.t_z, model.t_z*0, s=5, marker='o', c='black')
    plt.title(name)
    plt.xlabel('Time (s)')
    plt.xlim(0, 10)
    wbml.plot.tweak(legend=False)

    plt.subplot(3, 4, 3 + 4*i)
    plt.plot(t, np.stack(ks).T, lw=1)
    plt.scatter(model.t_u, model.t_u*0, s=5, marker='o', c='black')
    plt.title('Kernel')
    plt.xlabel('Lag (s)')
    plt.xlim(0, 6)
    wbml.plot.tweak(legend=False)

    # Estimate PSD.
    freqs, psds = zip(*[estimate_psd(t, k) for k in ks])
    freqs = freqs[0]
    psds = np.stack(psds).T

    plt.subplot(3, 4, 4 + 4*i)
    plt.title('PSD (dB)')
    inds = np.arange(int(len(freqs)/2))
    inds = inds[freqs[inds] <= 1]
    plt.plot(freqs[inds], psds[inds, :], lw=1)
    plt.xlabel('Frequency (Hz)')
    plt.xlim(0, 1)
    plt.ylim(-40, 20)
    wbml.plot.tweak(legend=False)

plt.tight_layout()
plt.savefig('experiments/samples.pdf')
plt.show()
