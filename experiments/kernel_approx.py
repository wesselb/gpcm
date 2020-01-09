import matplotlib.pyplot as plt
import numpy as np
import wbml.out
import wbml.plot
from lab import B

from gpcm.gprv import k_u, GPRV, determine_a_b
from gpcm.kernel_approx import kernel_approx_u

# Define some test parameters.
lam = 1/2  # Model length scale.
wbml.out.kv('Lambda', lam)

# Set window to twice the length scale of the model.
alpha = lam/2
wbml.out.kv('Window length scale', 1/alpha)
wbml.out.kv('Alpha', alpha)

t = np.linspace(0, 10, 200)
tu = np.linspace(0, 2/alpha, 20)

noise_f = np.random.randn(len(t), 1)
ks, fs = [], []

# Construct model.
a, b = determine_a_b(alpha, t)
model = GPRV(lam=lam, alpha=alpha, a=a, b=b, m_max=50, n_u=20)

with wbml.out.Progress(name='Sampling', total=5) as progress:
    for i in range(5):
        progress()

        f, K = None, None

        # Sample random u.
        while f is None:
            try:
                Ku = k_u(model, tu[:, None], tu[None, :])
                u = B.matmul(B.cholesky(Ku), np.random.randn(len(tu), 1))[:, 0]

                # Construct the kernel matrix.
                K = B.reg(kernel_approx_u(model, t, t, u))
                wbml.out.kv('Minimal eigenvalue', min(np.linalg.eigvals(K)))
                wbml.out.kv('Sampled variance', K[0, 0])
                K /= K[0, 0]  # Set to unity variance.

                # Draw sample function.
                f = B.matmul(B.cholesky(K), noise_f)[:, 0]
            except np.linalg.LinAlgError:
                wbml.out.out('Sampling failed. Trying again...')
                continue

        fs.append(f)
        ks.append(K[0, :])

# Compute PSDs.
psds = []
n_zero = 1000
for k in ks:
    k = B.concat(k, B.zeros(n_zero))
    k_symmetric = B.concat(k, k[1:-1][::-1])
    psd = np.fft.fft(k_symmetric)
    freqs = np.fft.fftfreq(len(psd))/(t[1] - t[0])

    # Should be real and positive, but the numerics may not be in our favour.
    psd = np.abs(np.real(psd))

    # Now scale appropriately.
    total_power = np.sum(psd*np.abs(freqs))
    psd /= total_power/k[0]

    # Convert to dB.
    psd = 10*np.log10(psd)

    psds.append(psd)

fs = np.stack(fs).T
ks = np.stack(ks).T
psds = np.stack(psds).T

# Plot.
plt.figure(figsize=(15, 3))

plt.subplot(1, 2, 1)
plt.plot(t, fs, lw=1)
plt.title('Function')
plt.xlabel('Time (s)')
wbml.plot.tweak(legend=False)

plt.subplot(1, 4, 3)
plt.plot(t, ks, lw=1)
plt.scatter(tu, tu*0, s=5, marker='o', c='black')
plt.title('Kernel')
plt.xlabel('Lag (s)')
wbml.plot.tweak(legend=False)

plt.subplot(1, 4, 4)
plt.title('PSD (dB)')
inds = np.arange(int(len(freqs)/2))
inds = inds[freqs[inds] <= 1]
plt.plot(freqs[inds], psds[inds, :], lw=1)
plt.xlabel('Frequency (Hz)')
plt.ylim(-40, 0)
wbml.plot.tweak(legend=False)

plt.tight_layout()
plt.show()
