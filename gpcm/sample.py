import time

import lab as B
import numpy.random as rng
import wbml.out


class ESS:
    """Elliptical slice sampling algorithm.

    Args:
        log_lik (function): Function that returns the log value of a quantity
            that is proportional to the likelihood of the target distribution.
        sample_prior (function): Function that generates a sample from the
            prior of the target distribution. The prior must be a zero
            mean Gaussian.
    """

    def __init__(self, log_lik, sample_prior):
        self.log_lik = log_lik
        self.sample_prior = sample_prior

        # Sample the initial state from the prior and compute log-likelihood.
        self.x = sample_prior()
        self.log_lik_x = log_lik(self.x)

    def _establish_ellipse(self):
        """Establish an ellipse, which is required to draw the new state."""
        self.y = self.sample_prior()

    def _draw_proposal(self, lower, upper):
        """Draw a proposal for the next state given a bracket for
        :math:`\\theta`.

        Args:
            lower (scalar): Lower bound of bracket.
            upper (scalar): Upper bound of bracket.
        """
        self.theta = rng.uniform(lower, upper)
        self.x_proposed = B.cos(self.theta)*self.x \
                          + B.sin(self.theta)*self.y
        self.log_lik_x_proposed = self.log_lik(self.x_proposed)

    def _draw_bracket(self):
        """Draw a bracket for :math:`\\theta`.

        Returns:
            tuple[scalar]: Tuple contain
        """
        theta = rng.uniform(0, 2*B.pi)
        return theta - 2*B.pi, theta

    def _draw(self, lower, upper, u, attempts=1):
        """Draw new state given a bracket for :math:`\\theta`.

        Args:
            lower (scalar): Lower bound of bracket.
            upper (scalar): Upper bound of bracket.
            u (scalar): Slice height.
            attempts (attempt, optional): Number of attempts so far. Defaults
                to one.

        Returns:
            tuple[scalar]: Tuple containing the proposed state, the
                corresponding log-likelihood, and the number of attempts.
        """
        self._draw_proposal(lower, upper)
        if self.log_lik_x_proposed > u:
            # Proposal is accepted.
            return self.x_proposed, self.log_lik_x_proposed, attempts + 1
        else:
            # Proposal rejected. Shrink bracket and try again.
            if self.theta > 0:
                return self._draw(lower, self.theta, u, attempts + 1)
            else:
                return self._draw(self.theta, upper, u, attempts + 1)

    def sample(self, num=1):
        """Generate samples from the target distribution.

        Args:
            num (int, optional): Number of samples. Defaults to one.

        Returns:
            list[tensor]: Samples.
        """
        samples = []

        with wbml.out.Progress(name='Sampling (ESS)',
                               total=num,
                               filter={'Attempts': None}) as progress:
            for i in range(num):
                # Draw a slice height.
                u = self.log_lik_x - rng.exponential(1.0)

                # Establish ellipse.
                self._establish_ellipse()

                # Draw a bracket.
                lower, upper = self._draw_bracket()

                # Draw a sample.
                start = time.time()
                self.x, self.log_lik_x, attempts = self._draw(lower, upper, u)

                # Record sample and time per attempt.
                ms_per_attempt = (time.time() - start)/attempts*1000
                samples.append(self.x)

                # Report.
                progress({'Pseudo-log-likelihood': self.log_lik_x,
                          'Attempts': attempts,
                          'Milliseconds per attempt': ms_per_attempt})

        return samples
