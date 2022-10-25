"""
Abstract SDE classes, Reverse SDE, and VE/VP SDEs.

Taken and adapted from https://github.com/yang-song/score_sde_pytorch/blob/1618ddea340f3e4a2ed7852a0694a809775cf8d0/sde_lib.py
"""
import abc
import warnings

import numpy as np
from sldgmse.util.tensors import batch_broadcast
import torch

from sldgmse.util.registry import Registry


SDERegistry = Registry("SDE")


class SDE(abc.ABC):
    """SDE abstract class. Functions are designed for a mini-batch of inputs."""

    def __init__(self, N):
        """Construct an SDE.

        Args:
            N: number of discretization time steps.
        """
        super().__init__()
        self.N = N

    @property
    @abc.abstractmethod
    def T(self):
        """End time of the SDE."""
        pass

    @abc.abstractmethod
    def sde(self, x, t, *args):
        pass

    @abc.abstractmethod
    def marginal_prob(self, x, t, *args):
        """Parameters to determine the marginal distribution of the SDE, $p_t(x|args)$."""
        pass

    @abc.abstractmethod
    def prior_sampling(self, shape, *args):
        """Generate one sample from the prior distribution, $p_T(x|args)$ with shape `shape`."""
        pass

    @abc.abstractmethod
    def prior_logp(self, z):
        """Compute log-density of the prior distribution.

        Useful for computing the log-likelihood via probability flow ODE.

        Args:
            z: latent code
        Returns:
            log probability density
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def add_argparse_args(parent_parser):
        """
        Add the necessary arguments for instantiation of this SDE class to an argparse ArgumentParser.
        """
        pass

    def discretize(self, x, t, *args):
        """Discretize the SDE in the form: x_{i+1} = x_i + f_i(x_i) + G_i z_i.

        Useful for reverse diffusion sampling and probabiliy flow sampling.
        Defaults to Euler-Maruyama discretization.

        Args:
            x: a torch tensor
            t: a torch float representing the time step (from 0 to `self.T`)

        Returns:
            f, G
        """
        dt = 1 / self.N
        drift, diffusion = self.sde(x, t, *args)
        f = drift * dt
        G = diffusion * torch.sqrt(torch.tensor(dt, device=t.device))
        return f, G

    def reverse(oself, score_model, probability_flow=False):
        """Create the reverse-time SDE/ODE.

        Args:
            score_model: A function that takes x, t and y and returns the score.
            probability_flow: If `True`, create the reverse-time ODE used for probability flow sampling.
        """
        N = oself.N
        T = oself.T
        sde_fn = oself.sde
        discretize_fn = oself.discretize

        # Build the class for reverse-time SDE.
        class RSDE(oself.__class__):
            def __init__(self):
                self.N = N
                self.probability_flow = probability_flow

            @property
            def T(self):
                return T

            def sde(self, x, t, *args):
                """Create the drift and diffusion functions for the reverse SDE/ODE."""
                rsde_parts = self.rsde_parts(x, t, *args)
                total_drift, diffusion = rsde_parts["total_drift"], rsde_parts["diffusion"]
                return total_drift, diffusion

            def rsde_parts(self, x, t, *args):
                sde_drift, sde_diffusion = sde_fn(x, t, *args)
                score = score_model(x, t, *args)
                score_drift = -sde_diffusion[:, None, None, None]**2 * score * (0.5 if self.probability_flow else 1.)
                diffusion = torch.zeros_like(sde_diffusion) if self.probability_flow else sde_diffusion
                total_drift = sde_drift + score_drift
                return {
                    'total_drift': total_drift, 'diffusion': diffusion, 'sde_drift': sde_drift,
                    'sde_diffusion': sde_diffusion, 'score_drift': score_drift, 'score': score,
                }

            def discretize(self, x, t, *args):
                """Create discretized iteration rules for the reverse diffusion sampler."""
                f, G = discretize_fn(x, t, *args)
                rev_f = f - G[:, None, None, None] ** 2 * score_model(x, t, *args) * (0.5 if self.probability_flow else 1.)
                rev_G = torch.zeros_like(G) if self.probability_flow else G
                return rev_f, rev_G

        return RSDE()

    @abc.abstractmethod
    def copy(self):
        pass


@SDERegistry.register("ouve")
class OUVESDE(SDE):
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--sde-n", type=int, default=1000, help="The number of timesteps in the SDE discretization. 30 by default")
        parser.add_argument("--theta", type=float, default=1.5, help="The constant stiffness of the Ornstein-Uhlenbeck process. 1.5 by default.")
        parser.add_argument("--sigma-min", type=float, default=0.05, help="The minimum sigma to use. 0.05 by default.")
        parser.add_argument("--sigma-max", type=float, default=0.5, help="The maximum sigma to use. 0.5 by default.")
        return parser

    def __init__(self, theta, sigma_min, sigma_max, N=1000, **ignored_kwargs):
        """Construct an Ornstein-Uhlenbeck Variance Exploding SDE.

        Note that the "steady-state mean" `y` is not provided at construction, but must rather be given as an argument
        to the methods which require it (e.g., `sde` or `marginal_prob`).

        dx = -theta (y-x) dt + sigma(t) dw

        with

        sigma(t) = sigma_min (sigma_max/sigma_min)^t * sqrt(2 log(sigma_max/sigma_min))

        Args:
            theta: stiffness parameter.
            sigma_min: smallest sigma.
            sigma_max: largest sigma.
            N: number of discretization steps
        """
        super().__init__(N)
        self.theta = theta
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.logsig = np.log(self.sigma_max / self.sigma_min)
        self.N = N

    def copy(self):
        return OUVESDE(self.theta, self.sigma_min, self.sigma_max, N=self.N)

    @property
    def T(self):
        return 1

    def sde(self, x, t, y):
        drift = self.theta * (y - x)
        # the sqrt(2*logsig) factor is required here so that logsig does not in the end affect the perturbation kernel
        # standard deviation. this can be understood from solving the integral of [exp(2s) * g(s)^2] from s=0 to t
        # with g(t) = sigma(t) as defined here, and seeing that `logsig` remains in the integral solution
        # unless this sqrt(2*logsig) factor is included.
        sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        diffusion = sigma * np.sqrt(2 * self.logsig)
        return drift, diffusion

    def _mean(self, x0, t, y):
        theta = self.theta
        exp_interp = torch.exp(-theta * t)[:, None, None, None]
        return exp_interp * x0 + (1 - exp_interp) * y

    def _std(self, t):
        # This is a full solution to the ODE for P(t) in our derivations, after choosing g(s) as in self.sde()
        sigma_min, theta, logsig = self.sigma_min, self.theta, self.logsig
        # could maybe replace the two torch.exp(... * t) terms here by cached values **t
        return torch.sqrt(
            (
                sigma_min**2
                * torch.exp(-2 * theta * t)
                * (torch.exp(2 * (theta + logsig) * t) - 1)
                * logsig
            )
            /
            (theta + logsig)
        )

    def marginal_prob(self, x0, t, y):
        return self._mean(x0, t, y), self._std(t)

    def prior_sampling(self, shape, y):
        if shape != y.shape:
            warnings.warn(f"Target shape {shape} does not match shape of y {y.shape}! Ignoring target shape.")
        std = self._std(torch.ones((y.shape[0],), device=y.device))
        x_T = y + torch.randn_like(y) * std[:, None, None, None]
        return x_T

    def prior_logp(self, z):
        raise NotImplementedError("prior_logp for OU SDE not yet implemented!")


@SDERegistry.register("ouvp")
class OUVPSDE(SDE):
    # !!! We do not utilize this SDE in our works due to observed instabilities around t=0.2. !!!
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--sde-n", type=int, default=1000,
            help="The number of timesteps in the SDE discretization. 1000 by default")
        parser.add_argument("--beta-min", type=float, required=True,
            help="The minimum beta to use.")
        parser.add_argument("--beta-max", type=float, required=True,
            help="The maximum beta to use.")
        parser.add_argument("--stiffness", type=float, default=1,
            help="The stiffness factor for the drift, to be multiplied by 0.5*beta(t). 1 by default.")
        return parser

    def __init__(self, beta_min, beta_max, stiffness=1, N=1000, **ignored_kwargs):
        """
        !!! We do not utilize this SDE in our works due to observed instabilities around t=0.2. !!!

        Construct an Ornstein-Uhlenbeck Variance Preserving SDE:

        dx = -1/2 * beta(t) * stiffness * (y-x) dt + sqrt(beta(t)) * dw

        with

        beta(t) = beta_min + t(beta_max - beta_min)

        Note that the "steady-state mean" `y` is not provided at construction, but must rather be given as an argument
        to the methods which require it (e.g., `sde` or `marginal_prob`).

        Args:
            beta_min: smallest sigma.
            beta_max: largest sigma.
            stiffness: stiffness factor of the drift. 1 by default.
            N: number of discretization steps
        """
        super().__init__(N)
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.stiffness = stiffness
        self.N = N

    def copy(self):
        return OUVPSDE(self.beta_min, self.beta_max, self.stiffness, N=self.N)

    @property
    def T(self):
        return 1

    def _beta(self, t):
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def sde(self, x, t, y):
        drift = 0.5 * self.stiffness * batch_broadcast(self._beta(t), y) * (y - x)
        diffusion = torch.sqrt(self._beta(t))
        return drift, diffusion

    def _mean(self, x0, t, y):
        b0, b1, s = self.beta_min, self.beta_max, self.stiffness
        x0y_fac = torch.exp(-0.25 * s * t * (t * (b1-b0) + 2 * b0))[:, None, None, None]
        return y + x0y_fac * (x0 - y)

    def _std(self, t):
        b0, b1, s = self.beta_min, self.beta_max, self.stiffness
        return (1 - torch.exp(-0.5 * s * t * (t * (b1-b0) + 2 * b0))) / s

    def marginal_prob(self, x0, t, y):
        return self._mean(x0, t, y), self._std(t)

    def prior_sampling(self, shape, y):
        if shape != y.shape:
            warnings.warn(f"Target shape {shape} does not match shape of y {y.shape}! Ignoring target shape.")
        std = self._std(torch.ones((y.shape[0],), device=y.device))
        x_T = y + torch.randn_like(y) * std[:, None, None, None]
        return x_T

    def prior_logp(self, z):
        raise NotImplementedError("prior_logp for OU SDE not yet implemented!")


@SDERegistry.register("poisson")
class Poisson:
    def __init__(self, z_min, z_max, upper_norm, vs, gamma=5, z_exp=5, N=100, f_dim=256, t_dim=256, channel=2, M=291, **ignored_kwargs):
        """Construct a PFGM.
        Args:
          config: configurations
        """
        self.z_min = z_min
        self.z_max = z_max
        self.upper_norm = upper_norm
        self.vs = vs
        self.gamma = gamma
        self.z_exp = 5
        self.f_dim = f_dim
        self.t_dim = t_dim
        self.channel = channel
        self.M = M
        self.N = N

    def add_argparse_args(parser):
        parser.add_argument("--sde-n", type=int, default=100,
                            help="The number of timesteps in the SDE discretization. 1000 by default")
        parser.add_argument("--z-min", type=float, default=1e-3,
                            help="The minimum beta to use.")
        parser.add_argument("--z-max", type=float, default=40,
                            help="The maximum beta to use.")
        parser.add_argument("--upper-norm", type=float, default=3000)
        parser.add_argument("--vs", type=bool, default=False)

        return parser

    @property
    def M(self):
        return self.M

    def prior_sampling(self, shape):
        """
        Sampling initial data from p_prior on z=z_max hyperplane.
        See Section 3.3 in PFGM paper
        """

        # Sample the radius from p_radius (details in Appendix A.4 in the PFGM paper)
        max_z = self.z_max
        N = self.channels * self.f_dim * self.t_dim + 1
        # Sampling form inverse-beta distribution
        samples_norm = np.random.beta(a=N / 2. - 0.5, b=0.5, size=shape[0])
        inverse_beta = samples_norm / (1 - samples_norm)
        # Sampling from p_radius(R) by change-of-variable
        samples_norm = np.sqrt(max_z ** 2 * inverse_beta)
        # clip the sample norm (radius)
        samples_norm = np.clip(samples_norm, 1, self.upper_norm)
        samples_norm = torch.from_numpy(samples_norm).cuda().view(len(samples_norm), -1)

        # Uniformly sample the angle direction
        gaussian = torch.randn(shape[0], N - 1).cuda()
        unit_gaussian = gaussian / torch.norm(gaussian, p=2, dim=1, keepdim=True)

        # Radius times the angle direction
        init_samples = unit_gaussian * samples_norm

        return init_samples.float().view(len(init_samples), self.channel,
                                         self.f_dim, self.t_dim)

    @property
    def T(self):
        return 1

    def ode(self, net_fn, x, t):
        z = np.exp(t.mean().cpu())
        if self.vs:
            print(z)
        x_drift, z_drift = net_fn(x, torch.ones((len(x))).cuda() * z)
        x_drift = x_drift.view(len(x_drift), -1)

        # Substitute the predicted z with the ground-truth
        # Please see Appendix B.2.3 in PFGM paper (https://arxiv.org/abs/2209.11178) for details
        z_exp = self.z_exp
        if z < z_exp and self.gamma > 0:
            data_dim = self.channels * self.f_dim * self.t_dim
            sqrt_dim = np.sqrt(data_dim)
            norm_1 = x_drift.norm(p=2, dim=1) / sqrt_dim
            x_norm = self.gamma * norm_1 / (1 - norm_1)
            x_norm = torch.sqrt(x_norm ** 2 + z ** 2)
            z_drift = -sqrt_dim * torch.ones_like(z_drift) * z / (x_norm + self.gamma)

        # Predicted normalized Poisson field
        v = torch.cat([x_drift, z_drift[:, None]], dim=1)

        dt_dz = 1 / (v[:, -1] + 1e-5)
        dx_dt = v[:, :-1].view(len(x), self.num_channels,
                               self.f_dim, self.t_dim)
        dx_dz = dx_dt * dt_dz.view(-1, *([1] * len(x.size()[1:])))
        # dx/dt_prime =  z * dx/dz
        dx_dt_prime = z * dx_dz
        return dx_dt_prime