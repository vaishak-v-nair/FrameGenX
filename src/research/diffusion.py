"""Independent small learned-variance DDPM for a CPU Cards feasibility study.

Mathematics: Nichol/Dhariwal I-DDPM and Bhosale et al. VSM (2026), equations
7,9,15. This is not an SDXL/PEFT checkpoint or the authors' full-scale model.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class CardsDenoiser(nn.Module):
    def __init__(self, width=16):
        super().__init__()
        self.time = nn.Sequential(nn.Linear(16, width), nn.SiLU(), nn.Linear(width, width))
        self.input = nn.Conv2d(3, width, 3, padding=1)
        self.high = nn.Sequential(nn.GroupNorm(4, width), nn.SiLU(),
                                  nn.Conv2d(width, width, 3, padding=1))
        self.down = nn.Conv2d(width, width*2, 4, stride=2, padding=1)
        self.low = nn.Sequential(nn.GroupNorm(4, width*2), nn.SiLU(),
                                 nn.Conv2d(width*2, width*2, 3, padding=1),
                                 nn.GroupNorm(4, width*2), nn.SiLU(),
                                 nn.Conv2d(width*2, width*2, 3, padding=1))
        self.up = nn.Conv2d(width*2, width, 3, padding=1)
        self.output = nn.Sequential(nn.GroupNorm(4, width*2), nn.SiLU(),
                                    nn.Conv2d(width*2, 6, 3, padding=1))

    def forward(self, pixels, timestep):
        frequencies = torch.exp(torch.arange(8, device=pixels.device) * (-math.log(10000)/7))
        angles = timestep.float()[:, None] * frequencies[None]
        embedding = self.time(torch.cat((angles.sin(), angles.cos()), dim=1))
        high = self.input(pixels) + embedding[:, :, None, None]
        high = high + self.high(high)
        low = self.down(high)
        low = low + self.low(low)
        up = self.up(F.interpolate(low, size=high.shape[-2:], mode="nearest"))
        return self.output(torch.cat((high, up), dim=1))


def mean_dimensions(value):
    return value.flatten(1).mean(1)


class LearnedVarianceDiffusion:
    def __init__(self, steps=100):
        if steps < 2:
            raise ValueError("At least two diffusion steps required")
        self.steps = steps
        points = torch.arange(steps+1, dtype=torch.float64) / steps
        curve = torch.cos((points+.008) / 1.008 * math.pi/2).square()
        curve = curve / curve[0]
        self.beta = (1-curve[1:]/curve[:-1]).clamp(max=.999).float()
        self.alpha = 1-self.beta
        self.bar = torch.cumprod(self.alpha, dim=0)
        self.previous = torch.cat((torch.ones(1), self.bar[:-1]))
        self.posterior_variance = self.beta * (1-self.previous)/(1-self.bar)
        self.posterior_log = torch.log(torch.cat((self.posterior_variance[1:2],
                                                 self.posterior_variance[1:])).clamp_min(1e-20))
        self.coefficient_x0 = self.beta * self.previous.sqrt()/(1-self.bar)
        self.coefficient_xt = (1-self.previous)*self.alpha.sqrt()/(1-self.bar)

    @staticmethod
    def take(values, timestep, reference):
        return values.to(reference.device)[timestep].view(-1, *([1]*(reference.ndim-1)))

    def noisy(self, clean, timestep, noise):
        bar = self.take(self.bar, timestep, clean)
        return bar.sqrt()*clean + (1-bar).sqrt()*noise

    def distribution(self, output, noisy, timestep, *, detach_mean=False, clip=False):
        if output.shape[1] != noisy.shape[1]*2:
            raise ValueError("Learned variance needs separate epsilon and variance channels")
        epsilon, raw_variance = output.chunk(2, dim=1)
        if detach_mean:
            epsilon = epsilon.detach()
        bar = self.take(self.bar, timestep, noisy)
        clean = (noisy-(1-bar).sqrt()*epsilon)/bar.sqrt()
        if clip:
            clean = clean.clamp(-1, 1)
        mean = self.take(self.coefficient_x0, timestep, noisy)*clean + self.take(
            self.coefficient_xt, timestep, noisy)*noisy
        fraction = (raw_variance.tanh()+1)/2
        log_variance = (1-fraction)*self.take(self.posterior_log, timestep, noisy) + fraction*self.take(
            self.beta.log(), timestep, noisy)
        return mean, log_variance

    def loss(self, output, clean, noisy, noise, timestep, rho):
        if rho < 0 or not math.isfinite(rho):
            raise ValueError("Invalid VSM coefficient")
        if any(not torch.isfinite(tensor).all() for tensor in (output, clean, noisy, noise)):
            raise ValueError("Nonfinite diffusion input")
        model_mean, log_variance = self.distribution(output, noisy, timestep, detach_mean=True)
        target_mean = self.take(self.coefficient_x0, timestep, noisy)*clean + self.take(
            self.coefficient_xt, timestep, noisy)*noisy
        target_log = self.take(self.posterior_log, timestep, noisy)
        kl = .5 * (-1+log_variance-target_log + (target_log-log_variance).exp()
                   + (target_mean-model_mean).square()*(-log_variance).exp())
        # Discretized Gaussian decoder likelihood for [-1,1] RGB values.
        inverse_std = (-.5*log_variance).exp()
        plus = torch.special.ndtr((clean-model_mean+1/255)*inverse_std)
        minus = torch.special.ndtr((clean-model_mean-1/255)*inverse_std)
        log_probability = torch.where(clean < -.999, plus.clamp_min(1e-12).log(),
            torch.where(clean > .999, (1-minus).clamp_min(1e-12).log(),
                        (plus-minus).clamp_min(1e-12).log()))
        vb = torch.where(timestep == 0, mean_dimensions(-log_probability), mean_dimensions(kl))/math.log(2)
        mse = mean_dimensions((output[:, :3]-noise).square())
        # Stable algebra for 1/(1+precision**2), precision=exp(-log_variance).
        curvature = mean_dimensions(torch.sigmoid(2*log_variance))
        schedule = rho/(1-self.bar[timestep]).sqrt()
        smooth = schedule*curvature
        return {"loss": (mse+vb+smooth).mean(), "mse": mse.mean(), "vlb": vb.mean(),
                "smooth": smooth.mean()}

    @torch.no_grad()
    def sample(self, model, seeds, check=None):
        if not seeds:
            raise ValueError("Sampling requires explicit seeds")
        generators = [torch.Generator(device="cpu").manual_seed(seed) for seed in seeds]
        pixels = torch.stack([torch.randn((3, 128, 128), generator=g) for g in generators])
        model.eval()
        for index in range(self.steps-1, -1, -1):
            if check is not None:
                check()
            timestep = torch.full((len(seeds),), index, dtype=torch.long)
            mean, log_variance = self.distribution(model(pixels, timestep), pixels, timestep, clip=True)
            if index:
                noise = torch.stack([torch.randn((3, 128, 128), generator=g) for g in generators])
                pixels = mean + (.5*log_variance).exp()*noise
            else:
                pixels = mean
            if not torch.isfinite(pixels).all():
                raise ValueError("Nonfinite sampled pixels")
        return ((pixels.clamp(-1, 1)+1)*127.5).round().byte().permute(0, 2, 3, 1).numpy()
