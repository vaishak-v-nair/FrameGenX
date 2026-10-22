"""Denoising objectives are training diagnostics, never contract pass scores."""
import torch


def denoising_loss(prediction, target, timesteps, scheduler, snr_gamma: float):
    prediction_type = scheduler.config.prediction_type
    if prediction_type not in {"epsilon", "v_prediction"}:
        raise ValueError(f"Unsupported prediction type: {prediction_type}")
    alphas = scheduler.alphas_cumprod.to(prediction.device)[timesteps].float()
    snr = alphas / (1 - alphas).clamp_min(1e-12)
    numerator = torch.minimum(snr, torch.full_like(snr, snr_gamma))
    weight = numerator / (snr if prediction_type == "epsilon" else snr + 1).clamp_min(1e-12)
    per_sample = (prediction.float() - target.float()).square().flatten(1).mean(1)
    return (per_sample * weight).mean(), per_sample.mean()


def make_target(scheduler, latent, noise, timesteps):
    if scheduler.config.prediction_type == "epsilon":
        return noise
    if scheduler.config.prediction_type == "v_prediction":
        return scheduler.get_velocity(latent, noise, timesteps)
    raise ValueError("Unsupported diffusion target; training must not guess")
