"""
ODE Term for reForge
Port of ComfyUI-RK-Sampler's TorchODEODETerm to reForge.
k-diffusion model API: model(x, sigma, **extra_args) -> denoised
"""
from __future__ import annotations

import logging
import math

import torch

logger = logging.getLogger(__name__)


class ReForgeODETerm:
    """
    ODE right-hand side function called by torchode's AutoDiffAdjoint.

    Probability flow ODE:
        dx/dsigma = (x - D(x, sigma)) / sigma
    D(x, sigma) is the denoised latent returned by CFGDenoiserKDiffusion.

    Args:
        model       : CFGDenoiserKDiffusion instance
        c_device    : device used by the ODE solver ("cpu" or "mps")
        c_dtype     : dtype used by the ODE solver (float64 or float32)
        o_device    : device of the original latent (e.g. CUDA)
        o_dtype     : dtype of the original latent
        o_shape     : shape of the original latent (batch, C, H, W)
        min_sigma   : pass through with zero gradient at or below this sigma
        t_max       : maximum sigma value
        t_min       : minimum sigma value
        n_steps     : number of steps in the sigma schedule (for the progress bar)
        extra_args  : dict of extra arguments passed to the model
        callback    : reForge callback function
        pbar        : tqdm progress bar
        is_adaptive : adaptive or fixed step (changes how the progress bar updates)
    """

    def __init__(
        self,
        model,
        c_device: str,
        c_dtype: torch.dtype,
        o_device,
        o_dtype: torch.dtype,
        o_shape: tuple,
        min_sigma: float,
        t_max: float,
        t_min: float,
        n_steps: int,
        extra_args: dict | None = None,
        callback=None,
        pbar=None,
        is_adaptive: bool = True,
        cfg_denoiser=None,   # for updating the step counter (apply_refiner support)
    ):
        self.model       = model
        self.c_device    = c_device
        self.c_dtype     = c_dtype
        self.o_device    = o_device
        self.o_dtype     = o_dtype
        self.o_shape     = o_shape
        self.min_sigma   = min_sigma
        self.t_max       = t_max
        self.t_min       = t_min
        self.n_steps     = n_steps
        self.extra_args  = extra_args or {}
        self.callback    = callback
        self.pbar        = pbar
        self.is_adaptive = is_adaptive
        self.cfg_denoiser = cfg_denoiser

        # Internal state
        self.n_callbacks   = 0
        self.pbar_step     = 0
        self.last_t        = None
        self.last_denoised = None

    # ------------------------------------------------------------------
    # Called as vf(t, y, stats, args) from torchode's ODETerm wrapper
    # ------------------------------------------------------------------
    def __call__(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t : (batch,)  current sigma value
            y : (batch, flat_dim)  current latent (flattened)
        Returns:
            dy/dt : (batch, flat_dim)
        """
        # If the actual shape of y does not match o_shape, update o_shape
        # (handles the case where the latent size changes after hires.fix upscaling)
        try:
            y_4d = y.reshape(self.o_shape)
        except RuntimeError:
            # o_shape does not match the actual y -> auto-correct to a reshapeable form.
            # Estimate the scale while preserving the original aspect ratio, and fall
            # back to a square only when it does not divide evenly (non-square latent support).
            b = y.shape[0]
            c = self.o_shape[1]
            hw = y.shape[1] // c
            h0, w0 = self.o_shape[2], self.o_shape[3]
            h = int(round(math.sqrt(hw * h0 / w0))) if h0 > 0 and w0 > 0 else 0
            if h <= 0 or hw % h != 0:
                h = int(math.isqrt(hw))
            w = hw // h
            if h * w != hw:
                raise RuntimeError(
                    f"[RK Sampler] cannot infer latent shape: "
                    f"flat_dim={y.shape[1]}, channels={c}"
                )
            self.o_shape = (b, c, h, w)
            logger.warning("[RK Sampler] auto-corrected o_shape to %s", self.o_shape)
            y_4d = y.reshape(self.o_shape)
        mask  = t <= self.min_sigma                              # (batch,)  rows whose sigma is too small

        denoised = torch.zeros_like(y_4d)

        if not mask.all():
            y_m = y_4d.to(self.o_device, dtype=self.o_dtype)
            t_m = t.to(self.o_device, dtype=self.o_dtype)

            # Pass the full batch to the model
            # (passing only part of it via a mask causes tensor size mismatches with LoRA etc.)
            # Pass t_m as (batch,) -- explicitly guarantee the sigma dimension
            if t_m.dim() == 0:
                t_m = t_m.unsqueeze(0).expand(y_m.shape[0])
            elif t_m.shape[0] != y_m.shape[0]:
                t_m = t_m[:1].expand(y_m.shape[0])

            with torch.no_grad():
                denoised_full = self.model(y_m, t_m, **self.extra_args)

            # Fill masked batches (sigma <= min_sigma) with zeros
            denoised_full = denoised_full.to(self.c_device, dtype=self.c_dtype)
            denoised_full[mask] = 0.0
            denoised = denoised_full

        # Zero gradient near sigma = 0
        d = torch.where(
            mask.view(-1, 1, 1, 1),
            torch.zeros_like(y_4d),
            (y_4d - denoised) / t.view(-1, 1, 1, 1),
        )

        self.last_t        = t
        self.last_denoised = denoised

        return d.flatten(start_dim=1)

    # ------------------------------------------------------------------
    # Called after each step from AutoDiffAdjoint
    # ------------------------------------------------------------------
    def trigger_callback(self, t: torch.Tensor, y: torch.Tensor):
        """Update the progress bar and fire the reForge callback."""
        if self.pbar is None and self.callback is None:
            return

        t_mean = t.mean().item()
        self.n_callbacks += 1

        # --- Update cfg_denoiser's step counter (apply_refiner support) ---
        # Adaptive steps can make n_callbacks exceed p.steps, so clamp it
        # to total_steps - 1 as the upper bound
        if self.cfg_denoiser is not None and hasattr(self.cfg_denoiser, "step"):
            total = getattr(self.cfg_denoiser, "total_steps", None)
            if total is not None:
                self.cfg_denoiser.step = min(self.n_callbacks, total - 1)
            else:
                self.cfg_denoiser.step = self.n_callbacks

        # --- Update the progress bar ---
        if self.pbar is not None:
            if self.is_adaptive:
                # Adaptive step: show progress as a percentage
                progress    = (self.t_max - t_mean) / max(self.t_max - self.t_min, 1e-8)
                percentage  = progress * 100
                self.pbar.update(percentage - self.pbar_step)
                self.pbar_step = percentage
                i = round(progress * self.n_steps)
            else:
                # Fixed step: count one step at a time
                self.pbar.update(1)
                self.pbar_step += 1
                i = self.pbar_step

            self.pbar.set_postfix({"sigma": f"{t_mean:.4f}"})
            self.pbar.refresh()
        else:
            progress = (self.t_max - t_mean) / max(self.t_max - self.t_min, 1e-8)
            i = round(progress * self.n_steps)

        # --- reForge callback ---
        if self.callback is not None:
            y_4d = y.reshape(self.o_shape)
            mask = self.last_t <= self.min_sigma if self.last_t is not None else torch.zeros(
                y_4d.shape[0], dtype=torch.bool, device=y_4d.device
            )
            samples = torch.where(
                mask.view(-1, 1, 1, 1),
                y_4d,
                self.last_denoised if self.last_denoised is not None else y_4d,
            ).to(self.o_device, dtype=self.o_dtype)

            self.callback({
                "x":         samples,
                "i":         i - 1,
                "sigma":     t_mean,
                "sigma_hat": t_mean,
                "denoised":  samples,
            })

    # ------------------------------------------------------------------
    # Called from torchode's ODETerm.init() (statistics initialization)
    # ------------------------------------------------------------------
    def init(self, problem, stats: dict):
        pass
