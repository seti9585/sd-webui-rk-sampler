"""
RK Sampler — reForge Extension
==============================
Location: extensions/rk-sampler/scripts/rk_sampler.py

Port of all ODE methods from ComfyUI-RK-Sampler
(https://github.com/memmaptensor/ComfyUI-RK-Sampler) as an independent sampler extension.

[Features]
- Registers 19 methods as individual samplers
  (ae_bosh3, ae_cash_karp5, ae_dopri5, ae_dopri8,
   ae_fehlberg2, ae_fehlberg5, ae_heun_euler2, ae_midpoint2,
   ae_ralston2, ae_tsit5,
   fe_euler1, fe_heun3, fe_kutta3, fe_kutta4, fe_kutta_38th4,
   fe_ralston3, fe_ralston4, fe_ssprk3, fe_wray3)
- Different methods can be selected for txt2img and hires.fix
  (reForge's Custom ODE Settings only allows one choice)
- Adaptive step (ae_*): automatic step size adjustment via PID controller
- Fixed step (fe_*): fixed step size following the sigma schedule
- rtol / atol / max_steps configurable per sampler in the Settings tab

Dependencies:
  torchode  (pip install torchode)
"""

from __future__ import annotations

import logging
import os
import sys

import gradio as gr
import torch
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
# Search for rk_core package in the parent directory of scripts
# ---------------------------------------------------------------------------
_EXT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _EXT_DIR not in sys.path:
    sys.path.insert(0, _EXT_DIR)

# ---------------------------------------------------------------------------
# reForge core modules
# ---------------------------------------------------------------------------
from modules import sd_samplers_common, shared, script_callbacks
from modules.shared import opts

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# torchode availability check
# ---------------------------------------------------------------------------
try:
    import torchode
    HAS_TORCHODE = True
except ModuleNotFoundError:
    HAS_TORCHODE = False
    logger.error(
        "[RK Sampler] torchode not found. Please run: pip install torchode"
    )

HAS_MPS = torch.backends.mps.is_available()
try:
    import torch_directml  # noqa: F401
    HAS_DML = True
except ModuleNotFoundError:
    HAS_DML = False


# ===========================================================================
# Section 0: Method Definitions and Registration Table
# ===========================================================================

ADAPTIVE_METHODS = {
    "ae_bosh3":      ("rk_core.methods.ae_bosh3",      "AEBosh3"),
    "ae_cash_karp5": ("rk_core.methods.ae_cash_karp5", "AECashKarp5"),
    "ae_dopri5":     ("rk_core.methods.ae_dopri5",     "AEDopri5"),
    "ae_dopri8":     ("rk_core.methods.ae_dopri8",     "AEDopri8"),
    "ae_fehlberg2":  ("rk_core.methods.ae_fehlberg2",  "AEFehlberg2"),
    "ae_fehlberg5":  ("rk_core.methods.ae_fehlberg5",  "AEFehlberg5"),
    "ae_heun_euler2":("rk_core.methods.ae_heun_euler2","AEHeunEuler2"),
    "ae_midpoint2":  ("rk_core.methods.ae_midpoint2",  "AEMidpoint2"),
    "ae_ralston2":   ("rk_core.methods.ae_ralston2",   "AERalston2"),
    "ae_tsit5":      ("rk_core.methods.ae_tsit5",      "AETsit5"),
}

FIXED_METHODS = {
    "fe_euler1":      ("rk_core.methods.fe_euler1",     "FEEuler1"),
    "fe_heun3":       ("rk_core.methods.fe_heun3",      "FEHeun3"),
    "fe_kutta3":      ("rk_core.methods.fe_kutta3",     "FEKutta3"),
    "fe_kutta4":      ("rk_core.methods.fe_kutta4",     "FEKutta4"),
    "fe_kutta_38th4": ("rk_core.methods.fe_kutta_38th4","FEKutta38th4"),
    "fe_ralston3":    ("rk_core.methods.fe_ralston3",   "FERalston3"),
    "fe_ralston4":    ("rk_core.methods.fe_ralston4",   "FERalston4"),
    "fe_ssprk3":      ("rk_core.methods.fe_ssprk3",     "FESSPRK3"),
    "fe_wray3":       ("rk_core.methods.fe_wray3",      "FEWray3"),
}

# scipy-based adaptive step methods (process one sample at a time)
SCIPY_METHODS = {
    "se_RK23":   {"order": 3,  "nfe_per_step": 3},
    "se_RK45":   {"order": 5,  "nfe_per_step": 6},
    "se_DOP853": {"order": 8,  "nfe_per_step": 12},
}

ALL_METHODS = {**ADAPTIVE_METHODS, **FIXED_METHODS, **SCIPY_METHODS}

# Options keys
OPT_LOG_RTOL  = "rk_sampler_log_rtol"    # relative tolerance for adaptive steps
OPT_LOG_ATOL  = "rk_sampler_log_atol"    # absolute tolerance for adaptive steps
OPT_MAX_STEPS = "rk_sampler_max_steps"   # max steps for adaptive methods
OPT_MIN_SIGMA = "rk_sampler_min_sigma"   # lower cutoff for sigma
OPT_PCOEFF    = "rk_sampler_pcoeff"      # PID P coefficient
OPT_ICOEFF    = "rk_sampler_icoeff"      # PID I coefficient
OPT_DCOEFF    = "rk_sampler_dcoeff"      # PID D coefficient

# Default values
DEF_LOG_RTOL  = -3.0
DEF_LOG_ATOL  = -4.0
DEF_MAX_STEPS = 1000
DEF_MIN_SIGMA = 1e-5
DEF_PCOEFF    = 0.0
DEF_ICOEFF    = 1.0
DEF_DCOEFF    = 0.0


def _get(key, default):
    return getattr(opts, key, default)


# ---------------------------------------------------------------------------
# Flow Matching detection and noise injection helpers  [patched]
# ---------------------------------------------------------------------------

def _is_flow_matching(model_wrap):
    # Check whether the loaded model uses Flow Matching
    # (Anima/DiT, FLUX, SD3, ...) or standard DDPM/EDM (SDXL, SD1.5, ...).
    #
    # Strategy 1: inspect the model_sampling class name.
    # Strategy 2: fallback - Flow Matching models have sigma_max <= 1.0,
    #             while DDPM/EDM models have sigma_max ~ 14.6.
    #             Using 1.5 as a safe threshold.
    #
    # NOTE: detection uses model_wrap.sigmas[-1] (the model's inherent sigma
    # table maximum), NOT sigma_sched[0] (which varies with denoising_strength).
    # This guarantees correct behaviour at all denoising_strength values.
    inner = getattr(model_wrap, 'inner_model', model_wrap)
    ms = getattr(inner, 'model_sampling', None)
    if ms is not None:
        ms_type = type(ms).__name__
        if any(kw in ms_type for kw in ('Flow', 'Flux')):
            return True
    sigs = getattr(model_wrap, 'sigmas', None)
    if sigs is not None and len(sigs) > 0:
        return sigs[-1].item() <= 1.5
    return False


def _flow_aware_noise_injection(x, noise, sigma, model_wrap):
    # Apply the correct noise injection for the model type.
    #
    # Flow Matching: x_t = (1 - t) * x + t * noise   (linear interpolation)
    # DDPM / EDM:    x_t = x + sigma * noise           (additive)
    if _is_flow_matching(model_wrap):
        t = sigma.clamp(0.0, 1.0)
        return (1.0 - t) * x + t * noise
    return x + noise * sigma



# ===========================================================================
# Section 1: Dynamic Method Class Loading
# ===========================================================================

def _load_method_class(method_name: str):
    """Dynamically import and return the class for a given method name. Returns None for scipy methods."""
    if method_name in SCIPY_METHODS:
        return None
    import importlib
    module_path, class_name = ALL_METHODS[method_name]
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


# ===========================================================================
# Section 2: Core Sampling Functions
# ===========================================================================

def _run_rk_sampler(
    method_name: str,
    model,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    extra_args: dict | None = None,
    callback=None,
    log_rtol: float  = DEF_LOG_RTOL,
    log_atol: float  = DEF_LOG_ATOL,
    max_steps: int   = DEF_MAX_STEPS,
    min_sigma: float = DEF_MIN_SIGMA,
    pcoeff: float    = DEF_PCOEFF,
    icoeff: float    = DEF_ICOEFF,
    dcoeff: float    = DEF_DCOEFF,
):
    """Dispatch to torchode or scipy implementation depending on the method."""
    if method_name in SCIPY_METHODS:
        return _run_scipy_sampler(
            method_name=method_name, model=model, x=x, sigmas=sigmas,
            extra_args=extra_args, callback=callback,
            log_rtol=log_rtol, log_atol=log_atol, max_steps=max_steps,
            min_sigma=min_sigma,
        )
    return _run_torchode_sampler(
        method_name=method_name, model=model, x=x, sigmas=sigmas,
        extra_args=extra_args, callback=callback,
        log_rtol=log_rtol, log_atol=log_atol, max_steps=max_steps,
        min_sigma=min_sigma, pcoeff=pcoeff, icoeff=icoeff, dcoeff=dcoeff,
    )


def _run_scipy_sampler(
    method_name: str,
    model,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    extra_args: dict | None = None,
    callback=None,
    log_rtol: float  = DEF_LOG_RTOL,
    log_atol: float  = DEF_LOG_ATOL,
    max_steps: int   = DEF_MAX_STEPS,
    min_sigma: float = DEF_MIN_SIGMA,
):
    """
    Adaptive step sampler using scipy.integrate.solve_ivp.
    Processes one sample at a time.
    Equivalent to ComfyUI-RK-Sampler's RungeKuttaSamplerImpl._call_scipy().
    """
    import scipy.integrate
    import scipy.integrate._ivp.rk
    import scipy.integrate._ivp.common
    import numpy as np
    from rk_core.scipy_ode_term import ReFormeSciPyODETerm
    from rk_core import scipy_step_impl

    scipy_method = method_name[3:]  # "se_RK45" -> "RK45"

    c_dtype  = np.float32 if (HAS_MPS or HAS_DML) else np.float64
    o_device = x.device
    o_dtype  = x.dtype
    o_shape  = x.shape           # (batch, C, H, W)
    o_shape_1 = o_shape[1:]      # (C, H, W)

    x_np     = x.cpu().numpy().astype(c_dtype)
    sigmas_np = sigmas.cpu().numpy().astype(c_dtype)
    t_max    = sigmas_np.max()
    t_min    = sigmas_np.min()
    n_steps  = len(sigmas_np) - 1
    batch    = o_shape[0]

    samples = []

    for i in range(batch):
        pbar = tqdm(
            total=100,
            desc=f"({i+1}/{batch}) [scipy] {method_name}",
            unit="%",
            bar_format="{desc}: {percentage:.2f}%|{bar}| [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
            postfix={"σ": f"{t_max:.4f}"},
        )

        with pbar:
            term = ReFormeSciPyODETerm(
                model      = model,
                o_device   = o_device,
                o_dtype    = o_dtype,
                o_shape_1  = o_shape_1,
                min_sigma  = min_sigma,
                t_max      = t_max,
                t_min      = t_min,
                n_steps    = n_steps,
                extra_args = extra_args,
                callback   = callback,
                pbar       = pbar,
                cfg_denoiser = model,
                c_dtype    = c_dtype,
            )

            # Replace scipy internals in the same way as ComfyUI-RK-Sampler
            scipy.integrate._ivp.rk.RungeKutta._step_impl = scipy_step_impl._step_impl
            scipy_step_impl.DT_MIN    = None
            scipy_step_impl.DT_MAX    = None
            scipy_step_impl.SAFETY    = 0.9
            scipy_step_impl.MIN_FACTOR = 0.2
            scipy_step_impl.MAX_FACTOR = 10.0
            scipy_step_impl.MAX_STEPS = max_steps
            scipy_step_impl.N_STEPS   = 0
            scipy_step_impl.TERM      = term

            try:
                result = scipy.integrate.solve_ivp(
                    fun         = term,
                    t_span      = [t_max, t_min],
                    y0          = x_np[i].reshape(-1),
                    method      = scipy_method,
                    t_eval      = None,
                    dense_output= False,
                    atol        = 10 ** log_atol,
                    rtol        = 10 ** log_rtol,
                )
            except sd_samplers_common.InterruptedException:
                logger.debug("[RK Sampler] scipy interrupted (InterruptedException)")
                sample = torch.from_numpy(x_np[i].reshape(o_shape_1)).to(o_device, dtype=o_dtype)
                samples.append(sample)
                break

            if result.success:
                sample = torch.from_numpy(
                    result.y[:, -1].reshape(o_shape_1)
                ).to(o_device, dtype=o_dtype)
                pbar.update(pbar.total - pbar.n)
            else:
                logger.warning(f"[RK Sampler] scipy Sample #{i} failed: {result.message}")
                sample = torch.full(o_shape_1, float("nan"), device=o_device, dtype=o_dtype)

        samples.append(sample)

    # Fill incomplete batch on interruption
    while len(samples) < batch:
        samples.append(torch.full(o_shape_1, float("nan"), device=o_device, dtype=o_dtype))

    return torch.stack(samples)


def _run_torchode_sampler(
    method_name: str,
    model,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    extra_args: dict | None = None,
    callback=None,
    log_rtol: float  = DEF_LOG_RTOL,
    log_atol: float  = DEF_LOG_ATOL,
    max_steps: int   = DEF_MAX_STEPS,
    min_sigma: float = DEF_MIN_SIGMA,
    pcoeff: float    = DEF_PCOEFF,
    icoeff: float    = DEF_ICOEFF,
    dcoeff: float    = DEF_DCOEFF,
):
    """
    Solve the ODE using torchode's AutoDiffAdjoint.
    Equivalent to ComfyUI-RK-Sampler's RungeKuttaSamplerImpl._call_torchode().
    """
    from rk_core.ode_term import ReForgeODETerm
    from rk_core.solvers.auto_diff_adjoint import AutoDiffAdjoint
    from rk_core.controllers.pid_controller import PIDController
    from rk_core.controllers.scheduled_controller import ScheduledController

    is_adaptive = method_name in ADAPTIVE_METHODS
    # Device and dtype setup
    c_device = "mps" if HAS_MPS else "cpu"
    c_dtype  = torch.float32 if (HAS_MPS or HAS_DML) else torch.float64
    o_device = x.device
    o_dtype  = x.dtype
    o_shape  = x.shape
    batch    = o_shape[0]

    x_c      = x.to(c_device, dtype=c_dtype)
    sigmas_c = sigmas.to(c_device, dtype=c_dtype)

    t_max = sigmas_c.max().item()
    t_min = sigmas_c.min().item()
    n_steps = len(sigmas_c) - 1

    # Progress bar
    if is_adaptive:
        pbar = tqdm(
            total=100,
            desc=f"[adaptive] {method_name}",
            unit="%",
            bar_format="{desc}: {percentage:.2f}%|{bar}| [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
            postfix={"σ": f"{t_max:.4f}"},
        )
    else:
        pbar = tqdm(
            total=n_steps,
            desc=f"[fixed] {method_name}",
            unit="step",
            bar_format="{desc}: {percentage:.2f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
            postfix={"σ": f"{t_max:.4f}"},
        )

    with pbar:
        # ODE term
        ode_fn = ReForgeODETerm(
            model      = model,
            c_device   = c_device,
            c_dtype    = c_dtype,
            o_device   = o_device,
            o_dtype    = o_dtype,
            o_shape    = o_shape,
            min_sigma  = min_sigma,
            t_max      = t_max,
            t_min      = t_min,
            n_steps    = n_steps,
            extra_args = extra_args,
            callback   = callback,
            pbar       = pbar,
            is_adaptive= is_adaptive,
            cfg_denoiser = model,   # passed to update step counter
        )
        term = torchode.ODETerm(ode_fn)

        # Step size controller
        if is_adaptive:
            controller = PIDController(
                atol       = 10 ** log_atol,
                rtol       = 10 ** log_rtol,
                pcoeff     = pcoeff,
                icoeff     = icoeff,
                dcoeff     = dcoeff,
                term       = term,
                norm       = torchode.step_size_controllers.rms_norm,
            )
        else:
            controller = ScheduledController(sigmas=sigmas_c)

        # Instantiate method class
        MethodClass = _load_method_class(method_name)
        step_method = MethodClass(term=term)

        # Solver
        adjoint = AutoDiffAdjoint(
            step_method,
            controller,
            max_steps                        = max_steps if is_adaptive else None,
            backprop_through_step_size_control= False,
            dense_output                     = False,
        )

        # Problem setup: t_start=sigma_max → t_end=sigma_min
        problem = torchode.InitialValueProblem(
            y0      = x_c.flatten(start_dim=1),
            t_start = torch.full((batch,), t_max, device=c_device, dtype=c_dtype),
            t_end   = torch.full((batch,), t_min, device=c_device, dtype=c_dtype),
            t_eval  = None,
        )

        try:
            result = adjoint.solve(problem)
        except sd_samplers_common.InterruptedException:
            # User interruption (Skip/Interrupt) — return x as-is
            logger.debug("[RK Sampler] Interrupted (InterruptedException)")
            return x.to(o_device, dtype=o_dtype)

    samples = result.ys[:, -1].reshape(o_shape).to(o_device, dtype=o_dtype)

    # Error check
    for i, status in enumerate(result.status):
        s = status.item()
        if s != 0:
            reason = torchode.status_codes.Status(s)
            logger.warning(f"[RK Sampler] Sample #{i} failed: {reason}")
            samples[i] = torch.full_like(samples[i], float("nan"))

    return samples


# ===========================================================================
# Section 3: reForge Sampler Wrapper Class (shared by all methods)
# ===========================================================================

class RKMethodSampler(sd_samplers_common.Sampler):
    """
    reForge sampler for a single RK method.
    One instance per method is created in _register().
    """

    def __init__(self, sd_model, method_name: str):
        # fe_* fixed-step methods require a sample_fn as sampler,
        # so pass a dummy callable (actual work is done by _run_rk_sampler)
        super().__init__(lambda *a, **k: None)
        self.funcname     = f"rk_{method_name}"
        self.method_name  = method_name
        self.extra_params = []

        from modules.sd_samplers_kdiffusion import CFGDenoiserKDiffusion
        self.model_wrap_cfg = CFGDenoiserKDiffusion(self)
        self.model_wrap     = self.model_wrap_cfg.inner_model

    def initialize(self, p) -> dict:
        self.p = p
        self.model_wrap_cfg.p               = p
        self.model_wrap_cfg.mask            = getattr(p, "mask",  None)
        self.model_wrap_cfg.nmask           = getattr(p, "nmask", None)
        self.model_wrap_cfg.step            = 0
        self.model_wrap_cfg.total_steps     = p.steps   # required for apply_refiner() to compute step/total_steps
        self.model_wrap_cfg.steps           = p.steps   # required for detail_daemon etc. that use // operator
        self.model_wrap_cfg.image_cfg_scale = getattr(p, "image_cfg_scale", None)
        self.eta          = p.eta if p.eta is not None else 0.0
        self.s_min_uncond = getattr(p, "s_min_uncond", 0.0)

        # TorchHijack (random seed reproducibility)
        try:
            from modules.sd_samplers_common import TorchHijack
            hijack = TorchHijack(p)
            if opts.sd_sampling == "A1111":
                from k_diff.k_diffusion import sampling as _s
            else:
                from ldm_patched.k_diffusion import sampling as _s
            _s.torch = hijack
            try:
                from ldm_patched.k_diffusion import sampling as _s2
                if _s2 is not _s:
                    _s2.torch = hijack
            except Exception:
                pass
        except Exception:
            pass

        return {}

    def get_sigmas(self, p, steps):
        from modules import sd_schedulers

        discard = (
            self.config is not None
            and self.config.options.get("discard_next_to_last_sigma", False)
        ) or opts.always_discard_next_to_last_sigma

        if discard:
            steps += 1
            p.extra_generation_params["Discard penultimate sigma"] = True

        scheduler_name = (p.hr_scheduler if p.is_hr_pass else p.scheduler) or "Automatic"
        if scheduler_name == "Automatic":
            scheduler_name = self.config.options.get("scheduler", None)

        scheduler = sd_schedulers.schedulers_map.get(scheduler_name)

        m_sigma_min = self.model_wrap.sigmas[0].item()
        m_sigma_max = self.model_wrap.sigmas[-1].item()
        sigma_min, sigma_max = (
            (0.1, 10)
            if getattr(opts, 'use_old_karras_scheduler_sigmas', False)
            else (m_sigma_min, m_sigma_max)
        )

        if p.sampler_noise_scheduler_override:
            sigmas = p.sampler_noise_scheduler_override(steps)
        elif scheduler is None or scheduler.function is None:
            sigmas = self.model_wrap.get_sigmas(steps)
        else:
            kwargs = {"sigma_min": sigma_min, "sigma_max": sigma_max}

            if scheduler.label != "Automatic" and not p.is_hr_pass:
                p.extra_generation_params["Schedule type"] = scheduler.label
            elif scheduler.label != p.extra_generation_params.get("Schedule type"):
                p.extra_generation_params["Hires schedule type"] = scheduler.label

            if opts.sigma_min != 0 and opts.sigma_min != m_sigma_min:
                kwargs["sigma_min"] = opts.sigma_min
                p.extra_generation_params["Schedule min sigma"] = opts.sigma_min
            if opts.sigma_max != 0 and opts.sigma_max != m_sigma_max:
                kwargs["sigma_max"] = opts.sigma_max
                p.extra_generation_params["Schedule max sigma"] = opts.sigma_max

            if scheduler.default_rho != -1 and opts.rho != 0 and opts.rho != scheduler.default_rho:
                kwargs["rho"] = opts.rho
                p.extra_generation_params["Schedule rho"] = opts.rho

            if scheduler.need_inner_model:
                kwargs["inner_model"] = self.model_wrap

            sigmas = scheduler.function(n=steps, **kwargs, device=shared.device)

        if discard:
            sigmas = torch.cat([sigmas[:-2], sigmas[-1:]])

        return sigmas.cpu()

    def _run(self, p, x, sigmas, n_steps):
        """Use Script UI values if available, otherwise fall back to Settings tab values."""
        log_rtol  = getattr(p, "_rk_log_rtol",  _get(OPT_LOG_RTOL,  DEF_LOG_RTOL))
        log_atol  = getattr(p, "_rk_log_atol",  _get(OPT_LOG_ATOL,  DEF_LOG_ATOL))
        max_steps = int(getattr(p, "_rk_max_steps", _get(OPT_MAX_STEPS, DEF_MAX_STEPS)))
        min_sigma = getattr(p, "_rk_min_sigma", _get(OPT_MIN_SIGMA, DEF_MIN_SIGMA))
        pcoeff    = getattr(p, "_rk_pcoeff",    _get(OPT_PCOEFF,    DEF_PCOEFF))
        icoeff    = getattr(p, "_rk_icoeff",    _get(OPT_ICOEFF,    DEF_ICOEFF))
        dcoeff    = getattr(p, "_rk_dcoeff",    _get(OPT_DCOEFF,    DEF_DCOEFF))

        # Record in infotext
        p.extra_generation_params["RK method"]    = self.method_name
        p.extra_generation_params["RK log_rtol"]  = log_rtol
        p.extra_generation_params["RK log_atol"]  = log_atol
        p.extra_generation_params["RK max_steps"] = max_steps

        return _run_rk_sampler(
            method_name = self.method_name,
            model       = self.model_wrap_cfg,
            x           = x,
            sigmas      = sigmas,
            extra_args  = self.sampler_extra_args,
            callback    = self.callback_state,
            log_rtol    = log_rtol,
            log_atol    = log_atol,
            max_steps   = max_steps,
            min_sigma   = min_sigma,
            pcoeff      = pcoeff,
            icoeff      = icoeff,
            dcoeff      = dcoeff,
        )

    # ------------------------------------------------------------------
    # txt2img
    # ------------------------------------------------------------------
    def sample(self, p, x, conditioning, unconditional_conditioning,
               steps=None, image_conditioning=None):
        logger.debug(
            "[RK Sampler] sample() called: is_hr_pass=%s _rk_txt2img_method=%s",
            getattr(p, "is_hr_pass", "N/A"),
            getattr(p, "_rk_txt2img_method", "NOT SET"),
        )
        # Delegate to reForge sampler when 'Use reForge sampler' is selected
        if getattr(p, "_rk_txt2img_method", "Use same sampler") in ("Use same sampler", "Use reForge sampler"):
            from modules import sd_samplers
            fallback_name = getattr(opts, "sampler_name", None) or "Euler"
            sampler = sd_samplers.create_sampler(fallback_name, p.sd_model)
            return sampler.sample(p, x, conditioning, unconditional_conditioning,
                                  steps=steps, image_conditioning=image_conditioning)

        # Delegate to TDE Sampler when '→ TDE Sampler' is selected
        if getattr(p, "_rk_txt2img_method", "") == "→ TDE Sampler":
            from modules import sd_samplers
            sampler = sd_samplers.create_sampler("TDE Sampler", shared.sd_model)
            return sampler.sample(p, x, conditioning, unconditional_conditioning,
                                  steps=steps, image_conditioning=image_conditioning)

        from backend.sampling.sampling_function import sampling_prepare, sampling_cleanup

        unet_patcher = self.model_wrap.inner_model.forge_objects.unet
        sampling_prepare(unet_patcher, x=x)


        steps  = steps or p.steps
        sigmas = self.get_sigmas(p, steps).to(x.device)

        if opts.sgm_noise_multiplier:
            p.extra_generation_params["SGM noise multiplier"] = True
            x = x * torch.sqrt(1.0 + sigmas[0] ** 2.0)
        else:
            x = x * sigmas[0]

        self.initialize(p)
        self.last_latent = x
        self.sampler_extra_args = {
            "cond":         conditioning,
            "image_cond":   image_conditioning,
            "uncond":       unconditional_conditioning,
            "cond_scale":   p.cfg_scale,
            "s_min_uncond": self.s_min_uncond,
        }

        samples = self._run(p, x, sigmas, steps)
        self.add_infotext(p)
        sampling_cleanup(unet_patcher)
        return samples

    # ------------------------------------------------------------------
    # img2img / hires.fix
    # ------------------------------------------------------------------
    def sample_img2img(self, p, x, noise, conditioning, unconditional_conditioning,
                       steps=None, image_conditioning=None):
        hr_method = getattr(p, "_rk_hr_method", "Use same sampler")
        logger.debug(
            "[RK Sampler] sample_img2img called: is_hr_pass=%s _rk_hr_method=%s",
            getattr(p, "is_hr_pass", "N/A"), hr_method
        )
        # Delegate to reForge sampler when 'Use same sampler' is selected
        if hr_method in ("Use same sampler", "Use reForge sampler"):
            from modules import sd_samplers
            fallback_name = getattr(opts, "sampler_name", None) or "Euler"
            sampler = sd_samplers.create_sampler(fallback_name, p.sd_model)
            return sampler.sample_img2img(p, x, noise, conditioning, unconditional_conditioning,
                                          steps=steps, image_conditioning=image_conditioning)

        # Delegate to TDE Sampler when '→ TDE Sampler' is selected
        if hr_method == "→ TDE Sampler":
            from modules import sd_samplers
            sampler = sd_samplers.create_sampler("TDE Sampler", shared.sd_model)
            return sampler.sample_img2img(p, x, noise, conditioning, unconditional_conditioning,
                                          steps=steps, image_conditioning=image_conditioning)

        from backend.sampling.sampling_function import sampling_prepare, sampling_cleanup

        unet_patcher = self.model_wrap.inner_model.forge_objects.unet
        sampling_prepare(unet_patcher, x=x)


        steps, t_enc = sd_samplers_common.setup_img2img_steps(p, steps)
        sigmas       = self.get_sigmas(p, steps).to(x.device)
        sigma_sched  = sigmas[steps - t_enc - 1:]

        x  = x.to(noise)
        xi = _flow_aware_noise_injection(x, noise, sigma_sched[0], self.model_wrap)

        if opts.img2img_extra_noise > 0:
            p.extra_generation_params["Extra noise"] = opts.img2img_extra_noise
            from modules.script_callbacks import ExtraNoiseParams, extra_noise_callback
            enp = ExtraNoiseParams(noise, x, xi)
            extra_noise_callback(enp)
            noise = enp.noise
            xi   += noise * opts.img2img_extra_noise

        self.initialize(p)
        self.model_wrap_cfg.init_latent = x
        self.last_latent = x
        self.sampler_extra_args = {
            "cond":         conditioning,
            "image_cond":   image_conditioning,
            "uncond":       unconditional_conditioning,
            "cond_scale":   p.cfg_scale,
            "s_min_uncond": self.s_min_uncond,
        }

        samples = self._run(p, xi, sigma_sched, t_enc + 1)
        self.add_infotext(p)
        sampling_cleanup(unet_patcher)
        return samples


# ===========================================================================
# Section 4: Settings Tab UI
# ===========================================================================

# --- Default values for advanced settings (change via Settings > Advanced Settings) ---
RK_DEFAULT_MIN_SIGMA = 1e-5
RK_PID_KP            = 0.0
RK_PID_KI            = 1.0
RK_PID_KD            = 0.0
# --------------------------------------------------------------------------


def _on_ui_settings():
    section = ("rk_sampler", "RK Sampler")

    # --- Standard Settings ---
    shared.opts.add_option(OPT_LOG_RTOL, shared.OptionInfo(
        default=DEF_LOG_RTOL,
        label="Log Relative Tolerance (10^x)",
        component=gr.Slider,
        component_args={"minimum": -7.0, "maximum": 0.0, "step": 0.5},
        section=section,
    ).info("Effective for ae_* / se_* methods. Smaller = more precise but slower. Set above atol."))

    shared.opts.add_option(OPT_LOG_ATOL, shared.OptionInfo(
        default=DEF_LOG_ATOL,
        label="Log Absolute Tolerance (10^x)",
        component=gr.Slider,
        component_args={"minimum": -7.0, "maximum": 0.0, "step": 0.5},
        section=section,
    ).info("Effective for ae_* / se_* methods. Smaller = more precise but slower. Set below rtol."))

    shared.opts.add_option(OPT_MAX_STEPS, shared.OptionInfo(
        default=DEF_MAX_STEPS,
        label="Max ODE Steps",
        component=gr.Slider,
        component_args={"minimum": 10, "maximum": 5000, "step": 10},
        section=section,
    ).info("Effective for ae_* / se_* methods. Failsafe upper limit when adaptive steps do not converge."))

    # --- Advanced Settings (accordion) ---
    shared.opts.add_option(OPT_MIN_SIGMA, shared.OptionInfo(
        default=RK_DEFAULT_MIN_SIGMA,
        label="[Advanced] Min Sigma (lower cutoff)",
        component=gr.Slider,
        component_args={"minimum": 0.0, "maximum": 0.1, "step": 1e-5},
        section=section,
        category_id="rk_advanced",
    ).info("Passes with zero gradient below this sigma. Normally no need to change."))

    shared.opts.add_option(OPT_PCOEFF, shared.OptionInfo(
        default=RK_PID_KP,
        label="[Advanced] PID P Coefficient (ae_* only)",
        component=gr.Slider,
        component_args={"minimum": 0.0, "maximum": 1.0, "step": 0.05},
        section=section,
        category_id="rk_advanced",
    ).info("Normally no need to change."))

    shared.opts.add_option(OPT_ICOEFF, shared.OptionInfo(
        default=RK_PID_KI,
        label="[Advanced] PID I Coefficient (ae_* only)",
        component=gr.Slider,
        component_args={"minimum": 0.0, "maximum": 2.0, "step": 0.05},
        section=section,
        category_id="rk_advanced",
    ).info("Normally no need to change."))

    shared.opts.add_option(OPT_DCOEFF, shared.OptionInfo(
        default=RK_PID_KD,
        label="[Advanced] PID D Coefficient (ae_* only)",
        component=gr.Slider,
        component_args={"minimum": 0.0, "maximum": 1.0, "step": 0.05},
        section=section,
        category_id="rk_advanced",
    ).info("Normally no need to change."))


script_callbacks.on_ui_settings(_on_ui_settings)


# ===========================================================================
# Section 5: Sampler Registration for reForge
# ===========================================================================

class RKScriptSampler(RKMethodSampler):
    """
    Sampler for the "RK Sampler" dropdown.
    The method to use is passed from Script UI via p._rk_txt2img_method / p._rk_hr_method.
    Falls back to ae_dopri5 if neither is set.
    """

    def __init__(self, sd_model):
        super().__init__(sd_model, "ae_dopri5")  # default (overwritten later)
        self.funcname = "rk_script_sampler"

    def _get_method(self, p) -> str:
        """Select method based on whether this is a hires.fix pass."""
        if getattr(p, "is_hr_pass", False):
            return getattr(p, "_rk_hr_method", "Use same sampler")
        else:
            return getattr(p, "_rk_txt2img_method", "Use same sampler")

    def _run(self, p, x, sigmas, n_steps):
        method = self._get_method(p)

        # Should not reach here when 'Use same sampler' is selected, but guard just in case
        if method in ("Use same sampler", None):
            raise sd_samplers_common.InterruptedException

        self.method_name = method

        log_rtol  = getattr(p, "_rk_log_rtol",  _get(OPT_LOG_RTOL,  DEF_LOG_RTOL))
        log_atol  = getattr(p, "_rk_log_atol",  _get(OPT_LOG_ATOL,  DEF_LOG_ATOL))
        max_steps = int(getattr(p, "_rk_max_steps", _get(OPT_MAX_STEPS, DEF_MAX_STEPS)))
        min_sigma = getattr(p, "_rk_min_sigma", _get(OPT_MIN_SIGMA, DEF_MIN_SIGMA))
        pcoeff    = getattr(p, "_rk_pcoeff",    _get(OPT_PCOEFF,    DEF_PCOEFF))
        icoeff    = getattr(p, "_rk_icoeff",    _get(OPT_ICOEFF,    DEF_ICOEFF))
        dcoeff    = getattr(p, "_rk_dcoeff",    _get(OPT_DCOEFF,    DEF_DCOEFF))

        p.extra_generation_params["RK method"]    = self.method_name
        p.extra_generation_params["RK log_rtol"]  = log_rtol
        p.extra_generation_params["RK log_atol"]  = log_atol
        p.extra_generation_params["RK max_steps"] = max_steps

        return _run_rk_sampler(
            method_name = self.method_name,
            model       = self.model_wrap_cfg,
            x           = x,
            sigmas      = sigmas,
            extra_args  = self.sampler_extra_args,
            callback    = self.callback_state,
            log_rtol    = log_rtol,
            log_atol    = log_atol,
            max_steps   = max_steps,
            min_sigma   = min_sigma,
            pcoeff      = pcoeff,
            icoeff      = icoeff,
            dcoeff      = dcoeff,
        )


def _register():
    from modules import sd_samplers

    added = 0

    # "RK Sampler" — unified sampler integrated with Script UI
    rk_script_data = sd_samplers_common.SamplerData(
        name        = "RK Sampler",
        constructor = lambda sd_model: RKScriptSampler(sd_model),
        aliases     = ["rk_sampler"],
        options     = {"scheduler": None},
    )
    if not any(s.name == "RK Sampler" for s in sd_samplers.all_samplers):
        sd_samplers.all_samplers.append(rk_script_data)
        added += 1
    sd_samplers.all_samplers_map["RK Sampler"] = rk_script_data

    sd_samplers.set_samplers()
    if added > 0:
        logger.warning(
            "[RK Sampler] Registered %d sampler(s) (torchode=%s)",
            added, HAS_TORCHODE
        )


# Re-register after model load (handles all_samplers_map reset on checkpoint switch)
def _on_model_loaded(sd_model):
    try:
        _register()
        from modules import sd_samplers
        rk_in_map = "RK Sampler" in sd_samplers.all_samplers_map
        logger.warning("[RK Sampler] on_model_loaded: RK Sampler in all_samplers_map = %s", rk_in_map)
    except Exception:
        import traceback
        logger.error("[RK Sampler] on_model_loaded error:\n%s", traceback.format_exc())


script_callbacks.on_model_loaded(_on_model_loaded)


try:
    _register()
    logger.warning("[RK Sampler] Startup registration complete")
except Exception as _exc:
    import traceback
    logger.error("[RK Sampler] Registration error:\n%s", traceback.format_exc())


# ===========================================================================
# Section 6: Script Class — adds UI to the Script pane in the generation tab
#   Displayed as "RK Sampler (Script)".
#   When "RK Sampler (Script)" is selected in the Sampling method dropdown,
#   it operates with the Method and rtol/atol set here.
# ===========================================================================

# Dropdown references for forced update on page load (module level)
_rk_method_dropdowns = []

try:
    from modules import scripts

    USE_REFORGE = "Use same sampler"
    TO_TDE      = "→ TDE Sampler"
    METHOD_NAMES = [USE_REFORGE, TO_TDE] + list(ALL_METHODS.keys())

    class RKSamplerScript(scripts.Script):

        def title(self):
            return "RK Sampler"

        def show(self, is_img2img):
            return scripts.AlwaysVisible

        def ui(self, is_img2img):
            from modules.ui_components import InputAccordion
            with InputAccordion(False, label="RK Sampler") as enabled:
                with gr.Row():
                    txt2img_method = gr.Dropdown(
                        choices=METHOD_NAMES,
                        value=USE_REFORGE,
                        label="txt2img Method",
                    )
                    hr_method = gr.Dropdown(
                        choices=METHOD_NAMES,
                        value=USE_REFORGE,
                        label="hires.fix Method",
                        visible=not is_img2img,
                    )
                    _rk_method_dropdowns.append((txt2img_method, hr_method))
                with gr.Row():
                    log_rtol = gr.Slider(
                        minimum=-7.0, maximum=0.0, step=0.5,
                        value=DEF_LOG_RTOL,
                        label="Log Relative Tolerance (10^x)",
                    )
                    log_atol = gr.Slider(
                        minimum=-7.0, maximum=0.0, step=0.5,
                        value=DEF_LOG_ATOL,
                        label="Log Absolute Tolerance (10^x)",
                    )
                with gr.Accordion("Advanced Settings", open=False):
                    max_steps = gr.Slider(
                        minimum=10, maximum=5000, step=10,
                        value=DEF_MAX_STEPS,
                        label="Max ODE Steps",
                    )
                    min_sigma = gr.Slider(
                        minimum=0.0, maximum=0.1, step=1e-5,
                        value=RK_DEFAULT_MIN_SIGMA,
                        label="Min Sigma",
                    )
                    with gr.Row():
                        pid_p = gr.Slider(
                            minimum=0.0, maximum=1.0, step=0.05,
                            value=RK_PID_KP,
                            label="PID P Coefficient",
                        )
                        pid_i = gr.Slider(
                            minimum=0.0, maximum=2.0, step=0.05,
                            value=RK_PID_KI,
                            label="PID I Coefficient",
                        )
                        pid_d = gr.Slider(
                            minimum=0.0, maximum=1.0, step=0.05,
                            value=RK_PID_KD,
                            label="PID D Coefficient",
                        )

            return [enabled, txt2img_method, hr_method, log_rtol, log_atol,
                    max_steps, min_sigma, pid_p, pid_i, pid_d]

        def process(self, p,
                    enabled, txt2img_method, hr_method, log_rtol, log_atol,
                    max_steps, min_sigma, pid_p, pid_i, pid_d):
            logger.debug(
                "[RK Sampler] process() called: sampler_name='%s' is_hr_pass=%s enabled=%s",
                getattr(p, "sampler_name", "N/A"),
                getattr(p, "is_hr_pass", "N/A"),
                enabled,
            )
            # Do nothing when the enable checkbox is OFF
            if not enabled:
                return

            p._rk_txt2img_method = txt2img_method
            p._rk_hr_method      = hr_method
            p._rk_log_rtol       = float(log_rtol)
            p._rk_log_atol       = float(log_atol)
            p._rk_max_steps      = int(max_steps)
            p._rk_min_sigma      = float(min_sigma)
            p._rk_pcoeff         = float(pid_p)
            p._rk_icoeff         = float(pid_i)
            p._rk_dcoeff         = float(pid_d)

except ImportError:
    pass


# Reset dropdowns to "Use same sampler" on page load
def _on_app_started(demo, app):
    if not _rk_method_dropdowns:
        return
    try:
        txt2img_dropdowns = [pair[0] for pair in _rk_method_dropdowns]
        hr_dropdowns      = [pair[1] for pair in _rk_method_dropdowns]
        all_dropdowns     = txt2img_dropdowns + hr_dropdowns

        def _reset_methods():
            return [USE_REFORGE] * len(all_dropdowns)

        with demo:
            demo.load(fn=_reset_methods, inputs=[], outputs=all_dropdowns)
        logger.warning("[RK Sampler] demo.load registered: %d dropdown(s)", len(all_dropdowns))
    except Exception:
        import traceback
        logger.error("[RK Sampler] demo.load registration error:\n%s", traceback.format_exc())


script_callbacks.on_app_started(_on_app_started)

