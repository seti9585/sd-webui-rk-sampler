"""
RK Sampler — reForge 拡張機能
==============================
配置場所: extensions/rk-sampler/scripts/rk_sampler.py

ComfyUI-RK-Sampler (https://github.com/memmaptensor/ComfyUI-RK-Sampler) の
全 ODE メソッドを reForge に移植した独立サンプラー拡張。

【特徴】
- 19 種のメソッドを個別のサンプラーとして登録
  (ae_bosh3, ae_cash_karp5, ae_dopri5, ae_dopri8,
   ae_fehlberg2, ae_fehlberg5, ae_heun_euler2, ae_midpoint2,
   ae_ralston2, ae_tsit5,
   fe_euler1, fe_heun3, fe_kutta3, fe_kutta4, fe_kutta_38th4,
   fe_ralston3, fe_ralston4, fe_ssprk3, fe_wray3)
- txt2img と hires.fix で別メソッドを選択可能
  → reForge の Custom ODE Settings は 1 つしか選べないため
- 適応ステップ (ae_*): PID コントローラーで自動刻み幅調整
- 固定ステップ (fe_*): sigma スケジュールに従って固定刻み幅
- Settings タブで rtol / atol / max_steps をサンプラーごとに設定可能

依存:
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
# rk_core パッケージを scripts の親ディレクトリから探す
# ---------------------------------------------------------------------------
_EXT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _EXT_DIR not in sys.path:
    sys.path.insert(0, _EXT_DIR)

# ---------------------------------------------------------------------------
# reForge コアモジュール
# ---------------------------------------------------------------------------
from modules import sd_samplers_common, shared, script_callbacks
from modules.shared import opts

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# torchode チェック
# ---------------------------------------------------------------------------
try:
    import torchode
    HAS_TORCHODE = True
except ModuleNotFoundError:
    HAS_TORCHODE = False
    logger.error(
        "[RK Sampler] torchode が見つかりません。pip install torchode を実行してください。"
    )

HAS_MPS = torch.backends.mps.is_available()
try:
    import torch_directml  # noqa: F401
    HAS_DML = True
except ModuleNotFoundError:
    HAS_DML = False


# ===========================================================================
# Section 0: メソッド定義と登録テーブル
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

# scipy ベースの適応ステップメソッド（バッチを1枚ずつ処理）
SCIPY_METHODS = {
    "se_RK23":   {"order": 3,  "nfe_per_step": 3},
    "se_RK45":   {"order": 5,  "nfe_per_step": 6},
    "se_DOP853": {"order": 8,  "nfe_per_step": 12},
}

ALL_METHODS = {**ADAPTIVE_METHODS, **FIXED_METHODS, **SCIPY_METHODS}

# opts キー
OPT_LOG_RTOL  = "rk_sampler_log_rtol"    # 適応ステップ用 relative tolerance
OPT_LOG_ATOL  = "rk_sampler_log_atol"    # 適応ステップ用 absolute tolerance
OPT_MAX_STEPS = "rk_sampler_max_steps"   # 適応ステップ上限
OPT_MIN_SIGMA = "rk_sampler_min_sigma"   # sigma の打ち切り下限
OPT_PCOEFF    = "rk_sampler_pcoeff"      # PID P 係数
OPT_ICOEFF    = "rk_sampler_icoeff"      # PID I 係数
OPT_DCOEFF    = "rk_sampler_dcoeff"      # PID D 係数

# デフォルト値
DEF_LOG_RTOL  = -3.0
DEF_LOG_ATOL  = -4.0
DEF_MAX_STEPS = 1000
DEF_MIN_SIGMA = 1e-5
DEF_PCOEFF    = 0.0
DEF_ICOEFF    = 1.0
DEF_DCOEFF    = 0.0


def _get(key, default):
    return getattr(opts, key, default)


# ===========================================================================
# Section 1: メソッドクラスの動的ロード
# ===========================================================================

def _load_method_class(method_name: str):
    """メソッド名からクラスを動的にインポートして返す。scipy メソッドは None を返す。"""
    if method_name in SCIPY_METHODS:
        return None
    import importlib
    module_path, class_name = ALL_METHODS[method_name]
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


# ===========================================================================
# Section 2: コアサンプリング関数
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
    """メソッドに応じて torchode 版または scipy 版にディスパッチする。"""
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
    scipy.integrate.solve_ivp を使った適応ステップサンプラー。
    バッチを1枚ずつ処理する。
    ComfyUI-RK-Sampler の RungeKuttaSamplerImpl._call_scipy() に相当。
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

            # ComfyUI-RK-Sampler と同じ方法で scipy の内部を差し替える
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
                logger.debug("[RK Sampler] scipy 中断されました（InterruptedException）")
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

    # 中断時にバッチが足りない場合を補完
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
    torchode の AutoDiffAdjoint を使って ODE を解く。
    ComfyUI-RK-Sampler の RungeKuttaSamplerImpl._call_torchode() に相当。
    """
    from rk_core.ode_term import ReForgeODETerm
    from rk_core.solvers.auto_diff_adjoint import AutoDiffAdjoint
    from rk_core.controllers.pid_controller import PIDController
    from rk_core.controllers.scheduled_controller import ScheduledController

    is_adaptive = method_name in ADAPTIVE_METHODS
    # デバイス・dtype 設定
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

    # プログレスバー
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
            cfg_denoiser = model,   # step カウンタ更新のため渡す
        )
        term = torchode.ODETerm(ode_fn)

        # ステップサイズコントローラー
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

        # メソッドクラスのインスタンス化
        MethodClass = _load_method_class(method_name)
        step_method = MethodClass(term=term)

        # ソルバー
        adjoint = AutoDiffAdjoint(
            step_method,
            controller,
            max_steps                        = max_steps if is_adaptive else None,
            backprop_through_step_size_control= False,
            dense_output                     = False,
        )

        # 問題設定: t_start=sigma_max → t_end=sigma_min
        problem = torchode.InitialValueProblem(
            y0      = x_c.flatten(start_dim=1),
            t_start = torch.full((batch,), t_max, device=c_device, dtype=c_dtype),
            t_end   = torch.full((batch,), t_min, device=c_device, dtype=c_dtype),
            t_eval  = None,
        )

        try:
            result = adjoint.solve(problem)
        except sd_samplers_common.InterruptedException:
            # ユーザーによる中断（Skip/Interrupt）— x をそのまま返して正常終了
            logger.debug("[RK Sampler] 中断されました（InterruptedException）")
            return x.to(o_device, dtype=o_dtype)

    samples = result.ys[:, -1].reshape(o_shape).to(o_device, dtype=o_dtype)

    # エラーチェック
    for i, status in enumerate(result.status):
        s = status.item()
        if s != 0:
            reason = torchode.status_codes.Status(s)
            logger.warning(f"[RK Sampler] Sample #{i} failed: {reason}")
            samples[i] = torch.full_like(samples[i], float("nan"))

    return samples


# ===========================================================================
# Section 3: reForge Sampler ラッパークラス（全メソッド共通）
# ===========================================================================

class RKMethodSampler(sd_samplers_common.Sampler):
    """
    1つの RK メソッドに対応する reForge サンプラー。
    _register() で全メソッド分のインスタンスが作られる。
    """

    def __init__(self, sd_model, method_name: str):
        # 固定ステップの fe_ はサンプラーとして sample_fn が必要なので
        # ダミーの callable を渡す（実際は _run_rk_sampler を使う）
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
        self.model_wrap_cfg.total_steps     = p.steps   # apply_refiner() が step/total_steps を計算するため必須
        self.model_wrap_cfg.steps           = p.steps   # detail_daemon 等が // 演算するため必須
        self.model_wrap_cfg.image_cfg_scale = getattr(p, "image_cfg_scale", None)
        self.eta          = p.eta if p.eta is not None else 0.0
        self.s_min_uncond = getattr(p, "s_min_uncond", 0.0)

        # TorchHijack（乱数再現性）
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
            if opts.use_old_karras_scheduler_sigmas
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
        """Script UI の値があればそちらを優先、なければ Settings タブの値を使う。"""
        log_rtol  = getattr(p, "_rk_log_rtol",  _get(OPT_LOG_RTOL,  DEF_LOG_RTOL))
        log_atol  = getattr(p, "_rk_log_atol",  _get(OPT_LOG_ATOL,  DEF_LOG_ATOL))
        max_steps = int(getattr(p, "_rk_max_steps", _get(OPT_MAX_STEPS, DEF_MAX_STEPS)))
        min_sigma = getattr(p, "_rk_min_sigma", _get(OPT_MIN_SIGMA, DEF_MIN_SIGMA))
        pcoeff    = getattr(p, "_rk_pcoeff",    _get(OPT_PCOEFF,    DEF_PCOEFF))
        icoeff    = getattr(p, "_rk_icoeff",    _get(OPT_ICOEFF,    DEF_ICOEFF))
        dcoeff    = getattr(p, "_rk_dcoeff",    _get(OPT_DCOEFF,    DEF_DCOEFF))

        # infotext に記録
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
        # 「Use reForge sampler」が選ばれている場合は reForge の sampler に委譲
        if getattr(p, "_rk_txt2img_method", "Use same sampler") in ("Use same sampler", "Use reForge sampler"):
            from modules import sd_samplers
            fallback_name = getattr(opts, "sampler_name", None) or "Euler"
            sampler = sd_samplers.create_sampler(fallback_name, p.sd_model)
            return sampler.sample(p, x, conditioning, unconditional_conditioning,
                                  steps=steps, image_conditioning=image_conditioning)

        # 「→ TDE Sampler」が選ばれている場合は TDE Sampler に委譲
        if getattr(p, "_rk_txt2img_method", "") == "→ TDE Sampler":
            from modules import sd_samplers
            sampler = sd_samplers.create_sampler("TDE Sampler", shared.sd_model)
            return sampler.sample(p, x, conditioning, unconditional_conditioning,
                                  steps=steps, image_conditioning=image_conditioning)

        from modules_forge.forge_sampler import sampling_prepare, sampling_cleanup

        unet_patcher = self.model_wrap.inner_model.forge_objects.unet
        sampling_prepare(unet_patcher, x=x)

        self.model_wrap.log_sigmas = self.model_wrap.log_sigmas.to(x.device)
        self.model_wrap.sigmas     = self.model_wrap.sigmas.to(x.device)

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
        # 「Use same sampler」が選ばれている場合は reForge の sampler に委譲
        if hr_method in ("Use same sampler", "Use reForge sampler"):
            from modules import sd_samplers
            fallback_name = getattr(opts, "sampler_name", None) or "Euler"
            sampler = sd_samplers.create_sampler(fallback_name, p.sd_model)
            return sampler.sample_img2img(p, x, noise, conditioning, unconditional_conditioning,
                                          steps=steps, image_conditioning=image_conditioning)

        # 「→ TDE Sampler」が選ばれている場合は TDE Sampler に委譲
        if hr_method == "→ TDE Sampler":
            from modules import sd_samplers
            sampler = sd_samplers.create_sampler("TDE Sampler", shared.sd_model)
            return sampler.sample_img2img(p, x, noise, conditioning, unconditional_conditioning,
                                          steps=steps, image_conditioning=image_conditioning)

        from modules_forge.forge_sampler import sampling_prepare, sampling_cleanup

        unet_patcher = self.model_wrap.inner_model.forge_objects.unet
        sampling_prepare(unet_patcher, x=x)

        self.model_wrap.log_sigmas = self.model_wrap.log_sigmas.to(x.device)
        self.model_wrap.sigmas     = self.model_wrap.sigmas.to(x.device)

        steps, t_enc = sd_samplers_common.setup_img2img_steps(p, steps)
        sigmas       = self.get_sigmas(p, steps).to(x.device)
        sigma_sched  = sigmas[steps - t_enc - 1:]

        x  = x.to(noise)
        xi = x + noise * sigma_sched[0]

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
# Section 4: Settings タブ UI
# ===========================================================================

# --- 拡張設定のデフォルト値（変更したい場合は Settings の「拡張設定」から） ---
RK_DEFAULT_MIN_SIGMA = 1e-5
RK_PID_KP            = 0.0
RK_PID_KI            = 1.0
RK_PID_KD            = 0.0
# --------------------------------------------------------------------------


def _on_ui_settings():
    section = ("rk_sampler", "RK Sampler")

    # --- 通常設定 ---
    shared.opts.add_option(OPT_LOG_RTOL, shared.OptionInfo(
        default=DEF_LOG_RTOL,
        label="Log Relative Tolerance (10^x)",
        component=gr.Slider,
        component_args={"minimum": -7.0, "maximum": 0.0, "step": 0.5},
        section=section,
    ).info("ae_* / se_* メソッドで有効。小さいほど精密・低速。atol 以上に設定してください。"))

    shared.opts.add_option(OPT_LOG_ATOL, shared.OptionInfo(
        default=DEF_LOG_ATOL,
        label="Log Absolute Tolerance (10^x)",
        component=gr.Slider,
        component_args={"minimum": -7.0, "maximum": 0.0, "step": 0.5},
        section=section,
    ).info("ae_* / se_* メソッドで有効。小さいほど精密・低速。rtol 以下に設定してください。"))

    shared.opts.add_option(OPT_MAX_STEPS, shared.OptionInfo(
        default=DEF_MAX_STEPS,
        label="Max ODE Steps",
        component=gr.Slider,
        component_args={"minimum": 10, "maximum": 5000, "step": 10},
        section=section,
    ).info("ae_* / se_* メソッドで有効。適応ステップが収束しない場合のフェイルセーフ上限。"))

    # --- 拡張設定（アコーディオン） ---
    shared.opts.add_option(OPT_MIN_SIGMA, shared.OptionInfo(
        default=RK_DEFAULT_MIN_SIGMA,
        label="[拡張設定] Min Sigma (打ち切り下限)",
        component=gr.Slider,
        component_args={"minimum": 0.0, "maximum": 0.1, "step": 1e-5},
        section=section,
        category_id="rk_advanced",
    ).info("この sigma 以下ではゼロ勾配で通過します。通常は変更不要。"))

    shared.opts.add_option(OPT_PCOEFF, shared.OptionInfo(
        default=RK_PID_KP,
        label="[拡張設定] PID P 係数 (ae_* のみ有効)",
        component=gr.Slider,
        component_args={"minimum": 0.0, "maximum": 1.0, "step": 0.05},
        section=section,
        category_id="rk_advanced",
    ).info("通常は変更不要。"))

    shared.opts.add_option(OPT_ICOEFF, shared.OptionInfo(
        default=RK_PID_KI,
        label="[拡張設定] PID I 係数 (ae_* のみ有効)",
        component=gr.Slider,
        component_args={"minimum": 0.0, "maximum": 2.0, "step": 0.05},
        section=section,
        category_id="rk_advanced",
    ).info("通常は変更不要。"))

    shared.opts.add_option(OPT_DCOEFF, shared.OptionInfo(
        default=RK_PID_KD,
        label="[拡張設定] PID D 係数 (ae_* のみ有効)",
        component=gr.Slider,
        component_args={"minimum": 0.0, "maximum": 1.0, "step": 0.05},
        section=section,
        category_id="rk_advanced",
    ).info("通常は変更不要。"))


script_callbacks.on_ui_settings(_on_ui_settings)


# ===========================================================================
# Section 5: reForge へのサンプラー登録
# ===========================================================================

class RKScriptSampler(RKMethodSampler):
    """
    「RK Sampler」ドロップダウン用サンプラー。
    実際に使う Method は Script UI から p._rk_txt2img_method / p._rk_hr_method で渡される。
    どちらも未設定の場合は ae_dopri5 にフォールバック。
    """

    def __init__(self, sd_model):
        super().__init__(sd_model, "ae_dopri5")  # デフォルト（後で上書きされる）
        self.funcname = "rk_script_sampler"

    def _get_method(self, p) -> str:
        """hires.fix パスかどうかで使うメソッドを選ぶ。"""
        if getattr(p, "is_hr_pass", False):
            return getattr(p, "_rk_hr_method", "Use same sampler")
        else:
            return getattr(p, "_rk_txt2img_method", "Use same sampler")

    def _run(self, p, x, sigmas, n_steps):
        method = self._get_method(p)

        # 「Use same sampler」が選ばれている場合はここに来ないはずだが念のため
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

    # 「RK Sampler」— Script UI と連携する統合サンプラー
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
            "[RK Sampler] %d サンプラーを登録しました (torchode=%s)",
            added, HAS_TORCHODE
        )


# モデルロード後にも再登録（チェックポイント切り替えで all_samplers_map がリセットされる場合の対策）
def _on_model_loaded(sd_model):
    try:
        _register()
        from modules import sd_samplers
        rk_in_map = "RK Sampler" in sd_samplers.all_samplers_map
        logger.warning("[RK Sampler] on_model_loaded: RK Sampler in all_samplers_map = %s", rk_in_map)
    except Exception:
        import traceback
        logger.error("[RK Sampler] on_model_loaded エラー:\n%s", traceback.format_exc())


script_callbacks.on_model_loaded(_on_model_loaded)


try:
    _register()
    logger.warning("[RK Sampler] 起動時登録完了")
except Exception as _exc:
    import traceback
    logger.error("[RK Sampler] 登録エラー:\n%s", traceback.format_exc())


# ===========================================================================
# Section 6: Script クラス — 生成タブの Script ペインに UI を追加
#   「RK Sampler (Script)」として表示される。
#   Sampling method ドロップダウンで「RK Sampler (Script)」を選んだとき、
#   ここで設定した Method・rtol/atol で動作する。
# ===========================================================================

# ページロード時の強制更新用ドロップダウン参照リスト（モジュールレベル）
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
                with gr.Accordion("拡張設定", open=False):
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
                            label="PID P 係数",
                        )
                        pid_i = gr.Slider(
                            minimum=0.0, maximum=2.0, step=0.05,
                            value=RK_PID_KI,
                            label="PID I 係数",
                        )
                        pid_d = gr.Slider(
                            minimum=0.0, maximum=1.0, step=0.05,
                            value=RK_PID_KD,
                            label="PID D 係数",
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
            # 有効化チェックボックスが OFF の場合は何もしない
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


# ページロード時にドロップダウンを "Use same sampler" にリセットする
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
        logger.warning("[RK Sampler] demo.load 登録完了: %d ドロップダウン", len(all_dropdowns))
    except Exception:
        import traceback
        logger.error("[RK Sampler] demo.load 登録エラー:\n%s", traceback.format_exc())


script_callbacks.on_app_started(_on_app_started)

