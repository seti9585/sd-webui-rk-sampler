"""
reForge 用 SciPy ODE Term
ComfyUI-RK-Sampler の SciPyODETerm を reForge 向けに移植。
バッチを1枚ずつ処理する（scipy.integrate.solve_ivp はベクトル化非対応）。
"""
from __future__ import annotations

import numpy as np
import torch


class ReFormeSciPyODETerm:
    """
    scipy.integrate.solve_ivp から呼ばれる ODE の右辺関数。
    1バッチサンプルずつ処理する。

    Args:
        model       : CFGDenoiserKDiffusion インスタンス
        o_device    : 元の latent のデバイス
        o_dtype     : 元の latent の dtype
        o_shape_1   : 1サンプル分の shape (C, H, W)
        min_sigma   : この sigma 以下はゼロ勾配で通過
        t_max       : sigma の最大値
        t_min       : sigma の最小値
        n_steps     : sigma スケジュールのステップ数（進捗計算用）
        extra_args  : model に渡す追加引数 dict
        callback    : reForge の callback 関数
        pbar        : tqdm プログレスバー
        cfg_denoiser: step カウンタ更新用
    """

    def __init__(
        self,
        model,
        o_device,
        o_dtype: torch.dtype,
        o_shape_1: tuple,       # (C, H, W) — バッチ次元なし
        min_sigma: float,
        t_max: float,
        t_min: float,
        n_steps: int,
        extra_args: dict | None = None,
        callback=None,
        pbar=None,
        cfg_denoiser=None,
        c_dtype=np.float64,
    ):
        self.model        = model
        self.o_device     = o_device
        self.o_dtype      = o_dtype
        self.o_shape_1    = o_shape_1
        self.min_sigma    = min_sigma
        self.t_max        = t_max
        self.t_min        = t_min
        self.n_steps      = n_steps
        self.extra_args   = extra_args or {}
        self.callback     = callback
        self.pbar         = pbar
        self.cfg_denoiser = cfg_denoiser
        self.c_dtype      = c_dtype

        # 内部状態
        self.n_callbacks   = 0
        self.pbar_step     = 0.0
        self.last_t        = None
        self.last_denoised = None

    def trigger_callback(self, t: float, y: np.ndarray):
        """scipy の _step_impl から各ステップ後に呼ばれる。"""
        self.n_callbacks += 1

        # cfg_denoiser.step を上限クランプしてインクリメント
        if self.cfg_denoiser is not None and hasattr(self.cfg_denoiser, "step"):
            total = getattr(self.cfg_denoiser, "total_steps", None)
            if total is not None:
                self.cfg_denoiser.step = min(self.n_callbacks, total - 1)
            else:
                self.cfg_denoiser.step = self.n_callbacks

        if self.pbar is None and self.callback is None:
            return

        progress   = (self.t_max - t) / max(self.t_max - self.t_min, 1e-8)
        percentage = progress * 100
        i          = round(progress * self.n_steps)

        if self.pbar is not None:
            self.pbar.update(percentage - self.pbar_step)
            self.pbar_step = percentage
            self.pbar.set_postfix({"σ": f"{t:.4f}"})
            self.pbar.refresh()

        if self.callback is not None:
            mask = (self.last_t is not None) and (self.last_t <= self.min_sigma)
            y_shaped = y.reshape(self.o_shape_1)
            if mask and self.last_denoised is not None:
                arr = self.last_denoised
            else:
                arr = y_shaped
            samples = torch.from_numpy(arr[np.newaxis, ...]).to(self.o_device, dtype=self.o_dtype)
            self.callback({
                "x":         samples,
                "i":         i - 1,
                "sigma":     t,
                "sigma_hat": t,
                "denoised":  samples,
            })

    def __call__(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        Args:
            t : float  現在の sigma 値
            y : (flat_dim,) numpy array
        Returns:
            dy/dt : (flat_dim,) numpy array
        """
        y_shaped = y.reshape(self.o_shape_1)
        mask = t <= self.min_sigma

        if mask:
            denoised = np.zeros_like(y_shaped)
            d = np.zeros_like(y_shaped)
        else:
            y_t  = torch.from_numpy(y_shaped[np.newaxis, ...]).to(self.o_device, dtype=self.o_dtype)
            t_t  = torch.tensor([t], device=self.o_device, dtype=self.o_dtype)
            with torch.no_grad():
                denoised_t = self.model(y_t, t_t, **self.extra_args)
            denoised = denoised_t[0].cpu().numpy().astype(self.c_dtype)
            d = (y_shaped - denoised) / t

        self.last_t        = t
        self.last_denoised = denoised
        return d.reshape(-1)
