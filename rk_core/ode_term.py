"""
reForge 用 ODE Term
ComfyUI-RK-Sampler の TorchODEODETerm を reForge 向けに移植。
k-diffusion モデル API: model(x, sigma, **extra_args) -> denoised
"""
from __future__ import annotations

import torch


class ReForgeODETerm:
    """
    torchode の AutoDiffAdjoint から呼ばれる ODE の右辺関数。

    probability flow ODE:
        dx/dσ = (x - D(x, σ)) / σ
    D(x, σ) は CFGDenoiserKDiffusion が返す denoised latent。

    Args:
        model       : CFGDenoiserKDiffusion インスタンス
        c_device    : ODE ソルバーが使うデバイス ("cpu" or "mps")
        c_dtype     : ODE ソルバーが使う dtype (float64 or float32)
        o_device    : 元の latent のデバイス (CUDA など)
        o_dtype     : 元の latent の dtype
        o_shape     : 元の latent の shape (batch, C, H, W)
        min_sigma   : この sigma 以下はゼロ勾配で通過
        t_max       : sigma の最大値
        t_min       : sigma の最小値
        n_steps     : sigma スケジュールのステップ数（プログレスバー用）
        extra_args  : model に渡す追加引数 dict
        callback    : reForge の callback 関数
        pbar        : tqdm プログレスバー
        is_adaptive : 適応ステップか固定ステップか（プログレスバー更新方法が変わる）
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
        cfg_denoiser=None,   # step カウンタ更新用（apply_refiner 対応）
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

        # 内部状態
        self.n_callbacks   = 0
        self.pbar_step     = 0
        self.last_t        = None
        self.last_denoised = None

    # ------------------------------------------------------------------
    # torchode の ODETerm ラッパーから vf(t, y, stats, args) として呼ばれる
    # ------------------------------------------------------------------
    def __call__(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t : (batch,)  現在の sigma 値
            y : (batch, flat_dim)  現在の latent（flatten 済み）
        Returns:
            dy/dt : (batch, flat_dim)
        """
        # y の実際の shape と o_shape が一致しない場合は o_shape を更新する
        # （hires.fix でアップスケール後に latent サイズが変わる場合への対処）
        actual_shape = (y.shape[0], self.o_shape[1], y.shape[1] // (self.o_shape[1] * self.o_shape[2] * self.o_shape[3]) ** 0, -1)
        try:
            y_4d = y.reshape(self.o_shape)
        except RuntimeError:
            # o_shape と実際の y が合わない → reshape できる形に自動修正
            import math
            b = y.shape[0]
            c = self.o_shape[1]
            hw = y.shape[1] // c
            h = w = int(math.isqrt(hw))
            self.o_shape = (b, c, h, w)
            logger.warning("[RK Sampler] o_shape を %s に自動修正しました", self.o_shape)
            y_4d = y.reshape(self.o_shape)
        mask  = t <= self.min_sigma                              # (batch,)  sigma が小さすぎる行

        denoised = torch.zeros_like(y_4d)

        if not mask.all():
            y_m = y_4d.to(self.o_device, dtype=self.o_dtype)
            t_m = t.to(self.o_device, dtype=self.o_dtype)

            # フルバッチをモデルに渡す
            # （マスクで一部だけ渡すと LoRA 等でテンソルサイズ不一致が起きる）
            # t_m は (batch,) で渡す — sigma の次元を明示的に保証
            if t_m.dim() == 0:
                t_m = t_m.unsqueeze(0).expand(y_m.shape[0])
            elif t_m.shape[0] != y_m.shape[0]:
                t_m = t_m[:1].expand(y_m.shape[0])

            with torch.no_grad():
                denoised_full = self.model(y_m, t_m, **self.extra_args)

            # マスクされたバッチ（sigma≦min_sigma）はゼロで埋める
            denoised_full = denoised_full.to(self.c_device, dtype=self.c_dtype)
            denoised_full[mask] = 0.0
            denoised = denoised_full

        # sigma = 0 付近はゼロ勾配
        d = torch.where(
            mask.view(-1, 1, 1, 1),
            torch.zeros_like(y_4d),
            (y_4d - denoised) / t.view(-1, 1, 1, 1),
        )

        self.last_t        = t
        self.last_denoised = denoised

        return d.flatten(start_dim=1)

    # ------------------------------------------------------------------
    # AutoDiffAdjoint から各ステップ後に呼ばれる
    # ------------------------------------------------------------------
    def trigger_callback(self, t: torch.Tensor, y: torch.Tensor):
        """プログレスバー更新と reForge callback の発火。"""
        if self.pbar is None and self.callback is None:
            return

        t_mean = t.mean().item()
        self.n_callbacks += 1

        # --- cfg_denoiser の step カウンタを更新（apply_refiner 対応）---
        # 適応ステップは n_callbacks が p.steps を超えることがあるため
        # total_steps - 1 を上限としてクランプする
        if self.cfg_denoiser is not None and hasattr(self.cfg_denoiser, "step"):
            total = getattr(self.cfg_denoiser, "total_steps", None)
            if total is not None:
                self.cfg_denoiser.step = min(self.n_callbacks, total - 1)
            else:
                self.cfg_denoiser.step = self.n_callbacks

        # --- プログレスバー更新 ---
        if self.pbar is not None:
            if self.is_adaptive:
                # 適応ステップ: パーセンテージで進捗表示
                progress    = (self.t_max - t_mean) / max(self.t_max - self.t_min, 1e-8)
                percentage  = progress * 100
                self.pbar.update(percentage - self.pbar_step)
                self.pbar_step = percentage
                i = round(progress * self.n_steps)
            else:
                # 固定ステップ: 1ステップずつカウント
                self.pbar.update(1)
                self.pbar_step += 1
                i = self.pbar_step

            self.pbar.set_postfix({"σ": f"{t_mean:.4f}"})
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
    # torchode の ODETerm.init() から呼ばれる（統計初期化）
    # ------------------------------------------------------------------
    def init(self, problem, stats: dict):
        pass
