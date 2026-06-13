"""
rk_core 数学的正しさ検証スクリプト（WebUI 起動不要・CPU のみ・数十秒）
=====================================================================

配置: 拡張機能リポジトリのルート（rk_core/ と同じ階層）に置いて
    python test_rk_core_verification.py

検証内容:
  [Test 1] 収束次数検証 — 固定ステップ法 (fe_*)
      解析解を持つ ODE dy/dt = -y を sigma 風スケジュール (1 → 0) で積分し、
      グリッドを倍化したときの誤差減少率から経験的収束次数を測定。
      理論次数 (ORDER) と一致すれば Butcher tableau は正しい。

  [Test 2] 許容誤差スイープ — 適応ステップ法 (ae_*)
      同じ ODE を PIDController で解き、rtol/atol を 10 倍ずつ厳しく
      したとき終端誤差が単調減少することを確認。

  [Test 3] 偽デノイザーによる実コードパス検証 (ReForgeODETerm)
      D(x, σ) を解析的に解ける関数に差し替え、probability flow ODE
      dx/dσ = (x - D)/σ を σ_max → 0 まで実際の配管
      (ReForgeODETerm + ScheduledController + AutoDiffAdjoint) で積分。
        (a) D ≡ c (定数):   厳密解 x(σ) = c + (x0 - c)·σ/σ0
        (b) D = x(1-σ/σmax): 厳密解 x(σ) = x0·exp((σ-σ0)/σmax)
      flatten/reshape・積分方向・min_sigma クランプ・バッチ並列を一括検証。

  [Test 4] fe_euler1 vs 手書き Euler の一致
      同一 sigma グリッド上で手書きの Euler 法と突き合わせ、
      ScheduledController がスケジュール通りに刻んでいるか
      (off-by-one がないか) をビット精度レベルで確認。

判定基準は各テスト内のコメント参照。全 PASS なら、RK の数学・
スケジュール追従・ODE 定式化の正しさが保証される。
torch / torchode をアップデートした際の退行検知にもそのまま使える。
"""
from __future__ import annotations

import math
import sys
import os

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torchode
from rk_core.solvers.auto_diff_adjoint import AutoDiffAdjoint
from rk_core.controllers.pid_controller import PIDController
from rk_core.controllers.scheduled_controller import ScheduledController
from rk_core.ode_term import ReForgeODETerm

torch.set_default_dtype(torch.float64)
DEVICE = "cpu"

FIXED_METHODS = {
    "fe_euler1":      ("rk_core.methods.fe_euler1",      "FEEuler1"),
    "fe_heun3":       ("rk_core.methods.fe_heun3",       "FEHeun3"),
    "fe_kutta3":      ("rk_core.methods.fe_kutta3",      "FEKutta3"),
    "fe_kutta4":      ("rk_core.methods.fe_kutta4",      "FEKutta4"),
    "fe_kutta_38th4": ("rk_core.methods.fe_kutta_38th4", "FEKutta38th4"),
    "fe_ralston3":    ("rk_core.methods.fe_ralston3",    "FERalston3"),
    "fe_ralston4":    ("rk_core.methods.fe_ralston4",    "FERalston4"),
    "fe_ssprk3":      ("rk_core.methods.fe_ssprk3",      "FESSPRK3"),
    "fe_wray3":       ("rk_core.methods.fe_wray3",       "FEWray3"),
}
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


def load_class(table, name):
    import importlib
    module_path, class_name = table[name]
    mod = importlib.import_module(module_path)
    # クラス名がファイルによって揺れている場合に備えて総当たりフォールバック
    if hasattr(mod, class_name):
        return getattr(mod, class_name)
    for attr in dir(mod):
        obj = getattr(mod, attr)
        if isinstance(obj, type) and getattr(obj, "TABLEAU", None) is not None:
            return obj
    raise ImportError(f"{module_path} に Method クラスが見つかりません")


# ---------------------------------------------------------------------------
# 共通: スケジュール駆動で単純な ODE を解く
# ---------------------------------------------------------------------------
class PlainODEFn:
    """この拡張の AutoDiffAdjoint は term.f.trigger_callback() を呼ぶため、
    素の関数をラップして no-op の trigger_callback を生やす。"""
    def __init__(self, f): self._f = f
    def __call__(self, t, y): return self._f(t, y)
    def trigger_callback(self, t, y): pass
    def init(self, problem, stats): pass


def solve_scheduled(method_cls, f, y0: torch.Tensor, sigmas: torch.Tensor):
    """ScheduledController（実運用と同じ）で f(t, y) を sigmas に沿って積分。"""
    f = PlainODEFn(f)
    term = torchode.ODETerm(f)
    controller = ScheduledController(sigmas=sigmas)
    method = method_cls(term=term)
    adjoint = AutoDiffAdjoint(
        method, controller,
        max_steps=None,
        backprop_through_step_size_control=False,
        dense_output=False,
    )
    batch = y0.shape[0]
    problem = torchode.InitialValueProblem(
        y0=y0,
        t_start=torch.full((batch,), sigmas[0].item()),
        t_end=torch.full((batch,), sigmas[-1].item()),
        t_eval=None,
    )
    result = adjoint.solve(problem)
    assert (result.status == 0).all(), f"solver status != 0: {result.status}"
    return result.ys[:, -1]


def solve_adaptive(method_cls, f, y0, t0, t1, rtol, atol, max_steps=10000):
    f = PlainODEFn(f)
    term = torchode.ODETerm(f)
    controller = PIDController(
        atol=atol, rtol=rtol, pcoeff=0.0, icoeff=1.0, dcoeff=0.0,
        term=term, norm=torchode.step_size_controllers.rms_norm,
    )
    method = method_cls(term=term)
    adjoint = AutoDiffAdjoint(
        method, controller,
        max_steps=max_steps,
        backprop_through_step_size_control=False,
        dense_output=False,
    )
    batch = y0.shape[0]
    problem = torchode.InitialValueProblem(
        y0=y0,
        t_start=torch.full((batch,), t0),
        t_end=torch.full((batch,), t1),
        t_eval=None,
    )
    dt0 = torch.full((batch,), (t1 - t0) / 50.0)  # 実運用同様 dt0 を明示
    result = adjoint.solve(problem, dt0=dt0)
    assert (result.status == 0).all(), f"solver status != 0: {result.status}"
    return result.ys[:, -1]


# ===========================================================================
# Test 1: 固定ステップ法の収束次数
# ===========================================================================
def test_fixed_order():
    print("=" * 76)
    print("[Test 1] 固定ステップ法: 経験的収束次数 (dy/dt = -y, t: 1 → 0)")
    print(f"  {'method':16s} {'理論':>4s} {'実測':>6s}  {'err(N=20)':>10s} {'err(N=40)':>10s}  判定")
    print("-" * 76)
    # dy/dt = -y, y(1) = 1, 厳密解 y(0) = e^{1-0... } y(t)=exp(1-t) → y(0)=e
    f = lambda t, y: -y
    y0 = torch.ones(1, 3)
    exact = math.e
    all_pass = True
    for name in FIXED_METHODS:
        cls = load_class(FIXED_METHODS, name)
        errs = []
        for n in (20, 40):
            sig = torch.linspace(1.0, 0.0, n + 1)
            y_end = solve_scheduled(cls, f, y0, sig)
            errs.append((y_end - exact).abs().max().item())
        order_emp = math.log2(errs[0] / errs[1]) if errs[1] > 0 else float("inf")
        order_thr = cls.ORDER
        # 判定: 実測次数が理論次数 - 0.3 以上（丸め誤差の床に当たる高次法は緩め）
        ok = order_emp >= order_thr - 0.3 or errs[1] < 1e-12
        all_pass &= ok
        print(f"  {name:16s} {order_thr:4d} {order_emp:6.2f}  {errs[0]:10.3e} {errs[1]:10.3e}  {'PASS' if ok else 'FAIL'}")
    print(f"  => {'ALL PASS' if all_pass else '*** FAIL あり ***'}")
    return all_pass


# ===========================================================================
# Test 2: 適応ステップ法の許容誤差スイープ
# ===========================================================================
def test_adaptive_tolerance():
    print("=" * 76)
    print("[Test 2] 適応ステップ法: tol を締めると誤差が減るか (dy/dt = -y)")
    print(f"  {'method':16s} {'err(tol=1e-3)':>14s} {'err(1e-5)':>11s} {'err(1e-7)':>11s}  判定")
    print("-" * 76)
    f = lambda t, y: -y
    y0 = torch.ones(1, 3)
    exact = math.e
    all_pass = True
    for name in ADAPTIVE_METHODS:
        cls = load_class(ADAPTIVE_METHODS, name)
        errs = []
        for tol in (1e-3, 1e-5, 1e-7):
            y_end = solve_adaptive(cls, f, y0, 1.0, 0.0, rtol=tol, atol=tol * 0.1)
            errs.append((y_end - exact).abs().max().item())
        # 判定: tol に応答して誤差が減ること（単調減少 かつ 1e-3→1e-7 で 1/50 以下）。
        # 低次法 (fehlberg2 等) は同じ tol でも大域誤差が大きいのが正常なので
        # 絶対値ではなく「応答性」を見る。高次法は最初から丸め誤差の床に
        # 当たることがあるため、その場合 (err < 1e-8) も合格。
        floor = errs[0] < 1e-8
        ok = floor or ((errs[2] <= errs[1] * 1.01) and (errs[1] <= errs[0] * 1.01)
                       and (errs[2] <= errs[0] / 50))
        all_pass &= ok
        print(f"  {name:16s} {errs[0]:14.3e} {errs[1]:11.3e} {errs[2]:11.3e}  {'PASS' if ok else 'FAIL'}")
    print(f"  => {'ALL PASS' if all_pass else '*** FAIL あり ***'}")
    return all_pass


# ===========================================================================
# Test 3: ReForgeODETerm + 偽デノイザー（実コードパス）
# ===========================================================================
class FakeDenoiserConst:
    """D(x, σ) ≡ c — 厳密解 x(σ) = c + (x0 - c)·σ/σ0"""
    def __init__(self, c): self.c = c
    def __call__(self, x, sigma, **kw): return torch.full_like(x, self.c)

class FakeDenoiserLinear:
    """D(x, σ) = x·(1 - σ/σmax) — 厳密解 x(σ) = x0·exp((σ - σ0)/σmax)"""
    def __init__(self, smax): self.smax = smax
    def __call__(self, x, sigma, **kw):
        s = sigma.view(-1, 1, 1, 1).to(x.dtype)
        return x * (1.0 - s / self.smax)

def karras_like_sigmas(n, smax=14.6, smin=0.03, rho=7.0):
    """Karras スケジュール風 (末尾 0 付き) — 実運用の非均等グリッドを再現"""
    i = torch.linspace(0, 1, n)
    sig = (smax ** (1 / rho) + i * (smin ** (1 / rho) - smax ** (1 / rho))) ** rho
    return torch.cat([sig, torch.zeros(1)])

def run_ode_term(method_cls, fake_model, x0_4d, sigmas, min_sigma=1e-5, t_end=None):
    o_shape = tuple(x0_4d.shape)
    ode_fn = ReForgeODETerm(
        model=fake_model,
        c_device=DEVICE, c_dtype=torch.float64,
        o_device=DEVICE, o_dtype=torch.float64,
        o_shape=o_shape, min_sigma=min_sigma,
        t_max=sigmas.max().item(), t_min=sigmas.min().item(),
        n_steps=len(sigmas) - 1,
        extra_args={}, callback=None, pbar=None, is_adaptive=False,
        cfg_denoiser=None,
    )
    term = torchode.ODETerm(ode_fn)
    controller = ScheduledController(sigmas=sigmas)
    method = method_cls(term=term)
    adjoint = AutoDiffAdjoint(method, controller, max_steps=None,
                              backprop_through_step_size_control=False,
                              dense_output=False)
    batch = o_shape[0]
    t_end_val = sigmas[-1].item() if t_end is None else t_end
    problem = torchode.InitialValueProblem(
        y0=x0_4d.flatten(start_dim=1),
        t_start=torch.full((batch,), sigmas[0].item()),
        t_end=torch.full((batch,), t_end_val),
        t_eval=None,
    )
    result = adjoint.solve(problem)
    assert (result.status == 0).all()
    return result.ys[:, -1].reshape(o_shape)

def test_ode_term_exact():
    print("=" * 76)
    print("[Test 3] ReForgeODETerm 実コードパス vs 厳密解 (fe_kutta4, Karras風 σ)")
    print("-" * 76)
    torch.manual_seed(42)
    cls = load_class(FIXED_METHODS, "fe_kutta4")
    sigmas = karras_like_sigmas(60)
    s0 = sigmas[0].item()
    smin = sigmas[-2].item()   # 0.03（末尾の 0 を除いた実質的な sigma_min）
    # バッチ2・4ch・8x8 latent 風テンソル（バッチ並列も同時に検証）
    x0 = torch.randn(2, 4, 8, 8) * s0
    all_pass = True

    # --- Part 1: 滑らかな領域 (σ_max → σ_min) — マシン精度での厳密一致を要求 ---
    # D≡c のとき厳密解 x(σ) = c + (x0-c)·σ/σ0 は σ について線形なので、
    # 全 RK ステージが厳密解の直線上に乗り、配管が正しければ任意の RK 法で
    # 「厳密」になる。わずかなズレも配線ミス (flatten/方向/スケジュール) を意味する
    # 鋭敏なテスト。t_end=σ_min とし、最終の σ→0 ジャンプを除外して評価する。
    c = 0.7
    out = run_ode_term(cls, FakeDenoiserConst(c), x0, sigmas, t_end=smin)
    exact = c + (x0 - c) * (smin / s0)
    err = (out - exact).abs().max().item()
    ok = err < 1e-10
    all_pass &= ok
    print(f"  (a) D≡{c}, σ→{smin}:     max|out - exact| = {err:.3e}   {'PASS' if ok else 'FAIL'} (< 1e-10)")

    # (b) D = x(1-σ/σmax): 解が指数関数の非自明ケース。RK4 の打ち切り誤差のみが残る。
    out = run_ode_term(cls, FakeDenoiserLinear(s0), x0, sigmas, t_end=smin)
    exact = x0 * math.exp((smin - s0) / s0)
    err = (out - exact).abs().max().item() / x0.abs().max().item()
    ok = err < 1e-6
    all_pass &= ok
    print(f"  (b) D=x(1-σ/σmax), σ→{smin}: rel max err  = {err:.3e}   {'PASS' if ok else 'FAIL'} (< 1e-6)")

    # --- Part 2: 最終ジャンプ込み (σ_max → 0) — 設計上のバイアスの定量化 ---
    # 最終ステップでは σ≦min_sigma のステージ評価がゼロ勾配にマスクされる
    # （PF-ODE は σ=0 に 1/σ 特異点を持つための処置。ComfyUI-RK-Sampler 由来の仕様）。
    # このため終端には O(b_last·σ_min·|d|) の小さなバイアスが乗る。
    # latent スケール比で ~1e-3 未満なら設計通り（画像上は不可視）。
    out = run_ode_term(cls, FakeDenoiserConst(c), x0, sigmas, t_end=0.0)
    exact = torch.full_like(x0, c)   # σ→0 の極限で x → c
    bias = (out - exact).abs().max().item() / x0.abs().max().item()
    ok = bias < 1e-3
    all_pass &= ok
    print(f"  (c) σ→0 最終ジャンプ:    終端バイアス(相対) = {bias:.3e}   {'PASS' if ok else 'FAIL'} (< 1e-3, 設計仕様)")
    print(f"  => {'ALL PASS' if all_pass else '*** FAIL あり ***'}")
    return all_pass


# ===========================================================================
# Test 4: fe_euler1 vs 手書き Euler（スケジュール追従の検証）
# ===========================================================================
def test_euler_equivalence():
    print("=" * 76)
    print("[Test 4] fe_euler1 vs 手書き Euler 法（同一 sigma グリッド・偽デノイザー）")
    print("-" * 76)
    torch.manual_seed(0)
    cls = load_class(FIXED_METHODS, "fe_euler1")
    sigmas = karras_like_sigmas(30)
    s0 = sigmas[0].item()
    x0 = torch.randn(2, 4, 8, 8) * s0
    fake = FakeDenoiserLinear(s0)
    min_sigma = 1e-5

    out = run_ode_term(cls, fake, x0, sigmas, min_sigma=min_sigma)

    # 手書き Euler: k-diffusion sample_euler と同じ d = (x - D)/σ, x += d·dσ
    x = x0.clone()
    for i in range(len(sigmas) - 1):
        s, s_next = sigmas[i], sigmas[i + 1]
        if s.item() <= min_sigma:
            continue
        d = (x - fake(x, s.expand(x.shape[0]))) / s
        x = x + d * (s_next - s)

    err = (out - x).abs().max().item()
    ok = err < 1e-10
    print(f"  max|RK fe_euler1 - 手書きEuler| = {err:.3e}   {'PASS' if ok else 'FAIL'} (< 1e-10)")
    print(f"  => {'PASS: ScheduledController はスケジュール通りに刻んでいる' if ok else '*** FAIL: 刻み or 定式化に不一致 ***'}")
    return ok


if __name__ == "__main__":
    results = []
    results.append(("Test 1 固定ステップ収束次数", test_fixed_order()))
    results.append(("Test 2 適応ステップ tol スイープ", test_adaptive_tolerance()))
    results.append(("Test 3 ReForgeODETerm 厳密解", test_ode_term_exact()))
    results.append(("Test 4 Euler 等価性", test_euler_equivalence()))
    print("=" * 76)
    print("総合結果:")
    for name, ok in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    sys.exit(0 if all(ok for _, ok in results) else 1)
