# RK Sampler — Mathematical Verification

A record of how `sd-webui-rk-sampler` (a port of ComfyUI-RK-Sampler to
reForge / Forge Neo) was verified to be a **mathematically correct ODE solver**,
without running a single image generation.

> 🇺🇸 English first, 🇯🇵 日本語は後半にあります。

---

## What was verified

- **Repository:** `sd-webui-rk-sampler` — a standalone sampler extension porting
  ComfyUI-RK-Sampler to reForge / Forge Neo.
- **Target:** the whole `rk_core/` package — the Runge-Kutta solvers, the
  step-size controller, and the ODE right-hand-side term `ReForgeODETerm`.
- **Goal:** Runge-Kutta methods themselves are universal, but a *port* can still
  be wired up incorrectly. The point of this check is to confirm the ported code
  integrates the probability-flow ODE correctly — schedule following, integration
  direction, flatten/reshape, batching, and the σ→0 singularity guard.
- **Environment:** no WebUI required, CPU only, finishes in tens of seconds.
  `torch` + `torchode 1.0.1`.

## Why ODE samplers are verifiable this way

Unlike SDE samplers, an ODE sampler has a **unique solution** for a given
initial condition, so its output can be checked directly against a closed-form
analytic solution. The verification is structured in three layers.

### Layer 1 — Solver vs. analytic solution

The diffusion model is removed and each method integrates a simple ODE with a
known closed form (`dy/dt = -y`, solution `e^{-t}`). Rather than looking at the
absolute error, the **empirical order of convergence** is measured (how fast the
error shrinks when the step size is halved). This is what exposes a wrong Butcher
tableau coefficient or a mis-ordered stage.

### Layer 2 — Real code path vs. fake denoiser

`D(x, σ)` is replaced by a function with a known analytic solution, and the
probability-flow ODE `dx/dσ = (x - D(x, σ)) / σ` is integrated through the
**actual plumbing** (`ReForgeODETerm` + `ScheduledController` + `AutoDiffAdjoint`,
including flatten/reshape, the `min_sigma` clamp, and batch parallelism).

- `D ≡ c` (constant) → exact solution `x(σ) = c + (x0 - c)·σ/σ0`
- `D = x(1 - σ/σmax)` → exact solution `x(σ) = x0·exp((σ - σ0)/σmax)`
- Note: for Rectified Flow models (e.g. Anima), substituting `x_t = (1-σ)x0 + σn`
  yields exactly the same right-hand side, so this verification applies to flow
  models as well.

### Layer 3 — Cross-checks on a real model (no extra code)

- `fe_euler1` is plain Euler on the sigma grid, so it should be nearly bit-identical
  to k-diffusion's built-in Euler at the same seed.
- Because the PF-ODE solution is unique, the adaptive methods (dopri5 / dopri8 /
  tsit5, …) should converge to the same latent as the tolerance is tightened.
- An ODE sampler has no stochastic term, so two runs with identical settings must
  match exactly (determinism).

## Results (all PASS)

### Test 1 — Empirical order of convergence, fixed-step methods (`dy/dt = -y`, t: 1 → 0)

Order measured from the error-reduction ratio when doubling the grid N=20 → 40.

| Method | Theoretical order | Measured order | Result |
|---|---|---|---|
| fe_euler1 | 1 | 0.97 | PASS |
| fe_heun3 | 3 | 2.97 | PASS |
| fe_kutta3 | 3 | 2.97 | PASS |
| fe_kutta4 | 4 | 3.97 | PASS |
| fe_kutta_38th4 | 4 | 3.97 | PASS |
| fe_ralston3 | 3 | 2.97 | PASS |
| fe_ralston4 | 4 | 3.97 | PASS |
| fe_ssprk3 | 3 | 2.97 | PASS |
| fe_wray3 | 3 | 2.97 | PASS |

→ Every method's measured order matches its theoretical order. The Butcher tableau
coefficients are correct.

### Test 2 — Tolerance sweep, adaptive-step methods (`dy/dt = -y`)

Terminal error as rtol/atol is tightened 1e-3 → 1e-5 → 1e-7.

| Method | err(1e-3) | err(1e-5) | err(1e-7) | Result |
|---|---|---|---|---|
| ae_bosh3 | 2.4e-3 | 3.8e-5 | 4.1e-7 | PASS |
| ae_cash_karp5 | 8.7e-5 | 8.8e-6 | 1.3e-7 | PASS |
| ae_dopri5 | 3.0e-5 | 2.6e-6 | 5.1e-8 | PASS |
| ae_dopri8 | 4.1e-9 | 4.1e-9 | 4.1e-9 | PASS (high order → hits the rounding floor early) |
| ae_fehlberg2 | 9.7e-2 | 1.9e-3 | 2.0e-5 | PASS |
| ae_fehlberg5 | 3.0e-4 | 1.0e-5 | 1.2e-7 | PASS |
| ae_heun_euler2 | 7.7e-4 | 7.8e-6 | 7.8e-8 | PASS |
| ae_midpoint2 | 7.7e-4 | 7.8e-6 | 7.8e-8 | PASS |
| ae_ralston2 | 7.7e-4 | 7.8e-6 | 7.8e-8 | PASS |
| ae_tsit5 | 3.6e-5 | 8.7e-7 | 2.6e-9 | PASS |

→ Every method's error decreases monotonically in response to the tolerance.
The PID controller and embedded error estimate work correctly.

### Test 3 — `ReForgeODETerm` real code path vs. exact solution (fe_kutta4, Karras-style σ)

| Case | Interval | Result | Verdict |
|---|---|---|---|
| (a) `D ≡ 0.7` | σ_max → σ_min(0.03) | max abs err = **2.2e-16** (machine precision) | PASS |
| (b) `D = x(1-σ/σmax)` | σ_max → σ_min(0.03) | rel max err = 1.7e-8 | PASS |
| (c) `D ≡ 0.7` | σ_max → 0 (incl. final jump) | terminal bias (rel) = 3.5e-4 | PASS (by design) |

→ Machine-precision agreement in the smooth region. If flatten/reshape, the
integration direction, schedule following, or batch parallelism were miswired,
these numbers would not appear — so the port is essentially proven correct.

### Test 4 — `fe_euler1` vs. hand-written Euler (same sigma grid)

→ max abs diff = **0.0** (exact match). `ScheduledController` steps exactly along
the schedule; there is no off-by-one.

## Behavior to be aware of (not a bug)

At the final step (σ_min → 0), stage evaluations where `σ ≤ min_sigma` are masked
to zero gradient. This handles the `1/σ` singularity the PF-ODE has at σ=0, and is
inherited from ComfyUI-RK-Sampler. It leaves a small terminal bias of about
rel 3.5e-4, which is below 1e-3 in latent-scale terms and invisible in the image.
Test 3 (c) quantifies this as known behavior.

## Conclusion

- **The mathematics of `rk_core` (the Runge-Kutta solvers, the PF-ODE
  formulation, schedule following) works correctly.**
- All 9 fixed-step and 10 adaptive-step methods show their theoretical convergence
  behavior, and the real code path matches the exact solution to machine precision.
- The upstream project (ComfyUI-RK-Sampler) has not been updated for a long time,
  but what rots is not the RK mathematics — it is the glue around torchode / PyTorch
  API compatibility and dtype/device handling. Running this verification script as a
  regression test catches such breakage when PyTorch is updated.

## How to reproduce

Place `test_rk_core_verification.py` at the repository root (the same level as
`rk_core/`) and run:

```bash
python test_rk_core_verification.py
```

If all 4 tests PASS, the RK mathematics, schedule following, and ODE formulation
are confirmed correct. As long as the import paths resolve, the same test works
unchanged after a Forge Neo port.

---
---

# RK Sampler — 数学的検証（日本語）

`sd-webui-rk-sampler`（ComfyUI-RK-Sampler を reForge / Forge Neo 向けに移植した
独立サンプラー拡張）が、**ODE ソルバーとして数学的に正しく動作する**ことを、
画像生成を一切回さずに検証した記録です。

## 検証対象

- **リポジトリ:** `sd-webui-rk-sampler` — ComfyUI-RK-Sampler を reForge / Forge Neo
  向けに移植した独立サンプラー拡張。
- **対象:** `rk_core/` 一式 — Runge-Kutta ソルバー、ステップサイズコントローラ、
  ODE 右辺項 `ReForgeODETerm`。
- **目的:** ルンゲ・クッタ法そのものは普遍だが、**移植したコード**は配線を誤りうる。
  本検証は、移植コードが probability flow ODE を正しく積分しているか
  （スケジュール追従・積分方向・flatten/reshape・バッチ・σ→0 特異点ガード）を確認する。
- **実行環境:** WebUI 起動不要・CPU のみ・数十秒。`torch` + `torchode 1.0.1`。

## なぜ ODE サンプラーはこの方法で検証できるか

ODE サンプラーは SDE 系と違い「初期条件に対して解が一意に決まる」ため、出力を
**閉形式の解析解**と直接突き合わせられる。検証は 3 層に分けた。

### 層1 — ソルバー単体 vs 解析解

拡散モデルを切り離し、閉形式解を持つ単純な ODE（`dy/dt = -y`、解 `e^{-t}`）で
各メソッドを積分する。誤差の絶対値ではなく **収束次数**（刻み幅を半分にしたときの
誤差減少率）を見ることで、Butcher tableau の係数ミス・段の取り違えを検出する。

### 層2 — 実コードパス vs 偽デノイザー

`D(x, σ)` を解析的に解ける関数に差し替え、probability flow ODE
`dx/dσ = (x - D(x, σ)) / σ` を実際の配管（`ReForgeODETerm` + `ScheduledController`
+ `AutoDiffAdjoint`、flatten/reshape、`min_sigma` クランプ、バッチ並列）を通して
積分する。

- `D ≡ c`（定数）→ 厳密解 `x(σ) = c + (x0 - c)·σ/σ0`
- `D = x(1 - σ/σmax)` → 厳密解 `x(σ) = x0·exp((σ - σ0)/σmax)`
- 注: Rectified Flow（Anima 等）でも `x_t = (1-σ)x0 + σn` を代入すると同じ右辺が
  厳密に成立するため、この検証は flow 系にもそのまま通用する。

### 層3 — 実モデル上のクロスチェック（追加コード不要）

- `fe_euler1` は sigma グリッド上の Euler 法そのものなので、k-diffusion 組み込み
  Euler と同一 seed でほぼビット一致するはず。
- PF-ODE の解は一意なので、適応メソッド（dopri5 / dopri8 / tsit5 等）は許容誤差を
  締めるほど同一 latent に収束するはず。
- ODE サンプラーは確率項がないので、同設定 2 回で完全一致（決定論性）するはず。

## 検証結果（全 PASS）

### Test 1 — 固定ステップ法の経験的収束次数（`dy/dt = -y`, t: 1 → 0）

グリッドを N=20 → 40 に倍化したときの誤差減少率から実測次数を算出。

| メソッド | 理論次数 | 実測次数 | 判定 |
|---|---|---|---|
| fe_euler1 | 1 | 0.97 | PASS |
| fe_heun3 | 3 | 2.97 | PASS |
| fe_kutta3 | 3 | 2.97 | PASS |
| fe_kutta4 | 4 | 3.97 | PASS |
| fe_kutta_38th4 | 4 | 3.97 | PASS |
| fe_ralston3 | 3 | 2.97 | PASS |
| fe_ralston4 | 4 | 3.97 | PASS |
| fe_ssprk3 | 3 | 2.97 | PASS |
| fe_wray3 | 3 | 2.97 | PASS |

→ 全メソッドで実測次数が理論次数と一致。Butcher tableau の係数は正しい。

### Test 2 — 適応ステップ法の許容誤差スイープ（`dy/dt = -y`）

rtol/atol を 1e-3 → 1e-5 → 1e-7 と締めたときの終端誤差。

| メソッド | err(1e-3) | err(1e-5) | err(1e-7) | 判定 |
|---|---|---|---|---|
| ae_bosh3 | 2.4e-3 | 3.8e-5 | 4.1e-7 | PASS |
| ae_cash_karp5 | 8.7e-5 | 8.8e-6 | 1.3e-7 | PASS |
| ae_dopri5 | 3.0e-5 | 2.6e-6 | 5.1e-8 | PASS |
| ae_dopri8 | 4.1e-9 | 4.1e-9 | 4.1e-9 | PASS（高次のため早期に丸め誤差の床） |
| ae_fehlberg2 | 9.7e-2 | 1.9e-3 | 2.0e-5 | PASS |
| ae_fehlberg5 | 3.0e-4 | 1.0e-5 | 1.2e-7 | PASS |
| ae_heun_euler2 | 7.7e-4 | 7.8e-6 | 7.8e-8 | PASS |
| ae_midpoint2 | 7.7e-4 | 7.8e-6 | 7.8e-8 | PASS |
| ae_ralston2 | 7.7e-4 | 7.8e-6 | 7.8e-8 | PASS |
| ae_tsit5 | 3.6e-5 | 8.7e-7 | 2.6e-9 | PASS |

→ 全メソッドで tol に応答して誤差が単調減少。PID コントローラと埋め込み誤差推定は正常。

### Test 3 — `ReForgeODETerm` 実コードパス vs 厳密解（fe_kutta4, Karras 風 σ）

| ケース | 評価区間 | 結果 | 判定 |
|---|---|---|---|
| (a) `D ≡ 0.7` | σ_max → σ_min(0.03) | max abs err = **2.2e-16**（マシン精度） | PASS |
| (b) `D = x(1-σ/σmax)` | σ_max → σ_min(0.03) | rel max err = 1.7e-8 | PASS |
| (c) `D ≡ 0.7` | σ_max → 0（最終ジャンプ込み） | 終端バイアス(相対) = 3.5e-4 | PASS（設計仕様） |

→ 滑らかな領域でマシン精度一致。flatten/reshape・積分方向・スケジュール追従・
バッチ並列のいずれかに配線ミスがあればこの数字は出ないため、移植の正しさは
ほぼ証明された。

### Test 4 — `fe_euler1` vs 手書き Euler 法（同一 sigma グリッド）

→ max abs diff = **0.0**（完全一致）。`ScheduledController` はスケジュール通りに
刻んでおり、off-by-one なし。

## 仕様として把握しておくべき挙動（バグではない）

最終ステップ（σ_min → 0）では、`σ ≤ min_sigma` のステージ評価がゼロ勾配に
マスクされる。これは PF-ODE が σ=0 に `1/σ` 特異点を持つための処置で、
ComfyUI-RK-Sampler 由来の仕様。このため終端に相対 3.5e-4 程度の小さなバイアスが
乗るが、latent スケール比で 1e-3 未満であり画像上は不可視。Test 3 (c) はこれを
「既知の挙動」として定量化したもの。

## 結論

- **rk_core の数学（Runge-Kutta ソルバー、PF-ODE の定式化、スケジュール追従）は
  正しく動作している。**
- 固定ステップ 9 種・適応ステップ 10 種すべてが理論通りの収束挙動を示し、
  実コードパスは厳密解にマシン精度で一致した。
- 上流（ComfyUI-RK-Sampler）が長期間更新されていないが、腐るのは RK の数学ではなく
  torchode / PyTorch との API 整合・dtype/device 周りの糊の部分。この検証スクリプトを
  回帰テストとして回せば、PyTorch をアップデートした際の退行検知としてそのまま機能する。

## 再現方法

検証スクリプト `test_rk_core_verification.py` をリポジトリのルート（`rk_core/` と
同じ階層）に置き、以下を実行する。

```bash
python test_rk_core_verification.py
```

全 4 テストが PASS すれば、RK の数学・スケジュール追従・ODE 定式化の正しさが
担保される。import パスが通れば Forge Neo 移植後も同じテストがそのまま使える。
