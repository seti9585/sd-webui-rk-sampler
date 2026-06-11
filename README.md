# sd-webui-rk-sampler

A reForge extension that ports all ODE solver methods from
[ComfyUI-RK-Sampler](https://github.com/memmaptensor/ComfyUI-RK-Sampler)
as an independent sampler.

> **Original:** memmaptensor/ComfyUI-RK-Sampler  
> **Dependency:** [torchode](https://github.com/martenlienen/torchode) (`pip install torchode`)

**日本語の説明は[こちら](#日本語) / Japanese documentation [below](#日本語).**

---

## Features

- Registers **22 methods** as the **"RK Sampler"** entry in the sampling method dropdown.
- **txt2img and Hires. fix can use different methods** via the Script accordion.
- Adaptive methods (`ae_*`) adjust step size automatically via PID controller.
- Fixed-step methods (`fe_*`) follow the sigma schedule like standard samplers.
- Scipy adaptive methods (`se_*`) use `scipy.integrate` as the ODE backend.
- Supports **Flow Matching models** (Anima / FLUX / SD3) with correct noise injection in Hires. fix.
- Parameters (rtol / atol / max_steps / PID) can be set per-generation in the Script UI, overriding the Settings tab defaults.
- Generation parameters are embedded in PNG infotext for reproducibility.

## Requirements

| Item | Value |
|---|---|
| WebUI | stable-diffusion-webui-reForge |
| Python | 3.10 / 3.12 |
| Required | `torchode` — `pip install torchode` |
| Optional | `scipy` — pre-installed in reForge |
| GPU | CUDA / MPS / DirectML |

A1111 is not supported.

## Installation

```
extensions/
└── sd-webui-rk-sampler/
    └── scripts/
        └── rk_sampler.py
```

1. Clone or copy this repository into your `extensions/` folder.
2. Install torchode: `pip install torchode`
3. Restart the WebUI.

## Usage

### Sampler dropdown

Select **"RK Sampler"** from the sampling method dropdown.
The default method when the Script accordion is not enabled is **`ae_dopri5`**.

### Script accordion

Expand the **"RK Sampler"** accordion in the script section to configure:

| Control | Description |
|---|---|
| **txt2img Method** | Method used for txt2img (and img2img). |
| **hires.fix Method** | Method used for the Hires. fix pass only. |
| **Log Relative Tolerance** | `10^x` rtol for adaptive methods (`ae_*` / `se_*`). |
| **Log Absolute Tolerance** | `10^x` atol for adaptive methods (`ae_*` / `se_*`). |
| **Max ODE Steps** | Step count cap for adaptive methods. |
| **Min Sigma** | Lower cutoff below which gradient is treated as zero. |
| **PID P / I / D** | Step-size controller coefficients (`ae_*` only). |

Special values for both method dropdowns:

- **Use same sampler** — Delegates to the sampler selected in the dropdown.
- **→ TDE Sampler** — Routes to TDE Sampler for that pass.

## Methods

### Adaptive Step — `ae_*` (torchode backend)

Step size is determined automatically by the PID controller. The scheduler type and step count are used only for the sigma range; the actual number of ODE evaluations is controlled by rtol / atol. In CUDA environments, integration is performed in float64 for accurate error estimation.

| Method | Algorithm | Order | NFEs/step | Notes |
|---|---|---|---|---|
| `ae_bosh3` | Bogacki–Shampine | 3 | 3 | Embedded 2/3 pair. **Recommended.** Efficient and accurate. |
| `ae_cash_karp5` | Cash–Karp | 5 | 6 | Embedded 4/5 pair. Classic method by Cash and Karp (1990). |
| `ae_dopri5` | Dormand–Prince | 5 | 6 | The most widely used adaptive ODE solver (≈ scipy RK45). Default method. |
| `ae_dopri8` | Dormand–Prince | 8 | 13 | 8th-order Dormand-Prince. Highest accuracy at highest cost. |
| `ae_fehlberg2` | Fehlberg | 2 | 3 | Low-order Fehlberg pair. Fast, lower accuracy. |
| `ae_fehlberg5` | Runge–Kutta–Fehlberg | 5 | 6 | Classic RKF45. Comparable quality to `ae_dopri5`. |
| `ae_heun_euler2` | Heun–Euler | 2 | 2 | Embedded 1/2 pair. Fastest adaptive method. |
| `ae_midpoint2` | Explicit Midpoint | 2 | 2 | Explicit midpoint rule. Fast but low accuracy. |
| `ae_ralston2` | Ralston | 2 | 2 | 2nd-order method with minimized truncation error bound. |
| `ae_tsit5` | Tsitouras | 5 | 6 | Modern efficient 5th-order (Tsitouras 2011). Comparable to or better than `ae_dopri5`. |

### Fixed Step — `fe_*` (torchode backend)

Step size follows the sigma schedule. Works like standard non-adaptive samplers. Integration runs in float32, matching standard k-diffusion sampler behavior.

| Method | Algorithm | Order | NFEs/step | Notes |
|---|---|---|---|---|
| `fe_euler1` | Forward Euler | 1 | 1 | Simplest method. Fastest, lowest accuracy. |
| `fe_heun3` | Heun (3rd order) | 3 | 3 | Heun's 3rd-order method. |
| `fe_kutta3` | Runge–Kutta | 3 | 3 | Classical 3rd-order Runge-Kutta. |
| `fe_kutta4` | Runge–Kutta | 4 | 4 | The classic RK4. Reliable all-rounder. |
| `fe_kutta_38th4` | Runge–Kutta 3/8-rule | 4 | 4 | RK4 variant using the 3/8-rule. Slightly different error characteristics. |
| `fe_ralston3` | Ralston | 3 | 3 | Minimized truncation error bound. **Recommended for fixed-step.** |
| `fe_ralston4` | Ralston | 4 | 4 | 4th-order Ralston's method. |
| `fe_ssprk3` | SSPRK3 | 3 | 3 | Strong Stability Preserving RK3. Good for high CFG scales. |
| `fe_wray3` | Wray | 3 | 3 | Low-storage 3rd-order Runge-Kutta (Wray 1986). Memory-efficient. |

### Scipy Adaptive — `se_*` (scipy backend)

Uses `scipy.integrate.solve_ivp`. Processes one sample at a time (not batched). Slower than `ae_*` for batch sizes > 1.

| Method | Algorithm | Order | NFEs/step | Notes |
|---|---|---|---|---|
| `se_RK23` | Bogacki–Shampine | 3 | 3 | scipy `RK23`. Embedded 2/3 pair. |
| `se_RK45` | Dormand–Prince | 5 | 6 | scipy `RK45`. Good accuracy. |
| `se_DOP853` | Dormand–Prince | 8 | 12 | scipy `DOP853`. Highest accuracy, slowest. |

## Quality Ranking

Benchmark from the original ComfyUI-RK-Sampler README. Tested with SDXL, 896×1152, CFG 30, batch 1, fixed scheduled, Align Your Steps, 28 steps. Not all methods are included; `ae_tsit5`, `ae_dopri8`, `ae_fehlberg2`, `ae_midpoint2`, `fe_wray3`, `fe_heun3`, and `se_*` are not listed in the original benchmark.

| Rank | Method | Algorithm | Order | NFEs/step | Time (RTX 3090) |
|---|---|---|---|---|---|
| 1 | `fe_ralston3` | Ralston | 3 | 3 | 23.1 s |
| 2 | `ae_bosh3` | Bogacki–Shampine | 3 | 3 | 23.0 s |
| 3 | `fe_ssprk3` | SSPRK3 | 3 | 3 | 23.0 s |
| 4 | `fe_kutta4` | Runge–Kutta | 4 | 4 | 29.9 s |
| 5 | `fe_kutta_38th4` | Runge–Kutta 3/8 | 4 | 4 | 30.1 s |
| 6 | `ae_dopri5` | Dormand–Prince | 5 | 6 | 44.0 s |
| 7 | `ae_fehlberg5` | Fehlberg | 5 | 6 | 44.3 s |
| 8 | `ae_heun_euler2` | Heun–Euler | 2 | 2 | 15.9 s |
| 9 | `fe_kutta3` | Runge–Kutta | 3 | 3 | 22.8 s |
| 10 | `ae_ralston2` | Ralston | 2 | 2 | 16.1 s |
| 11 | `ae_cash_karp5` | Cash–Karp | 5 | 6 | 44.5 s |
| 12 | `fe_ralston4` | Ralston | 4 | 4 | 29.7 s |
| 13 | `fe_euler1` | Forward Euler | 1 | 1 | 9.0 s |

## Parameters

Parameters can be set in two places:
- **Script accordion** — per-generation, overrides Settings tab
- **Settings → RK Sampler** — global defaults

### Common

| Parameter | Default | Range | Description |
|---|---|---|---|
| **Log Relative Tolerance** | −3.0 | −7.0 〜 0.0 | `10^x` rtol for `ae_*` / `se_*`. Smaller = more precise, slower. Must be ≥ atol. |
| **Log Absolute Tolerance** | −4.0 | −7.0 〜 0.0 | `10^x` atol for `ae_*` / `se_*`. Smaller = more precise, slower. Must be ≤ rtol. |
| **Max ODE Steps** | 1000 | 10 〜 5000 | Failsafe step count cap for adaptive methods. Has no effect on `fe_*`. |

### Advanced

| Parameter | Default | Range | Description |
|---|---|---|---|
| **Min Sigma** | 1×10⁻⁵ | 0.0 〜 0.1 | Sigma below which ODE gradient is treated as zero. Normally unchanged. |
| **PID P** | 0.0 | 0.0 〜 1.0 | Proportional coefficient for adaptive step controller (`ae_*` only). |
| **PID I** | 1.0 | 0.0 〜 2.0 | Integral coefficient. `P=0 / I=1 / D=0` = basic integral controller. |
| **PID D** | 0.0 | 0.0 〜 1.0 | Derivative coefficient (`ae_*` only). |

## Technical Notes

### Float64 / Float32 separation
Adaptive `ae_*` methods run the ODE integration in **float64** on CUDA for accurate step-size error estimation. Fixed `fe_*` methods run in **float32**, matching standard k-diffusion sampler behavior. All model calls (CFG denoiser and pre/post-CFG hooks) always receive **float32** tensors regardless, via an internal dtype-safe wrapper.

### Flow Matching support
Models are automatically detected as Flow Matching (Anima / FLUX / SD3) or DDPM/EDM (SDXL / SD1.5) via `model_wrap.sigmas[-1]` (≤ 1.5 → Flow Matching). Hires. fix applies the correct noise injection formula for each model type:

```
Flow Matching:  x_t = (1 − t) × x + t × noise    # linear interpolation
DDPM / EDM:     x_t = x + sigma × noise            # additive
```

### PNG infotext
The following parameters are embedded in generated PNG metadata:

```
RK method, RK log_rtol, RK log_atol, RK max_steps
```

## File Structure

```
sd-webui-rk-sampler/
├── scripts/
│   └── rk_sampler.py       # Sampler registration + Script UI
└── rk_core/
    ├── methods/             # ae_* / fe_* method classes (torchode)
    ├── ode_term.py          # ODE term for torchode
    ├── scipy_ode_term.py    # ODE term for scipy
    ├── scipy_step_impl.py   # scipy step implementation
    ├── solvers/             # AutoDiffAdjoint solver
    └── controllers/         # PIDController / ScheduledController
```

## Compatibility

| Environment | Status |
|---|---|
| reForge + SDXL | ✅ Confirmed |
| reForge + SD1.5 | ✅ Confirmed |
| Forge Neo + SDXL | ✅ Confirmed |
| Forge Neo + Anima (Flow Matching) | ✅ Confirmed (including Hires. fix) |
| A1111 | ❌ Not supported |

<br>

---
---

<br>

# 日本語

reForge の独立したサンプラー拡張として、
[ComfyUI-RK-Sampler](https://github.com/memmaptensor/ComfyUI-RK-Sampler)
の全 ODE ソルバーメソッドを移植したものです。

> **原作:** memmaptensor/ComfyUI-RK-Sampler  
> **依存ライブラリ:** [torchode](https://github.com/martenlienen/torchode) (`pip install torchode`)

**English documentation is [at the top](#sd-webui-rk-sampler) of this page.**

---

## 特徴

- サンプラードロップダウンに **"RK Sampler"** として登録、**22 メソッド**を搭載。
- Script アコーディオンから **txt2img と Hires. fix に異なるメソッドを設定可能**。
- 適応ステップ (`ae_*`) は PID コントローラで刻み幅を自動調整。
- 固定ステップ (`fe_*`) は通常のサンプラーと同様に sigma スケジュールに従う。
- scipy 適応 (`se_*`) は `scipy.integrate` をバックエンドとして使用。
- Hires. fix での **Flow Matching モデル**（Anima / FLUX / SD3 等）の正しいノイズ注入に対応。
- パラメータ（rtol / atol / max_steps / PID）は Script UI で生成ごとに上書き可能（Settings タブのデフォルトを優先度で上書き）。
- 生成パラメータは PNG の infotext に記録され再現性を保持。

## 動作環境

| 項目 | 値 |
|---|---|
| WebUI | stable-diffusion-webui-reForge |
| Python | 3.10 / 3.12 |
| 必須 | `torchode` — `pip install torchode` |
| 任意 | `scipy` — reForge に同梱済み |
| GPU | CUDA / MPS / DirectML |

A1111 非対応。

## インストール

```
extensions/
└── sd-webui-rk-sampler/
    └── scripts/
        └── rk_sampler.py
```

1. このリポジトリを `extensions/` フォルダにクローンまたはコピー。
2. torchode をインストール: `pip install torchode`
3. WebUI を再起動。

## 使い方

### サンプラードロップダウン

サンプラードロップダウンから **"RK Sampler"** を選択します。
Script アコーディオンが無効の場合のデフォルトメソッドは **`ae_dopri5`** です。

### Script アコーディオン

Script セクションの **"RK Sampler"** アコーディオンを展開して設定します。

| コントロール | 説明 |
|---|---|
| **txt2img Method** | txt2img（および img2img）で使用するメソッド。 |
| **hires.fix Method** | Hires. fix パスのみで使用するメソッド。 |
| **Log Relative Tolerance** | 適応メソッド (`ae_*` / `se_*`) の `10^x` rtol。 |
| **Log Absolute Tolerance** | 適応メソッド (`ae_*` / `se_*`) の `10^x` atol。 |
| **Max ODE Steps** | 適応メソッドのステップ数上限。 |
| **Min Sigma** | これ以下で勾配をゼロとみなす下限。 |
| **PID P / I / D** | ステップ幅コントローラの係数 (`ae_*` のみ)。 |

メソッドドロップダウンの特殊値:

- **Use same sampler** — reForge のドロップダウンで選択したサンプラーに委譲。
- **→ TDE Sampler** — そのパスを TDE Sampler にルーティング。

## メソッド一覧

### 適応ステップ — `ae_*`（torchode バックエンド）

刻み幅は PID コントローラが自動決定。スケジューラの種類やステップ数は sigma の範囲にのみ使用され、実際の ODE 評価回数は rtol / atol で制御される。CUDA 環境では精度の高い誤差推定のため float64 で積分を実行。

| メソッド | アルゴリズム | 次数 | NFEs/ステップ | 備考 |
|---|---|---|---|---|
| `ae_bosh3` | Bogacki–Shampine | 3 | 3 | 2/3 次埋め込みペア。**推奨**。効率と精度のバランスが優秀。 |
| `ae_cash_karp5` | Cash–Karp | 5 | 6 | 4/5 次埋め込みペア。Cash と Karp (1990) の古典手法。 |
| `ae_dopri5` | Dormand–Prince | 5 | 6 | 最も広く使われる適応 ODE ソルバー（≈ scipy RK45）。デフォルトメソッド。 |
| `ae_dopri8` | Dormand–Prince | 8 | 13 | 8 次 Dormand-Prince。最高精度・最高コスト。 |
| `ae_fehlberg2` | Fehlberg | 2 | 3 | 低次 Fehlberg ペア。高速・低精度。 |
| `ae_fehlberg5` | Runge–Kutta–Fehlberg | 5 | 6 | 古典的な RKF45。`ae_dopri5` と同等の品質。 |
| `ae_heun_euler2` | Heun–Euler | 2 | 2 | 1/2 次埋め込みペア。最速の適応メソッド。 |
| `ae_midpoint2` | Explicit Midpoint | 2 | 2 | 陽的中点法。高速・低精度。 |
| `ae_ralston2` | Ralston | 2 | 2 | 打ち切り誤差上界を最小化した 2 次法。 |
| `ae_tsit5` | Tsitouras | 5 | 6 | 現代的な効率的 5 次法（Tsitouras 2011）。`ae_dopri5` と同等以上。 |

### 固定ステップ — `fe_*`（torchode バックエンド）

刻み幅は sigma スケジュールに従う。通常の非適応サンプラーと同様の動作。積分は float32 で実行（k-diffusion サンプラーの挙動に準拠）。

| メソッド | アルゴリズム | 次数 | NFEs/ステップ | 備考 |
|---|---|---|---|---|
| `fe_euler1` | Forward Euler | 1 | 1 | 最もシンプルな手法。最速・最低精度。 |
| `fe_heun3` | Heun（3 次） | 3 | 3 | Heun の 3 次法。 |
| `fe_kutta3` | Runge–Kutta | 3 | 3 | 古典的 3 次 Runge-Kutta。 |
| `fe_kutta4` | Runge–Kutta | 4 | 4 | 古典的 RK4。信頼性の高い万能手法。 |
| `fe_kutta_38th4` | Runge–Kutta 3/8-rule | 4 | 4 | 3/8 則を用いた RK4 変形。誤差特性がやや異なる。 |
| `fe_ralston3` | Ralston | 3 | 3 | 打ち切り誤差上界を最小化。**固定ステップの推奨**。 |
| `fe_ralston4` | Ralston | 4 | 4 | 4 次 Ralston 法。 |
| `fe_ssprk3` | SSPRK3 | 3 | 3 | Strong Stability Preserving RK3。高 CFG スケールで安定。 |
| `fe_wray3` | Wray | 3 | 3 | 低ストレージ 3 次 Runge-Kutta（Wray 1986）。メモリ効率が高い。 |

### scipy 適応 — `se_*`（scipy バックエンド）

`scipy.integrate.solve_ivp` を使用。サンプルを 1 つずつ処理（バッチ並列なし）。バッチサイズ > 1 の場合は `ae_*` より低速。

| メソッド | アルゴリズム | 次数 | NFEs/ステップ | 備考 |
|---|---|---|---|---|
| `se_RK23` | Bogacki–Shampine | 3 | 3 | scipy `RK23`。2/3 次埋め込みペア。 |
| `se_RK45` | Dormand–Prince | 5 | 6 | scipy `RK45`。高精度。 |
| `se_DOP853` | Dormand–Prince | 8 | 12 | scipy `DOP853`。最高精度・最低速。 |

## 品質ランキング

原作 ComfyUI-RK-Sampler README のベンチマーク。SDXL / 896×1152 / CFG 30 / バッチ 1 / fixed scheduled / Align Your Steps / 28 ステップで測定。全メソッドは含まれていない（`ae_tsit5`・`ae_dopri8`・`ae_fehlberg2`・`ae_midpoint2`・`fe_wray3`・`fe_heun3`・`se_*` は原作ベンチマーク未収録）。

| 順位 | メソッド | アルゴリズム | 次数 | NFEs/ステップ | 時間 (RTX 3090) |
|---|---|---|---|---|---|
| 1 | `fe_ralston3` | Ralston | 3 | 3 | 23.1 s |
| 2 | `ae_bosh3` | Bogacki–Shampine | 3 | 3 | 23.0 s |
| 3 | `fe_ssprk3` | SSPRK3 | 3 | 3 | 23.0 s |
| 4 | `fe_kutta4` | Runge–Kutta | 4 | 4 | 29.9 s |
| 5 | `fe_kutta_38th4` | Runge–Kutta 3/8 | 4 | 4 | 30.1 s |
| 6 | `ae_dopri5` | Dormand–Prince | 5 | 6 | 44.0 s |
| 7 | `ae_fehlberg5` | Fehlberg | 5 | 6 | 44.3 s |
| 8 | `ae_heun_euler2` | Heun–Euler | 2 | 2 | 15.9 s |
| 9 | `fe_kutta3` | Runge–Kutta | 3 | 3 | 22.8 s |
| 10 | `ae_ralston2` | Ralston | 2 | 2 | 16.1 s |
| 11 | `ae_cash_karp5` | Cash–Karp | 5 | 6 | 44.5 s |
| 12 | `fe_ralston4` | Ralston | 4 | 4 | 29.7 s |
| 13 | `fe_euler1` | Forward Euler | 1 | 1 | 9.0 s |

## パラメータ

パラメータは 2 箇所で設定できます。
- **Script アコーディオン** — 生成ごと、Settings タブを上書き
- **Settings → RK Sampler** — グローバルデフォルト

### 共通

| パラメータ | デフォルト | 範囲 | 説明 |
|---|---|---|---|
| **Log Relative Tolerance** | −3.0 | −7.0 〜 0.0 | `ae_*` / `se_*` の `10^x` rtol。小さいほど高精度・低速。atol 以上にする。 |
| **Log Absolute Tolerance** | −4.0 | −7.0 〜 0.0 | `ae_*` / `se_*` の `10^x` atol。小さいほど高精度・低速。rtol 以下にする。 |
| **Max ODE Steps** | 1000 | 10 〜 5000 | 適応メソッドのステップ数上限（フェイルセーフ）。`fe_*` には影響なし。 |

### 詳細設定

| パラメータ | デフォルト | 範囲 | 説明 |
|---|---|---|---|
| **Min Sigma** | 1×10⁻⁵ | 0.0 〜 0.1 | これ以下の sigma で ODE 勾配をゼロとみなす。通常は変更不要。 |
| **PID P** | 0.0 | 0.0 〜 1.0 | 適応ステップコントローラの比例係数（`ae_*` のみ）。 |
| **PID I** | 1.0 | 0.0 〜 2.0 | 積分係数。`P=0 / I=1 / D=0` で基本的な積分コントローラ。 |
| **PID D** | 0.0 | 0.0 〜 1.0 | 微分係数（`ae_*` のみ）。 |

## 技術メモ

### Float64 / Float32 の使い分け
適応 `ae_*` メソッドは CUDA 上で誤差推定の精度確保のため **float64** で ODE 積分を実行。固定 `fe_*` メソッドは標準 k-diffusion サンプラーの挙動に合わせ **float32** で実行。CFG デノイザーおよび pre/post-CFG フック等は常に **float32** テンソルを受け取る（内部の dtype-safe ラッパーによる）。

### Flow Matching 対応
モデルは `model_wrap.sigmas[-1]`（≤ 1.5 → Flow Matching）により Flow Matching（Anima / FLUX / SD3）か DDPM/EDM（SDXL / SD1.5）かを自動判別。Hires. fix で各モデルタイプに正しいノイズ注入式を適用する。

```
Flow Matching:  x_t = (1 − t) × x + t × noise    # 線形補間
DDPM / EDM:     x_t = x + sigma × noise            # 加算
```

### PNG infotext
生成 PNG のメタデータに以下のパラメータが記録される。

```
RK method, RK log_rtol, RK log_atol, RK max_steps
```

## ファイル構成

```
sd-webui-rk-sampler/
├── scripts/
│   └── rk_sampler.py       # サンプラー登録 + Script UI
└── rk_core/
    ├── methods/             # ae_* / fe_* メソッドクラス (torchode)
    ├── ode_term.py          # torchode 用 ODE term
    ├── scipy_ode_term.py    # scipy 用 ODE term
    ├── scipy_step_impl.py   # scipy ステップ実装
    ├── solvers/             # AutoDiffAdjoint ソルバー
    └── controllers/         # PIDController / ScheduledController
```

## 互換性

| 環境 | 状態 |
|---|---|
| reForge + SDXL | ✅ 確認済み |
| reForge + SD1.5 | ✅ 確認済み |
| Forge Neo + SDXL | ✅ 確認済み |
| Forge Neo + Anima (Flow Matching) | ✅ 確認済み（Hires. fix 含む） |
| A1111 | ❌ 非対応 |

## License

This project is a port of [memmaptensor/ComfyUI-RK-Sampler](https://github.com/memmaptensor/ComfyUI-RK-Sampler), which is licensed under **GPL-3.0**. As a derivative work, this repository is therefore also distributed under **GPL-3.0**.

- Original: [memmaptensor/ComfyUI-RK-Sampler](https://github.com/memmaptensor/ComfyUI-RK-Sampler) (GPL-3.0)
- ODE solver backend: [torchode](https://github.com/martenlienen/torchode)

See the `LICENSE` file for the full text.

---

## ライセンス（日本語）

本プロジェクトは [memmaptensor/ComfyUI-RK-Sampler](https://github.com/memmaptensor/ComfyUI-RK-Sampler) の移植版です。原作は **GPL-3.0** ライセンスで公開されているため、その派生物である本リポジトリも **GPL-3.0** で配布されます。

- 原作: [memmaptensor/ComfyUI-RK-Sampler](https://github.com/memmaptensor/ComfyUI-RK-Sampler)（GPL-3.0）
- ODE ソルバーバックエンド: [torchode](https://github.com/martenlienen/torchode)

全文は `LICENSE` ファイルを参照してください。
