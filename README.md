# sd-webui-RK-Sampler

**EN** | [日本語](#日本語)

ODE sampler extension for Stable Diffusion WebUI (Forge-based),  
powered by [torchode](https://github.com/martenlienen/torchode).

Port of [ComfyUI-RK-Sampler](https://github.com/memmaptensor/ComfyUI-RK-Sampler) by memmaptensor.

> Unlike reForge's built-in samplers, this extension registers as an independent sampler,  
> allowing different solvers to be selected for txt2img and hires.fix separately.

---

## Features

- Registers **22 methods** as the **"RK Sampler"** entry in the sampling method dropdown.
- **txt2img and Hires.fix can use different methods** via the Script accordion.
- **Relative/absolute tolerance can also be set independently for txt2img and Hires.fix.**
- Adaptive methods (`ae_*`) adjust step size automatically via PID controller.
- Fixed-step methods (`fe_*`) follow the sigma schedule like standard samplers.
- Scipy adaptive methods (`se_*`) use `scipy.integrate` as the ODE backend.
- Supports **Flow Matching models** (Anima / FLUX / SD3) with correct noise injection in Hires.fix.
- Parameters (relative tolerance / absolute tolerance / max_steps / PID) can be set per-generation in the Script UI, overriding the Settings tab defaults.
- Generation parameters are embedded in PNG infotext for reproducibility.

---

## Dependency

```bash
pip install torchode
```

**scipy** is optional; required only for `se_*` methods:

```bash
pip install scipy
```

No other packages are needed beyond what is already included in your WebUI.  
GPU support: CUDA / MPS / DirectML.

> This extension uses `set_model_sampler_*_function`, which is part of the Forge
> backend. It is not available in A1111 (AUTOMATIC1111).

---

## Installation

**Extensions → Install from URL:**

```
https://github.com/seti9585/sd-webui-rk-sampler
```

---

## Usage

### Sampler dropdown

Select **"RK Sampler"** from the sampling method dropdown.  
When the Script accordion is disabled or set to **"Use same sampler"**, sampling is
delegated to the method selected in the main sampler dropdown.

### Script accordion

Expand the **"RK Sampler"** accordion in the script section to configure:

| Control | Description |
|---|---|
| **txt2img Method** | Method used for txt2img (and img2img). |
| **hires.fix Method** | Method used for the Hires.fix pass only. |
| **txt2img Log Relative Tolerance** | `10^x` relative tolerance (rtol) for txt2img, effective for adaptive methods (`ae_*` / `se_*`). |
| **txt2img Log Absolute Tolerance** | `10^x` absolute tolerance (atol) for txt2img. |
| **hires.fix Log Relative Tolerance** | `10^x` relative tolerance (rtol) for the Hires.fix pass only. |
| **hires.fix Log Absolute Tolerance** | `10^x` absolute tolerance (atol) for the Hires.fix pass only. |
| **Max ODE Steps** | Step count cap for adaptive methods (shared across both passes). |
| **Min Sigma** | Lower cutoff below which gradient is treated as zero (shared). |
| **PID P / I / D** | Step-size controller coefficients (`ae_*` only, shared across both passes). |

Special values for both method dropdowns:

- **Use same sampler** — Delegates to the sampler selected in the dropdown.
- **→ TDE Sampler** — Routes to TDE Sampler for that pass.

> **Why split by pass?** Most users can safely leave both at the same tolerance —
> this is not a control the average workflow needs. It exists because the author
> personally runs txt2img and Hires.fix with deliberately different methods (e.g.
> a low-order method at a high step count for txt2img, paired with a different
> high-order method at a low step count for Hires.fix, sometimes even with a
> different checkpoint per pass), and wanted matching independent tolerance control
> to go with it.

---

## Methods

### Adaptive Step — `ae_*` (torchode backend)

Step size is determined automatically by the PID controller. The scheduler type and step count are used only for the sigma range; the actual number of ODE evaluations is controlled by rtol / atol. In CUDA environments, integration is performed in float64 for accurate error estimation.

| Method | Algorithm | Order | NFEs/step | Notes |
|---|---|---|---|---|
| `ae_bosh3` | Bogacki–Shampine | 3 | 3 | Embedded 2/3 pair. **Recommended.** Efficient and accurate. |
| `ae_cash_karp5` | Cash–Karp | 5 | 6 | Embedded 4/5 pair. Classic method by Cash and Karp (1990). |
| `ae_dopri5` | Dormand–Prince | 5 | 6 | The most widely used adaptive ODE solver (≈ scipy RK45). |
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

---

## Parameters

Parameters can be set in two places:
- **Script accordion** — per-generation, overrides Settings tab
- **Settings → RK Sampler** — global defaults

### Common

| Parameter | Default | Range | Description |
|---|---|---|---|
| **Log Relative Tolerance** | −3.0 | −7.0 〜 0.0 | `10^x` rtol for `ae_*` / `se_*`. Smaller = more precise, slower. Must be ≥ the Log Absolute Tolerance. Set independently for txt2img and Hires.fix in the Script accordion. |
| **Log Absolute Tolerance** | −4.0 | −7.0 〜 0.0 | `10^x` atol for `ae_*` / `se_*`. Smaller = more precise, slower. Must be ≤ the Log Relative Tolerance. Set independently for txt2img and Hires.fix in the Script accordion. |
| **Max ODE Steps** | 1000 | 10 〜 5000 | Failsafe step count cap for adaptive methods. Has no effect on `fe_*`. Shared across both passes. |

### Advanced

| Parameter | Default | Range | Description |
|---|---|---|---|
| **Min Sigma** | 0.00001 | 0.0 〜 0.1 | Sigma below which ODE gradient is treated as zero. Normally unchanged. Shared across both passes. |
| **PID P** | 0.0 | 0.0 〜 1.0 | Proportional coefficient for adaptive step controller (`ae_*` only). Shared across both passes. |
| **PID I** | 1.0 | 0.0 〜 2.0 | Integral coefficient. `P=0 / I=1 / D=0` = basic integral controller. Shared across both passes. |
| **PID D** | 0.0 | 0.0 〜 1.0 | Derivative coefficient (`ae_*` only). Shared across both passes. |

---

## Technical Notes

### Float64 / Float32 separation

Adaptive `ae_*` methods run the ODE integration in **float64** on CUDA for accurate step-size error estimation. Fixed `fe_*` methods run in **float32**, matching standard k-diffusion sampler behavior. All model calls (CFG denoiser and pre/post-CFG hooks) always receive **float32** tensors regardless, via an internal dtype-safe wrapper.

### Flow Matching support

Models are automatically detected as Flow Matching (Anima / FLUX / SD3) or DDPM/EDM (SDXL / SD1.5) via `model_wrap.sigmas[-1]` (≤ 1.5 → Flow Matching). Hires.fix applies the correct noise injection formula for each model type:

```
Flow Matching:  x_t = (1 − t) × x + t × noise    # linear interpolation
DDPM / EDM:     x_t = x + sigma × noise            # additive
```

### PNG infotext

The following parameters are embedded in generated PNG metadata:

```
RK method,
RK txt2img log_rtol, RK txt2img log_atol,
RK hires log_rtol, RK hires log_atol,
RK max_steps, RK min_sigma, RK pid_p, RK pid_i, RK pid_d
```

> **Known limitation:** `RK method` is a single shared key — when the Hires.fix pass
> uses a different method than txt2img, the Hires.fix value overwrites it in the
> saved PNG, so the txt2img method cannot currently be recovered from infotext
> alone. The tolerance keys above are not affected by this.

---

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

---

## Compatibility

| Environment | Status |
|---|---|
| reForge + SDXL | ✅ Confirmed |
| reForge + SD1.5 | ✅ Confirmed |
| Forge Neo + SDXL | ✅ Confirmed |
| Forge Neo + Anima (Flow Matching) | ✅ Confirmed (including Hires.fix) |
| A1111 | ❌ Not supported |

---

## ODE Formulation

```
dx/dσ = (x − D(x, σ)) / σ
```

`D(x, σ)` is the denoised latent predicted by the model at noise level σ.

---

## License

This project is a port of [memmaptensor/ComfyUI-RK-Sampler](https://github.com/memmaptensor/ComfyUI-RK-Sampler), which is licensed under **GPL-3.0**. As a derivative work, this repository is therefore also distributed under **GPL-3.0**.

- Original: [memmaptensor/ComfyUI-RK-Sampler](https://github.com/memmaptensor/ComfyUI-RK-Sampler) (GPL-3.0)
- ODE solver backend: [torchode](https://github.com/martenlienen/torchode)

See the `LICENSE` file for the full text.

---
---

# 日本語

**[English](#sd-webui-rk-sampler)** | 日本語

[torchode](https://github.com/martenlienen/torchode) を使った ODE サンプラー拡張機能（Forge 系 WebUI 向け）。

[ComfyUI-RK-Sampler](https://github.com/memmaptensor/ComfyUI-RK-Sampler)（memmaptensor 作）を reForge 向けに移植。

> reForge 組み込みのサンプラーとは独立したサンプラーとして登録されるため、  
> txt2img と hires.fix で異なるソルバーを選択できます。

---

## 特徴

- サンプラードロップダウンに **"RK Sampler"** として登録、**22 メソッド**を搭載。
- Script アコーディオンから **txt2img と Hires.fix に異なるメソッドを設定可能**。
- **相対/絶対許容誤差も txt2img と Hires.fix で独立に設定可能**。
- 可変ステップ (`ae_*`) は PID コントローラで刻み幅を自動調整。
- 固定ステップ (`fe_*`) は通常のサンプラーと同様に sigma スケジュールに従う。
- scipy 可変ステップ (`se_*`) は `scipy.integrate` をバックエンドとして使用。
- Hires.fix での **Flow Matching モデル**（Anima / FLUX / SD3 等）の正しいノイズ注入に対応。
- パラメータ（相対許容誤差 / 絶対許容誤差 / max_steps / PID）は Script UI で生成ごとに上書き可能（Settings タブのデフォルトより優先）。
- 生成パラメータは PNG の infotext に記録され再現性を保持。

---

## 依存ライブラリ

```bash
pip install torchode
```

**scipy** は任意です。`se_*` メソッドを使用する場合のみ必要です。

```bash
pip install scipy
```

その他のパッケージは、WebUI に同梱のものがそのまま使用されます。  
GPU サポート: CUDA / MPS / DirectML。

> 本拡張は `set_model_sampler_*_function` を使用しており、これは Forge 系バックエンド
> 固有のフックです。このフックを持たない A1111（AUTOMATIC1111）では動作しません。

---

## インストール

**Extensions → Install from URL:**

```
https://github.com/seti9585/sd-webui-rk-sampler
```

---

## 使い方

### サンプラードロップダウン

サンプラードロップダウンから **"RK Sampler"** を選択します。  
Script アコーディオンが無効、または **"Use same sampler"** が選択されている場合は、
メインのサンプラードロップダウンで選択されているサンプラーに委譲されます。

### Script アコーディオン

Script セクションの **"RK Sampler"** アコーディオンを展開して設定します。

| コントロール | 説明 |
|---|---|
| **txt2img Method** | txt2img（および img2img）で使用するメソッド。 |
| **hires.fix Method** | Hires.fix パスのみで使用するメソッド。 |
| **txt2img Log Relative Tolerance** | txt2img の `10^x` 相対許容誤差（rtol）。可変ステップメソッド (`ae_*` / `se_*`) で有効。 |
| **txt2img Log Absolute Tolerance** | txt2img の `10^x` 絶対許容誤差（atol）。 |
| **hires.fix Log Relative Tolerance** | Hires.fix パスのみの `10^x` 相対許容誤差（rtol）。 |
| **hires.fix Log Absolute Tolerance** | Hires.fix パスのみの `10^x` 絶対許容誤差（atol）。 |
| **Max ODE Steps** | 可変ステップメソッドのステップ数上限（両パス共通）。 |
| **Min Sigma** | これ以下で勾配をゼロとみなす下限（両パス共通）。 |
| **PID P / I / D** | ステップ幅コントローラの係数（`ae_*` のみ、両パス共通）。 |

メソッドドロップダウンの特殊値:

- **Use same sampler** — サンプラードロップダウンで選択したサンプラーに委譲。
- **→ TDE Sampler** — そのパスを TDE Sampler にルーティング。

> **なぜパスごとに分けているか:** ほとんどのユーザーは両パスで同じ許容誤差のままで
> 問題ありません、平均的なワークフローには不要な設定です。これは作者自身が
> txt2img と Hires.fix で意図的に異なるメソッドを使い分けている（例: txt2img は
> 低次メソッドを高ステップ数で、Hires.fix は別の高次メソッドを低ステップ数で、
> 場合によってはパスごとにチェックポイントも切り替える）ため、それに合わせて
> 許容誤差も独立して制御できるようにと付けた機能です。

---

## メソッド一覧

### 可変ステップ — `ae_*`（torchode バックエンド）

刻み幅は PID コントローラが自動決定。スケジューラの種類やステップ数は sigma の範囲にのみ使用され、実際の ODE 評価回数は rtol / atol で制御される。CUDA 環境では精度の高い誤差推定のため float64 で積分を実行。

| メソッド | アルゴリズム | 次数 | NFEs/ステップ | 備考 |
|---|---|---|---|---|
| `ae_bosh3` | Bogacki–Shampine | 3 | 3 | 2/3 次埋め込みペア。**推奨**。効率と精度のバランスが優秀。 |
| `ae_cash_karp5` | Cash–Karp | 5 | 6 | 4/5 次埋め込みペア。Cash と Karp (1990) の古典手法。 |
| `ae_dopri5` | Dormand–Prince | 5 | 6 | 最も広く使われる可変ステップ ODE ソルバー（≈ scipy RK45）。 |
| `ae_dopri8` | Dormand–Prince | 8 | 13 | 8 次 Dormand-Prince。最高精度・最高コスト。 |
| `ae_fehlberg2` | Fehlberg | 2 | 3 | 低次 Fehlberg ペア。高速・低精度。 |
| `ae_fehlberg5` | Runge–Kutta–Fehlberg | 5 | 6 | 古典的な RKF45。`ae_dopri5` と同等の品質。 |
| `ae_heun_euler2` | Heun–Euler | 2 | 2 | 1/2 次埋め込みペア。最速の可変ステップメソッド。 |
| `ae_midpoint2` | Explicit Midpoint | 2 | 2 | 陽的中点法。高速・低精度。 |
| `ae_ralston2` | Ralston | 2 | 2 | 打ち切り誤差上界を最小化した 2 次法。 |
| `ae_tsit5` | Tsitouras | 5 | 6 | 現代的な効率的 5 次法（Tsitouras 2011）。`ae_dopri5` と同等以上。 |

### 固定ステップ — `fe_*`（torchode バックエンド）

刻み幅は sigma スケジュールに従う。通常の固定ステップサンプラーと同様の動作。積分は float32 で実行（k-diffusion サンプラーの挙動に準拠）。

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

### scipy 可変ステップ — `se_*`（scipy バックエンド）

`scipy.integrate.solve_ivp` を使用。サンプルを 1 つずつ処理（バッチ並列なし）。バッチサイズ > 1 の場合は `ae_*` より低速。

| メソッド | アルゴリズム | 次数 | NFEs/ステップ | 備考 |
|---|---|---|---|---|
| `se_RK23` | Bogacki–Shampine | 3 | 3 | scipy `RK23`。2/3 次埋め込みペア。 |
| `se_RK45` | Dormand–Prince | 5 | 6 | scipy `RK45`。高精度。 |
| `se_DOP853` | Dormand–Prince | 8 | 12 | scipy `DOP853`。最高精度・最低速。 |

---

## パラメータ

パラメータは 2 箇所で設定できます。
- **Script アコーディオン** — 生成ごと、Settings タブを上書き
- **Settings → RK Sampler** — グローバルデフォルト

### 共通

| パラメータ | デフォルト | 範囲 | 説明 |
|---|---|---|---|
| **Log Relative Tolerance** | −3.0 | −7.0 〜 0.0 | `ae_*` / `se_*` の `10^x` rtol。小さいほど高精度・低速。Log Absolute Tolerance 以上にする。Script アコーディオンで txt2img と Hires.fix を独立に設定可能。 |
| **Log Absolute Tolerance** | −4.0 | −7.0 〜 0.0 | `ae_*` / `se_*` の `10^x` atol。小さいほど高精度・低速。Log Relative Tolerance 以下にする。Script アコーディオンで txt2img と Hires.fix を独立に設定可能。 |
| **Max ODE Steps** | 1000 | 10 〜 5000 | 可変ステップメソッドのステップ数上限（フェイルセーフ）。`fe_*` には影響なし。両パス共通。 |

### 詳細設定

| パラメータ | デフォルト | 範囲 | 説明 |
|---|---|---|---|
| **Min Sigma** | 0.00001 | 0.0 〜 0.1 | これ以下の sigma で ODE 勾配をゼロとみなす。通常は変更不要。両パス共通。 |
| **PID P** | 0.0 | 0.0 〜 1.0 | 可変ステップコントローラの比例係数（`ae_*` のみ）。両パス共通。 |
| **PID I** | 1.0 | 0.0 〜 2.0 | 積分係数。`P=0 / I=1 / D=0` で基本的な積分コントローラ。両パス共通。 |
| **PID D** | 0.0 | 0.0 〜 1.0 | 微分係数（`ae_*` のみ）。両パス共通。 |

---

## 技術メモ

### Float64 / Float32 の使い分け

可変ステップ `ae_*` メソッドは CUDA 上で誤差推定の精度確保のため **float64** で ODE 積分を実行。固定 `fe_*` メソッドは標準 k-diffusion サンプラーの挙動に合わせ **float32** で実行。CFG デノイザーおよび pre/post-CFG フック等は常に **float32** テンソルを受け取る（内部の dtype-safe ラッパーによる）。

### Flow Matching 対応

モデルは `model_wrap.sigmas[-1]`（≤ 1.5 → Flow Matching）により Flow Matching（Anima / FLUX / SD3）か DDPM/EDM（SDXL / SD1.5）かを自動判別。Hires.fix で各モデルタイプに正しいノイズ注入式を適用する。

```
Flow Matching:  x_t = (1 − t) × x + t × noise    # 線形補間
DDPM / EDM:     x_t = x + sigma × noise            # 加算
```

### PNG infotext

生成 PNG のメタデータに以下のパラメータが記録される。

```
RK method,
RK txt2img log_rtol, RK txt2img log_atol,
RK hires log_rtol, RK hires log_atol,
RK max_steps, RK min_sigma, RK pid_p, RK pid_i, RK pid_d
```

> **既知の制限:** `RK method` は単一の共有キーです。Hires.fixパスでtxt2imgと異なる
> methodを使った場合、保存されるPNGではHires.fixの値で上書きされてしまうため、
> infotextだけからtxt2img側のmethodを復元することは現状できません。上記の
> 許容誤差キーはこの影響を受けません。

---

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

---

## 互換性

| 環境 | 状態 |
|---|---|
| reForge + SDXL | ✅ 確認済み |
| reForge + SD1.5 | ✅ 確認済み |
| Forge Neo + SDXL | ✅ 確認済み |
| Forge Neo + Anima (Flow Matching) | ✅ 確認済み（Hires.fix 含む） |
| A1111 | ❌ 非対応 |

---

## ODE の定式化

```
dx/dσ = (x − D(x, σ)) / σ
```

`D(x, σ)` はノイズレベル σ におけるモデルの denoised latent 予測値。

---

## ライセンス

本プロジェクトは [memmaptensor/ComfyUI-RK-Sampler](https://github.com/memmaptensor/ComfyUI-RK-Sampler) の移植版です。原作は **GPL-3.0** ライセンスで公開されているため、その派生物である本リポジトリも **GPL-3.0** で配布されます。

- 原作: [memmaptensor/ComfyUI-RK-Sampler](https://github.com/memmaptensor/ComfyUI-RK-Sampler)（GPL-3.0）
- ODE ソルバーバックエンド: [torchode](https://github.com/martenlienen/torchode)

全文は `LICENSE` ファイルを参照してください。
