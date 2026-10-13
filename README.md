# sd-webui-RK-Sampler

**EN** | [日本語](#日本語)

ODE sampler extension for Stable Diffusion WebUI (reForge / Forge Neo),  
powered by [torchode](https://github.com/martenlienen/torchode).

Port of [ComfyUI-RK-Sampler](https://github.com/memmaptensor/ComfyUI-RK-Sampler) by memmaptensor.

> In reForge, this extension registers as an independent sampler, separate from the built-in ODE Custom,  
> allowing different methods to be selected for txt2img and hires.fix separately.

---

## Features

- Registers **22 methods** as the **"RK Sampler"** entry in the sampling method dropdown.
- **txt2img and Hires.fix can use different methods** via the Script accordion.
- **rtol / atol can also be set independently** for txt2img and hires.fix.
- Adaptive-step methods (`ae_*`) size each step with a PID controller driven by the tolerances.
- SciPy methods (`se_*`) use their own internal step control, reading the same tolerances.
- Fixed-step methods (`fe_*`) follow the sigma schedule and use the WebUI step count directly.
- The tolerance sliders are **greyed out automatically** when the selected method does not use them (any fixed-step method, `Use same sampler`, or `→ TDE Sampler`).
- Max ODE Steps, Min Sigma and the PID coefficients live under **Advanced Settings** and are shared across both passes.
- Generation parameters are embedded in PNG infotext for reproducibility.

---

## Dependency

```bash
pip install torchode
```

No other packages are needed beyond what is already included in your WebUI.

> This extension relies on the Forge backend sampling functions.  
> It is not available in A1111 (AUTOMATIC1111).

---

## Tested on

- reForge, Python 3.10.11, PyTorch 2.9.0+cu128
- Forge Neo 2.29.2, Python 3.13.12, PyTorch 2.13.0+cu130

---

## Installation

**Extensions → Install from URL:**

```
https://github.com/seti9585/sd-webui-RK-Sampler
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
| **txt2img Log Relative Tolerance** | `10^x` relative tolerance (rtol) for the txt2img pass. |
| **txt2img Log Absolute Tolerance** | `10^x` absolute tolerance (atol) for the txt2img pass. |
| **hires.fix Log Relative Tolerance** | `10^x` relative tolerance (rtol) for the Hires.fix pass. |
| **hires.fix Log Absolute Tolerance** | `10^x` absolute tolerance (atol) for the Hires.fix pass. |

Under **Advanced Settings** (shared across both passes):

| Control | Description |
|---|---|
| **Max ODE Steps** | Step count cap for `ae_*` / `se_*` methods. |
| **Min Sigma** | Lower sigma cutoff. Normally no need to change. |
| **PID P / I / D Coefficient** | Step-size controller gains for `ae_*` methods only. |

Special values for both method dropdowns:

- **Use same sampler** — Delegates to the sampler selected in the dropdown.
- **→ TDE Sampler** — Routes to TDE Sampler for that pass.

> The tolerance sliders are enabled only while an `ae_*` or `se_*` method is
> selected. For fixed-step `fe_*` methods, `Use same sampler`, and
> `→ TDE Sampler`, the sliders are greyed out because rtol / atol are not used.
> Their values are preserved and reappear when a tolerance-driven method is
> selected again.

---

## ODE Formulation

```
dx/dσ = (x − D(x, σ)) / σ
```

`D(x, σ)` is the denoised latent predicted by the model at noise level σ.  
The probability-flow ODE is integrated from σ_max down to σ_min.

---

## Methods

### Adaptive Step (`ae_*`)

Step size is determined by a PID controller so that the relative tolerance (rtol)
and absolute tolerance (atol) targets are met. The scheduler type and step count
are used only for the sigma range; the actual number of ODE evaluations is
controlled by the tolerances. In CUDA environments, integration is performed in
float64 for accurate error estimation.

| Method | Algorithm | Order | Notes |
|---|---|---|---|
| `ae_dopri5` | Dormand–Prince | 5 | The most widely used adaptive ODE method. Well-balanced general-purpose choice. |
| `ae_dopri8` | Dormand–Prince | 8 | Higher-order variant. Highest accuracy at highest cost. |
| `ae_tsit5` | Tsitouras | 5 | Modern 5th-order alternative to Dormand–Prince. |
| `ae_cash_karp5` | Cash–Karp | 5 | 5th-order variant with a different error estimator. |
| `ae_fehlberg5` | Fehlberg | 5 | Classical RKF45. |
| `ae_bosh3` | Bogacki–Shampine | 3 | Lightweight. Efficient when looser tolerances are acceptable. |
| `ae_fehlberg2` | Fehlberg | 2 | Low order. Fast, but needs tighter tolerances to converge cleanly. |
| `ae_heun_euler2` | Heun–Euler | 2 | Simplest adaptive pair. Benefits from tighter tolerances. |
| `ae_midpoint2` | Midpoint | 2 | 2nd-order adaptive variant. |
| `ae_ralston2` | Ralston | 2 | 2nd-order with reduced truncation error. |

### Fixed Step (`fe_*`)

Fixed-step methods ignore the tolerances and step exactly along the sigma
schedule, using the WebUI step count directly. Because they have no per-step
error control, their accuracy depends strongly on the step count. Integration is
performed in float32 to match standard k-diffusion sampler behaviour.

| Method | Algorithm | Order | Notes |
|---|---|---|---|
| `fe_euler1` | Euler | 1 | Simplest method. Fastest per step. |
| `fe_heun3` | Heun | 3 | Heun's 3rd-order method. |
| `fe_kutta3` | Kutta | 3 | Kutta's 3rd-order method. |
| `fe_ralston3` | Ralston | 3 | 3rd-order with reduced truncation error. |
| `fe_wray3` | van der Houwen / Wray | 3 | 3rd-order variant. |
| `fe_ssprk3` | Strong Stability Preserving RK | 3 | 3rd-order SSP method. |
| `fe_kutta4` | Classical Runge–Kutta | 4 | Classical RK4 (weights 1/6, 1/3, 1/3, 1/6). Good quality at moderate step counts. |
| `fe_kutta_38th4` | Runge–Kutta 3/8-rule | 4 | 4th-order 3/8 variant (weights 1/8, 3/8, 3/8, 1/8). |
| `fe_ralston4` | Ralston | 4 | 4th-order minimizing truncation error. |

### SciPy (`se_*`)

These methods delegate step-size control to SciPy's own implementation, which
reads the same rtol / atol values. They process one sample at a time, so batch
generation is slower than with the `ae_*` methods.

| Method | Order | NFE per step |
|---|---|---|
| `se_RK23` | 3 | 3 |
| `se_RK45` | 5 | 6 |
| `se_DOP853` | 8 | 12 |

---

## Settings

Persistent defaults are available under **Settings → RK Sampler**.

| Parameter | Default | Range | Description |
|---|---|---|---|
| **Log Relative Tolerance** | −3.0 | −7.0 〜 0.0 | `10^x` rtol for `ae_*` / `se_*` methods. Smaller = more precise, slower. Must be ≥ the Log Absolute Tolerance. |
| **Log Absolute Tolerance** | −4.0 | −7.0 〜 0.0 | `10^x` atol for `ae_*` / `se_*` methods. Smaller = more precise, slower. Must be ≤ the Log Relative Tolerance. |
| **Max ODE Steps** | 250 | 1 〜 500 | Failsafe step count cap for `ae_*` / `se_*` methods. Has no effect on `fe_*` methods. |
| **[Advanced] Min Sigma** | 0.00001 | 0.0 〜 0.1 | Passes with zero gradient below this sigma. Normally no need to change. |
| **[Advanced] PID P Coefficient** | 0.0 | 0.0 〜 1.0 | Proportional gain of the step-size controller (`ae_*` only). |
| **[Advanced] PID I Coefficient** | 1.0 | 0.0 〜 2.0 | Integral gain of the step-size controller (`ae_*` only). |
| **[Advanced] PID D Coefficient** | 0.0 | 0.0 〜 1.0 | Derivative gain of the step-size controller (`ae_*` only). |

> **Note:** Always set the Log Relative Tolerance greater than or equal to the Log Absolute Tolerance.

> **If an adaptive method stops after reaching the step cap:** the tolerances are
> too strict for the sigma range (this is common in the narrow Hires.fix range).
> Loosen `log_rtol` / `log_atol`, or raise **Max ODE Steps**.

---

## Reproducibility

The following parameters are embedded in generated PNG metadata:

```
RK txt2img method, RK txt2img log_rtol, RK txt2img log_atol
RK hires method,   RK hires log_rtol,   RK hires log_atol
RK max_steps
```

`RK method` is also written for the txt2img pass, for backward compatibility with
images generated before the per-pass keys were introduced.

---

---

# 日本語

**[English](#sd-webui-rk-sampler)** | 日本語

[torchode](https://github.com/martenlienen/torchode) を使った ODE サンプラー拡張機能（reForge / Forge Neo 向け）。

[ComfyUI-RK-Sampler](https://github.com/memmaptensor/ComfyUI-RK-Sampler)（memmaptensor 作）を Forge 系 WebUI 向けに移植。

> reForge では、組み込みの ODE Custom とは独立したサンプラーとして登録されるため、  
> txt2img と hires.fix で異なるメソッドを選択できます。

---

## 特徴

- Sampling method ドロップダウンに **22 種類のメソッド**を **「RK Sampler」** として登録。
- **txt2img と Hires.fix で異なるメソッド**を Script アコーディオンで選択可能。
- **rtol / atol も txt2img と Hires.fix で個別に設定可能。**
- 可変ステップ法（`ae_*`）は許容誤差を用いた PID コントローラでステップ幅を自動調整。
- SciPy メソッド（`se_*`）は SciPy 側の刻み幅制御を使用し、同じ許容誤差を参照。
- 固定ステップ法（`fe_*`）は sigma スケジュールに沿って刻み、WebUI のステップ数をそのまま使用。
- 選択中のメソッドが許容誤差を使わない場合（すべての固定ステップ法、`Use same sampler`、`→ TDE Sampler`）、許容誤差スライダーは**自動的にグレーアウト**されます。
- Max ODE Steps・Min Sigma・PID 係数は **Advanced Settings** に配置され、両パス共通です。
- 生成パラメータは PNG の infotext に記録され再現性を保持。

---

## 依存ライブラリ

```bash
pip install torchode
```

WebUI に既に含まれるもの以外、追加パッケージは不要です。

> 本拡張機能は Forge バックエンドの sampling 関数に依存します。  
> A1111（AUTOMATIC1111）では使用できません。

---

## 動作確認環境

- reForge、Python 3.10.11、PyTorch 2.9.0+cu128
- Forge Neo 2.29.2、Python 3.13.12、PyTorch 2.13.0+cu130

---

## インストール

**Extensions → Install from URL:**

```
https://github.com/seti9585/sd-webui-RK-Sampler
```

---

## 使い方

### Sampler ドロップダウン

Sampling method ドロップダウンから **「RK Sampler」** を選択します。  
Script アコーディオンが無効、または **「Use same sampler」** の場合は、メインの
サンプラードロップダウンで選択中のメソッドに処理を委譲します。

### Script アコーディオン

Script セクションの **「RK Sampler」** アコーディオンを展開して設定します。

| 項目 | 説明 |
|---|---|
| **txt2img Method** | txt2img（および img2img）で使うメソッド。 |
| **hires.fix Method** | Hires.fix パスでのみ使うメソッド。 |
| **txt2img Log Relative Tolerance** | txt2img パスの `10^x` 相対許容誤差（rtol）。 |
| **txt2img Log Absolute Tolerance** | txt2img パスの `10^x` 絶対許容誤差（atol）。 |
| **hires.fix Log Relative Tolerance** | Hires.fix パスの `10^x` 相対許容誤差（rtol）。 |
| **hires.fix Log Absolute Tolerance** | Hires.fix パスの `10^x` 絶対許容誤差（atol）。 |

**Advanced Settings**（両パス共通）：

| 項目 | 説明 |
|---|---|
| **Max ODE Steps** | `ae_*` / `se_*` メソッドのステップ数上限。 |
| **Min Sigma** | sigma の下限カットオフ。通常は変更不要。 |
| **PID P / I / D Coefficient** | ステップ幅コントローラのゲイン（`ae_*` のみ有効）。 |

両メソッドドロップダウンの特殊値：

- **Use same sampler** — ドロップダウンで選択中のサンプラーに委譲。
- **→ TDE Sampler** — そのパスを TDE Sampler に委譲。

> 許容誤差スライダーは `ae_*` または `se_*` メソッドを選択している間だけ有効です。
> 固定ステップ法（`fe_*`）・`Use same sampler`・`→ TDE Sampler` では rtol / atol が
> 使われないため、スライダーはグレーアウトされます。値は保持され、許容誤差を使う
> メソッドを再び選ぶと元の値が復活します。

---

## ODE の定式化

```
dx/dσ = (x − D(x, σ)) / σ
```

`D(x, σ)` はノイズレベル σ におけるモデルの denoised latent 予測値。  
probability flow ODE を σ_max から σ_min へ降順に積分します。

---

## メソッド一覧

### 可変ステップ法（`ae_*`）

ステップ幅は PID コントローラにより、相対許容誤差（rtol）と絶対許容誤差（atol）の
目標値を満たすよう決定されます。スケジューラの種類やステップ数は sigma の範囲にのみ
使用され、実際の ODE 評価回数は許容誤差で制御されます。CUDA 環境では精度の高い誤差
推定のため float64 で積分を実行します。

| メソッド | アルゴリズム | 次数 | 特徴 |
|---|---|---|---|
| `ae_dopri5` | Dormand–Prince | 5 | 最も広く使われる可変ステップ法。バランスの取れた汎用メソッド。 |
| `ae_dopri8` | Dormand–Prince | 8 | 高次版。最高精度だが最も低速。 |
| `ae_tsit5` | Tsitouras | 5 | Dormand–Prince に代わる現代的な 5 次法。 |
| `ae_cash_karp5` | Cash–Karp | 5 | 誤差推定の異なる 5 次のバリアント。 |
| `ae_fehlberg5` | Fehlberg | 5 | 古典的な RKF45。 |
| `ae_bosh3` | Bogacki–Shampine | 3 | 軽量。許容誤差を緩めても許容できる場面で効率的。 |
| `ae_fehlberg2` | Fehlberg | 2 | 低次。高速だが、きれいに収束させるには許容誤差を厳しめにする必要がある。 |
| `ae_heun_euler2` | Heun–Euler | 2 | 最もシンプルな可変ステップ対。許容誤差を厳しめにすると安定する。 |
| `ae_midpoint2` | Midpoint | 2 | 2 次の可変ステップ版。 |
| `ae_ralston2` | Ralston | 2 | 打ち切り誤差を抑えた 2 次法。 |

### 固定ステップ法（`fe_*`）

固定ステップ法は許容誤差を無視し、sigma スケジュールに沿って正確に刻みます。WebUI の
ステップ数をそのまま使用します。1 ステップごとの誤差制御を持たないため、精度はステップ
数への依存度が高くなります。標準の k-diffusion サンプラーと挙動を揃えるため float32 で
積分します。

| メソッド | アルゴリズム | 次数 | 特徴 |
|---|---|---|---|
| `fe_euler1` | Euler | 1 | 最もシンプル。1 ステップあたりが最速。 |
| `fe_heun3` | Heun | 3 | Heun の 3 次法。 |
| `fe_kutta3` | Kutta | 3 | Kutta の 3 次法。 |
| `fe_ralston3` | Ralston | 3 | 打ち切り誤差を抑えた 3 次法。 |
| `fe_wray3` | van der Houwen / Wray | 3 | 3 次のバリアント。 |
| `fe_ssprk3` | Strong Stability Preserving RK | 3 | 3 次の SSP 法。 |
| `fe_kutta4` | 古典的 Runge–Kutta | 4 | 古典的 RK4（重み 1/6, 1/3, 1/3, 1/6）。中程度のステップ数で高品質。 |
| `fe_kutta_38th4` | Runge–Kutta 3/8 則 | 4 | 4 次の 3/8 則バリアント（重み 1/8, 3/8, 3/8, 1/8）。 |
| `fe_ralston4` | Ralston | 4 | 打ち切り誤差を最小化する 4 次法。 |

### SciPy メソッド（`se_*`）

刻み幅の制御を SciPy 側の実装に委ねるメソッドです。参照する rtol / atol は同じもの
です。サンプルを 1 枚ずつ処理するため、バッチ生成では `ae_*` より低速になります。

| メソッド | 次数 | 1 ステップあたりの評価回数 |
|---|---|---|
| `se_RK23` | 3 | 3 |
| `se_RK45` | 5 | 6 |
| `se_DOP853` | 8 | 12 |

---

## Settings

**Settings → RK Sampler** でデフォルト値を永続的に設定できます。

| パラメータ | デフォルト | 範囲 | 説明 |
|---|---|---|---|
| **Log Relative Tolerance** | −3.0 | −7.0 〜 0.0 | `ae_*` / `se_*` メソッドの `10^x` rtol。小さいほど高精度・低速。Log Absolute Tolerance 以上にする。 |
| **Log Absolute Tolerance** | −4.0 | −7.0 〜 0.0 | `ae_*` / `se_*` メソッドの `10^x` atol。小さいほど高精度・低速。Log Relative Tolerance 以下にする。 |
| **Max ODE Steps** | 250 | 1 〜 500 | `ae_*` / `se_*` メソッドのステップ数上限（フェイルセーフ）。`fe_*` には影響しない。 |
| **[Advanced] Min Sigma** | 0.00001 | 0.0 〜 0.1 | この sigma を下回る領域では勾配ゼロとして通過。通常は変更不要。 |
| **[Advanced] PID P Coefficient** | 0.0 | 0.0 〜 1.0 | ステップ幅コントローラの比例ゲイン（`ae_*` のみ）。 |
| **[Advanced] PID I Coefficient** | 1.0 | 0.0 〜 2.0 | ステップ幅コントローラの積分ゲイン（`ae_*` のみ）。 |
| **[Advanced] PID D Coefficient** | 0.0 | 0.0 〜 1.0 | ステップ幅コントローラの微分ゲイン（`ae_*` のみ）。 |

> **注意：** Log Relative Tolerance は必ず Log Absolute Tolerance 以上にして下さい。

> **可変ステップ法がステップ数上限で停止した場合：** 許容誤差が sigma の範囲に対して
> 厳しすぎます（狭い Hires.fix の範囲で起こりやすい）。`log_rtol` / `log_atol` を緩めるか、
> **Max ODE Steps** を上げて下さい。

---

## 再現性

生成された PNG のメタデータには以下のパラメータが記録されます。

```
RK txt2img method, RK txt2img log_rtol, RK txt2img log_atol
RK hires method,   RK hires log_rtol,   RK hires log_atol
RK max_steps
```

per-pass キー導入前に生成された画像との互換のため、txt2img パスでは `RK method`
も併せて記録されます。

---

## License

MIT License — Original port: [ComfyUI-RK-Sampler](https://github.com/memmaptensor/ComfyUI-RK-Sampler) © memmaptensor

Adaptive and fixed-step integration is provided by [torchode](https://github.com/martenlienen/torchode) © Marten Lienen.

The fixed-step Runge–Kutta tableaux (`fe_kutta4`, `fe_kutta_38th4`, `fe_ralston3`, `fe_ralston4`, `fe_wray3`, `fe_ssprk3`) are also available in [sd-webui-TDE-Sampler](https://github.com/seti9585/sd-webui-TDE-Sampler), so each has a torchdiffeq-path counterpart there.
