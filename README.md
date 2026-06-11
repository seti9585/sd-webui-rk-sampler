# sd-webui-rk-sampler

A reForge extension that ports all ODE solver methods from
[ComfyUI-RK-Sampler](https://github.com/memmaptensor/ComfyUI-RK-Sampler)
as an independent sampler.

> **Original:** memmaptensor/ComfyUI-RK-Sampler  
> **Dependency:** [torchode](https://github.com/martenlienen/torchode) (`pip install torchode`)

---

reForge の独立したサンプラー拡張として、
[ComfyUI-RK-Sampler](https://github.com/memmaptensor/ComfyUI-RK-Sampler)
の全 ODE ソルバーメソッドを移植したものです。

> **原作:** memmaptensor/ComfyUI-RK-Sampler  
> **依存ライブラリ:** [torchode](https://github.com/martenlienen/torchode) (`pip install torchode`)

---

## Features / 特徴

- Registers **22 methods** as the **"RK Sampler"** entry in the sampling method dropdown.  
  サンプラードロップダウンに **"RK Sampler"** として登録、**22 メソッド**を搭載。

- **txt2img and Hires. fix can use different methods** via the Script accordion.  
  Script アコーディオンから **txt2img と Hires. fix に異なるメソッドを設定可能**。

- Adaptive methods (`ae_*`) adjust step size automatically via PID controller.  
  適応ステップ (`ae_*`) は PID コントローラで刻み幅を自動調整。

- Fixed-step methods (`fe_*`) follow the sigma schedule like standard samplers.  
  固定ステップ (`fe_*`) は通常のサンプラーと同様に sigma スケジュールに従う。

- Scipy adaptive methods (`se_*`) use `scipy.integrate` as the ODE backend.  
  scipy 適応 (`se_*`) は `scipy.integrate` をバックエンドとして使用。

- Supports **Flow Matching models** (Anima / FLUX / SD3) with correct noise injection in Hires. fix.  
  Hires. fix での **Flow Matching モデル**（Anima / FLUX / SD3 等）の正しいノイズ注入に対応。

- Parameters (rtol / atol / max_steps / PID) can be set per-generation in the Script UI,  
  overriding the Settings tab defaults.  
  パラメータ（rtol / atol / max_steps / PID）は Script UI で生成ごとに上書き可能。Settings タブのデフォルト値を優先度で上書きします。

- Generation parameters are embedded in PNG infotext for reproducibility.  
  生成パラメータは PNG の infotext に記録され再現性を保持。

---

## Requirements / 動作環境

| Item | Value |
|---|---|
| WebUI | stable-diffusion-webui-reForge |
| Python | 3.10 / 3.12 |
| Required | `torchode` — `pip install torchode` |
| Optional | `scipy` — pre-installed in reForge |
| GPU | CUDA / MPS / DirectML |

A1111 非対応。

---

## Installation / インストール

```
extensions/
└── sd-webui-rk-sampler/
    └── scripts/
        └── rk_sampler.py
```

1. Clone or copy this repository into your `extensions/` folder.  
2. Install torchode: `pip install torchode`  
3. Restart the WebUI.

1. このリポジトリを `extensions/` フォルダにクローンまたはコピー。  
2. torchode をインストール: `pip install torchode`  
3. WebUI を再起動。

---

## Usage / 使い方

### Sampler dropdown / サンプラードロップダウン

Select **"RK Sampler"** from the sampling method dropdown.  
The default method when the Script accordion is not enabled is **`ae_dopri5`**.

サンプラードロップダウンから **"RK Sampler"** を選択します。  
Script アコーディオンが無効の場合のデフォルトメソッドは **`ae_dopri5`** です。

### Script accordion / Script アコーディオン

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

Script セクションの **"RK Sampler"** アコーディオンを展開して設定します。

Special values for both method dropdowns / メソッドドロップダウンの特殊値:

- **Use same sampler** — Delegates to the sampler selected in the dropdown. reForge のドロップダウンで選択したサンプラーに委譲。
- **→ TDE Sampler** — Routes to TDE Sampler for that pass. そのパスを TDE Sampler にルーティング。

---

## Methods / メソッド一覧

### Adaptive Step — `ae_*` (torchode backend)

Step size is determined automatically by the PID controller.  
The scheduler type and step count are used only for the sigma range;  
the actual number of ODE evaluations is controlled by rtol / atol.  
In CUDA environments, integration is performed in float64 for accurate error estimation.

刻み幅は PID コントローラが自動決定。スケジューラの種類やステップ数は sigma の範囲にのみ使用され、  
実際の ODE 評価回数は rtol / atol で制御される。CUDA 環境では精度の高い誤差推定のため float64 で積分を実行。

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

| メソッド | アルゴリズム | 次数 | NFEs/ステップ | 備考 |
|---|---|---|---|---|
| `ae_bosh3` | Bogacki–Shampine | 3 | 3 | 2/3 次埋め込みペア。**推奨**。効率と精度のバランスが優秀。 |
| `ae_cash_karp5` | Cash–Karp | 5 | 6 | 4/5 次埋め込みペア。Cash と Karp (1990) の古典手法。 |
| `ae_dopri5` | Dormand–Prince | 5 | 6 | 最も広く使われる適応 ODE ソルバー（≈ scipy RK45）。デフォルトメソッド。 |
| `ae_dopri8` | Dormand–Prince | 8 | 13 | 8次 Dormand-Prince。最高精度・最高コスト。 |
| `ae_fehlberg2` | Fehlberg | 2 | 3 | 低次 Fehlberg ペア。高速・低精度。 |
| `ae_fehlberg5` | Runge–Kutta–Fehlberg | 5 | 6 | 古典的な RKF45。`ae_dopri5` と同等の品質。 |
| `ae_heun_euler2` | Heun–Euler | 2 | 2 | 1/2 次埋め込みペア。最速の適応メソッド。 |
| `ae_midpoint2` | Explicit Midpoint | 2 | 2 | 陽的中点法。高速・低精度。 |
| `ae_ralston2` | Ralston | 2 | 2 | 打ち切り誤差上界を最小化した 2 次法。 |
| `ae_tsit5` | Tsitouras | 5 | 6 | 現代的な効率的 5 次法（Tsitouras 2011）。`ae_dopri5` と同等以上。 |

---

### Fixed Step — `fe_*` (torchode backend)

Step size follows the sigma schedule. Works like standard non-adaptive samplers.  
Integration runs in float32, matching standard k-diffusion sampler behavior.

刻み幅は sigma スケジュールに従う。通常の非適応サンプラーと同様の動作。  
積分は float32 で実行。k-diffusion サンプラーの挙動に準拠。

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

---

### Scipy Adaptive — `se_*` (scipy backend)

Uses `scipy.integrate.solve_ivp`. Processes one sample at a time (not batched).  
Slower than `ae_*` for batch sizes > 1.

`scipy.integrate.solve_ivp` を使用。サンプルを 1 つずつ処理（バッチ並列なし）。  
バッチサイズ > 1 の場合は `ae_*` より低速。

| Method | Algorithm | Order | NFEs/step | Notes |
|---|---|---|---|---|
| `se_RK23` | Bogacki–Shampine | 3 | 3 | scipy `RK23`. Embedded 2/3 pair. |
| `se_RK45` | Dormand–Prince | 5 | 6 | scipy `RK45`. Good accuracy. |
| `se_DOP853` | Dormand–Prince | 8 | 12 | scipy `DOP853`. Highest accuracy, slowest. |

| メソッド | アルゴリズム | 次数 | NFEs/ステップ | 備考 |
|---|---|---|---|---|
| `se_RK23` | Bogacki–Shampine | 3 | 3 | scipy `RK23`。2/3 次埋め込みペア。 |
| `se_RK45` | Dormand–Prince | 5 | 6 | scipy `RK45`。高精度。 |
| `se_DOP853` | Dormand–Prince | 8 | 12 | scipy `DOP853`。最高精度・最低速。 |

---

## Quality Ranking / 品質ランキング

Benchmark from the original ComfyUI-RK-Sampler README.  
Tested with SDXL, 896×1152, CFG 30, batch 1, fixed scheduled, Align Your Steps, 28 steps.  
Not all methods are included; `ae_tsit5`, `ae_dopri8`, `ae_fehlberg2`, `ae_midpoint2`,  
`fe_wray3`, `fe_heun3`, and `se_*` are not listed in the original benchmark.

原作 ComfyUI-RK-Sampler README のベンチマーク。  
SDXL / 896×1152 / CFG 30 / バッチ 1 / fixed scheduled / Align Your Steps / 28 ステップで測定。  
全メソッドは含まれていない。

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

---

## Parameters / パラメータ

Parameters can be set in two places:
- **Script accordion** — per-generation, overrides Settings tab  
- **Settings → RK Sampler** — global defaults

パラメータは 2 箇所で設定できる。Script アコーディオン（生成ごと、Settings タブを上書き）と Settings → RK Sampler（グローバルデフォルト）。

### Common / 共通

| Parameter | Default | Range | Description |
|---|---|---|---|
| **Log Relative Tolerance** | −3.0 | −7.0 〜 0.0 | `10^x` rtol for `ae_*` / `se_*`. Smaller = more precise, slower. Must be ≥ atol. |
| **Log Absolute Tolerance** | −4.0 | −7.0 〜 0.0 | `10^x` atol for `ae_*` / `se_*`. Smaller = more precise, slower. Must be ≤ rtol. |
| **Max ODE Steps** | 1000 | 10 〜 5000 | Failsafe step count cap for adaptive methods. Has no effect on `fe_*`. |

### Advanced / 詳細設定

| Parameter | Default | Range | Description |
|---|---|---|---|
| **Min Sigma** | 1×10⁻⁵ | 0.0 〜 0.1 | Sigma below which ODE gradient is treated as zero. Normally unchanged. |
| **PID P** | 0.0 | 0.0 〜 1.0 | Proportional coefficient for adaptive step controller (`ae_*` only). |
| **PID I** | 1.0 | 0.0 〜 2.0 | Integral coefficient. `P=0 / I=1 / D=0` = basic integral controller. |
| **PID D** | 0.0 | 0.0 〜 1.0 | Derivative coefficient (`ae_*` only). |

---

## Technical Notes / 技術メモ

### Float64 / Float32 separation
Adaptive `ae_*` methods run the ODE integration in **float64** on CUDA for accurate step-size error estimation.  
Fixed `fe_*` methods run in **float32**, matching standard k-diffusion sampler behavior.  
All model calls (CFG denoiser and pre/post-CFG hooks) always receive **float32** tensors regardless,  
via an internal dtype-safe wrapper.

適応 `ae_*` メソッドは CUDA 上で誤差推定の精度確保のため **float64** で ODE 積分を実行。  
固定 `fe_*` メソッドは標準 k-diffusion サンプラーの挙動に合わせ **float32** で実行。  
CFG デノイザーおよび pre/post-CFG フック等は常に **float32** テンソルを受け取る（内部の dtype-safe ラッパーによる）。

### Flow Matching support
Models are automatically detected as Flow Matching (Anima / FLUX / SD3) or DDPM/EDM (SDXL / SD1.5)  
via `model_wrap.sigmas[-1]` (≤ 1.5 → Flow Matching).  
Hires. fix applies the correct noise injection formula for each model type:

モデルは `model_wrap.sigmas[-1]`（≤ 1.5 → Flow Matching）により自動判別。  
Hires. fix で各モデルタイプに正しいノイズ注入式を適用する。

```
Flow Matching:  x_t = (1 − t) × x + t × noise    # linear interpolation
DDPM / EDM:     x_t = x + sigma × noise            # additive
```

### PNG infotext
The following parameters are embedded in generated PNG metadata:

生成 PNG のメタデータに以下のパラメータが記録される。

```
RK method, RK log_rtol, RK log_atol, RK max_steps
```

---

## File Structure / ファイル構成

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

## Compatibility / 互換性

| Environment | Status |
|---|---|
| reForge + SDXL | ✅ Confirmed |
| reForge + SD1.5 | ✅ Confirmed |
| Forge Neo + SDXL | ✅ Confirmed |
| Forge Neo + Anima (Flow Matching) | ✅ Confirmed (including Hires. fix) |
| A1111 | ❌ Not supported |
