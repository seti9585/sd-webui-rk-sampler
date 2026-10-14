# sd-webui-RK-Sampler

English | [日本語](#日本語)

RK Sampler is a sampler extension for reForge / Forge Neo that generates images using numerical methods for solving ODEs (ordinary differential equations). It uses [torchode](https://github.com/martenlienen/torchode) and provides 22 methods.

It is designed to use ODE methods in both txt2img and Hires.fix, with a separate choice of method for each pass. You can also use RK Sampler for only one pass alongside another sampler, or combine RK Sampler and TDE Sampler across the two passes.

[Installation and tested environments](#installation-and-tested-environments) · [Usage](#usage) · [Adjusting computation and accuracy](#adjusting-computation-and-accuracy) · [Method reference](#method-reference)

## Installation and tested environments

In Extensions → Install from URL, enter:

```text
https://github.com/seti9585/sd-webui-rk-sampler
```

The dependency is `torchode`. If needed, install it in the Python environment used by your WebUI:

```bash
pip install torchode
```

Restart the WebUI after installation. This extension uses Forge backend sampling functions and does not support A1111 (AUTOMATIC1111). In reForge, it registers separately from the built-in ODE Custom.

Tested environments:

- reForge, Python 3.10.11, PyTorch 2.9.0+cu128
- Forge Neo 2.29.2, Python 3.13.12, PyTorch 2.13.0+cu130

## Compatibility with Spectrum extensions

When using U-Net models, [Forge Neo's built-in Spectrum (`sd_forge_spectrum`)](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo/extensions-builtin/sd_forge_spectrum) and hirorohi03's [sd-forge-spectrum-faithful](https://github.com/hirorohi03/sd-forge-spectrum-faithful) are not designed for higher-order sampling methods that evaluate the U-Net multiple times within one solver step. Using them with these methods can stop generation with an error.

These Spectrum extensions manage forecasting progress by counting U-Net calls, without grouping the internal stages of a solver step. In multi-stage ODE methods, one U-Net call is not one solver step, so their progress and forecast-history handling do not match the sampler's integration.

If you want to use Spectrum-style acceleration alongside this sampler, try my [sd-webui-ChebyCast](https://github.com/seti9585/sd-webui-ChebyCast) instead. Disable the built-in Spectrum and spectrum-faithful extensions before enabling ChebyCast. It distinguishes solver steps from individual U-Net calls and supports fixed-step multi-stage methods. Use `Stage grouping = auto` for TDE / RK Sampler. ChebyCast does not support adaptive ODE methods.

## Usage

### Choose the sampler, then its ODE method

The WebUI sampler dropdowns select which sampler runs. The Script controls select which ODE method RK Sampler uses.

1. Select `RK Sampler` in Sampling method. To use it for Hires.fix too, select `RK Sampler` there as well, or let Hires.fix use the same sampler as txt2img.
2. Enable the `RK Sampler` accordion in the Script section.
3. Choose an ODE method in `txt2img Method` and `hires.fix Method` for each pass you use.
4. Adjust the WebUI step count for a fixed-step method, or the tolerances for an adaptive-step method, then generate.

Selecting a method in the Script section does not change the WebUI sampler selection.

### Combine txt2img and Hires.fix samplers

| Combination | WebUI txt2img sampler | WebUI Hires.fix sampler | Script controls to configure |
|---|---|---|---|
| RK Sampler for both passes | `RK Sampler` | `RK Sampler` or same as txt2img | Choose the method for each pass |
| RK Sampler for txt2img only | `RK Sampler` | Another sampler | Choose the txt2img method |
| RK Sampler for Hires.fix only | Another sampler | `RK Sampler` | Choose the Hires.fix method |
| RK for txt2img, TDE for Hires.fix | `RK Sampler` | `TDE Sampler` | RK's txt2img Method and TDE's hires.fix Solver |

To combine RK and TDE, install both extensions and enable both Script accordions. The reverse combination—TDE for txt2img and RK for Hires.fix—also works.

### Special dropdown choices

| Choice | Behavior |
|---|---|
| `Use same sampler` | Delegates to the default sampler instead of using this extension's ODE methods |
| `→ TDE Sampler` | Hands this pass to TDE Sampler, using that extension's method and settings for the same pass |

The fallback for `Use same sampler` is the WebUI `sampler_name` setting, or Euler if that setting is absent. It does not automatically read the txt2img or Hires.fix sampler dropdown. To combine this extension with another sampler, select the samplers explicitly as shown in the table.

When using `→ TDE Sampler`, enable the receiving extension's Script accordion and select a concrete method for that pass. Do not create a circular route from RK to TDE and back to RK for the same pass.

## Adjusting computation and accuracy

### Fixed and adaptive steps

| Type | What to adjust | Behavior |
|---|---|---|
| Fixed-step (`fe_*`) | WebUI step count and scheduler for each pass | Follows the supplied sigma schedule |
| Adaptive-step (`ae_*`) | Relative and absolute tolerances | Adjusts step size using an error estimate |
| SciPy (`se_*`) | Relative and absolute tolerances | SciPy controls the step size internally |

For fixed-step methods, the step count determines how finely the calculation is divided. Tolerances and Max ODE Steps are not used. Different methods perform different numbers of model evaluations per step, so equal step counts do not imply equal generation times.

Adaptive methods do not use the WebUI step count as their integration step count. They take the sigma range from the WebUI settings, then choose their steps using the tolerances. The `ae_*` methods use a PID controller; `se_*` methods use SciPy's internal step control.

### Reading the tolerances

txt2img and Hires.fix each have their own Log Relative Tolerance and Log Absolute Tolerance controls. The Script values apply to the current generation.

- Log Relative Tolerance: `rtol = 10^x`, an error target relative to the size of the values being integrated.
- Log Absolute Tolerance: `atol = 10^x`, an absolute error target that also applies near zero.

For example, `−3` means `0.001`, and `−4` means `0.0001`. More negative values set stricter tolerances and usually require more computation. These control numerical accuracy, not the visual quality of an image directly. Set Log Relative Tolerance greater than or equal to Log Absolute Tolerance.

The tolerance sliders are greyed out for fixed-step methods, `Use same sampler`, and `→ TDE Sampler`, because those choices do not use this extension's tolerances. Values are preserved when you switch back to an adaptive method.

### Advanced settings and defaults

Persistent defaults are available under Settings → RK Sampler. The Script's Advanced Settings are shared by txt2img and Hires.fix.

| Control | Default | Range | Purpose |
|---|---|---|---|
| Log Relative Tolerance | −3.0 | −7.0 to 0.0 | Relative tolerance for adaptive methods |
| Log Absolute Tolerance | −4.0 | −7.0 to 0.0 | Absolute tolerance for adaptive methods |
| Max ODE Steps | 250 | 1 to 500 | Step cap for adaptive / SciPy methods |
| Min Sigma | 0.00001 | 0.0 to 0.1 | Uses a zero gradient below this sigma |
| PID P Coefficient | 0.0 | 0.0 to 1.0 | Proportional gain (`ae_*` only) |
| PID I Coefficient | 1.0 | 0.0 to 2.0 | Integral gain (`ae_*` only) |
| PID D Coefficient | 0.0 | 0.0 to 1.0 | Derivative gain (`ae_*` only) |

Min Sigma and the PID coefficients can normally be left at their defaults.

If an adaptive method reaches the step cap, loosen the tolerances or raise Max ODE Steps. Strict tolerances can exhaust the cap, particularly in the narrow sigma range used by Hires.fix.

## Method reference

“Order” describes a mathematical accuracy property of the numerical method. A higher order does not directly imply a better-looking image.

### Adaptive-step (`ae_*`)

| Method | Algorithm | Order |
|---|---|---|
| `ae_dopri5` | Dormand–Prince | 5 |
| `ae_dopri8` | Dormand–Prince | 8 |
| `ae_tsit5` | Tsitouras | 5 |
| `ae_cash_karp5` | Cash–Karp | 5 |
| `ae_fehlberg5` | Fehlberg | 5 |
| `ae_bosh3` | Bogacki–Shampine | 3 |
| `ae_fehlberg2` | Fehlberg | 2 |
| `ae_heun_euler2` | Heun–Euler | 2 |
| `ae_midpoint2` | Midpoint | 2 |
| `ae_ralston2` | Ralston | 2 |

### Fixed-step (`fe_*`)

| Method | Algorithm | Order |
|---|---|---|
| `fe_euler1` | Euler | 1 |
| `fe_heun3` | Heun | 3 |
| `fe_kutta3` | Kutta | 3 |
| `fe_ralston3` | Ralston | 3 |
| `fe_wray3` | van der Houwen / Wray | 3 |
| `fe_ssprk3` | Strong Stability Preserving RK | 3 |
| `fe_kutta4` | Classical Runge–Kutta | 4 |
| `fe_kutta_38th4` | Runge–Kutta 3/8-rule | 4 |
| `fe_ralston4` | Ralston | 4 |

### SciPy (`se_*`)

SciPy methods process one image at a time, so batch generation can be slower than with `ae_*`. The evaluation counts below are per step, not totals for an entire generation.

| Method | Order | NFE per step |
|---|---|---|
| `se_RK23` | 3 | 3 |
| `se_RK45` | 5 | 6 |
| `se_DOP853` | 8 | 12 |

## Technical details

### ODE formulation

```text
dx/dσ = (x − D(x, σ)) / σ
```

`x` is the current latent, `σ` is the noise level, and `D(x, σ)` is the denoised latent predicted by the model. The probability-flow ODE is integrated from higher to lower noise levels.

Adaptive methods integrate in float64 in CUDA environments. Fixed-step methods use float32 to match standard k-diffusion samplers. Inputs to the denoising model are cast to float32.

### Settings recorded in PNGs

The method and tolerances are written to PNG infotext separately for txt2img and Hires.fix.

```text
RK txt2img method, RK txt2img log_rtol, RK txt2img log_atol
RK hires method, RK hires log_rtol, RK hires log_atol
RK max_steps
```

`RK method` is also recorded for txt2img for compatibility with older images. Min Sigma and the PID coefficients are not recorded. When regenerating under the same conditions, also match the advanced settings that are not recorded.

---

# 日本語

[English](#sd-webui-rk-sampler)

RK Sampler は、ODE（常微分方程式）の数値解法を利用して画像を生成する、reForge / Forge Neo 向けのサンプラー拡張機能です。[torchode](https://github.com/martenlienen/torchode) を使い、22種類のメソッドから解法を選べます。

txt2img と Hires.fix の両方で ODE を使い、それぞれ異なる解法を選べる設計です。片方だけを RK Sampler にして他のサンプラーと組み合わせたり、RK Sampler と TDE Sampler を両段階で使い分けたりすることもできます。

[導入と動作確認環境](#導入と動作確認環境) · [使い方](#使い方) · [計算量と精度の調整](#計算量と精度の調整) · [解法一覧](#解法一覧)

## 導入と動作確認環境

Extensions → Install from URL に次のURLを入力します。

```text
https://github.com/seti9585/sd-webui-rk-sampler
```

依存ライブラリは `torchode` です。必要な場合は、WebUI が使う Python 環境でインストールしてください。

```bash
pip install torchode
```

導入後は WebUI を再起動してください。Forge バックエンドの sampling 関数を使うため、A1111（AUTOMATIC1111）では使用できません。reForge では、組み込みの ODE Custom とは別のサンプラーとして登録されます。

動作確認環境：

- reForge、Python 3.10.11、PyTorch 2.9.0+cu128
- Forge Neo 2.29.2、Python 3.13.12、PyTorch 2.13.0+cu130

## Spectrum 拡張機能との併用

U-Net モデルでは、[Forge Neo 内蔵の Spectrum（`sd_forge_spectrum`）](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo/extensions-builtin/sd_forge_spectrum) と、ひろろひ氏（hirorohi03）の [sd-forge-spectrum-faithful](https://github.com/hirorohi03/sd-forge-spectrum-faithful) は、1つの積分ステップ内で U-Net を複数回呼び出す高次の Sampling Method を想定していません。これらのメソッドと併用すると、生成がエラー停止することがあります。

これらの Spectrum 拡張機能は、U-Net の呼び出し回数をもとに予測の進行を管理し、1ステップ内の複数の内部段階（stage）をまとめて扱いません。多段の ODE 解法では「U-Net 呼び出し1回」と「積分ステップ1回」が一致しないため、進行や予測履歴の管理がサンプラーの計算と合わなくなります。

スペクトラム拡張機能を併用する場合は、拙作の [sd-webui-ChebyCast](https://github.com/seti9585/sd-webui-ChebyCast) を試してみてください。内蔵 Spectrum と spectrum-faithful を無効にし、代わりに ChebyCast を有効にします。ChebyCast は積分ステップと個々の U-Net 呼び出しを分けて扱い、固定ステップの多段法に対応しています。TDE / RK Sampler では `Stage grouping = auto` を使ってください。可変ステップの ODE 解法には対応していません。

## 使い方

### サンプラーと解法を選ぶ

WebUI のサンプラー欄では「どのサンプラーを使うか」、Script 欄では「RK Sampler がどの解法を使うか」を設定します。

1. Sampling method で `RK Sampler` を選びます。Hires.fix にも使う場合は、Hires.fix のサンプラーも `RK Sampler` にするか、txt2img と同じサンプラーを使う設定にします。
2. Script 欄の `RK Sampler` アコーディオンを有効にします。
3. `txt2img Method` と `hires.fix Method` で、それぞれ使う解法を選びます。
4. 固定ステップ法なら WebUI のステップ数、可変ステップ法なら許容誤差を調整して生成します。

Script 欄で解法を選ぶだけでは、WebUI のサンプラー選択は切り替わりません。

### txt2img と Hires.fix を使い分ける

| 使いたい組み合わせ | WebUI の txt2img サンプラー | WebUI の Hires.fix サンプラー | Script 欄で設定する解法 |
|---|---|---|---|
| 両方 RK Sampler | `RK Sampler` | `RK Sampler` または txt2img と同じ | txt2img／Hires.fix のメソッドをそれぞれ選ぶ |
| txt2img だけ RK Sampler | `RK Sampler` | 他のサンプラー | txt2img のメソッドを選ぶ |
| Hires.fix だけ RK Sampler | 他のサンプラー | `RK Sampler` | Hires.fix のメソッドを選ぶ |
| txt2img は RK、Hires.fix は TDE | `RK Sampler` | `TDE Sampler` | RK の txt2img Method と TDE の hires.fix Solver を選ぶ |

RK と TDE を組み合わせる場合は、両拡張機能を導入し、両方の Script アコーディオンを有効にします。上の表の逆に、txt2img を TDE、Hires.fix を RK にすることもできます。

### 解法ドロップダウンの特殊な選択肢

| 選択肢 | 動作 |
|---|---|
| `Use same sampler` | この拡張機能の ODE 解法を使わず、既定のサンプラーへ処理を渡す |
| `→ TDE Sampler` | その段階の処理を TDE Sampler に渡し、渡した先の同じ段階の解法と設定を使う |

`Use same sampler` の委譲先は WebUI の `sampler_name` 設定で、設定がなければ Euler です。txt2img／Hires.fix のサンプラー欄を自動的に参照して切り替える選択肢ではありません。他のサンプラーと組み合わせる場合は、上の表のように WebUI の各サンプラー欄で選ぶと明確です。

`→ TDE Sampler` を使う場合は、渡した先の Script アコーディオンも有効にし、該当する段階で具体的な解法を選んでください。同じ段階で RK → TDE → RK と相互に渡す設定にはしないでください。

## 計算量と精度の調整

### 固定ステップ法と可変ステップ法

| 種類 | 調整する項目 | 動作 |
|---|---|---|
| 固定ステップ法（`fe_*`） | WebUI の txt2img／Hires.fix のステップ数とスケジューラ | 指定した sigma スケジュールに沿って計算する |
| 可変ステップ法（`ae_*`） | 相対・絶対許容誤差 | 推定誤差に応じてステップ幅を調整する |
| SciPy メソッド（`se_*`） | 相対・絶対許容誤差 | SciPy 側でステップ幅を調整する |

固定ステップ法では、ステップ数が計算の細かさを決めます。許容誤差と Max ODE Steps は使いません。解法ごとに1ステップで行うモデル評価の回数が違うため、同じステップ数でも生成時間は異なります。

可変ステップ法では、WebUI のステップ数をそのまま積分ステップ数として使いません。sigma の範囲は WebUI の設定から取得し、その範囲をどう刻むかは許容誤差で決めます。`ae_*` は PID コントローラ、`se_*` は SciPy 内部の制御を使います。

### 許容誤差の読み方

txt2img と Hires.fix にはそれぞれ、Log Relative Tolerance と Log Absolute Tolerance があります。Script 欄の値は、その生成の設定として使われます。

- Log Relative Tolerance：相対許容誤差 `rtol = 10^x`。値の大きさに応じた誤差の基準。
- Log Absolute Tolerance：絶対許容誤差 `atol = 10^x`。値がゼロに近いときにも使う誤差の基準。

例えば `−3` は `0.001`、`−4` は `0.0001` です。負の値を小さくするほど許容誤差が厳しくなり、通常は計算量が増えます。ODE の計算精度を指定する値であり、画像の見栄えを直接保証する値ではありません。Log Relative Tolerance は Log Absolute Tolerance 以上に設定してください。

固定ステップ法、`Use same sampler`、`→ TDE Sampler` を選ぶと、こちらの許容誤差スライダーは使われないためグレーアウトします。値は保持され、可変ステップ法に戻すと再び使えます。

### 詳細設定と既定値

Settings → RK Sampler で既定値を設定できます。Script 欄の Advanced Settings は txt2img と Hires.fix で共通です。

| 項目 | 既定値 | 範囲 | 用途 |
|---|---|---|---|
| Log Relative Tolerance | −3.0 | −7.0 ～ 0.0 | 可変ステップ法の相対許容誤差 |
| Log Absolute Tolerance | −4.0 | −7.0 ～ 0.0 | 可変ステップ法の絶対許容誤差 |
| Max ODE Steps | 250 | 1 ～ 500 | 可変ステップ法／SciPy メソッドのステップ数上限 |
| Min Sigma | 0.00001 | 0.0 ～ 0.1 | この値より小さい sigma で勾配をゼロにする |
| PID P Coefficient | 0.0 | 0.0 ～ 1.0 | 比例ゲイン（`ae_*` のみ） |
| PID I Coefficient | 1.0 | 0.0 ～ 2.0 | 積分ゲイン（`ae_*` のみ） |
| PID D Coefficient | 0.0 | 0.0 ～ 1.0 | 微分ゲイン（`ae_*` のみ） |

Min Sigma と PID 係数は、通常は既定値のままで使えます。

可変ステップ法がステップ数上限で停止する場合は、許容誤差を緩めるか、Max ODE Steps を増やしてください。特に Hires.fix の狭い sigma 範囲では、厳しい許容誤差で上限に達することがあります。

## 解法一覧

「次数」は数値解法の精度に関する数学的な性質です。高い次数が、そのまま画像の見栄えの良さを意味するわけではありません。

### 可変ステップ法（`ae_*`）

| メソッド | アルゴリズム | 次数 |
|---|---|---|
| `ae_dopri5` | Dormand–Prince | 5 |
| `ae_dopri8` | Dormand–Prince | 8 |
| `ae_tsit5` | Tsitouras | 5 |
| `ae_cash_karp5` | Cash–Karp | 5 |
| `ae_fehlberg5` | Fehlberg | 5 |
| `ae_bosh3` | Bogacki–Shampine | 3 |
| `ae_fehlberg2` | Fehlberg | 2 |
| `ae_heun_euler2` | Heun–Euler | 2 |
| `ae_midpoint2` | Midpoint | 2 |
| `ae_ralston2` | Ralston | 2 |

### 固定ステップ法（`fe_*`）

| メソッド | アルゴリズム | 次数 |
|---|---|---|
| `fe_euler1` | Euler | 1 |
| `fe_heun3` | Heun | 3 |
| `fe_kutta3` | Kutta | 3 |
| `fe_ralston3` | Ralston | 3 |
| `fe_wray3` | van der Houwen / Wray | 3 |
| `fe_ssprk3` | Strong Stability Preserving RK | 3 |
| `fe_kutta4` | 古典的 Runge–Kutta | 4 |
| `fe_kutta_38th4` | Runge–Kutta 3/8 則 | 4 |
| `fe_ralston4` | Ralston | 4 |

### SciPy メソッド（`se_*`）

SciPy メソッドはバッチ内の画像を1枚ずつ処理するため、バッチ生成では `ae_*` より低速になることがあります。表の評価回数は1ステップあたりのもので、生成全体の回数ではありません。

| メソッド | 次数 | 1 ステップあたりの評価回数 |
|---|---|---|
| `se_RK23` | 3 | 3 |
| `se_RK45` | 5 | 6 |
| `se_DOP853` | 8 | 12 |

## 技術的な詳細

### ODE の定式化

```text
dx/dσ = (x − D(x, σ)) / σ
```

`x` は計算中の latent、`σ` はノイズレベル、`D(x, σ)` はモデルが予測した denoised latent です。この probability-flow ODE を、高いノイズレベルから低いノイズレベルへ積分します。

可変ステップ法は CUDA 環境では float64 で積分し、固定ステップ法は標準の k-diffusion サンプラーに合わせて float32 を使います。モデルへの入力は float32 に揃えます。

### PNG に記録する設定

使用した解法と許容誤差を txt2img／Hires.fix 別に PNG の infotext に記録します。

```text
RK txt2img method, RK txt2img log_rtol, RK txt2img log_atol
RK hires method, RK hires log_rtol, RK hires log_atol
RK max_steps
```

以前の画像との互換性のため、txt2img では `RK method` も記録します。Min Sigma と PID 係数は記録されません。同じ条件で再生成する場合は、記録されない詳細設定も揃えてください。

## License / ライセンス

MIT License — Original port: [ComfyUI-RK-Sampler](https://github.com/memmaptensor/ComfyUI-RK-Sampler) © memmaptensor

Adaptive and fixed-step integration is provided by [torchode](https://github.com/martenlienen/torchode) © Marten Lienen.

The fixed-step Runge–Kutta tableaux (`fe_kutta4`, `fe_kutta_38th4`, `fe_ralston3`, `fe_ralston4`, `fe_wray3`, `fe_ssprk3`) are also available in [sd-webui-TDE-Sampler](https://github.com/seti9585/sd-webui-TDE-Sampler), so each has a torchdiffeq-path counterpart there.

