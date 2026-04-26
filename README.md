# E(3)-Equivariant Diffusion Propagator for Argon MD

このリポジトリは、GROMACS の `tpr/trr` 軌道から「現在状態から次状態を予測する」E(3) 同変な拡散プロパゲーターを学習する最小実装です。

モデルは以下の設計です。

- 条件付き DDPM: 現在の位置 `x_t` と速度 `v_t` を条件に、次時刻の変位 `Δx` と速度差分 `Δv` を生成
- E(3) 同変ネットワーク: EGNN 系の座標更新則を使い、並進・回転・反転に対して同変
- Argon 向け前提: 単一元素系を想定し、ノード属性は速度とノルム特徴を中心に構成
- MacBook 向け: `torch_geometric` 非依存、`torch` + `MDAnalysis` のみで動作

## ディレクトリ構成

```text
configs/argon.yaml              学習設定
scripts/train.py                学習エントリポイント
scripts/sample.py               1 ステップ伝播サンプラ
scripts/evaluate.py             1 ステップ予測の評価
scripts/rollout.py              複数ステップの予測軌道出力
src/diffusion_models/data/      tpr/trr ローダ
src/diffusion_models/models/    EGNN と拡散モデル
src/diffusion_models/training/  学習ループ
```

## セットアップ

```bash
./scripts/setup_mac.sh
source venvs/bin/activate
```

Apple Silicon の場合、`torch` は MPS を使います。利用不可なら CPU にフォールバックします。

## Google Colab / CUDA

GitHub にアップロード後、Colab では `notebooks/colab_workflow.ipynb` を開いてください。ノートブック内の `REPO_URL` を自分のリポジトリ URL に変更すると、clone、install、学習、評価、rollout まで実行できます。

Colab 用の設定ファイルは `configs/argon_colab.yaml` です。MD ファイルは以下に置く想定です。

```text
data/argon/nve.tpr
data/argon/nve.trr
```

CLI から直接 CUDA を使う場合:

```bash
python scripts/train.py --config configs/argon_colab.yaml --device cuda
python scripts/evaluate.py --config configs/argon_colab.yaml --checkpoint outputs/argon_diffusion/best.pt --device cuda
python scripts/rollout.py --config configs/argon_colab.yaml --checkpoint outputs/argon_diffusion/best.pt --device cuda
```

`--device auto` では CUDA、MPS、CPU の順で自動選択します。

## 学習

サンプル設定では、近傍軌道として以下を参照します。

```text
../machine_learning_for_molecular_dynamics_propagator/md/sample_argon/nve.tpr
../machine_learning_for_molecular_dynamics_propagator/md/sample_argon/nve.trr
```

必要に応じて `configs/argon.yaml` を書き換えてください。

```bash
python scripts/train.py --config configs/argon.yaml
```

## サンプリング

学習後に 1 ステップの伝播を試す例です。

```bash
python scripts/sample.py \
  --config configs/argon.yaml \
  --checkpoint outputs/argon_diffusion/best.pt \
  --frame-index 0
```

## 時間幅

学習する時間幅は `configs/argon.yaml` の `data.stride` と `data.time_lag` で決まります。

```yaml
data:
  stride: 1
  time_lag: 1
```

- `stride`: TRR から何フレームおきに読むか
- `time_lag`: 読み込んだフレーム列で何フレーム先を予測するか

実際の予測時間幅は、おおよそ `元のTRR保存間隔 × stride × time_lag` です。`stride` を大きくすると読み込むフレーム数が減るので、学習サンプル数も大きく減ります。`time_lag` だけを大きくした場合は、サンプル数は `読み込んだフレーム数 - time_lag` になります。

## 評価

学習済みモデルの 1 ステップ予測誤差を CSV に出力します。

```bash
python scripts/evaluate.py \
  --config configs/argon.yaml \
  --checkpoint outputs/argon_diffusion/best.pt \
  --output-csv outputs/argon_diffusion/evaluation.csv \
  --num-samples 100
```

CSV には `index`, `dt`, `position_rmse`, `velocity_rmse` が出力されます。

## 予測軌道の出力

学習済みモデルで複数ステップの時間発展を予測し、`.npz` と `.xyz` を出力します。

```bash
python scripts/rollout.py \
  --config configs/argon.yaml \
  --checkpoint outputs/argon_diffusion/best.pt \
  --frame-index 0 \
  --steps 20 \
  --output-prefix outputs/argon_diffusion/rollout
```

出力:

```text
outputs/argon_diffusion/rollout.npz
outputs/argon_diffusion/rollout.xyz
```

`.npz` には位置、速度、box、dt が含まれます。`.xyz` は位置のみで、可視化用です。

## モデルの考え方

通常の MD 積分器を直接置き換えるには、座標だけではなく速度も予測し、さらに対称性を壊さない必要があります。本実装では:

- 入力状態: `x_t, v_t`
- 予測対象: `Δx = x_{t+1} - x_t`, `Δv = v_{t+1} - v_t`
- 学習: `Δx, Δv` にノイズを加え、そのノイズを予測
- 推論: 逆拡散で `Δx, Δv` を再構成し、`x_{t+1}, v_{t+1}` を得る

`Δx` と `Δv` はベクトル量なので、E(3) 同変なネットワークを使うと回転した系でも整合した予測ができます。

## 実務上の注意

- 大規模系では近傍グラフ計算が支配的です。`neighbor_k` と `cutoff` を調整してください。
- TRR に速度が含まれない場合、有限差分で近似します。
- 長時間ロールアウトでは誤差蓄積が起きるため、次段階では multi-step loss や corrector を追加するのが自然です。
