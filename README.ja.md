# mdm-unlearning

> English README is at [`README.md`](README.md).

**マスク拡散言語モデル（MDM）**における**訓練データ帰属（attribution）**と**選択的逆学習（selective unlearning）**の実証研究です。自己回帰モデル（ARM）および Encoder-Decoder MDM（E2D2 系）と並べて比較しています。

このリポジトリには、テクニカルレポート [`docs/untrac_results.ja.md`](docs/untrac_results.ja.md) で報告した実験のコード、生 JSON 結果、図がすべて含まれています。

## 核心的発見

> UnTrac の NLL ベース帰属スコアは MDM でも原論文の ARM と同等に機能するが、トークンレベルの復元分析を行うと、その背後で起きているのは「選択的忘却」ではなく**「差分的崩壊の検出」**であることが分かる。実際に選択的逆学習が成功するか否かは、**アーキテクチャではなく忘却対象の粒度**によって決まり、事実レベル（〜100 件）では ARM/MDM ともに成功し、コーパスレベル（〜40,000 件）では両者とも失敗する。

詳細は [`docs/untrac_results.ja.md`](docs/untrac_results.ja.md)、対照実験設計は [`docs/experiment_setup.ja.md`](docs/experiment_setup.ja.md) を参照してください。

## リポジトリ構成

- **`src/mdm_unlearning/`** — インストール可能な Python パッケージ（`src/` レイアウト、PEP 621）
  - `models/` — MDM (`TransEncoder`)、ARM (`GPT`)、E2D2 系 Encoder-Decoder MDM (`TransEncoderDecoder`)、`Config`、`RMSNorm`/`RoPE` など
  - `data/` — `PackedDataset` / `CombinedDataset` と TinyLlama トークナイザのラッパー
  - `train/` — MDM/ARM/E2D2 のスクラッチ訓練ループ（Lightning Fabric、アーキテクチャごとに 1 ファイル）
  - `evaluate/` — UnTrac NLL 帰属とマスクトークン復元分析（3 アーキテクチャすべて対応）
  - `analysis/` — コーパス固有トリグラムの抽出、知識ニューロン局在性、ニューロン抑制、事実レベル EU
  - `utils/` — `SpeedMonitor`、fused cross-entropy、チェックポイント補助
- **`scripts/`** — エンドツーエンド実行用のシェルスクリプト（`train/`, `untrac/`, `reconstruction/`）と `prepare_data.py`
- **`results/`** — 実験ファミリーごとの生 JSON 結果（`ground_truth`, `attribution/{compare,fisher_meta,eu,e2d2,ar,mdm_kl}`, `reconstruction/{mdm_kl,mdm_fm,mdm_eu,e2d2,ar}`, `analysis/`）
- **`docs/`** — テクニカルレポート（英語＋日本語）、実験設定（英語＋日本語）、図（PNG）
- **`configs/`** — モデル/訓練設定の YAML（プレースホルダ）
- **`tests/`** — 単体テスト用ディレクトリ（プレースホルダ）

## アーキテクチャと手法

| アーキテクチャ | クラス | 損失 | 場所 |
| --- | --- | --- | --- |
| MDM（単一双方向 Transformer） | `TransEncoder` | マスク拡散 ELBO | `models/diffmodel.py` |
| ARM（causal LLaMA 系） | `GPT` | NLL | `models/arm.py` |
| Encoder-Decoder MDM（E2D2 系） | `TransEncoderDecoder` | block-causal ELBO | `models/enc_dec_diffmodel.py` |

| 逆学習手法 | 安定？ | 崩壊耐性？ | 実装場所 |
| --- | --- | --- | --- |
| Gradient Ascent (GA) | × | × | `evaluate/untrac_*.py` |
| KL 制約付き GA (KL) | ○ | ○（正則化） | `evaluate/untrac_*.py` |
| NPO 風適応重み | ○ | ○（step 1 で飽和） | `evaluate/untrac_*.py` |
| VDU L2 アンカー | γ 次第 | γ=0.01 では弱い | `evaluate/untrac_*.py` |
| Fisher-Meta（Selective Amnesia + SalUn + Meta-Unlearning） | ○ | ○ | `evaluate/untrac_mdm.py`（`--unlearn_method fisher_meta`） |
| 排他的逆学習 (EU) | ○ | `log V` で上界 | `evaluate/untrac_*.py`（`--unlearn_method eu`） |

## インストール

**Python 3.11+**, **CUDA 12**, **flash-attention 2** が必要です。

```sh
git clone https://github.com/Udon0729/mdm-unlearning.git
cd mdm-unlearning

# 推奨: uv (https://docs.astral.sh/uv/)
uv venv -p 3.11
source .venv/bin/activate
uv pip install -e ".[viz,dev]"

# flash-attn は CUDA toolchain に合わせて別途インストール
uv pip install flash-attn --no-build-isolation
```

uv を使わない場合は `pip install -e .` でも構いません。実行時依存は [`pyproject.toml`](pyproject.toml) に固定されています。`[viz]` extra で matplotlib/seaborn、`[dev]` extra で ruff/mypy/pytest/pre-commit が入ります。

## データ準備

訓練コーパスは公開ウェブテキストの 8 種類（BookCorpus, StackExchange, CCNewsV2, Gutenberg, HackerNews, OpenWebText, Pile-CC, Wikipedia）。すべて TinyLlama トークナイザで符号化し、1024 トークンのブロックにパックします。各コーパスの取得元は [`docs/experiment_setup.ja.md`](docs/experiment_setup.ja.md) §5 を参照してください。

```sh
python scripts/prepare_data.py --output_dir data/untrac
```

期待される配置: `data/untrac/train_<corpus>_*.bin`。テストセット（ToxiGen, WinoBias, TruthfulQA）は評価スクリプトが自動でダウンロードします。

## 実験の再現

すべてのコマンドはリポジトリルートから実行します。**チェックポイントはリポジトリに含まれていません**（[`.gitignore`](.gitignore) で除外）。自分で再訓練するか、`MDM_CKPT` / `AR_CKPT` / `E2D2_CKPT` 環境変数で既存チェックポイントを指してください。

### 1. 事前訓練

```sh
# MDM (113M non-embedding params, 40k steps; GPU 1 枚の例)
python -m mdm_unlearning.train.train_mdm \
    --model 113 --max_steps 40000 --data_dir data/untrac \
    --save_interval 40000 --num_devices 1 \
    --batch_size 8 --micro_batch_size 4 --grad_clip 1.0 --seq_len 1024

# ARM
python -m mdm_unlearning.train.train_ar --model 113 --max_steps 40000 ...

# E2D2 Encoder-Decoder MDM
python -m mdm_unlearning.train.train_e2d2 --model 113 --max_steps 40000 ...
```

Leave-One-Out 訓練（除外コーパスごとに 8 回）はラッパーを使用：

```sh
bash scripts/train/run_loo.sh
```

### 2. UnTrac NLL 帰属

```sh
# 単一コーパス、KL 制約付き GA
MDM_CKPT=workdir/.../iter-040000-ckpt.pth \
python -m mdm_unlearning.evaluate.untrac_mdm \
    --mode untrac --model 113 \
    --ckpt_path "$MDM_CKPT" --data_dir data/untrac \
    --unlearn_method kl --untrac_corpus bookcorpus \
    --output results/attribution/compare/kl_bookcorpus.json

# 全コーパス・全手法のスイープ
bash scripts/untrac/run_compare.sh         # GA / KL / NPO / VDU
bash scripts/untrac/run_fisher_meta.sh     # Fisher-Meta
bash scripts/untrac/run_eu.sh              # 排他的逆学習
bash scripts/untrac/run_e2d2.sh            # E2D2 KL
bash scripts/untrac/run_ar.sh              # ARM KL/EU
```

### 3. マスクトークン復元分析

```sh
python -m mdm_unlearning.evaluate.reconstruction_mdm \
    --model 113 --ckpt_path "$MDM_CKPT" --data_dir data/untrac \
    --unlearn_corpus bookcorpus --unlearn_steps 10000 \
    --unlearn_method kl --num_samples 100 --mask_ratio 0.15 \
    --output results/reconstruction/mdm_kl/recon_bookcorpus.json

# スイープ
bash scripts/reconstruction/run_mdm_kl.sh
bash scripts/reconstruction/run_mdm_fm.sh
bash scripts/reconstruction/run_mdm_eu.sh
bash scripts/reconstruction/run_e2d2.sh
bash scripts/reconstruction/run_ar.sh
```

### 4. 知識局在性分析と事実レベル逆学習

```sh
# Phase 0: コーパス固有トリグラムを抽出し、MDM が中央トークンを予測できるか検証
python -m mdm_unlearning.analysis.extract_corpus_trigrams \
    --ckpt_path "$MDM_CKPT" \
    --output_trigrams results/analysis/corpus_specific_trigrams.json \
    --output_prediction results/analysis/trigram_prediction_results.json

# Phase 1: MLP ニューロンごとの知識重要度
python -m mdm_unlearning.analysis.knowledge_localization \
    --ckpt_path "$MDM_CKPT" \
    --output results/analysis/knowledge_localization.json

# Phase 2a: 上位 k ニューロンをゼロ化
python -m mdm_unlearning.analysis.neuron_suppression \
    --ckpt_path "$MDM_CKPT" \
    --output results/analysis/neuron_suppression_results.json

# Phase 2b: 事実レベルへの勾配ベース EU 適用（粒度仮説の検証）
python -m mdm_unlearning.analysis.fact_level_eu \
    --ckpt_path "$MDM_CKPT" \
    --output results/analysis/fact_level_eu_results.json
```

## 主な実験結果

| 実験 | 結果 |
| --- | --- |
| SMDM NLL 帰属（KL / Fisher-Meta / EU） | r ≈ 0.48-0.49（ARM 原論文と同等） |
| SMDM トークン復元（KL, 10K） | 選択性 -0.001（一律崩壊） |
| SMDM トークン復元（EU, 10K） | 選択性 -0.075（コーパスレベルでは最良、残存 18.3%） |
| ARM トークン復元（EU, 10K） | 選択性 -0.073（≈ MDM EU） |
| E2D2 トークン復元（KL, 10K） | 選択性 +0.001（cross-attention だけでは不十分） |
| 事実レベル 勾配ベース EU（200 steps） | **選択性 -0.663**（gutenberg: ターゲット -95%、他 +1.3%） |
| 事実レベル ニューロン抑制（k=200） | 選択性 -0.135 |

すべての数値・アブレーション・考察は [`docs/untrac_results.ja.md`](docs/untrac_results.ja.md) を参照してください。

## 引用

このコードや分析を利用される場合は、以下の形式で引用してください。

```bibtex
@misc{munaoka2026mdmunlearning,
  author       = {Munaoka, Kazuki},
  title        = {{mdm-unlearning}: empirical study of training data attribution and selective unlearning in masked diffusion language models},
  year         = {2026},
  howpublished = {\url{https://github.com/Udon0729/mdm-unlearning}}
}
```

機械可読な [`CITATION.cff`](CITATION.cff) も用意しています。

## ライセンスと派生関係

本プロジェクトは [MIT License](LICENSE) で公開しています。複数の研究コードベース（SMDM, E2D2, UnTrac, 排他的逆学習, NPO 等）の上に構築されており、出典は [`NOTICE.md`](NOTICE.md) を参照してください。

## 連絡先

著者: **宗岡 康太**（静岡大学） — `kmunaoka@gmail.com`
Issues: <https://github.com/Udon0729/mdm-unlearning/issues>
