# UnTrac on Masked Diffusion Models: 実験設定

## 1. 研究目的

Unlearningベースの訓練データ帰属手法 UnTrac/UnTrac-Inv (Misonuma et al., 2024) を、自己回帰モデル（ARM）から**マスク拡散言語モデル（MDM）**に拡張し、その有効性を検証する対照実験を行う。

---

## 2. 対照実験設計

### 2.1 原論文の実験 5.2（ARM側）

| 項目 | 設定 |
| --- | --- |
| モデル | OPT-125M（Decoder-only, 125Mパラメータ） |
| 訓練 | スクラッチから事前訓練 |
| 損失関数 | 自己回帰交差エントロピー（= NLL） |
| 訓練データ | 8コーパス × 40,000サンプル × 1,024トークン |
| 総トークン数 | 327,680,000（約328Mトークン） |

### 2.2 本実験（MDM側）

| 項目 | 設定 |
| --- | --- |
| モデル | Diff_LLaMA_113M（Bidirectional Transformer, 162Mパラメータ） |
| 訓練 | スクラッチから事前訓練 |
| 損失関数 | ELBO（マスク拡散損失の変分下界） |
| 訓練データ | 8コーパス × 40,000サンプル × 1,024トークン |
| 総トークン数 | 327,680,000（約328Mトークン） |

---

## 3. モデルアーキテクチャ比較

| 項目 | OPT-125M（ARM） | Diff_LLaMA_113M（MDM） |
| --- | --- | --- |
| アーキテクチャ | Decoder-only（因果的） | Bidirectional Transformer |
| 層数 | 12 | 12 |
| ヘッド数 | 12 | 12 |
| 隠れ次元 | 768 | 768 |
| FFN次元 | 3072 | 3072 |
| 正規化 | LayerNorm | RMSNorm |
| 活性化関数 | GeLU | SwiGLU |
| 位置埋め込み | 学習済み | RoPE |
| 非埋め込みパラメータ | 約85M | 113M |
| 総パラメータ | 約125M | 約162M |
| 語彙数 | 50,272 | 32,000 |
| トークナイザ | GPT-2 BPE（OPT同梱） | LLaMA SentencePiece（TinyLlama同梱） |

**備考**: 層数・ヘッド数・隠れ次元・FFN次元は一致。正規化・活性化・位置埋め込みの差異はARMとMDMで一般に使用されるコンポーネントの違いに起因し、本実験の制限事項として位置づける。

---

## 4. 訓練ハイパーパラメータ

| 項目 | 原論文（OPT-125M） | 本実験（MDM） | 一致 |
| --- | --- | --- | --- |
| 訓練ステップ数 | 40,000 | 40,000 | ✓ |
| グローバルバッチサイズ | 8 | 8 | ✓ |
| シーケンス長 | 1,024 | 1,024 | ✓ |
| オプティマイザ | AdamW（weight_decay=0） | Adam（weight_decay=0） | ✓（等価） |
| 学習率 | 5 × 10⁻⁵ | 5 × 10⁻⁵ | ✓ |
| LRスケジュール | 定数 | 定数 | ✓ |
| β₁ / β₂ | 0.9 / 0.999 | 0.9 / 0.999 | ✓ |
| 勾配クリッピング | 1.0 | 1.0 | ✓ |
| ウォームアップ | 0 | 0 | ✓ |
| Weight decay | 0.0 | 0.0 | ✓ |

**出典**: 原論文のGitHubリポジトリ `misonuma/untrac` の `arguments.py` および `pretrain_opt.sh` から確認。

---

## 5. 訓練データ

### 5.1 コーパス構成（Equal設定）

| コーパス | サンプル数 | ソース | 原論文との対応 |
| --- | --- | --- | --- |
| BookCorpus | 40,000 | bookcorpus/bookcorpus | BookCorpus ✓ |
| StackExchange | 40,000 | The Pile（pile-uncopyrighted） | CC-Stories（代替）△ |
| CCNewsV2 | 40,000 | vblagoje/cc_news | CCNewsV2 ✓ |
| Gutenberg | 40,000 | deepmind/pg19 | PJ Gutenberg ✓ |
| HackerNews | 40,000 | The Pile（pile-uncopyrighted） | HackerNews ✓ |
| OpenWebText | 40,000 | Skylion007/openwebtext | OpenWebText2（代替）△ |
| Pile-CC | 40,000 | The Pile（pile-uncopyrighted） | Pile-CC ✓ |
| Wikipedia | 40,000 | The Pile（pile-uncopyrighted） | Wikipedia ✓ |

- **合計**: 320,000サンプル × 1,024トークン = 327,680,000トークン
- **データ形式**: TinyLlamaトークナイザで符号化、PackedDataset形式（セパレータなし連結）
- **代替コーパス**: CC-Stories（原版消失）→ StackExchange、OpenWebText2（著作権問題）→ OpenWebText

### 5.2 データ前処理

原論文に準拠: テキストをトークナイズ後、エンドツーエンドで連結し、1,024トークンのブロックに分割。端数は破棄。データ前処理スクリプトは `scripts/prepare_data.py` を参照。

---

## 6. 評価データ（テストセット）

原論文の `preprocess_test.py` に準拠した前処理を実施。

| データセット | サブセット数 | 各サブセットのサンプル数 | 合計サンプル数 | フィルタ条件 |
| --- | --- | --- | --- | --- |
| ToxiGen | 13グループ | 256 | 3,328 | prompt_label==1、語数8〜24 |
| WinoBias | 4カテゴリ | 256 | 1,024 | type1_pro/anti × 男女分類 |
| TruthfulQA | 9カテゴリ | 256 | 約1,852 | 不正解回答、カテゴリ内128件以上 |

---

## 7. 実験パイプライン

### 7.1 Ground Truth: Leave-One-Out帰属

各テストサブセット s に対する訓練コーパス Zk の影響度:

$$
\text{Influence}_{\text{LOO}}(s, Z_k) = \mathcal{L}_{\text{ELBO}}(s, \theta_{\setminus k}) - \mathcal{L}_{\text{ELBO}}(s, \theta_{\text{full}})
$$

- θ_full: 全8コーパスで訓練したモデル
- θ_\k: コーパス Zk を除外して訓練したモデル
- 正の値 = そのコーパスがテストデータに有益（除外でELBO悪化）

`scripts/train/run_loo.sh` を参照。

### 7.2 UnTrac

訓練済みモデルに対し、各コーパスを勾配上昇（unlearning）で忘却させ、テスト損失の変化を測定:

$$
I_{\text{UnTrac}}(s, Z_k) = \mathcal{L}_{\text{ELBO}}(s, \theta_T^{(k)}) - \mathcal{L}_{\text{ELBO}}(s, \theta_0)
$$

- θ₀: 訓練済みモデル
- θ_T^(k): コーパス Zk で勾配上昇した後のモデル
- Unlearningハイパーパラメータ: Adam, lr = 5 × 10⁻⁵, batch = 1, 1 epoch, 勾配クリッピングなし

実装は `src/mdm_unlearning/evaluate/untrac_mdm.py` 等。

### 7.3 UnTrac-Inv

テストデータをunlearnし、各訓練コーパスの損失変化を測定（UnTracの逆方向）:

$$
I_{\text{UnTrac-Inv}}(s, Z_k) = \mathcal{L}_{\text{ELBO}}(Z_k, \theta_T'^{(s)}) - \mathcal{L}_{\text{ELBO}}(Z_k, \theta_0)
$$

- テストデータで勾配上昇: effective batch = 256, max_steps = 50

### 7.4 MDM固有の考慮事項

ARMとMDMの損失関数の違いにより、UnTracの適用に以下の差異がある:

| 側面 | ARM | MDM |
| --- | --- | --- |
| 損失関数 L | NLL（決定的） | ELBO（確率的、t とマスク m に依存） |
| ELBO と NLL の関係 | 一致 | ELBO ≤ NLL（一般に不一致） |
| 評価の再現性 | 決定的 | モンテカルロ推定（分散あり） |
| 影響度推定の精度 | 高い | MCサンプル数 N に依存（分散は O(1/N) で減少） |

本実験では N = 128 のMCサンプルでELBOを推定する。

---

## 8. 評価指標

原論文に準拠し、各テストサブセットについて:

$$
r = \text{Pearson}(\mathbf{I}_{\text{method}}, \mathbf{I}_{\text{LOO}})
$$

- I ∈ ℝ⁸: 8コーパスの影響度ベクトル
- method ∈ {UnTrac, UnTrac-Inv}
- テストサブセットごと（ToxiGen 13 + WinoBias 4 + TruthfulQA 9 = 26サブセット）にPearson相関を計算

---

## 9. 制限事項

1. **アーキテクチャの差異**: OPT-125MとDiff_LLaMA_113Mは層数・次元は一致するが、正規化（LayerNorm vs RMSNorm）、活性化（GeLU vs SwiGLU）、位置符号化（学習済み vs RoPE）が異なる。本実験は訓練パラダイム（ARM vs MDM）の違いを主眼とするが、これらアーキテクチャ差異は交絡因子となりうる。
2. **コーパスの代替**: 原論文のCC-StoriesとOpenWebText2が入手不可のため、StackExchangeとOpenWebTextで代替。コーパスの性質が異なるため、原論文のARM結果との直接比較には注意が必要。
3. **トークナイザの差異**: GPT-2 BPEトークナイザ（語彙50,272、OPT同梱）とLLaMA SentencePieceトークナイザ（語彙32,000、TinyLlama同梱）の違いにより、同一テキストのトークン分割が異なる。
4. **ELBOの確率性**: MDMの損失関数はマスクサンプリングに依存する確率的なELBOであり、ARMの決定的なNLLと異なる。影響度推定にモンテカルロ分散が加わるため、UnTracの有効性がARMより低下する可能性がある。

---

## 10. 計算環境

| 項目 | 値 |
| --- | --- |
| GPU | NVIDIA RTX PRO 6000 Blackwell Max-Q（97GB） |
| GPU数（Full訓練） | 2 |
| GPU数（LOO訓練） | 1 / run |
| PyTorch | 2.11.0+cu128 |
| Flash-Attention | 2.8.3 |
| Lightning Fabric | 2.6.1 |
| Python | 3.11.13 |

---

## 参考文献

- Misonuma et al. (2024). "Unlearning Traces the Influential Training Data of Language Models." arXiv:2401.15241
- Nie et al. (2024). "Scaling up Masked Diffusion Models on Text." arXiv:2410.18514
