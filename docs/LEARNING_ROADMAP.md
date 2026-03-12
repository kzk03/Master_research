# 📚 初学者向け学習ロードマップ

このドキュメントは、本研究プロジェクトのコードを理解するための学習パスを示します。

---

## 🎯 プロジェクト概要

**目的**: OSS開発者のレビュー継続行動を予測し、最適な推薦を行う

**手法**:

- **Phase 1 (IRL)**: 逆強化学習で開発者の報酬関数を推定し、継続確率を予測
- **Phase 2 (RL)**: 強化学習でタスク推薦戦略を最適化

---

## 📖 Phase 0: 全体像の把握（30分）

### ✅ やること

1. [research_design_roadmap.md](research_design_roadmap.md)を読む
2. 研究の2フェーズ構成を理解
3. RQ1-RQ5の研究課題を把握

### 🎓 理解すべきポイント

- なぜIRLとRLの2段階が必要なのか
- 「継続予測」と「推薦最適化」の違い
- データソース: OpenStack 5年分のレビューデータ

---

## 📊 Phase 1: データの理解（1-2時間）

### ✅ やること

#### 1.1 データ構造の確認

```bash
# CSVファイルの中身を確認
head -20 data/review_requests_openstack_multi_5y_detail.csv

# カラムを確認
uv run python -c "
import pandas as pd
df = pd.read_csv('data/review_requests_openstack_multi_5y_detail.csv')
print(df.columns.tolist())
print(df.head())
"
```

#### 1.2 データ生成プロセスの理解

**読むファイル**: `scripts/pipeline/build_dataset.py`

```bash
# コードを開いて読む
code scripts/pipeline/build_dataset.py
```

### 🎓 理解すべきポイント

- **label=1**: 14日以内に応答（承諾）
- **label=0**: 14日以内に応答なし（拒否/無視）
- レビュー依頼のタイムスタンプとレスポンスタイムの関係
- プロジェクトごとのデータ分離

### 📝 確認問題

- [ ] ラベルの定義を説明できる
- [ ] どのプロジェクトのデータが含まれているか
- [ ] データの時間範囲を理解している

---

## 🔍 Phase 2: 特徴量エンジニアリング（2-3時間）

### ✅ やること

#### 2.1 特徴量の定義を理解

**最重要ファイル**: `src/review_predictor/IRL/features/common_features.py`

```bash
# コードを開く
code src/review_predictor/IRL/features/common_features.py

# 関数のドキュメントを確認
uv run python -c "
from review_predictor.IRL.features.common_features import extract_common_features
help(extract_common_features)
"
```

#### 2.2 14次元の特徴量リスト

| #   | 特徴量名                    | 説明                 |
| --- | --------------------------- | -------------------- |
| 1   | days_since_last_activity    | 最後の活動からの日数 |
| 2   | recent_review_load          | 直近のレビュー負荷   |
| 3   | acceptance_rate             | 承諾率               |
| 4   | avg_response_time           | 平均応答時間         |
| 5   | total_requests              | 総依頼数             |
| 6   | accepted_requests           | 承諾数               |
| 7   | activity_streak             | 連続活動日数         |
| 8   | last_7d_requests            | 直近7日の依頼数      |
| 9   | last_30d_requests           | 直近30日の依頼数     |
| 10  | response_rate               | 応答率               |
| 11  | avg_reviews_per_week        | 週平均レビュー数     |
| 12  | active_days                 | 活動日数             |
| 13  | max_consecutive_acceptances | 最大連続承諾数       |
| 14  | recent_acceptance_trend     | 直近の承諾トレンド   |

### 🎓 理解すべきポイント

- 時系列データ（活動履歴）から特徴量を計算
- ウィンドウベースの集約（7日、30日）
- なぜこれらの特徴が「継続」を予測するのか

### 📝 ハンズオン

```python
# Jupyter notebookまたはPythonインタラクティブシェルで試す
uv run python

>>> import pandas as pd
>>> from review_predictor.IRL.features.common_features import extract_common_features

# サンプルデータで試す
>>> activity_history = [...]  # データを用意
>>> features = extract_common_features(activity_history)
>>> print(features)
```

### 📝 確認問題

- [ ] 14次元の特徴量を説明できる
- [ ] なぜ時系列データが必要か理解している
- [ ] データリーク（未来のデータの混入）を防ぐ方法を理解している

---

## 🧠 Phase 3: IRLモデルの理解（3-4時間）

### ✅ やること

#### 3.1 ベースライン: Random Forest

**読むファイル**: `src/review_predictor/IRL/model/rf_predictor.py`

```bash
code src/review_predictor/IRL/model/rf_predictor.py
```

**ポイント**:

- シンプルな教師あり学習
- 特徴量 → RandomForestClassifier → 予測
- scikit-learnの標準的な使い方

#### 3.2 本命: IRL (Inverse Reinforcement Learning)

**最重要ファイル**: `src/review_predictor/IRL/model/irl_predictor.py`

```bash
code src/review_predictor/IRL/model/irl_predictor.py
```

**アーキテクチャ**:

```
入力: 時系列活動履歴 (可変長)
  ↓
LSTM (128次元)
  ↓
全結合層 (64次元)
  ↓
出力: 継続確率 (0-1)
```

**重要な実装ポイント**:

1. **LSTMの使用**

   ```python
   self.lstm = nn.LSTM(
       input_size=14,  # 14次元特徴量
       hidden_size=128,
       num_layers=2,
       batch_first=True,
       dropout=0.3
   )
   ```

   - なぜ? → 時系列パターンを学習

2. **Focal Loss**

   ```python
   class FocalLoss(nn.Module):
       def __init__(self, alpha=0.25, gamma=2.0):
   ```

   - なぜ? → クラス不均衡対策（継続者は少数）

3. **pack_padded_sequence**
   - 可変長シーケンスの効率的な処理

### 🎓 理解すべきポイント

- IRLの基本: 専門家の行動から報酬関数を逆推定
- LSTMによる時系列モデリング
- クラス不均衡問題と対処法
- なぜRFよりIRLが優れているか

### 📝 確認問題

- [ ] LSTMの入力・出力の形状を説明できる
- [ ] Focal Lossの役割を理解している
- [ ] IRLとRFの違いを説明できる

---

## 🚀 Phase 4: 訓練パイプラインの理解（2-3時間）

### ✅ やること

#### 4.1 訓練スクリプトを読む

**最重要ファイル**: `scripts/train/train_model.py`

```bash
code scripts/train/train_model.py
```

**処理の流れ**:

```
1. データ読み込み (load_review_requests)
   ↓
2. 軌跡抽出 (extract_review_acceptance_trajectories)
   訓練期間内の各開発者の活動を月次で集約
   ↓
3. モデル訓練
   - LSTM + Focal Loss
   - Adam optimizer
   - 学習率スケジューリング
   ↓
4. 最適閾値決定 (find_optimal_threshold)
   F1スコア最大化
   ↓
5. モデル保存 (irl_model.pt)
   ↓
6. 評価軌跡抽出 (extract_evaluation_trajectories)
   評価期間のスナップショット特徴量
   ↓
7. 予測実行
   ↓
8. メトリクス計算・保存
   - AUC-ROC, AUC-PR
   - F1, Precision, Recall
```

#### 4.2 重要な関数を理解

**継続判定ロジック**:

```python
def extract_review_acceptance_trajectories(...):
    # 評価期間内の行動を判定
    # - 依頼を受けていない → 除外
    # - 少なくとも1つ承諾 → 継続 (True)
    # - 全て拒否 → 離脱 (False)
```

**時系列整合性**:

- 訓練期間内のデータのみ使用（データリーク防止）
- 各月時点での「過去のデータのみ」で特徴量計算

### 🎓 理解すべきポイント

- 訓練期間と評価期間の分離
- 継続判定の定義（重要！）
- クロスバリデーションの考え方
- メトリクスの選択理由

### 📝 実行してみる

```bash
# ヘルプを確認
uv run python scripts/train/train_model.py --help

# 小規模テスト実行（データがある場合）
uv run python scripts/train/train_model.py \
  --reviews data/review_requests_openstack_multi_5y_detail.csv \
  --train-start 2020-01-01 \
  --train-end 2020-12-31 \
  --epochs 5 \
  --output outputs/IRL/learning_test

# 出力を確認
ls -lh outputs/IRL/learning_test/
cat outputs/IRL/learning_test/metrics.json
```

### 📝 確認問題

- [ ] データリークを防ぐ方法を理解している
- [ ] 継続判定ロジックを説明できる
- [ ] 訓練・評価の流れを図示できる

---

## 📈 Phase 5: 結果分析（1-2時間）

### ✅ やること

#### 5.1 特徴量重要度の可視化

```bash
code scripts/analyze/plot_feature_importance.py

# 実行（訓練後）
uv run python scripts/analyze/plot_feature_importance.py \
  --output outputs/IRL/learning_test
```

#### 5.2 ヒートマップ分析

```bash
code scripts/analyze/plot_heatmaps.py

# クロス評価結果を可視化
uv run python scripts/analyze/plot_heatmaps.py \
  --input outputs/IRL/review_continuation_cross_eval_nova
```

#### 5.3 予測比較

```bash
code scripts/analyze/plot_prediction_comparison.py
```

### 🎓 理解すべきポイント

- どの特徴量が予測に効いているか
- モデルの時系列汎化性能
- プロジェクト間の転移可能性

---

## 🎮 Phase 6: RLシステムの理解（進行中）

### ✅ やること

Phase 2（修論）の内容です。現在開発中。

#### 6.1 環境の理解

**読むファイル**: `src/review_predictor/RL/env/review_env.py`

**MDPの定義**:

- **状態 (State)**: 開発者の活動状態 + プロジェクト状態
- **行動 (Action)**: タスク推薦 or 推薦しない
- **報酬 (Reward)**: IRLで推定した報酬関数を審判として利用

#### 6.2 エージェントの理解

**読むファイル**: `src/review_predictor/RL/agent/agent.py`

強化学習アルゴリズム（候補）:

- PPO (Proximal Policy Optimization)
- DQN (Deep Q-Network)
- A3C (Asynchronous Actor-Critic)

### 🎓 理解すべきポイント

- IRLで学習した報酬関数をRLで活用
- シミュレーション環境の設計
- オンライン学習 vs オフライン学習

---

## 🗺️ 推奨学習パス

### ⚡ 最短パス（半日）

Phase 0 → Phase 1 → Phase 2 → Phase 3（IRL部分のみ）

### 📘 標準パス（2-3日）

Phase 0-5 + 小規模実験の実行

### 🎓 完全理解パス（1週間）

全Phase + 実験の再現 + RLシステムの設計理解

---

## 💡 学習のコツ

### 1. トップダウンアプローチ

```
全体像 → データフロー → モデル → 実装詳細
```

いきなりコードを読まず、まずドキュメントで文脈を理解

### 2. 手を動かす

```bash
# 関数のヘルプを見る
uv run python -c "from module import function; help(function)"

# インタラクティブに試す
uv run python
>>> import review_predictor
>>> # 試す
```

### 3. 図を描く

データの流れ、モデルの構造を紙に書いて整理

### 4. 質問リストを作成

わからない部分をメモ → 後で調べる or 質問する

---

## 📚 参考資料

### 論文・教科書

- **IRL**: Abbeel & Ng (2004) "Apprenticeship Learning via Inverse Reinforcement Learning"
- **LSTM**: Hochreiter & Schmidhuber (1997) "Long Short-Term Memory"
- **Focal Loss**: Lin et al. (2017) "Focal Loss for Dense Object Detection"

### オンライン資料

- PyTorch公式チュートリアル: https://pytorch.org/tutorials/
- scikit-learn User Guide: https://scikit-learn.org/stable/user_guide.html
- Stable-Baselines3 Docs: https://stable-baselines3.readthedocs.io/

---

## ✅ チェックリスト

### Phase 1-2 完了

- [ ] データの構造を理解した
- [ ] 14次元の特徴量を説明できる
- [ ] ラベルの定義を理解した

### Phase 3-4 完了

- [ ] IRLとRFの違いを説明できる
- [ ] LSTMの役割を理解した
- [ ] 訓練パイプラインを理解した
- [ ] スクリプトを実行できた

### Phase 5-6 完了

- [ ] 分析スクリプトを使える
- [ ] RLシステムの設計を理解した
- [ ] 研究全体の流れを説明できる

---

## 🆘 困ったときは

### デバッグのコツ

```bash
# エラーの詳細を見る
uv run python scripts/train/train_model.py --help

# インポートテスト
uv run python -c "from review_predictor.IRL.model.irl_predictor import RetentionIRLSystem"

# 環境確認
uv pip list
```

### よくある質問

**Q: データファイルがない**
A: `scripts/pipeline/build_dataset.py`で生成するか、サンプルデータを用意

**Q: メモリエラー**
A: バッチサイズを小さくする、エポック数を減らす

**Q: 精度が上がらない**
A: データ量、特徴量、ハイパーパラメータを見直す

---

**頑張ってください！** 🚀

質問があれば遠慮なく聞いてください。
