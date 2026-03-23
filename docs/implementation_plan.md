# 実装プラン（改訂版）

## 全体フロー

```
Phase 0: データ収集（Gerrit API から再取得・再現可能な形で）
    ↓
Phase 1: 特徴量拡張（ミクロ14次元 → ミクロ14 + マクロ3 + パス4 = 21次元）
    ↓
Phase 2: IRL 再学習（拡張特徴量で再訓練）
    ↓
Phase 3: RL 実装（拡張 IRL を報酬関数として使用）
    ↓
Phase 4: 評価（行動模倣 / シミュレーション）
```

> **単一プロジェクト (Nova) のみで全フェーズを通す。**
> Cross-project 対応は後フェーズで追加。

---

## Phase 0: データ収集（再現可能な形で）

### 現状の問題

- `build_dataset.py` は存在するが、実際に使ったパラメータ（期間・プロジェクト・URL）が未記録
- `common_features.py` が `df['email']` / `df['timestamp']` を参照しているが、出力 CSV は `developer_email` / `request_time` → カラム名の不整合

### やること

**0-1. `build_dataset.py` の実行と結果記録**

```bash
uv run python scripts/pipeline/build_dataset.py \
    --gerrit-url https://review.opendev.org \
    --project openstack/nova \
    --start-date 2013-01-01 \
    --end-date 2017-12-31 \
    --output data/nova_raw.csv
```

- 期間・プロジェクト・URL を `data/collection_config.json` に記録して再現可能にする
- 実行ログを `data/collection_log.txt` に保存

**0-2. カラム名の統一（build_dataset.py の出力段階で修正）**

コード全体で統一するカラム名:

| 現在の出力 | 統一後 | 意味 |
|-----------|-------|------|
| `developer_email` | `email` | レビュアーのメール |
| `request_time` | `timestamp` | レビュー依頼発生時刻 |

→ `build_dataset.py` の最終出力前にリネームを追加する（1箇所だけの修正）

**0-3. データ分割の記録**

```python
# data/split_config.json に記録
{
  "train":    {"start": "2013-01-01", "end": "2016-01-01"},
  "eval":     {"start": "2016-01-01", "end": "2017-01-01"},
  "project":  "openstack/nova"
}
```

---

## Phase 1: 特徴量拡張

### 現状

```
ミクロ（14次元）= 状態10次元 + 行動4次元
  状態: experience_days, total_changes, total_reviews, recent_activity_frequency,
        avg_activity_gap, activity_trend, collaboration_score, code_quality_score,
        recent_acceptance_rate, review_load
  行動: avg_action_intensity, avg_collaboration, avg_response_time, avg_review_size
```

### 追加する特徴量

**マクロ（3次元）** — プロジェクト全体の状況

| 特徴量名 | 説明 | 計算方法 |
|---------|------|---------|
| `task_queue_length` | 直近30日のオープンレビュー数（正規化） | `len(recent) / 100.0` （上限1.0） |
| `release_cycle_phase` | リリースサイクル内の位相（0.0〜1.0） | Gerrit のタグ情報 or 月次周期で近似 |
| `project_acceptance_rate` | プロジェクト全体の直近30日承諾率 | `recent['label'].mean()` |

**パス類似度（4次元）** — このレビュアーが過去に触ったファイルとの類似度

データには review 単位でパス類似度が存在するので、開発者単位に集約:

| 特徴量名 | 元カラム | 集約方法 |
|---------|---------|---------|
| `avg_path_cosine_files` | `path_cosine_files_global` | 直近30日の平均 |
| `avg_path_jaccard_files` | `path_jaccard_files_global` | 直近30日の平均 |
| `avg_path_dice_files` | `path_dice_files_global` | 直近30日の平均 |
| `avg_path_overlap_files` | `path_overlap_coeff_files_global` | 直近30日の平均 |

### 変更ファイル

```
src/review_predictor/IRL/features/common_features.py
  - FEATURE_NAMES に 7次元追加（マクロ3 + パス4）
  - extract_common_features() にマクロ・パス特徴量の計算を追加
  - 14次元 → 21次元

src/review_predictor/RL/state/state_builder.py
  - MICRO_FEATURES が自動的に21次元になる（FEATURE_NAMES を参照しているため変更不要）
  - MACRO_FEATURES は空のままでよい（マクロを MICRO に統合したため）
```

### 拡張後の特徴量インデックス

```
[0]  experience_days
[1]  total_changes
[2]  total_reviews
[3]  recent_activity_frequency
[4]  avg_activity_gap
[5]  activity_trend
[6]  collaboration_score
[7]  code_quality_score
[8]  recent_acceptance_rate
[9]  review_load
[10] avg_action_intensity       ← 行動特徴量
[11] avg_collaboration
[12] avg_response_time
[13] avg_review_size
[14] task_queue_length          ← マクロ（新規）
[15] release_cycle_phase
[16] project_acceptance_rate
[17] avg_path_cosine_files      ← パス類似度（新規）
[18] avg_path_jaccard_files
[19] avg_path_dice_files
[20] avg_path_overlap_files
```

---

## Phase 2: IRL 再学習

### やること

IRL のネットワークは `state_dim` / `action_dim` で初期化するので、次元変更に合わせて再学習する。

**変更ファイル**

```
src/review_predictor/IRL/model/irl_predictor.py
  - RetentionIRLNetwork: state_dim=17 (10 + マクロ3 + パス4), action_dim=4
  - state_to_tensor(): 21次元の特徴量ベクトルを state(17) + action(4) に分割
```

**再学習コマンド（Nova 単一プロジェクト）**

```bash
uv run python scripts/train/train_model.py \
    --data data/nova_raw.csv \
    --project openstack/nova \
    --output outputs/IRL/nova_extended/
```

**検証（IRL 単体の精度比較）**

| モデル | 特徴量 | AUC-ROC |
|--------|--------|---------|
| IRL（卒論、14次元） | ミクロのみ | 0.72（既存） |
| IRL（修論、21次元） | ミクロ + マクロ + パス | ??（再学習後） |

→ パス特徴量・マクロ追加による精度改善を RQ として実証

---

## Phase 3: RL 実装

### 3-1. `reward.py` の実装

**変更内容**: `_load_model()` と `compute()` の TODO を埋める

```python
# _load_model() の実装
from IRL.model.irl_predictor import RetentionIRLNetwork

network = RetentionIRLNetwork(state_dim=17, action_dim=4)
network.load_state_dict(torch.load(self.model_path, map_location=self.device))
network.eval()
self._model = network

# compute() の実装
# state_vector は 21 次元
# [0:17] = state 入力, [17:21] = action 入力 (4次元)
state_t = torch.FloatTensor(state_vector[:17]).unsqueeze(0).unsqueeze(0)   # [1,1,17]
action_t = torch.FloatTensor(state_vector[17:21]).unsqueeze(0).unsqueeze(0) # [1,1,4]
lengths  = torch.tensor([1], dtype=torch.long)

with torch.no_grad():
    _, continuation_prob = self._model(state_t, action_t, lengths)

irl_score = float(continuation_prob)
```

### 3-2. `review_env.py` の修正

**修正内容**:
- `_build_event_queue()` のカラム参照: Phase 0 でカラム名を統一済みなので変更不要
- `step()` の TODO を埋める

```python
# step() Step 3 の実装
state_vec = self._current_state.get(
    recommended_dev,
    np.zeros(self.state_builder.obs_dim, dtype=np.float32)
)
acceptance_prob = float(np.clip(
    self.reward_fn.compute(state_vec, recommended_dev, action_dict), 0.0, 1.0
))
accepted = bool(np.random.rand() < acceptance_prob)
```

### 3-3. `agent.py` の実装

TODO メソッドをすべて埋める（SB3 の PPO を使用）:

```python
# __init__
from stable_baselines3 import PPO
self._model = PPO("MlpPolicy", env, verbose=1)

# train
self._model.learn(total_timesteps=total_timesteps)

# evaluate
# ドキュメントに記載のコードをそのまま実装

# save / load
self._model.save(path) / PPO.load(path, env=self.env)
```

### 3-4. 訓練スクリプト作成

**新規ファイル**: `scripts/train/train_rl_agent.py`

```
処理の流れ:
1. data/nova_raw.csv を読み込み
2. IRLReward(model_path="outputs/IRL/nova_extended/irl_model.pt") を初期化
3. StateBuilder(window_days=90, normalize=True) を初期化
4. ReviewEnv を初期化（eval_start〜eval_end の期間を指定）
5. ReviewAgent(env, algorithm="PPO") で学習
6. outputs/RL/nova/ に保存
```

---

## Phase 4: 評価

### 評価①: 行動模倣精度（モード③）

**新規ファイル**: `scripts/evaluate/evaluate_cloning.py`

過去の史実（「誰が実際に担当したか」）を正解ラベルとして、RL エージェントの推薦が一致するかを測定。

```
入力: テスト期間の全レビュー依頼 × 実際の担当者
出力: Top-1 Accuracy, Top-3 Accuracy, MRR

比較対象:
  - RF（スナップショット、時系列無視）
  - IRL（時系列考慮、報酬関数推定）
  - RL（時系列考慮 + 最適化）
```

### 評価②: 貢献可能性予測（モード②）

卒論の「期間ベース予測」をパス条件付きに拡張:

```
P(contribute | developer, time_window)              ← 卒論
P(contribute | developer, time_window, file_path)   ← 修論拡張
```

**新規ファイル**: `scripts/evaluate/conditional_prediction.py`

---

## ファイル変更・作成の全体像

### 変更するファイル

| ファイル | 変更内容 | Phase |
|--------|--------|-------|
| `scripts/pipeline/build_dataset.py` | 出力カラム名を `email`/`timestamp` に統一 | 0 |
| `src/.../IRL/features/common_features.py` | 特徴量を14 → 21次元に拡張 | 1 |
| `src/.../IRL/model/irl_predictor.py` | state_dim/action_dim を更新 | 2 |
| `src/.../RL/reward/reward.py` | `_load_model()` / `compute()` を実装 | 3 |
| `src/.../RL/env/review_env.py` | `step()` の承諾シミュレーションを実装 | 3 |
| `src/.../RL/agent/agent.py` | 全 TODO メソッドを実装 | 3 |

### 新規作成するファイル

| ファイル | 内容 | Phase |
|--------|------|-------|
| `data/collection_config.json` | データ収集パラメータの記録 | 0 |
| `scripts/train/train_rl_agent.py` | RL 訓練スクリプト | 3 |
| `scripts/evaluate/evaluate_cloning.py` | 行動模倣評価 | 4 |
| `scripts/evaluate/conditional_prediction.py` | 条件付き貢献予測 | 4 |

---

## 実装順序とチェックリスト

```
Phase 0: データ収集
  [ ] build_dataset.py にカラム名リネームを追加
  [ ] nova_raw.csv を再取得（Gerrit API）
  [ ] collection_config.json に収集パラメータを記録

Phase 1: 特徴量拡張
  [ ] common_features.py にマクロ3次元を追加
  [ ] common_features.py にパス4次元を追加
  [ ] 21次元で単体テスト（全開発者に対して計算できるか確認）

Phase 2: IRL 再学習
  [ ] irl_predictor.py の state_dim/action_dim を更新
  [ ] train_model.py で再学習実行
  [ ] 既存（14次元）vs 拡張（21次元）の精度比較

Phase 3: RL 実装
  [ ] reward.py の _load_model() / compute() 実装
  [ ] review_env.py の step() 実装
  [ ] agent.py の train/evaluate/save/load 実装
  [ ] train_rl_agent.py を作成して学習実行

Phase 4: 評価
  [ ] evaluate_cloning.py: RF vs IRL vs RL の比較表を出力
  [ ] conditional_prediction.py: 期間×パス条件付き予測
```

---

## 未決事項

- [ ] データ収集期間: 2013〜2017 年を使う？（卒論と同じ）
- [ ] `release_cycle_phase`: Gerrit のタグデータから取得するか、月次で近似するか
- [ ] 行動模倣評価の train/test 分割: IRL の訓練期間と RL のテスト期間を分離する
- [ ] RL の action space サイズ: Nova の活動開発者は何人か（数十〜数百規模）
