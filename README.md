# Master_research

逆強化学習(IRL)と強化学習(RL)を用いたOSS開発者のレビュー継続予測と推薦最適化の研究プロジェクト

## 🚀 Quick Start

```bash
# 環境構築
uv venv              # 仮想環境作成
uv pip install -e .  # パッケージインストール

# スクリプト実行
uv run python scripts/train/train_model.py --help
```

## 🐳 Docker

リポジトリ直下の `compose.yaml` でコンテナ `irl_dir_v2` を起動し、ホスト側のリポジトリを
`/app/` に bind mount して実行する。学習・評価はこの中で回すのが基本。

`compose.yaml` が参照する `UID` / `GID` / `USERNAME` はリポジトリ直下の `.env`
(gitignore 対象) で管理する。新しい環境では `id -u` / `id -g` / `whoami` の値を
`.env` に書いておく。

```bash
# 1. コンテナをバックグラウンドで起動
docker compose up -d --build      # 初回 or Dockerfile 変更時
docker compose up -d              # 2 回目以降

# 2. コンテナに入る
docker exec -it irl_dir_v2 bash

# 以降、コンテナ内で uv コマンド・パイプラインを実行
#   例: bash scripts/run_mce_pipeline.sh main32

# 3. 停止
docker compose down
```

以降の `scripts/...` の例はコンテナ内 / ホスト側どちらでも同じコマンドで動く。

## 🔁 End-to-End パイプライン (データ収集 → 評価)

研究対象スコープは現在 **32 main repos**（OpenStack governance の service teams、sunbeam を除く）。
プロジェクト集合は任意のサブセットに切り替え可能（10 旧スコープ / 32 main / 244 full / tier 別など）。

### TL;DR

データが揃っていれば、以下 1 コマンドで filter → MCE-IRL 学習 → 評価まで一気通貫:

```bash
# 長時間実行になるため nohup とマルチスレッド制限を入れてバックグラウンド実行を推奨
# ※ スレッド数は (搭載CPUコア数 ÷ 並列プロセス数) を目安に調整してください (40コア・4並列の場合は 8 程度)
nohup env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
    bash scripts/run_mce_pipeline.sh main32 outputs/main32_mce 0 > logs/main32_mce.log 2>&1 &

# 特徴量重要度も保存する場合 (第4引数に true)
nohup env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
    bash scripts/run_mce_pipeline.sh main32 outputs/main32_mce 0 true > logs/main32_mce.log 2>&1 &

# ログの確認
tail -f logs/main32_mce.log
```

デフォルトで variant 0 (LSTM) のみ実行。LSTM+Attention / Transformer はスクリプト末尾で
コメントアウトしてあるので必要時に復元する。詳細は下の Step 1〜7。

### Step 1. Gerrit データ収集

`scripts/pipeline/collect_service_teams.sh` が `data/service_teams_repos.csv` (245 repos) を読み、
Gerrit API から per-repo の JSON と整形済み CSV を取得する。既に取得済みの repo はスキップされる。

```bash
# 全 244 repos を取得 (sunbeam-charms を除く)
bash scripts/pipeline/collect_service_teams.sh

# tier や並列度を絞って取得
bash scripts/pipeline/collect_service_teams.sh 大              # 大規模 tier のみ
bash scripts/pipeline/collect_service_teams.sh 中,小  6        # 中・小、並列 6

# 失敗ログ: logs/collect_service_teams/<repo>.log
```

出力:
- `data/raw_json/openstack__<repo>.json` … Gerrit API 生データ (change → file path マッピング用)
- `data/raw_csv/openstack__<repo>.csv` … per-repo 整形済みレビュー依頼データ

### Step 2. 全 repo を 1 ファイルに統合

```bash
uv run python scripts/prepare_combined_data.py
# → data/combined_raw_<N>.csv  (N = 統合できた repo 数。例: 231)
```

`data/service_teams_repos.csv` の `excluded_reason` が空の repo のみ取り込む。
tier 別行数・欠損 repo などの統計が stdout に表示される。

### Step 3. プロジェクト集合を絞ってサブセットを生成

`scripts/pipeline/filter_combined.py` で、統合済み CSV から任意のサブセットを切り出す。
学習対象を切り替えるたびに収集をやり直す必要はない。

```bash
# 32 main repos のみ抽出 (推奨デフォルト)
uv run python scripts/pipeline/filter_combined.py --main
# → data/combined_raw_main32.csv
# → data/combined_raw_main32.raw_json_list.txt  (対応する raw_json パス一覧)

# tier 別 / team 別 / プロジェクト直接指定も可
uv run python scripts/pipeline/filter_combined.py --tier 大
uv run python scripts/pipeline/filter_combined.py --teams nova,neutron
uv run python scripts/pipeline/filter_combined.py \
    --projects openstack/nova,openstack/neutron \
    --output data/combined_raw_pair.csv
```

副産物の `*.raw_json_list.txt` はスペース区切りの JSON パス列なので、`$(cat ...)` で
そのまま `--raw-json` に流せる。


### Step 4. 本番: バリアント比較 (4 訓練窓 × 10 評価パターン)

TL;DR の `run_mce_pipeline.sh` が呼び出している本体。scope を切り替えた `REVIEWS` /
`RAW_JSON_LIST_FILE` を環境変数で渡せる。

```bash
REVIEWS=data/combined_raw_main32.csv \
RAW_JSON_LIST_FILE=data/combined_raw_main32.raw_json_list.txt \
bash scripts/variant/run_mce_irl_variant_single.sh 0 lstm_baseline \
    outputs/main32_mce_variant 0
# variant_id: 0=LSTM, 1=LSTM+Attention, 2=Transformer (MCE-IRL は二値 action なので 3-5 は非対応)

# 特徴量重要度も保存する場合 (第5引数に true)
REVIEWS=data/combined_raw_main32.csv \
RAW_JSON_LIST_FILE=data/combined_raw_main32.raw_json_list.txt \
bash scripts/variant/run_mce_irl_variant_single.sh 0 lstm_baseline \
    outputs/main32_mce_variant 0 true
```

特徴量重要度の出力:
- `<eval_dir>/irl_feature_importance.json` … gradient × input の絶対値重要度 (28次元)
- `<eval_dir>/irl_feature_importance_signed.json` … 符号付き重要度 (正=継続促進, 負=離脱促進)
- `<eval_dir>/rf_dir_feature_importance.json` … RF_Dir の Gini importance

#### 既存モデルから重要度だけ取得（再学習不要）

学習済みモデル（`.pt`）があれば、評価スクリプトだけ再実行すれば重要度を取得できる。
学習はスキップされ（`.pt` が既にあるため）、評価のみ再実行される。

```bash
# 既存の outputs/main32_mce_ep150 に対して重要度だけ取得
nohup env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
    bash scripts/run_mce_pipeline.sh main32 outputs/main32_mce_ep150 0 true \
    > logs/main32_mce_importance.log 2>&1 &
```

> ℹ️ `run_mce_irl_variant_single.sh` は学習済み `.pt` が存在すれば学習をスキップし、
> 評価のみ再実行する。そのため既存モデルからの重要度取得は `--save-importance` を
> 付けて同じパイプラインを再実行するだけでよい。

固定値: `TRAIN_START=2019-01-01`, `TRAIN_END=2022-01-01`, `EVAL_CUTOFF=2023-01-01`,
`EPOCHS=150`, `PATIENCE=5`, 評価は `--calibrate --n-jobs 8` で並列。

学習スクリプトには `--skip-threshold` を常時付与している（訓練データ上の F1 最適閾値計算は
46928 軌跡の逐次推論で 1 時間以上かかり、かつメイン評価指標 AUC/Spearman は閾値非依存・
eval 側で別途閾値を再計算しているため不要）。F1 動作点が必要になった場合のみ外す。

> 教師あり版のバリアント比較は `run_variant_single.sh` (参考比較用)。

### Step 5. 可視化

```bash
uv run python scripts/analyze/plot/visualize_results.py \
    --input-dir outputs/main32_mce_variant \
    --output-dir outputs/figures_main32
```

出力は PDF（PNG は使用しない方針）。

### 単発デバッグ用例 (1 訓練窓・1 評価パターン)

> ⚠ **本番評価には使わない**。スクリプトの引数確認・小さく回して動作チェックする
> ための例。本番の学習・評価は Step 6 (= TL;DR の `run_mce_pipeline.sh`) を使う。

主手法は Maximum Causal Entropy IRL (`train_mce_irl.py`)。教師あり Focal Loss を使わない、
逆強化学習の本来の定式化。Phase 1 の評価・論文の基準モデルはこちら。

```bash
REVIEWS=data/combined_raw_main32.csv
RAWJSON=$(cat data/combined_raw_main32.raw_json_list.txt)

# 学習 (デフォ: train 2021-2023, future 0-3m の 1 窓のみ)
# --skip-threshold: 訓練データ上の F1 最適閾値計算（46928軌跡を逐次推論、1h+）をスキップ。
# AUC/Spearman は閾値非依存・eval 側で別途閾値を計算するので外して問題なし。
uv run python scripts/train/train_mce_irl.py \
    --directory-level --model-type 0 \
    --reviews "$REVIEWS" --raw-json $RAWJSON \
    --skip-threshold \
    --output outputs/main32_mce_irl_debug

# 評価 (1 パターン)
uv run python scripts/analyze/eval/eval_mce_irl_path_prediction.py \
    --data "$REVIEWS" --raw-json $RAWJSON \
    --irl-dir-model outputs/main32_mce_irl_debug/mce_irl_model.pt \
    --output-dir outputs/main32_mce_eval_debug
```

`--model-type`: 0=LSTM, 1=LSTM+Attention, 2=Transformer。保存ファイルは `mce_irl_model.pt`。

> **参考比較用 (教師あり Focal-IRL)**: `scripts/train/train_model.py --model-type 0` で
> `lstm_baseline` を学習。主結果には用いず、MCE-IRL との対照として参照のみ。


### プロジェクト集合の切り替え方

学習・評価に渡す `--reviews` と `--raw-json` を差し替えるだけ。各サブセットを
`data/combined_raw_<tag>.csv` として命名しておけば、Step 4 以降の出力ディレクトリも
`outputs/<tag>_*` で揃えられ、結果の比較がしやすい。

| 用途             | 生成コマンド                                              | 出力 CSV                       |
|------------------|-----------------------------------------------------------|--------------------------------|
| 旧スコープ 10    | `--projects openstack/nova,openstack/neutron,...`         | `combined_raw_custom10.csv`    |
| **main 32 (推奨)** | `--main`                                                  | `combined_raw_main32.csv`      |
| 大規模 tier      | `--tier 大`                                               | `combined_raw_tier_大.csv`     |
| 全 service 244   | (Step 2 の出力)                                           | `combined_raw_231.csv`         |

## 📚 初学者向けガイド

**[👉 学習ロードマップを読む](docs/LEARNING_ROADMAP.md)**

コードの読み方、学習の進め方を段階的に解説しています。


## 特徴量分布調査方法

### 1. Raw 特徴量分布（月次スナップショット）

全レビュアー × 全月でスナップショットを取り、生の特徴量分布を可視化する。
活動がない月もゼロとしてカウントされるため、ゼロ率が高く出る点に注意。

```bash
# 基本: main32 scope の特徴量分布を集計
uv run python scripts/analyze/plot/plot_feature_distributions.py \
    --reviews data/combined_raw_main32.csv \
    --train-start 2019-01-01 --train-end 2022-01-01 \
    --output-dir outputs/feature_dist_main32

# ラベル別（pos / neg / no-request）に分けて重ね描き
uv run python scripts/analyze/plot/plot_feature_distributions.py \
    --reviews data/combined_raw_main32.csv \
    --train-start 2019-01-01 --train-end 2022-01-01 \
    --output-dir outputs/feature_dist_main32_labeled --label-split
```

### 2. 実際のモデル入力分布（軌跡キャッシュから）

MCE-IRL が学習時に使うのと同じロジック (`_precompute_trajectories`) で特徴量を計算し、
**実際にモデルが見る分布**を可視化する。ディレクトリ親和度（3次元）も含む全 28 次元を出力。

```bash
# 全体分布（デフォルト）
uv run python scripts/analyze/plot/plot_trajectory_feature_dist.py \
    --cache outputs/mce_irl_trajectory_cache/main32/mce_traj_0-3.pkl \
    --output-dir outputs/feature_dist_trajectory_main32 \
    --n-jobs -1

# 正例/負例の色分けあり
uv run python scripts/analyze/plot/plot_trajectory_feature_dist.py \
    --cache outputs/mce_irl_trajectory_cache/main32/mce_traj_0-3.pkl \
    --output-dir outputs/feature_dist_trajectory_main32_split \
    --label-split --n-jobs -1
```

出力:
- `trajectory_feature_values.csv`: 全ステップの特徴量値
- `trajectory_feature_percentiles.csv`: パーセンタイル統計（ゼロ率を含む）
- 各特徴量の分布プロット（`.png` / `.pdf`）           

## 📁 プロジェクト構造

```
Master_research/
├── src/
│   └── review_predictor/      # メインパッケージ
│       ├── IRL/               # 予測モデル (Phase 1)
│       │   ├── features/      # 特徴量エンジニアリング
│       │   └── model/         # IRLモデル実装
│       └── RL/                # 強化学習 (Phase 2)
│           ├── agent/         # RLエージェント
│           ├── env/           # シミュレーション環境
│           ├── reward/        # 報酬関数
│           └── state/         # 状態表現
├── scripts/
│   ├── pipeline/              # データ生成
│   ├── train/                 # モデル訓練
│   ├── analyze/               # 結果分析・可視化
│   └── run_cross_temporal_dir.sh # クロス時間・プロジェクト評価一括スクリプト
├── data/                      # データセット (raw_json/, csv など)
├── outputs/                   # 実験結果一覧・保存済みモデル (eval_results/ など)
└── tests/                     # テストコード ( pytest 用 )
```

## 🎯 研究フェーズ

### Previous Work: 個人単位のレビュー継続予測 (卒論 / 完了)

- 逆強化学習（IRL）を用いて各開発者のモチベーション（報酬関数）を推定
- LSTM + Focal Loss を用いて「ある人が、このタイミングで継続するか（継続確率）」を予測

### Phase 1: 機能単位の貢献者数予測 (修士研究 柱1 / 進行中)

- 卒論モデルの継続確率をディレクトリ（機能・パス）単位で拡張・集計
- 将来のプロジェクトモジュールごとの貢献者数を予測
- 「どの機能の開発者が不足するか（無人化の危険）」を事前に検知

### Phase 2: タスク発生時のレビュアー推薦 (修士研究 柱2 / 進行中)

- 提出されたPR（タスク内容）と、各開発者の動的な状態（経験・負荷・パス親和度）を考慮
- 強化学習（RL）エージェントが、プロジェクト全体の健全性（負荷分散・応答速度・バス係数）を最適化するような推薦戦略を学習
- 従来の静的なレビュアー推薦ではなく、時間的変化を伴うシミュレーション評価を実施


---
### memo
--skip-threshold フラグ。メイン評価指標は          
  AUC/Spearman で閾値非依存（CLAUDE.md にも「IRL 出力は未校正スコア」「clf_auc_roc / spearman
  が本来の評価指標」とある）ので、run_mce_irl_variant_single.sh:192-205 の uv run python            
  scripts/train/train_mce_irl.py ... 行に --skip-threshold                                        
  を足せば閾値計算ステップ自体が消えます。F1
  ベースの閾値が要らない実験なら、これが一番手っ取り早い。

  ④ 4 訓練窓を並列実行                                                                              
   
  run_mce_irl_variant_single.sh:170 の for i in 0 1 2 3 が GPU                                      
  共有のため逐次ですが、①でメモリ余裕ができれば & + wait で 2〜4 並列に出来ます（同 GPU           
  上で複数プロセス）。                                                      