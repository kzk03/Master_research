# Master_research

逆強化学習(IRL)と強化学習(RL)を用いたOSS開発者のレビュー継続予測と推薦最適化の研究プロジェクト

## 🚀 Quick Start

```bash
# 環境構築
uv venv              # 仮想環境作成
uv pip install -e .  # パッケージインストール

# スクリプト実行
uv run python scripts/train/train_model.py --help

# 予測結果の評価（複数プロジェクトのJSON指定やRFの将来窓の個別設定に対応）
uv run python scripts/analyze/eval_path_prediction.py \
    --raw-json data/raw_json/openstack__nova.json data/raw_json/other_project.json \
    --rf-future-start-months 1
```

## 📚 初学者向けガイド

**[👉 学習ロードマップを読む](docs/LEARNING_ROADMAP.md)**

コードの読み方、学習の進め方を段階的に解説しています。

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
