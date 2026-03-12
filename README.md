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
│   └── analyze/               # 結果分析・可視化
├── data/                      # データセット
├── outputs/                   # 実験結果
└── docs/                      # ドキュメント
    ├── LEARNING_ROADMAP.md    # 学習ガイド ⭐
    └── research_design_roadmap.md  # 研究設計書
```

## 🎯 研究フェーズ

### Phase 1: IRL予測モデル (完了)

- 開発者の報酬関数を推定
- レビュー継続確率を予測
- LSTM + Focal Loss

### Phase 2: RL推薦最適化 (進行中)

- タスク推薦戦略の最適化
- シミュレーション評価
- マルチエージェント環境

## 📖 ドキュメント

- [学習ロードマップ](docs/LEARNING_ROADMAP.md) - 初学者向け学習パス
- [研究設計書](docs/research_design_roadmap.md) - 研究全体の設計
- [RL実装ガイド](docs/rl_implementation_guide.md) - Phase 2の実装方針
