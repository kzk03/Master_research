# RL 実装ガイド（初学者向け）

新規作成した `src/RL/` 配下の4つのファイルについて、
概念・コードの読み方・実装手順を順番に解説します。

---

## 全体像：4つのファイルの関係

```
┌──────────────────────────────────────────────────────┐
│  あなたが書くスクリプト（例: scripts/train/train_rl.py）  │
│                                                      │
│   env = ReviewEnv(reward_fn=..., state_builder=...)  │
│   agent = ReviewAgent(env=env)                       │
│   agent.train()                                      │
└──────┬───────────────────────────────────────────────┘
       │
       ▼
┌─────────────────┐    使う    ┌──────────────────┐
│  RL/env/        │ ─────────▶ │  RL/state/       │
│  review_env.py  │            │  state_builder.py│
│  （環境の司令塔） │            │  （状態の作り方）  │
└────────┬────────┘            └──────────────────┘
         │ 使う
         ▼
┌─────────────────┐            ┌──────────────────┐
│  RL/reward/     │            │  RL/agent/       │
│  reward.py      │            │  agent.py        │
│  （報酬の計算）  │            │  （学習エンジン） │
└─────────────────┘            └──────────────────┘
         ▲
         │ IRL の学習済みモデルを流用
┌─────────────────┐
│  IRL/model/     │
│  irl_predictor  │
│  （卒論資産）   │
└─────────────────┘
```

---

## 1. `RL/state/state_builder.py` — 状態の作り方

### 1.1 「状態」とは何か

強化学習では、エージェントが行動を決める前に「今の状況」を数値で把握します。
これを **状態ベクトル** と呼びます。

```
開発者A の状態ベクトル（14次元）
= [経験日数, 総変更数, 直近活動頻度, ..., 平均応答時間]
   ↑ これを使って「この人に仕事を振るべきか？」を判断する
```

### 1.2 StateBuilder クラスの読み方

```python
class StateBuilder:
    def __init__(
        self,
        window_days: int = 90,   # 何日分の過去データを見るか
        normalize: bool = True,  # 値を 0〜1 に揃えるか
        use_macro: bool = False, # プロジェクト全体の情報も使うか（M3以降）
    ):
```

**コンストラクタ**: クラスを作るときの設定。
`window_days=90` なら「過去90日のデータで状態を計算する」という意味。

```python
    def build(self, df, developer_id, current_time) -> np.ndarray:
```

**メインメソッド**: 1人の開発者の状態ベクトルを返す。

```python
    def build_all(self, df, developer_ids, current_time) -> Dict[str, np.ndarray]:
```

全員分をまとめて計算して `{"dev@example.com": array([...]), ...}` を返す。

### 1.3 実装するべき TODO

#### TODO 1: マクロ特徴量（M3 で実装）

`_build_macro()` メソッドと `MACRO_FEATURES` リストを同時に更新する。

```python
# 現在（空リスト）
MACRO_FEATURES: List[str] = []

# M3 で追加するもの
MACRO_FEATURES: List[str] = [
    "task_queue_length",      # オープンタスクの数
    "release_cycle_phase",   # リリースサイクルの位置（0.0〜1.0）
    "project_acceptance_rate",  # プロジェクト全体の承諾率
]
```

```python
def _build_macro(self, df, current_time) -> np.ndarray:
    # 実装例：
    recent = df[df["timestamp"] >= current_time - timedelta(days=30)]

    # オープンタスク数（正規化）
    task_queue_length = len(recent) / 100.0  # 100件で1.0

    # リリースサイクル位相（別途リリース日程データが必要）
    release_cycle_phase = 0.5  # TODO: 実際のリリース日程から計算

    # プロジェクト全体の承諾率
    if "label" in recent.columns and len(recent) > 0:
        project_acceptance_rate = recent["label"].mean()
    else:
        project_acceptance_rate = 0.5

    return np.array([
        task_queue_length,
        release_cycle_phase,
        project_acceptance_rate,
    ], dtype=np.float32)
```

---

## 2. `RL/reward/reward.py` — 報酬の計算

### 2.1 「報酬関数」とは何か

RL エージェントは「報酬が高くなる行動」を学習します。
ここでは「全開発者の継続確率の合計」を報酬にします。

```
報酬 = 開発者A の継続確率 + 開発者B の継続確率 + ...
     ← これを最大化するような推薦を学習する
```

### 2.2 抽象クラス（RewardFunction）の読み方

```python
class RewardFunction(ABC):  # ABC = Abstract Base Class（抽象基底クラス）

    @abstractmethod  # このデコレータがついたメソッドは「必ず実装しなければならない」
    def compute(self, state_vector, developer_id, action) -> float:
        ...
```

**なぜ抽象クラスを使うのか？**

```python
# 将来、報酬関数を差し替えたいとき...

# 今: IRL の継続確率を使う
reward_fn = IRLReward(model_path="...")

# 将来: 全く別の設計にしたい
reward_fn = MyNewReward(some_param=...)

# ReviewEnv は変更ゼロ！ どちらも同じ compute() を持つから
env = ReviewEnv(reward_fn=reward_fn, ...)
```

### 2.3 IRLReward の実装 TODO（最初にやること）

`_load_model()` に IRL モデルのロード処理を実装します。

```python
def _load_model(self) -> None:
    if self._model is not None:
        return  # 既にロード済みなら何もしない

    # IRL/model/irl_predictor.py の IRLPredictor を使う
    from IRL.model.irl_predictor import IRLPredictor

    predictor = IRLPredictor()  # IRLPredictor の引数は irl_predictor.py を確認
    predictor.load_model(self.model_path)
    self._model = predictor
    logger.info(f"IRL モデルをロード: {self.model_path}")
```

次に `compute()` の TODO 部分を実装します。

```python
def compute(self, state_vector, developer_id, action) -> float:
    self._load_model()

    # IRL モデルに状態ベクトルを渡してスコアを取得
    # irl_predictor.py の predict() や score() に相当するメソッドを確認して使う
    tensor = torch.FloatTensor(state_vector).unsqueeze(0).to(self.device)
    with torch.no_grad():
        irl_score = self._model.predict_score(tensor)  # メソッド名は要確認

    irl_score = float(irl_score)

    # 負荷ペナルティ（review_load が高い人に推薦すると罰則）
    penalty = 0.0
    if self.workload_penalty_weight > 0.0 and action.get("developer_id") == developer_id:
        REVIEW_LOAD_IDX = 9  # common_features.py の FEATURE_NAMES の順番
        if len(state_vector) > REVIEW_LOAD_IDX:
            penalty = self.workload_penalty_weight * float(state_vector[REVIEW_LOAD_IDX])

    return irl_score - penalty
```

### 2.4 実装の確認方法

```python
# irl_predictor.py を読んで、以下を確認する：
# 1. IRLPredictor はどのように初期化するか？
# 2. モデルのロードは load_model()? それとも from_pretrained()?
# 3. スコア計算は predict()? score()? forward()?
```

---

## 3. `RL/env/review_env.py` — 環境（シミュレータの司令塔）

### 3.1 gymnasium とは

gymnasium は RL の標準ライブラリです。
`env.reset()` と `env.step(action)` の2つのメソッドを実装するだけで、
どんな RL ライブラリ（Stable-Baselines3 など）とも連携できます。

```
エピソードの流れ:

obs, info = env.reset()                  # 初期化
while True:
    action = agent.predict(obs)          # エージェントが行動を選ぶ
    obs, reward, terminated, truncated, info = env.step(action)  # 1ステップ進む
    if terminated or truncated:
        break                            # エピソード終了
```

### 3.2 ReviewEnv の行動空間と観測空間

```python
# 行動空間: 推薦する開発者のインデックス（整数）
self.action_space = spaces.Discrete(self.n_developers)
# → action=0 なら developer_ids[0] に推薦
# → action=2 なら developer_ids[2] に推薦

# 観測空間: 全開発者の状態ベクトルを連結した1次元配列
# 例: 開発者30人 × 14次元 = 420次元のベクトル
self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,))
```

### 3.3 step() の実装 TODO

最重要の TODO は「承諾確率のシミュレーション」です。

```python
def step(self, action: int):
    recommended_dev = self.developer_ids[action]
    action_dict = {"developer_id": recommended_dev, "step": self._step_idx}

    # ── TODO: ここを実装 ──────────────────────────────
    # 推薦された開発者が承諾するかどうかを確率的に決める

    # 推薦された開発者の状態ベクトルを取得
    state_vec = self._current_state.get(
        recommended_dev,
        np.zeros(self.state_builder.obs_dim, dtype=np.float32)
    )

    # IRL スコアを承諾確率として使う（0〜1 の範囲になるように設計されている）
    acceptance_prob = self.reward_fn.compute(state_vec, recommended_dev, action_dict)
    acceptance_prob = float(np.clip(acceptance_prob, 0.0, 1.0))

    # 確率的に承諾/拒否を決定
    accepted = bool(np.random.rand() < acceptance_prob)
    # ─────────────────────────────────────────────────

    # 状態を更新（変わらない部分）
    self._current_state = self.state_builder.build_all(
        df=self.df,
        developer_ids=self.developer_ids,
        current_time=self._current_time,
    )
    ...
```

### 3.4 開発者リストの絞り込み TODO

現状は df 内の全メールアドレスを使っていますが、
活動中の開発者だけに絞る処理を追加すると効率的です。

```python
def _build_event_queue(self) -> pd.DataFrame:
    df = self.df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    mask = (df["timestamp"] >= self.eval_start) & (df["timestamp"] < self.eval_end)
    events = df[mask].sort_values("timestamp").reset_index(drop=True)
    return events
```

上記はすでに実装済みです。必要に応じてここでイベントの種類絞り込み
（例: `action_type == "review"` のみ）を追加できます。

---

## 4. `RL/agent/agent.py` — RL エージェント（M2 で実装）

### 4.1 Stable-Baselines3 を使う方法（推奨）

自前でアルゴリズムを実装する必要はありません。
**Stable-Baselines3（SB3）** という RL ライブラリを使うと、
PPO や SAC を数行で動かせます。

```bash
pip install stable-baselines3
```

### 4.2 ReviewAgent の実装

```python
from stable_baselines3 import PPO, SAC, DQN

ALGORITHMS = {"PPO": PPO, "SAC": SAC, "DQN": DQN}

class ReviewAgent:
    def __init__(self, env, algorithm: str = "PPO") -> None:
        self.env = env
        AlgorithmClass = ALGORITHMS[algorithm]

        # SB3 にそのまま ReviewEnv を渡すだけで学習できる
        self._model = AlgorithmClass(
            policy="MlpPolicy",  # 状態ベクトル → 行動 の全結合ネットワーク
            env=env,
            verbose=1,           # 学習ログを表示
        )

    def train(self, total_timesteps: int = 100_000) -> None:
        self._model.learn(total_timesteps=total_timesteps)

    def evaluate(self, n_episodes: int = 10) -> dict:
        results = {"rewards": [], "accepted_counts": []}
        for _ in range(n_episodes):
            obs, _ = self.env.reset()
            total_reward = 0.0
            accepted_count = 0
            done = False
            while not done:
                action, _ = self._model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.env.step(action)
                total_reward += reward
                if info.get("accepted"):
                    accepted_count += 1
                done = terminated or truncated
            results["rewards"].append(total_reward)
            results["accepted_counts"].append(accepted_count)
        return results

    def save(self, path: str) -> None:
        self._model.save(path)

    def load(self, path: str) -> None:
        AlgorithmClass = type(self._model)
        self._model = AlgorithmClass.load(path, env=self.env)
```

### 4.3 アルゴリズム選定の指針

| アルゴリズム | 行動空間 | 特徴 | 推奨場面 |
|------------|--------|------|---------|
| **PPO** | 離散・連続 | 安定して学習しやすい | **最初はこれ** |
| DQN | 離散のみ | シンプル、開発者数が少ない場合 | 開発者 < 50人 |
| SAC | 連続のみ | 高効率だが実装複雑 | 行動を連続値にしたい場合 |

ReviewEnv の行動空間は `Discrete`（整数）なので、**PPO か DQN** が適切です。

---

## 5. 実装の順番（ロードマップ）

```
Step 1: IRL モデルとの接続（最初にやること）
  └─ RL/reward/reward.py の _load_model() と compute() を実装
  └─ IRL/model/irl_predictor.py を読んで使い方を確認

Step 2: 環境の動作確認
  └─ RL/env/review_env.py の step() の承諾確率部分を実装
  └─ 以下のテストコードで動作確認:

      from RL.env.review_env import ReviewEnv
      from RL.reward.reward import IRLReward
      from RL.state.state_builder import StateBuilder

      reward_fn = IRLReward(model_path="outputs/train_0-3m/irl_model.pt")
      state_builder = StateBuilder(window_days=90)
      env = ReviewEnv(df=df, reward_fn=reward_fn, state_builder=state_builder)

      obs, info = env.reset()
      print("obs shape:", obs.shape)  # (n_developers * 14,) になるはず

      action = env.action_space.sample()  # ランダムな行動
      obs, reward, terminated, truncated, info = env.step(action)
      print("reward:", reward)

Step 3: エージェントの学習（M2）
  └─ RL/agent/agent.py を実装
  └─ scripts/train/train_rl_agent.py を作成して学習を実行

Step 4: 特徴量の拡張（M3）
  └─ RL/state/state_builder.py の _build_macro() を実装
  └─ MACRO_FEATURES に新しい特徴量名を追加
```

---

## 6. よくある疑問

**Q: `np.ndarray` とは？**
A: NumPy の配列（数値のリスト）。`[0.1, 0.5, 0.3, ...]` のような形。
   `shape=(14,)` なら14個の数値が入った1次元配列。

**Q: `@abstractmethod` とは？**
A: 「このメソッドは必ずサブクラスで実装しなさい」という強制マーク。
   実装せずに使うと `TypeError` が出る。

**Q: `gymnasium.Env` を継承するとは？**
A: `ReviewEnv` が `gym.Env` のルールに従うという宣言。
   `reset()` と `step()` を実装すれば、SB3 などの RL ライブラリが自動で使える。

**Q: エピソードとは？**
A: シミュレーションの1回分（開始〜終了まで）。
   この研究では「eval_start から eval_end までの全タスク処理」が1エピソード。
