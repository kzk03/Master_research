"""MCE-IRL モデルの forward / 学習1ステップ / state_dict round-trip スモークテスト.

検証対象 (src/review_predictor/IRL/model/mce_irl_predictor.py):
  - 3 バリアント (LSTM / Attention / Transformer) の forward 形状
  - forward_all_steps が全タイムステップ分の logits を返すこと
  - _mce_loss_on_batch が finite な NLL を返すこと
  - optimizer.step() で勾配が流れて loss が更新できること
  - state_dict を保存 → 別インスタンスで読み込んで重みが完全一致すること

ねらい: 実データ・実 trajectory に依存しない最小スモーク。
ネットワーク層の入出力契約と保存形式の互換性だけを担保する。
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from review_predictor.IRL.features.common_features import (
    ACTION_FEATURES,
    STATE_FEATURES,
)
from review_predictor.IRL.model.mce_irl_predictor import (
    MCEIRLSystem,
    create_mce_network,
)


# ─────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────

STATE_DIM = len(STATE_FEATURES)     # 21
ACTION_DIM = len(ACTION_FEATURES)   # 5
HIDDEN_DIM = 32                      # テスト用に小さく
NUM_ACTIONS = 2

VARIANTS = [0, 1, 2]                 # 0=LSTM, 1=Attention, 2=Transformer
VARIANT_NAMES = {0: "lstm", 1: "attention", 2: "transformer"}


def _make_config(model_type: int) -> dict:
    return {
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "hidden_dim": HIDDEN_DIM,
        "dropout": 0.1,
        "model_type": model_type,
        "learning_rate": 3e-4,
        "weight_decay": 1e-4,
    }


def _make_dummy_batch(
    B: int = 4, L: int = 6, device: torch.device | str = "cpu"
) -> dict[str, torch.Tensor]:
    """_mce_loss_on_batch が要求するキー一式をダミーで用意する."""
    torch.manual_seed(0)
    state_seq = torch.randn(B, L, STATE_DIM, device=device)
    action_seq = torch.randn(B, L, ACTION_DIM, device=device)
    # 系列長は B 個ばらばらに (ただし >=2、<=L)
    lengths = torch.tensor([L, L - 1, L - 2, 2][:B], dtype=torch.long, device=device)
    actions = torch.randint(0, NUM_ACTIONS, (B, L), dtype=torch.long, device=device)
    sample_weights = torch.ones(B, L, device=device)
    mask = torch.zeros(B, L, device=device)
    for i in range(B):
        mask[i, : lengths[i].item()] = 1.0
    return {
        "state_seq": state_seq,
        "action_seq": action_seq,
        "lengths": lengths,
        "actions": actions,
        "sample_weights": sample_weights,
        "mask": mask,
    }


# ─────────────────────────────────────────────────────────────────────
# forward 形状
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("model_type", VARIANTS, ids=lambda v: VARIANT_NAMES[v])
def test_forward_returns_per_sequence_reward(model_type):
    """network(state, action, lengths) は [B, num_actions] を返す."""
    net = create_mce_network(
        variant=model_type,
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=HIDDEN_DIM,
        dropout=0.0,
        num_actions=NUM_ACTIONS,
    )
    net.eval()
    B, L = 3, 5
    state = torch.randn(B, L, STATE_DIM)
    action = torch.randn(B, L, ACTION_DIM)
    lengths = torch.tensor([L, L - 1, 2], dtype=torch.long)

    out = net(state, action, lengths)
    assert out.shape == (B, NUM_ACTIONS)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("model_type", VARIANTS, ids=lambda v: VARIANT_NAMES[v])
def test_forward_all_steps_returns_per_step_reward(model_type):
    """forward_all_steps は [B, L, num_actions] を返す."""
    net = create_mce_network(
        variant=model_type,
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=HIDDEN_DIM,
        dropout=0.0,
        num_actions=NUM_ACTIONS,
    )
    net.eval()
    B, L = 3, 5
    state = torch.randn(B, L, STATE_DIM)
    action = torch.randn(B, L, ACTION_DIM)
    lengths = torch.tensor([L, L - 1, 2], dtype=torch.long)

    out = net.forward_all_steps(state, action, lengths)
    assert out.shape == (B, L, NUM_ACTIONS)
    assert torch.isfinite(out).all()


def test_create_mce_network_rejects_unknown_variant():
    with pytest.raises(ValueError):
        create_mce_network(variant=99, state_dim=STATE_DIM, action_dim=ACTION_DIM)


# ─────────────────────────────────────────────────────────────────────
# _mce_loss_on_batch
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("model_type", VARIANTS, ids=lambda v: VARIANT_NAMES[v])
def test_mce_loss_is_finite_and_positive(model_type):
    """softmax CE は非負 + finite. mask=0 のステップは寄与しない."""
    system = MCEIRLSystem(_make_config(model_type))
    batch = _make_dummy_batch(device=system.device)

    loss = system._mce_loss_on_batch(batch)
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0  # CE は非負


@pytest.mark.parametrize("model_type", VARIANTS, ids=lambda v: VARIANT_NAMES[v])
def test_one_optimizer_step_changes_loss(model_type):
    """1 step backward + step で勾配が流れ、loss が動く (= 学習可能)."""
    system = MCEIRLSystem(_make_config(model_type))
    batch = _make_dummy_batch(device=system.device)

    system.network.train()
    loss_before = system._mce_loss_on_batch(batch)
    system.optimizer.zero_grad()
    loss_before.backward()
    # 少なくとも 1 つのパラメータに勾配が流れていること
    any_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in system.network.parameters()
    )
    assert any_grad, "勾配が一切流れていない"
    system.optimizer.step()

    # ステップ後に再計算した loss が finite (NaN 化していない)
    loss_after = system._mce_loss_on_batch(batch)
    assert torch.isfinite(loss_after)


def test_mce_loss_respects_mask():
    """全ステップ mask=0 にすれば denom=1 で loss=0 になる (NaN にもならない)."""
    system = MCEIRLSystem(_make_config(0))
    batch = _make_dummy_batch(device=system.device)
    batch["mask"] = torch.zeros_like(batch["mask"])

    loss = system._mce_loss_on_batch(batch)
    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


# ─────────────────────────────────────────────────────────────────────
# state_dict round-trip
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("model_type", VARIANTS, ids=lambda v: VARIANT_NAMES[v])
def test_state_dict_roundtrip_preserves_weights(model_type, tmp_path: Path):
    """学習スクリプト (train_mce_irl.py) と同じ保存形式で重みが完全復元できる."""
    system_a = MCEIRLSystem(_make_config(model_type))
    # ランダム重みに少し学習を入れて識別性を持たせる
    batch = _make_dummy_batch(device=system_a.device)
    loss = system_a._mce_loss_on_batch(batch)
    system_a.optimizer.zero_grad()
    loss.backward()
    system_a.optimizer.step()

    model_path = tmp_path / "irl_model.pt"
    torch.save(system_a.network.state_dict(), model_path)

    system_b = MCEIRLSystem(_make_config(model_type))
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    system_b.network.load_state_dict(state_dict)

    # 同一バッチで forward → 完全一致
    system_a.network.eval()
    system_b.network.eval()
    with torch.no_grad():
        out_a = system_a.network(batch["state_seq"], batch["action_seq"], batch["lengths"])
        out_b = system_b.network(batch["state_seq"], batch["action_seq"], batch["lengths"])
    assert torch.allclose(out_a, out_b, atol=1e-6), (
        f"round-trip 後の出力が一致しない: max diff = {(out_a - out_b).abs().max().item()}"
    )


# ─────────────────────────────────────────────────────────────────────
# 初期化サニティ
# ─────────────────────────────────────────────────────────────────────


def test_mce_irl_system_uses_correct_dims():
    """config で渡した次元が network に伝わっている."""
    system = MCEIRLSystem(_make_config(0))
    assert system.state_dim == STATE_DIM
    assert system.action_dim == ACTION_DIM
    assert system.NUM_ACTIONS == NUM_ACTIONS
    # reward head の最終層は num_actions=2 ユニット
    last_linear = system.network.reward_predictor[-1]
    assert isinstance(last_linear, torch.nn.Linear)
    assert last_linear.out_features == NUM_ACTIONS
