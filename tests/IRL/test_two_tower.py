"""
MCEIRLNetworkLSTMTwoTower (model_type=3) の動作確認テスト。

Two-tower アーキテクチャ:
    state[:, :, :state_only_dim] → state_encoder + LSTM
    state[:, :, state_only_dim:] → path_encoder (バイパス、最終ステップ直結)
    両者を concat → reward_predictor

動機: B-10 で IRL が path 特徴量を 17.6% しか拾えていなかった "LSTM dilution"
問題を回避し、path 系を LSTM をバイパスさせて最終層に直結する。
"""
from __future__ import annotations

import pytest
import torch

from review_predictor.IRL.model.mce_irl_predictor import (
    MCEIRLNetworkLSTMTwoTower,
    MCEIRLSystem,
    create_mce_network,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Forward / forward_all_steps の shape 検証
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_two_tower_forward_shape():
    net = MCEIRLNetworkLSTMTwoTower(
        state_dim=27, action_dim=5, hidden_dim=128, dropout=0.1,
        num_actions=2, path_dim=6,
    )
    B, L = 4, 12
    state = torch.randn(B, L, 27)
    action = torch.randn(B, L, 5)
    lengths = torch.tensor([12, 8, 12, 5])
    out = net(state, action, lengths)
    assert out.shape == (B, 2)


def test_two_tower_forward_all_steps_shape():
    net = MCEIRLNetworkLSTMTwoTower(
        state_dim=27, action_dim=5, hidden_dim=128, dropout=0.1,
        num_actions=2, path_dim=6,
    )
    B, L = 3, 8
    state = torch.randn(B, L, 27)
    action = torch.randn(B, L, 5)
    lengths = torch.tensor([8, 5, 3])
    out = net.forward_all_steps(state, action, lengths)
    assert out.shape == (B, L, 2)


def test_two_tower_internal_dims():
    net = MCEIRLNetworkLSTMTwoTower(
        state_dim=27, action_dim=5, hidden_dim=128,
        num_actions=2, path_dim=6,
    )
    assert net.total_state_dim == 27
    assert net.state_only_dim == 21
    assert net.path_dim == 6
    assert net.path_encoder is not None
    # state_encoder は state_only_dim (21) を入力にする
    assert net.state_encoder[0].in_features == 21
    # path_encoder は path_dim (6) を入力にする
    assert net.path_encoder[0].in_features == 6


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# path_dim=0 で LSTM ベースラインと同形状を返す (path bypass 無効)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_two_tower_path_dim_zero_equivalent():
    """path_dim=0 を指定すると path_encoder=None、forward 出力は state_dim 全部を
    LSTM に通したのと同形状になる (実質 LSTM ベースライン相当)。"""
    net = MCEIRLNetworkLSTMTwoTower(
        state_dim=21, action_dim=5, hidden_dim=128,
        num_actions=2, path_dim=0,
    )
    assert net.path_encoder is None
    assert net.state_only_dim == 21
    B, L = 2, 6
    state = torch.randn(B, L, 21)
    action = torch.randn(B, L, 5)
    lengths = torch.tensor([6, 4])
    out = net(state, action, lengths)
    assert out.shape == (B, 2)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 入力検証
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_two_tower_invalid_path_dim_negative():
    with pytest.raises(ValueError, match="path_dim must be >= 0"):
        MCEIRLNetworkLSTMTwoTower(state_dim=27, action_dim=5, path_dim=-1)


def test_two_tower_invalid_path_dim_too_large():
    with pytest.raises(ValueError, match="must be <"):
        MCEIRLNetworkLSTMTwoTower(state_dim=27, action_dim=5, path_dim=27)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# create_mce_network / MCEIRLSystem 経由
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_create_mce_network_variant_3():
    net = create_mce_network(
        variant=3, state_dim=27, action_dim=5,
        hidden_dim=128, dropout=0.1, num_actions=2, path_dim=6,
    )
    assert isinstance(net, MCEIRLNetworkLSTMTwoTower)
    assert net.path_dim == 6


def test_mce_irl_system_auto_path_dim():
    """MCEIRLSystem が state_dim と STATE_FEATURES から path_dim を自動推定."""
    from review_predictor.IRL.features.common_features import STATE_FEATURES
    sys = MCEIRLSystem({"state_dim": len(STATE_FEATURES) + 6, "model_type": 3})
    assert sys.path_dim == 6
    assert isinstance(sys.network, MCEIRLNetworkLSTMTwoTower)


def test_mce_irl_system_explicit_path_dim():
    """config で path_dim を明示すれば優先される."""
    sys = MCEIRLSystem({"state_dim": 30, "path_dim": 9, "model_type": 3})
    assert sys.path_dim == 9
    assert sys.network.state_only_dim == 21


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Path-bypass の動作確認: path が違うと出力が変わる
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_two_tower_path_affects_output():
    """path 部分を変えると forward 出力が変化することを確認 (path bypass が
    実際に最終層に伝わっているかの検証)。"""
    torch.manual_seed(0)
    net = MCEIRLNetworkLSTMTwoTower(
        state_dim=27, action_dim=5, hidden_dim=64, dropout=0.0,
        num_actions=2, path_dim=6,
    )
    net.eval()
    B, L = 1, 5
    state_only = torch.randn(B, L, 21)
    action = torch.randn(B, L, 5)
    lengths = torch.tensor([5])

    # 同じ state_only + action だが path だけ違う 2 ケース
    path_a = torch.zeros(B, L, 6)
    path_b = torch.ones(B, L, 6)
    state_a = torch.cat([state_only, path_a], dim=-1)
    state_b = torch.cat([state_only, path_b], dim=-1)

    with torch.no_grad():
        out_a = net(state_a, action, lengths)
        out_b = net(state_b, action, lengths)

    # path が違えば出力も違うはず
    assert not torch.allclose(out_a, out_b, atol=1e-6), (
        "path 入力を変えても出力が変わらない = path bypass が機能していない"
    )


def test_two_tower_state_only_path_zero_isolates_lstm():
    """path_dim=6 で path 部分を全て 0 にすると、path_encoder の出力は LayerNorm
    と ReLU 経由でも (非ゼロ vs ゼロ入力で) 異なるが、出力の主要寄与は state-only
    側になる。これは architecture 正常性の sanity check (path bypass が完全に
    主役を奪っていないことの確認)."""
    torch.manual_seed(0)
    net = MCEIRLNetworkLSTMTwoTower(
        state_dim=27, action_dim=5, hidden_dim=64, dropout=0.0,
        num_actions=2, path_dim=6,
    )
    net.eval()
    B, L = 1, 5
    state_zero = torch.zeros(B, L, 21)
    action = torch.zeros(B, L, 5)
    state_normal = torch.randn(B, L, 21)
    path = torch.zeros(B, L, 6)
    lengths = torch.tensor([5])

    state_a = torch.cat([state_normal, path], dim=-1)
    state_b = torch.cat([state_zero, path], dim=-1)

    with torch.no_grad():
        out_a = net(state_a, action, lengths)
        out_b = net(state_b, action, lengths)

    # state 入力を変えれば出力も変わる
    assert not torch.allclose(out_a, out_b, atol=1e-6)
