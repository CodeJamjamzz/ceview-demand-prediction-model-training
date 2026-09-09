import copy
import io
import json

import numpy as np
import pytest
import torch
from torch import nn

from model.components import (LearnedPositions, PredictionHead,
                              PreNormTransformerBlock, SharedEmbeddings)
from model.experiment import (file_digest, load_protocol, protocol_digest,
                             verify_dataset)
from model.metrics import forecast_metrics


@pytest.fixture(autouse=True)
def small_cpu_workload():
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(42)
    yield
    torch.set_num_threads(old_threads)


def test_identity_order_repeated_over_history_and_invalid_ids():
    layer = SharedEmbeddings().eval()
    history = torch.randn(3, 52, 3)
    market = torch.tensor([2, 0, 1])
    category = torch.tensor([6, 2, 0])
    actual = layer(history, market, category)
    assert actual.shape == (3, 52, 11)
    torch.testing.assert_close(actual[:, :, :3], history)
    torch.testing.assert_close(actual[:, :, 3:7], layer.market(market)[:, None].expand(-1, 52, -1))
    torch.testing.assert_close(actual[:, :, 7:], layer.category(category)[:, None].expand(-1, 52, -1))
    with pytest.raises(ValueError, match="vocabulary"):
        layer(history, torch.tensor([-1, 0, 1]), category)
    with pytest.raises(ValueError, match="int64"):
        layer(history, market.float(), category)
    with pytest.raises(ValueError, match="52"):
        layer(torch.randn(3, 64, 3), market, category)


def test_positions_are_position_specific_and_shared_across_batch():
    layer = LearnedPositions().eval()
    actual = layer(torch.zeros(2, 52, 64))
    torch.testing.assert_close(actual[0], actual[1])
    torch.testing.assert_close(actual[0], layer.embedding.weight)
    assert not torch.equal(actual[0, 0], actual[0, 1])


def test_prediction_head_pools_time_and_preserves_raw_output():
    head = PredictionHead().eval()
    x = torch.randn(2, 52, 64)
    torch.testing.assert_close(head(x), head(x.flip(1)))
    with torch.no_grad():
        head.layers[-1].weight.zero_()
        head.layers[-1].bias.fill_(1.5)
    torch.testing.assert_close(head(x), torch.full((2, 12), 1.5))


def test_zero_sublayers_preserve_exact_residual_no_final_norm():
    block = PreNormTransformerBlock().eval()
    with torch.no_grad():
        for module in (block.attention, block.feedforward):
            for parameter in module.parameters():
                parameter.zero_()
    x = 4 + torch.randn(2, 52, 64) * 3
    torch.testing.assert_close(block(x), x, rtol=0, atol=0)


def test_attention_receives_normalized_history_and_batches_do_not_mix():
    block = PreNormTransformerBlock().eval()
    x = torch.randn(2, 52, 64)
    captured = []
    handle = block.attention.register_forward_pre_hook(
        lambda module, inputs: captured.append(inputs[0].detach().clone()))
    together = block(x)
    handle.remove()
    torch.testing.assert_close(captured[0], block.attention_norm(x))
    separately = torch.cat([block(x[i:i + 1]) for i in range(2)])
    torch.testing.assert_close(together, separately, atol=1e-6, rtol=1e-5)


def test_shared_components_have_finite_gradients_and_restore_state():
    embeddings, projection = SharedEmbeddings(), nn.Linear(11, 64)
    encoder = nn.Sequential(LearnedPositions(), PreNormTransformerBlock(),
                            nn.LayerNorm(64, eps=1e-5), PredictionHead())
    modules = nn.ModuleList([embeddings, projection, encoder])
    optimizer = torch.optim.AdamW(modules.parameters(), lr=0.001)
    history = torch.rand(3, 52, 3)
    market, category = torch.tensor([0, 1, 2]), torch.tensor([0, 3, 6])
    before = projection.weight.detach().clone()
    prediction = encoder(projection(embeddings(history, market, category)))
    assert prediction.shape == (3, 12)
    loss = torch.abs(prediction - torch.rand(3, 12)).mean()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in modules.parameters())
    optimizer.step()
    assert not torch.equal(before, projection.weight)
    modules.eval()
    expected = encoder(projection(embeddings(history, market, category)))
    state = io.BytesIO()
    torch.save(modules.state_dict(), state)
    state.seek(0)
    restored = copy.deepcopy(modules)
    restored.load_state_dict(torch.load(state, weights_only=True))
    actual = restored[2](restored[1](restored[0](history, market, category)))
    torch.testing.assert_close(expected, actual)
    assert torch.equal(actual, restored[2](restored[1](restored[0](history, market, category))))


@pytest.mark.parametrize("layer", [LearnedPositions, PredictionHead, PreNormTransformerBlock])
def test_encoded_layers_reject_wrong_width_or_nonfinite(layer):
    module = layer()
    with pytest.raises(ValueError):
        module(torch.zeros(2, 52, 11))
    with pytest.raises(ValueError):
        module(torch.full((2, 52, 64), float("nan")))


def test_metric_zero_policy_and_mape_coverage():
    actual = np.array([[0.] * 6 + [.02] * 6])
    predicted = np.full((1, 12), .01)
    report = forecast_metrics(actual, predicted)
    assert report["overall"]["mae"] == pytest.approx(1)
    assert report["overall"]["rmse"] == pytest.approx(1)
    assert report["overall"]["smape_percent"] == pytest.approx(400 / 3)
    assert report["overall"]["mape_percent"] == pytest.approx(50)
    assert report["overall"]["mape_coverage"] == .5
    assert report["overall"]["mape_included"] == 6
    zero = forecast_metrics(np.zeros((1, 12)), np.zeros((1, 12)))["overall"]
    assert zero["smape_percent"] == 0
    assert zero["mape_percent"] is None
    assert zero["mape_coverage"] == 0


def test_metric_raw_predictions_averages_and_grouping():
    raw = forecast_metrics(np.ones((1, 12)), np.full((1, 12), 1.5))
    assert raw["overall"]["mae"] == 50
    actual = np.full((2, 12), .5)
    predicted = np.array([[.4, .6] * 6, [.5] * 12])
    result = forecast_metrics(actual, predicted, market_id=[0, 1], category_id=[0, 2])
    assert result["overall"]["mae"] == pytest.approx(5)
    assert result["demand_mean_4w"]["mae"] == pytest.approx(0)
    assert result["demand_mean_12w"]["mae"] == pytest.approx(0)
    assert result["by_market"]["JP"]["overall"]["mae"] == pytest.approx(10)
    assert result["by_category"]["coastal_island"]["overall"]["mae"] == 0
    assert len(result["by_series"]) == 2
    actual = np.zeros((1, 12))
    predicted = np.zeros((1, 12))
    predicted[0, 0] = .12
    result = forecast_metrics(actual, predicted)
    assert result["overall"]["rmse"] == pytest.approx(np.sqrt(12))
    assert result["mean_of_horizon_metrics"]["rmse"] == pytest.approx(1)


@pytest.mark.parametrize("actual,predicted", [
    (np.zeros((0, 12)), np.zeros((0, 12))),
    (np.zeros((2, 12)), np.zeros((2, 1))),
    (np.full((1, 12), np.nan), np.zeros((1, 12))),
    (np.zeros((1, 12)), np.full((1, 12), np.inf)),
    (np.full((1, 12), 50), np.zeros((1, 12))),
])
def test_metrics_reject_invalid_data(actual, predicted):
    with pytest.raises(ValueError):
        forecast_metrics(actual, predicted)


def test_protocol_checksum_and_dataset_bytes(tmp_path, monkeypatch):
    blob = tmp_path / "test.npz"
    blob.write_bytes(b"opaque sealed test archive")
    protocol = load_protocol()
    protocol["dataset"]["files_sha256"] = {"test.npz": file_digest(blob)}
    protocol["protocol_sha256"] = protocol_digest(protocol)
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol), encoding="utf-8")
    monkeypatch.setattr(np, "load", lambda *args, **kwargs: pytest.fail("Test arrays opened"))
    assert verify_dataset(tmp_path, path) == ["test.npz"]
    blob.write_bytes(b"changed")
    with pytest.raises(ValueError, match="missing or changed"):
        verify_dataset(tmp_path, path)
    protocol["training"]["batch_size"] = 64
    path.write_text(json.dumps(protocol), encoding="utf-8")
    with pytest.raises(ValueError, match="Protocol checksum"):
        load_protocol(path)


def test_protocol_rejects_external_dataset_paths(tmp_path):
    protocol = load_protocol()
    protocol["dataset"]["files_sha256"] = {"../outside.npz": "not_a_hash"}
    protocol["protocol_sha256"] = protocol_digest(protocol)
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol), encoding="utf-8")
    with pytest.raises(ValueError, match="escapes"):
        verify_dataset(tmp_path, path)
