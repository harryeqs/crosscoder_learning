import pytest
import torch as th

from dictionary_learning.dictionary import CodeNormalization
from dictionary_learning.hetero import HeteroBatchTopK, hetero_collate
from dictionary_learning.trainers.hetero_batch_topk import HeteroBatchTopKTrainer


def _model(dims=(32, 64, 48), dict_size=16, k=4, **kwargs):
    return HeteroBatchTopK(
        activation_dims=dims,
        dict_size=dict_size,
        k=k,
        **kwargs,
    )


def _batch(dims=(32, 64, 48), batch=8):
    return [th.randn(batch, d) for d in dims]


def test_encode_decode_shapes():
    dims = (32, 64, 48)
    model = _model(dims)
    xs = _batch(dims, batch=8)
    f = model.encode(xs, use_threshold=False)
    assert f.shape == (8, 16)
    xhat = model.decode(f, denormalize_activations=False)
    assert len(xhat) == 3
    for x, xh, d in zip(xs, xhat, dims):
        assert xh.shape == (8, d)
        assert xh.shape == x.shape


def test_batch_topk_keeps_exactly_k_times_batch():
    dims = (32, 64)
    k = 3
    batch = 10
    model = _model(dims, dict_size=32, k=k)
    xs = _batch(dims, batch=batch)
    f = model.encode(xs, use_threshold=False, normalize_activations=False)
    assert int((f != 0).sum().item()) == k * batch


def test_inference_threshold_zeros_small_codes():
    model = _model()
    model.threshold.fill_(1e9)
    xs = _batch()
    f = model.encode(xs, use_threshold=True, normalize_activations=False)
    assert th.all(f == 0)


def test_code_normalization_crosscoder_is_sum_of_decoder_col_norms():
    model = _model(dims=(8, 16), dict_size=5, k=2)
    w = model.get_code_normalization()
    expected = model.decoder.weights[0].norm(dim=-1) + model.decoder.weights[1].norm(
        dim=-1
    )
    assert w.shape == (1, 5)
    assert th.allclose(w.squeeze(0), expected)


def test_code_normalization_sae_is_concat_norm():
    model = _model(
        dims=(8, 16),
        dict_size=5,
        k=2,
        code_normalization=CodeNormalization.SAE,
    )
    w = model.get_code_normalization().squeeze(0)
    expected = th.sqrt(
        model.decoder.weights[0].norm(dim=-1) ** 2
        + model.decoder.weights[1].norm(dim=-1) ** 2
    )
    assert th.allclose(w, expected)


def test_activation_normalization_roundtrip():
    dims = (4, 7)
    means = [th.randn(d) for d in dims]
    stds = [th.rand(d) + 0.5 for d in dims]
    model = _model(dims, activation_means=means, activation_stds=stds, target_rms=1.0)
    xs = _batch(dims, batch=5)
    xs_n = model.normalize_activations(xs)
    xs_hat = model.denormalize_activations(xs_n)
    for a, b in zip(xs, xs_hat):
        assert th.allclose(a, b, atol=1e-5)


def test_decoupled_rejected():
    with pytest.raises(NotImplementedError):
        _model(code_normalization="decoupled")


def test_from_pretrained_roundtrip(tmp_path):
    dims = (12, 20)
    model = _model(dims, dict_size=9, k=3)
    path = tmp_path / "hetero.pt"
    th.save(model.state_dict(), path)
    loaded = HeteroBatchTopK.from_pretrained(str(path))
    assert loaded.activation_dims == list(dims)
    assert loaded.dict_size == 9
    xs = _batch(dims, batch=4)
    f1 = model.encode(xs, use_threshold=False, normalize_activations=False)
    f2 = loaded.encode(xs, use_threshold=False, normalize_activations=False)
    assert th.allclose(f1, f2)


def test_hetero_collate():
    sample0 = [th.randn(8), th.randn(16)]
    sample1 = [th.randn(8), th.randn(16)]
    batch = hetero_collate([sample0, sample1])
    assert len(batch) == 2
    assert batch[0].shape == (2, 8)
    assert batch[1].shape == (2, 16)


def test_trainer_one_step_reduces_or_finite_loss():
    dims = (16, 32)
    dict_size = 24
    trainer = HeteroBatchTopKTrainer(
        steps=10,
        activation_dims=dims,
        dict_size=dict_size,
        k=4,
        layer=0,
        lm_name="toy",
        warmup_steps=1,
        threshold_start_step=100,
        device="cpu",
        seed=0,
        lr=1e-3,
    )
    xs = _batch(dims, batch=16)
    loss0 = trainer.update(0, xs)
    loss1 = trainer.update(1, xs)
    assert th.isfinite(th.tensor(loss0))
    assert th.isfinite(th.tensor(loss1))
    # After geometric-median init on step 0, step 1 should still be a valid train step
    f = trainer.ae.encode(xs, use_threshold=False, normalize_activations=False)
    assert int((f != 0).sum().item()) == 4 * 16
