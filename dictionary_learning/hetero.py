"""
Heterogeneous BatchTopK crosscoder: shared sparse codes across sides
with different hidden dimensions.

Unlike CrossCoder / BatchTopKCrossCoder, this does not stack activations
into (batch, num_layers, d). Each side has its own encoder/decoder Linear
with width d_ℓ.

Typical usage::

    from dictionary_learning.hetero import (
        HeteroBatchTopK,
        HeteroActivationCache,
        hetero_collate,
    )
    from dictionary_learning.trainers.hetero_batch_topk import HeteroBatchTopKTrainer

    cache = HeteroActivationCache(dir_a, dir_b, dir_c)
    loader = DataLoader(cache, batch_size=2048, collate_fn=hetero_collate)
    # each batch is a list of tensors [(B, d0), (B, d1), (B, d2)]
"""

from __future__ import annotations

from typing import Sequence
from warnings import warn

import torch as th
import torch.nn as nn
import torch.nn.init as init
from torch.nn.functional import relu

from .cache import ActivationCache
from .dictionary import CodeNormalization, Dictionary


def hetero_collate(batch: list[list[th.Tensor]]) -> list[th.Tensor]:
    """Collate a batch of per-side activation lists into a list of (B, d_ℓ) tensors."""
    if len(batch) == 0:
        raise ValueError("Empty batch")
    num_sides = len(batch[0])
    return [
        th.stack([sample[i] for sample in batch], dim=0) for i in range(num_sides)
    ]


def _scale_columns(weight: th.Tensor, dim: int, scale: float | None) -> th.Tensor:
    if scale is None:
        return weight
    return weight / (weight.norm(dim=dim, keepdim=True) + 1e-8) * scale


class HeteroCrossCoderEncoder(nn.Module):
    """Per-side encoder maps (B, d_ℓ) → dict_size, then sums and applies ReLU."""

    def __init__(
        self,
        activation_dims: Sequence[int],
        dict_size: int,
        norm_init_scale: float | None = 1.0,
    ):
        super().__init__()
        self.activation_dims = list(activation_dims)
        self.dict_size = dict_size
        self.num_layers = len(self.activation_dims)
        weights = []
        for d in self.activation_dims:
            w = init.kaiming_uniform_(th.empty(d, dict_size))
            w = _scale_columns(w, dim=0, scale=norm_init_scale)
            weights.append(nn.Parameter(w))
        self.weights = nn.ParameterList(weights)
        self.bias = nn.Parameter(th.zeros(dict_size))

    def forward(
        self,
        xs: list[th.Tensor],
        select_features: list[int] | None = None,
    ) -> th.Tensor:
        if len(xs) != self.num_layers:
            raise ValueError(
                f"Expected {self.num_layers} activation tensors, got {len(xs)}"
            )
        f = None
        for x, W in zip(xs, self.weights):
            w = W if select_features is None else W[:, select_features]
            contrib = x @ w
            f = contrib if f is None else f + contrib
        bias = self.bias if select_features is None else self.bias[select_features]
        return relu(f + bias)


class HeteroCrossCoderDecoder(nn.Module):
    """Per-side decoder maps shared codes (B, dict_size) → (B, d_ℓ)."""

    def __init__(
        self,
        activation_dims: Sequence[int],
        dict_size: int,
        norm_init_scale: float | None = 1.0,
    ):
        super().__init__()
        self.activation_dims = list(activation_dims)
        self.dict_size = dict_size
        self.num_layers = len(self.activation_dims)
        weights = []
        biases = []
        for d in self.activation_dims:
            w = init.kaiming_uniform_(th.empty(dict_size, d))
            w = _scale_columns(w, dim=1, scale=norm_init_scale)
            weights.append(nn.Parameter(w))
            biases.append(nn.Parameter(th.zeros(d)))
        self.weights = nn.ParameterList(weights)
        self.biases = nn.ParameterList(biases)

    def forward(
        self,
        f: th.Tensor,
        select_features: list[int] | None = None,
        add_bias: bool = True,
    ) -> list[th.Tensor]:
        outs = []
        for W, b in zip(self.weights, self.biases):
            w = W if select_features is None else W[select_features]
            x = f @ w
            if add_bias:
                x = x + b
            outs.append(x)
        return outs


class HeteroBatchTopK(Dictionary):
    """
    BatchTopK crosscoder for sides with different hidden sizes.

    Encode sums per-side linear maps into a shared ReLU code, then keeps the
    top-k (training) or thresholded (inference) latents, ranked by
    f * code_normalization (decoder-column norms). Decode is per-side.

    Args:
        activation_dims: Hidden size of each side, e.g. [2048, 4096].
        dict_size: Shared dictionary size.
        k: Target number of active latents per token (exact on a batch in train).
        norm_init_scale: Column-norm used at init for encoder and decoder.
        code_normalization: How decoder norms weight f for top-k / threshold.
        activation_means: Optional per-side mean vectors for input normalization.
        activation_stds: Optional per-side std vectors (same length as means).
        target_rms: After centering, scale each side so E[||x||^2] ≈ target_rms^2.
        keep_relative_variance: If True, one scalar scale per side (default).
            If False, z-score each neuron.
    """

    def __init__(
        self,
        activation_dims: Sequence[int],
        dict_size: int,
        k: int | th.Tensor = 100,
        norm_init_scale: float | None = 1.0,
        code_normalization: CodeNormalization | str = CodeNormalization.CROSSCODER,
        code_normalization_alpha_sae: float | None = 1.0,
        code_normalization_alpha_cc: float | None = 0.1,
        activation_means: Sequence[th.Tensor] | None = None,
        activation_stds: Sequence[th.Tensor] | None = None,
        target_rms: float | None = 1.0,
        keep_relative_variance: bool = True,
    ):
        super().__init__()
        self.activation_dims = [int(d) for d in activation_dims]
        if len(self.activation_dims) < 1:
            raise ValueError("activation_dims must be non-empty")
        self.activation_dim = self.activation_dims  # alias used in some trainer configs
        self.dict_size = int(dict_size)
        self.num_layers = len(self.activation_dims)
        self.keep_relative_variance = keep_relative_variance
        if target_rms is None:
            target_rms = 1.0
        self.register_buffer("target_rms", th.tensor(float(target_rms)))
        self.register_buffer(
            "activation_dims_buf",
            th.tensor(self.activation_dims, dtype=th.long),
        )

        if isinstance(code_normalization, str):
            code_normalization = CodeNormalization.from_string(code_normalization)
        if getattr(self, "_hub_mixin_config", None) is not None:
            self._hub_mixin_config["code_normalization"] = code_normalization.name
        if code_normalization == CodeNormalization.DECOUPLED:
            raise NotImplementedError(
                "HeteroBatchTopK does not support DECOUPLED code normalization"
            )
        self.code_normalization = code_normalization
        self.code_normalization_alpha_sae = code_normalization_alpha_sae
        self.code_normalization_alpha_cc = code_normalization_alpha_cc
        self.register_buffer(
            "code_normalization_id", th.tensor(code_normalization.value)
        )

        self.encoder = HeteroCrossCoderEncoder(
            self.activation_dims, self.dict_size, norm_init_scale=norm_init_scale
        )
        self.decoder = HeteroCrossCoderDecoder(
            self.activation_dims, self.dict_size, norm_init_scale=norm_init_scale
        )

        if not isinstance(k, th.Tensor):
            k = th.tensor(k, dtype=th.int)
        self.register_buffer("k", k)
        self.register_buffer("threshold", th.tensor(-1.0, dtype=th.float32))

        self._init_activation_normalizer(activation_means, activation_stds)

    def _init_activation_normalizer(
        self,
        activation_means: Sequence[th.Tensor] | None,
        activation_stds: Sequence[th.Tensor] | None,
    ) -> None:
        n = self.num_layers
        if activation_means is None or activation_stds is None:
            for i, d in enumerate(self.activation_dims):
                self.register_buffer(f"activation_mean_{i}", th.full((d,), th.nan))
                self.register_buffer(f"activation_std_{i}", th.full((d,), th.nan))
                self.register_buffer(f"activation_global_scale_{i}", th.ones(()))
            return
        if len(activation_means) != n or len(activation_stds) != n:
            raise ValueError(
                "activation_means and activation_stds must have one tensor per side"
            )
        for i, (mean, std, d) in enumerate(
            zip(activation_means, activation_stds, self.activation_dims)
        ):
            mean = mean.detach().float().reshape(-1)
            std = std.detach().float().reshape(-1)
            if mean.numel() != d or std.numel() != d:
                raise ValueError(
                    f"Side {i}: expected mean/std of length {d}, got {mean.numel()}/{std.numel()}"
                )
            if th.isnan(mean).any() or th.isnan(std).any():
                raise ValueError(f"Side {i}: mean/std contain NaN")
            self.register_buffer(f"activation_mean_{i}", mean)
            self.register_buffer(f"activation_std_{i}", std)
            if self.keep_relative_variance:
                total_var = (std**2).sum()
                scale = self.target_rms / th.sqrt(total_var + 1e-8)
            else:
                scale = th.ones(())
            self.register_buffer(f"activation_global_scale_{i}", scale.reshape(()))

    @property
    def has_activation_normalizer(self) -> bool:
        return all(
            not th.isnan(getattr(self, f"activation_mean_{i}")).any()
            and not th.isnan(getattr(self, f"activation_std_{i}")).any()
            for i in range(self.num_layers)
        )

    def normalize_activations(
        self, xs: list[th.Tensor], inplace: bool = False
    ) -> list[th.Tensor]:
        if not self.has_activation_normalizer:
            return xs
        out = []
        for i, x in enumerate(xs):
            if not inplace:
                x = x.clone()
            mean = getattr(self, f"activation_mean_{i}").to(dtype=x.dtype, device=x.device)
            x = x - mean
            if self.keep_relative_variance:
                scale = getattr(self, f"activation_global_scale_{i}").to(
                    dtype=x.dtype, device=x.device
                )
                x = x * scale
            else:
                std = getattr(self, f"activation_std_{i}").to(
                    dtype=x.dtype, device=x.device
                )
                x = x / (std + 1e-8)
            out.append(x)
        return out

    def denormalize_activations(
        self, xs: list[th.Tensor], inplace: bool = False
    ) -> list[th.Tensor]:
        if not self.has_activation_normalizer:
            return xs
        out = []
        for i, x in enumerate(xs):
            if not inplace:
                x = x.clone()
            mean = getattr(self, f"activation_mean_{i}").to(dtype=x.dtype, device=x.device)
            if self.keep_relative_variance:
                scale = getattr(self, f"activation_global_scale_{i}").to(
                    dtype=x.dtype, device=x.device
                )
                x = x / (scale + 1e-8)
            else:
                std = getattr(self, f"activation_std_{i}").to(
                    dtype=x.dtype, device=x.device
                )
                x = x * (std + 1e-8)
            out.append(x + mean)
        return out

    def get_code_normalization(
        self, select_features: list[int] | None = None
    ) -> th.Tensor:
        """Decoder-norm weights for ranking / L1. Shape (1, n_features) except NONE."""
        norms = []
        for W in self.decoder.weights:
            w = W if select_features is None else W[select_features]
            norms.append(w.norm(dim=-1))
        if self.code_normalization == CodeNormalization.NONE:
            return th.tensor(1.0, device=norms[0].device, dtype=norms[0].dtype)
        if self.code_normalization == CodeNormalization.CROSSCODER:
            return sum(norms).unsqueeze(0)
        if self.code_normalization == CodeNormalization.SAE:
            sq = sum(n**2 for n in norms)
            return sq.sqrt().unsqueeze(0)
        if self.code_normalization == CodeNormalization.MIXED:
            weight_norm_sae = sum(n**2 for n in norms).sqrt().unsqueeze(0)
            weight_norm_cc = sum(norms).unsqueeze(0)
            return (
                weight_norm_sae * self.code_normalization_alpha_sae
                + weight_norm_cc * self.code_normalization_alpha_cc
            )
        raise NotImplementedError(
            f"Code normalization {self.code_normalization} not implemented for HeteroBatchTopK"
        )

    def encode(
        self,
        xs: list[th.Tensor],
        return_active: bool = False,
        use_threshold: bool = True,
        select_features: list[int] | None = None,
        normalize_activations: bool = True,
        inplace_normalize: bool = False,
    ):
        if not isinstance(xs, (list, tuple)):
            raise TypeError(
                "HeteroBatchTopK.encode expects a list of tensors [(B, d_ℓ), ...], "
                f"got {type(xs)}"
            )
        if normalize_activations:
            xs = self.normalize_activations(xs, inplace=inplace_normalize)
        batch_size = xs[0].size(0)
        post_relu_f = self.encoder(xs, select_features=select_features)
        code_norm = self.get_code_normalization(select_features)
        post_relu_f_scaled = post_relu_f * code_norm
        if use_threshold:
            f = post_relu_f * (post_relu_f_scaled > self.threshold)
        else:
            flattened_acts_scaled = post_relu_f_scaled.flatten()
            post_topk = flattened_acts_scaled.topk(
                int(self.k.item()) * batch_size, sorted=False, dim=-1
            )
            post_topk_values = post_relu_f.flatten()[post_topk.indices]
            f = (
                th.zeros_like(flattened_acts_scaled)
                .scatter_(-1, post_topk.indices, post_topk_values)
                .reshape(post_relu_f.shape)
            )
        if return_active:
            return (
                f,
                f * code_norm,
                f.sum(0) > 0,
                post_relu_f,
                post_relu_f_scaled,
            )
        return f

    def decode(
        self,
        f: th.Tensor,
        denormalize_activations: bool = True,
        **kwargs,
    ) -> list[th.Tensor]:
        xs = self.decoder(f, **kwargs)
        if denormalize_activations:
            xs = self.denormalize_activations(xs, inplace=True)
        return xs

    def forward(
        self,
        xs: list[th.Tensor],
        output_features: bool = False,
        normalize_activations: bool = True,
        use_threshold: bool = True,
    ):
        f = self.encode(
            xs,
            normalize_activations=normalize_activations,
            use_threshold=use_threshold,
        )
        x_hat = self.decode(f, denormalize_activations=normalize_activations)
        if output_features:
            weight_norm = self.get_code_normalization()
            return x_hat, f * weight_norm
        return x_hat

    def get_activations(
        self,
        xs: list[th.Tensor],
        use_threshold: bool = True,
        select_features=None,
        normalize_activations: bool = True,
        inplace_normalize: bool = False,
        **kwargs,
    ):
        _, f_scaled, *_ = self.encode(
            xs,
            use_threshold=use_threshold,
            return_active=True,
            select_features=select_features,
            normalize_activations=normalize_activations,
            inplace_normalize=inplace_normalize,
            **kwargs,
        )
        return f_scaled

    @classmethod
    def from_pretrained(
        cls,
        path: str,
        dtype: th.dtype = th.float32,
        device: th.device | None = None,
        from_hub: bool = False,
        **kwargs,
    ):
        if from_hub:
            return super().from_pretrained(
                path, device=device, dtype=dtype, from_hub=True, **kwargs
            )

        state_dict = th.load(path, map_location="cpu", weights_only=True)
        if "encoder.weights.0" not in state_dict:
            if any("_orig_mod." in k for k in state_dict):
                warn(
                    "hetero state dict was saved while torch.compile was enabled. Fixing..."
                )
                state_dict = {
                    k.split("_orig_mod.")[-1] if "_orig_mod." in k else k: v
                    for k, v in state_dict.items()
                }
            else:
                raise KeyError(
                    "Could not find encoder.weights.0 in state dict. "
                    "Is this a HeteroBatchTopK checkpoint?"
                )

        activation_dims = []
        i = 0
        while f"encoder.weights.{i}" in state_dict:
            activation_dims.append(int(state_dict[f"encoder.weights.{i}"].shape[0]))
            i += 1
        dict_size = int(state_dict["encoder.weights.0"].shape[1])

        if "code_normalization" in kwargs:
            code_normalization = kwargs.pop("code_normalization")
        elif "code_normalization_id" in state_dict:
            code_normalization = CodeNormalization._value2member_map_[
                int(state_dict["code_normalization_id"].item())
            ]
        else:
            warn(
                f"No code normalization id found in {path}; assuming CROSSCODER"
            )
            code_normalization = CodeNormalization.CROSSCODER

        if "k" in kwargs:
            assert state_dict["k"] == kwargs["k"], (
                f"k in kwargs ({kwargs['k']}) does not match k in state_dict "
                f"({state_dict['k']})"
            )
            kwargs.pop("k")

        means = None
        stds = None
        if f"activation_mean_0" in state_dict and not th.isnan(
            state_dict["activation_mean_0"]
        ).any():
            means = [
                state_dict[f"activation_mean_{i}"] for i in range(len(activation_dims))
            ]
            stds = [
                state_dict[f"activation_std_{i}"] for i in range(len(activation_dims))
            ]

        target_rms = kwargs.pop("target_rms", None)
        if target_rms is None and "target_rms" in state_dict:
            target_rms = float(state_dict["target_rms"].item())

        model = cls(
            activation_dims,
            dict_size,
            k=state_dict["k"],
            code_normalization=code_normalization,
            activation_means=means,
            activation_stds=stds,
            target_rms=target_rms,
            **kwargs,
        )
        model.load_state_dict(state_dict)
        if device is not None:
            model = model.to(device)
        return model.to(dtype=dtype)


class HeteroActivationCache:
    """
    Token-aligned activation caches from N models/layers with possibly different d.

    ``__getitem__`` returns a list of 1-D tensors, one per side. Use
    ``collate_fn=hetero_collate`` (also ``HeteroActivationCache.collate_fn``)
    with a DataLoader.
    """

    def __init__(self, *store_dirs: str, submodule_name: str | None = None):
        if len(store_dirs) == 0:
            raise ValueError("Need at least one store_dir")
        self.activation_caches = [
            ActivationCache(store_dir, submodule_name) for store_dir in store_dirs
        ]
        n0 = len(self.activation_caches[0])
        for i, cache in enumerate(self.activation_caches[1:], start=1):
            if len(cache) != n0:
                raise AssertionError(
                    f"Cache {i} length {len(cache)} != cache 0 length {n0}"
                )
        tokens0 = self.activation_caches[0].tokens
        if tokens0 is not None:
            for i, cache in enumerate(self.activation_caches[1:], start=1):
                if cache.tokens is None:
                    continue
                if not th.equal(tokens0, cache.tokens):
                    raise AssertionError(f"Tokens of cache {i} do not match cache 0")

    def __len__(self) -> int:
        return len(self.activation_caches[0])

    collate_fn = staticmethod(hetero_collate)

    def __getitem__(self, index: int) -> list[th.Tensor]:
        return [cache[index] for cache in self.activation_caches]

    @property
    def tokens(self):
        return [cache.tokens for cache in self.activation_caches]

    @property
    def mean(self) -> list[th.Tensor]:
        return [cache.mean for cache in self.activation_caches]

    @property
    def std(self) -> list[th.Tensor]:
        return [cache.std for cache in self.activation_caches]

    @property
    def activation_dims(self) -> list[int]:
        return [int(cache[0].shape[-1]) for cache in self.activation_caches]
