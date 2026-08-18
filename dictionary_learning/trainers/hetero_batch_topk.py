"""Trainer for HeteroBatchTopK crosscoders with per-side hidden sizes."""

from collections import namedtuple
from typing import Optional, Sequence

import torch as th

from ..hetero import HeteroBatchTopK
from ..trainers.trainer import SAETrainer, get_lr_schedule


class HeteroBatchTopKTrainer(SAETrainer):
    """
    BatchTopK trainer for HeteroBatchTopK.

    Activations are a list of tensors ``[(B, d0), (B, d1), ...]``. Reconstruction
    loss is the unweighted mean of per-side L2, so a wider residual cannot dominate
    simply by having more coordinates. Prefer also passing per-side mean/std so
    each side is scaled to the same ``target_rms``.
    """

    def __init__(
        self,
        steps: int,
        activation_dims: Sequence[int],
        dict_size: int,
        k: int,
        layer: int,
        lm_name: str,
        k_max: Optional[int] = None,
        k_annealing_steps: int = 0,
        dict_class: type = HeteroBatchTopK,
        lr: Optional[float] = None,
        auxk_alpha: float = 1 / 32,
        warmup_steps: int = 1000,
        decay_start: Optional[int] = None,
        threshold_beta: float = 0.999,
        threshold_start_step: int = 1000,
        seed: Optional[int] = None,
        device: Optional[str] = None,
        wandb_name: str = "HeteroBatchTopK",
        submodule_name: Optional[str] = None,
        pretrained_ae: Optional[HeteroBatchTopK] = None,
        dict_class_kwargs: dict | None = None,
        activation_means: Optional[Sequence[th.Tensor]] = None,
        activation_stds: Optional[Sequence[th.Tensor]] = None,
        target_rms: float = 1.0,
        recon_normalize_by_sqrt_d: bool = True,
    ):
        super().__init__(seed)
        assert layer is not None and lm_name is not None
        self.layer = layer
        self.lm_name = lm_name
        self.submodule_name = submodule_name
        self.wandb_name = wandb_name
        self.steps = steps
        self.decay_start = decay_start
        self.warmup_steps = warmup_steps
        self.recon_normalize_by_sqrt_d = recon_normalize_by_sqrt_d

        self.k_target = k
        self.k_initial = k_max if k_max is not None else k
        self.k_annealing_total_steps = k_annealing_steps
        self.threshold_beta = threshold_beta
        self.threshold_start_step = threshold_start_step
        self.target_rms = target_rms
        self.activation_dims = [int(d) for d in activation_dims]
        dict_class_kwargs = dict_class_kwargs or {}

        if seed is not None:
            th.manual_seed(seed)
            th.cuda.manual_seed_all(seed)

        if pretrained_ae is None:
            self.ae = dict_class(
                self.activation_dims,
                dict_size,
                self.k_initial,
                activation_means=activation_means,
                activation_stds=activation_stds,
                target_rms=target_rms,
                **dict_class_kwargs,
            )
        else:
            self.ae = pretrained_ae

        if device is None:
            self.device = "cuda" if th.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.ae.to(self.device)

        if lr is not None:
            self.lr = lr
        else:
            scale = dict_size / (2**14)
            self.lr = 2e-4 / scale**0.5

        self.auxk_alpha = auxk_alpha
        self.dead_feature_threshold = 10_000_000
        self.top_k_aux = max(self.activation_dims) // 2
        self.num_tokens_since_fired = th.zeros(dict_size, dtype=th.long, device=self.device)
        self.logging_parameters = [
            "effective_l0",
            "running_deads",
            "pre_norm_auxk_loss",
            "k_current_value",
        ]
        self.dict_class_kwargs = dict_class_kwargs
        self.effective_l0 = -1
        self.running_deads = -1
        self.pre_norm_auxk_loss = -1
        self.k_current_value = self.k_initial

        self.optimizer = th.optim.Adam(
            self.ae.parameters(), lr=self.lr, betas=(0.9, 0.999)
        )
        lr_fn = get_lr_schedule(steps, warmup_steps, decay_start=decay_start)
        self.scheduler = th.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_fn)

    def _maybe_anneal_k(self, step: int | None) -> None:
        if step is None:
            return
        if self.k_annealing_total_steps > 0 and self.k_initial != self.k_target:
            if step < self.k_annealing_total_steps:
                progress = float(step) / self.k_annealing_total_steps
                current_k_float = (
                    self.k_initial - (self.k_initial - self.k_target) * progress
                )
                new_k_val = max(1, int(round(current_k_float)))
                if self.ae.k.item() != new_k_val:
                    self.ae.k.fill_(new_k_val)
            elif self.ae.k.item() != self.k_target:
                self.ae.k.fill_(self.k_target)
        elif self.k_annealing_total_steps == 0 and self.ae.k.item() != self.k_initial:
            self.ae.k.fill_(self.k_initial)
        self.k_current_value = self.ae.k.item()

    def _per_side_l2(self, residuals: list[th.Tensor]) -> th.Tensor:
        terms = []
        for e in residuals:
            l2 = th.linalg.norm(e, dim=-1).mean()
            if self.recon_normalize_by_sqrt_d:
                l2 = l2 / (e.shape[-1] ** 0.5)
            terms.append(l2)
        return sum(terms) / len(terms)

    def get_auxiliary_loss(
        self,
        residuals: list[th.Tensor],
        post_relu_f: th.Tensor,
        post_relu_f_scaled: th.Tensor,
    ):
        dead_features = self.num_tokens_since_fired >= self.dead_feature_threshold
        self.running_deads = int(dead_features.sum())
        if dead_features.sum() == 0:
            self.pre_norm_auxk_loss = -1
            return th.tensor(0.0, dtype=residuals[0].dtype, device=residuals[0].device)

        k_aux = min(self.top_k_aux, int(dead_features.sum()))
        auxk_latents_scaled = th.where(
            dead_features[None], post_relu_f_scaled, -th.inf
        ).detach()
        _, auxk_indices = auxk_latents_scaled.topk(k_aux, sorted=False)
        auxk_buffer_BF = th.zeros_like(post_relu_f)
        row_indices = (
            th.arange(post_relu_f.size(0), device=post_relu_f.device)
            .view(-1, 1)
            .expand(-1, auxk_indices.size(1))
        )
        auxk_acts_BF = auxk_buffer_BF.scatter_(
            dim=-1, index=auxk_indices, src=post_relu_f[row_indices, auxk_indices]
        )
        x_reconstruct_aux = self.ae.decoder(auxk_acts_BF, add_bias=False)

        aux_terms = []
        denom_terms = []
        for e, x_aux in zip(residuals, x_reconstruct_aux):
            e_flat = e.reshape(e.size(0), -1).float()
            x_aux_flat = x_aux.reshape(e.size(0), -1).float()
            l2_aux = (e_flat - x_aux_flat).pow(2).sum(dim=-1).mean()
            residual_mu = e_flat.mean(dim=0, keepdim=True)
            denom = (e_flat - residual_mu).pow(2).sum(dim=-1).mean()
            aux_terms.append(l2_aux)
            denom_terms.append(denom)

        l2_loss_aux = sum(aux_terms) / len(aux_terms)
        self.pre_norm_auxk_loss = l2_loss_aux.detach()
        loss_denom = sum(denom_terms) / len(denom_terms)
        return (l2_loss_aux / (loss_denom + 1e-8)).nan_to_num(0.0)

    def update_threshold(self, f_scaled: th.Tensor):
        active = f_scaled[f_scaled > 0]
        if active.size(0) == 0:
            min_activation = 0.0
        else:
            min_activation = active.min().detach().to(dtype=th.float32)
        if self.ae.threshold < 0:
            self.ae.threshold = min_activation
        else:
            self.ae.threshold = (self.threshold_beta * self.ae.threshold) + (
                (1 - self.threshold_beta) * min_activation
            )

    def loss(
        self,
        xs,
        step=None,
        logging=False,
        use_threshold=False,
        normalize_activations=True,
        inplace_normalize=True,
        **kwargs,
    ):
        xs = (
            self.ae.normalize_activations(xs, inplace=inplace_normalize)
            if normalize_activations
            else xs
        )
        self._maybe_anneal_k(step)

        f, f_scaled, active_indices_F, post_relu_f, post_relu_f_scaled = self.ae.encode(
            xs,
            return_active=True,
            use_threshold=use_threshold,
            normalize_activations=False,
        )

        if step is not None and step > self.threshold_start_step and not logging:
            self.update_threshold(f_scaled)

        x_hat = self.ae.decode(f, denormalize_activations=False)
        residuals = [x - xh for x, xh in zip(xs, x_hat)]

        self.effective_l0 = self.ae.k.item()
        num_tokens_in_step = xs[0].size(0)
        did_fire = th.zeros_like(self.num_tokens_since_fired, dtype=th.bool)
        did_fire[active_indices_F] = True
        self.num_tokens_since_fired += num_tokens_in_step
        self.num_tokens_since_fired[did_fire] = 0

        l2_loss = self._per_side_l2(residuals)
        mse_loss = sum(e.pow(2).sum(dim=-1).mean() for e in residuals) / len(residuals)
        auxk_loss = self.get_auxiliary_loss(
            [e.detach() for e in residuals], post_relu_f, post_relu_f_scaled
        )
        loss = l2_loss + self.auxk_alpha * auxk_loss

        if not logging:
            return loss

        log = {
            "mse_loss": mse_loss.item(),
            "l2_loss": l2_loss.item(),
            "auxk_loss": auxk_loss.item()
            if isinstance(auxk_loss, th.Tensor)
            else float(auxk_loss),
            "loss": loss.item(),
            "deads": ~did_fire,
            "threshold": self.ae.threshold.item(),
            "sparsity_weight": self.ae.get_code_normalization().mean().item(),
        }
        for i, e in enumerate(residuals):
            log[f"l2_loss_l{i}"] = th.linalg.norm(e, dim=-1).mean().item()
        return namedtuple("LossLog", ["x", "x_hat", "f", "losses"])(xs, x_hat, f, log)

    def update(self, step, xs):
        xs = [x.to(self.device) for x in xs]
        xs = self.ae.normalize_activations(xs, inplace=True)
        if step == 0:
            for i, x in enumerate(xs):
                median = self.geometric_median(x).to(self.device)
                self.ae.decoder.biases[i].data.copy_(median)
        loss = self.loss(xs, step=step, normalize_activations=False)
        loss.backward()
        th.nn.utils.clip_grad_norm_(self.ae.parameters(), 1.0)
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()
        return loss.item()

    @property
    def config(self):
        return {
            "trainer_class": "HeteroBatchTopKTrainer",
            "dict_class": "HeteroBatchTopK",
            "lr": self.lr,
            "steps": self.steps,
            "auxk_alpha": self.auxk_alpha,
            "warmup_steps": self.warmup_steps,
            "decay_start": self.decay_start,
            "threshold_beta": self.threshold_beta,
            "threshold_start_step": self.threshold_start_step,
            "top_k_aux": self.top_k_aux,
            "seed": self.seed,
            "activation_dims": self.activation_dims,
            "dict_size": self.ae.dict_size,
            "k": self.ae.k.item(),
            "k_target": self.k_target,
            "k_initial": self.k_initial,
            "k_annealing_steps": self.k_annealing_total_steps,
            "code_normalization": str(self.ae.code_normalization),
            "code_normalization_alpha_sae": self.ae.code_normalization_alpha_sae,
            "code_normalization_alpha_cc": self.ae.code_normalization_alpha_cc,
            "device": self.device,
            "layer": self.layer,
            "lm_name": self.lm_name,
            "wandb_name": self.wandb_name,
            "submodule_name": self.submodule_name,
            "dict_class_kwargs": {k: str(v) for k, v in self.dict_class_kwargs.items()},
            "target_rms": self.target_rms,
            "recon_normalize_by_sqrt_d": self.recon_normalize_by_sqrt_d,
        }

    @staticmethod
    def geometric_median(points: th.Tensor, max_iter: int = 100, tol: float = 1e-5):
        """Geometric median over a batch of vectors, shape (B, d) → (d,)."""
        guess = points.mean(dim=0)
        prev = th.zeros_like(guess)
        for _ in range(max_iter):
            prev = guess
            dist = th.norm(points - guess, dim=-1, keepdim=True).clamp_min(1e-8)
            weights = 1 / dist
            weights = weights / weights.sum(dim=0, keepdim=True)
            guess = (weights * points).sum(dim=0)
            if th.norm(guess - prev) < tol:
                break
        return guess
