import os

import numpy as np
import optax
import pytest
import jax
from jax import numpy as jnp

from classifier.utils import vqc_training
from classifier.utils.vqcs import scale_logits


class _DummyVQC:
    def __init__(
        self,
        N_QUBITS,
        DEPTH,
        building_block_tag,
        temperature,
        temperature_mode="multiply",
    ):
        self.temperature = temperature
        self.temperature_mode = temperature_mode
        self.params = jnp.zeros(1)

    def setup(self):
        def model(params, state):
            return jnp.concatenate([params, jnp.zeros(9)])

        model_vmap = jax.vmap(jax.jit(model), in_axes=(None, 0))
        cost_fn = lambda params, state, target: optax.softmax_cross_entropy_with_integer_labels(
            scale_logits(model(params, state), self.temperature, self.temperature_mode),
            target,
        )
        cost_fn_vmap = jax.vmap(jax.jit(cost_fn), in_axes=(None, 0, 0))
        grad_fn = jax.vmap(jax.jit(jax.grad(cost_fn)), in_axes=(None, 0, 0))
        return {
            "model_vmap": model_vmap,
            "params": self.params,
            "loss_fn": cost_fn_vmap,
            "grad_fn": grad_fn,
        }


def test_early_stopping_uses_scaled_loss(tmp_path, monkeypatch):
    states = np.zeros((20, 2), dtype=np.float32)
    labels = np.array([0, 1] * 10, dtype=np.int64)

    monkeypatch.setattr(vqc_training, "_load_dataset", lambda config: (states, labels))
    monkeypatch.setattr(vqc_training, "LinearVQC", _DummyVQC)

    scaled_sequence = iter([(1.0, 0.0, 4), (0.5, 0.0, 4)])

    def _fake_scaled_metrics(*_args, **_kwargs):
        return next(scaled_sequence)

    monkeypatch.setattr(vqc_training, "_evaluate_scaled_metrics", _fake_scaled_metrics)

    config = {
        "dataset_name": "dummy",
        "data_dir": os.fspath(tmp_path),
        "basepath": os.fspath(tmp_path),
        "trial_dir": os.fspath(tmp_path),
        "fold": 0,
        "seed": 0,
        "model_name": "LinearVQC",
        "building_block_tag": "su4",
        "depth": 1,
        "epochs": 2,
        "batch_size": 8,
        "optimizer": "adam",
        "learning_rate": 0.01,
        "temperature": 1.0,
        "temperature_mode": "multiply",
        "compression_depth": 0,
        "n_patches": 1,
    }

    training = vqc_training.TrainingVQC(config)
    _, _, summary = training.train()

    assert summary["best_epoch"] == 2
    assert summary["best_val_loss"] == 0.5
