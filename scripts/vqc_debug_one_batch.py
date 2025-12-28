#!/usr/bin/env python3
import argparse
import os
import sys

import numpy as np
import optax
import jax
from jax import numpy as jnp
from sklearn.model_selection import StratifiedKFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from classifier.utils.vqc_training import _load_dataset
from classifier.utils.vqcs import LinearVQC, NonLinearVQC


def _infer_n_qubits(states: np.ndarray) -> int:
    dim = int(states.shape[1])
    n_qubits = int(np.log2(dim))
    if 2**n_qubits != dim:
        raise ValueError(f"State dimension {dim} is not a power of two.")
    return n_qubits


def main() -> int:
    parser = argparse.ArgumentParser(description="Debug VQC gradients on a single minibatch.")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, choices=["linear", "nonlinear"], default="linear")
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--building_block_tag", type=str, default="su4")
    parser.add_argument("--temperature", type=float, default=1.0 / 128.0)
    parser.add_argument(
        "--temperature_mode",
        type=str,
        default="multiply",
        choices=["multiply", "divide"],
    )
    args = parser.parse_args()

    data_dir = args.data_dir or os.path.join(ROOT, "data")
    config = {
        "dataset_name": args.dataset,
        "data_dir": data_dir,
        "basepath": ROOT,
        "compression_depth": 0,
        "n_patches": 1,
    }
    states, labels = _load_dataset(config)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(skf.split(states, labels))
    train_idx, _ = splits[int(args.fold)]

    states_train = states[train_idx]
    targets_train = labels[train_idx]
    n_qubits = _infer_n_qubits(states_train)

    np.random.seed(args.seed)
    jax.random.PRNGKey(args.seed)

    if args.model == "linear":
        vqc = LinearVQC(
            N_QUBITS=n_qubits,
            DEPTH=args.depth,
            building_block_tag=args.building_block_tag,
            temperature=args.temperature,
            temperature_mode=args.temperature_mode,
        )
    else:
        vqc = NonLinearVQC(
            N_QUBITS=n_qubits,
            DEPTH=args.depth,
            use_initial_state=False,
            building_block_tag=args.building_block_tag,
            temperature=args.temperature,
            temperature_mode=args.temperature_mode,
        )

    model = vqc.setup()
    params = model["params"]

    batch_states = jnp.asarray(states_train[: args.batch_size])
    batch_targets = jnp.asarray(targets_train[: args.batch_size])

    def _loss(params):
        return jnp.mean(model["loss_fn"](params, batch_states, batch_targets))

    grad_per_sample = model["grad_fn"](params, batch_states, batch_targets)
    grad = jnp.mean(grad_per_sample, axis=0)

    n_network = int(getattr(vqc, "N_PARAMS_NETWORK"))
    n_last = int(getattr(vqc, "N_LAST_LINEAR"))
    grad_network = grad[:n_network]
    grad_last = grad[-n_last:]

    optimizer = optax.adam(args.lr)
    opt_state = optimizer.init(params)
    updates, opt_state = optimizer.update(grad, opt_state, params)
    params_next = optax.apply_updates(params, updates)

    update_network = params_next[:n_network] - params[:n_network]
    update_last = params_next[-n_last:] - params[-n_last:]

    print("Gradient norms")
    print(f"  circuit_params_l2: {float(jnp.linalg.norm(grad_network)):.6e}")
    print(f"  last_linear_l2:    {float(jnp.linalg.norm(grad_last)):.6e}")
    print("Update norms (1 step)")
    print(f"  circuit_update_l2: {float(jnp.linalg.norm(update_network)):.6e}")
    print(f"  last_update_l2:    {float(jnp.linalg.norm(update_last)):.6e}")

    losses = [float(_loss(params))]
    params_step = params
    opt_state = optimizer.init(params_step)
    for _ in range(args.steps):
        grad_per_sample = model["grad_fn"](params_step, batch_states, batch_targets)
        grad = jnp.mean(grad_per_sample, axis=0)
        updates, opt_state = optimizer.update(grad, opt_state, params_step)
        params_step = optax.apply_updates(params_step, updates)
        losses.append(float(_loss(params_step)))

    print("Losses on fixed minibatch")
    print("  " + ", ".join(f"{loss:.6f}" for loss in losses))
    print(f"  loss_decreased: {losses[-1] < losses[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
