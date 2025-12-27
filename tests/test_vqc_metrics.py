import unittest

import numpy as np
from jax import numpy as jnp

from classifier.utils.vqc_training import _evaluate_scaled_metrics
from classifier.utils.vqcs import LinearVQC


class TestVQCMetrics(unittest.TestCase):
    def test_corrected_eval_uses_full_val_set(self):
        n_samples = 100
        batch_size = 32
        n_batches = max(1, n_samples // batch_size)
        states = np.zeros((n_samples, 2), dtype=np.float32)
        targets = np.arange(n_samples) % 10
        states_batches = np.array_split(states, n_batches)
        targets_batches = np.array_split(targets, n_batches)

        def predict_fn(_params, batch_states):
            return jnp.zeros((batch_states.shape[0], 10))

        _, _, total_samples = _evaluate_scaled_metrics(
            predict_fn,
            None,
            states_batches,
            targets_batches,
            temperature=1.0,
        )
        self.assertEqual(total_samples, n_samples)

    def test_seed_changes_initial_params(self):
        np.random.seed(42)
        model_a = LinearVQC(N_QUBITS=5, DEPTH=1, building_block_tag="su4", temperature=1.0)
        params_a = np.asarray(model_a.params)

        np.random.seed(43)
        model_b = LinearVQC(N_QUBITS=5, DEPTH=1, building_block_tag="su4", temperature=1.0)
        params_b = np.asarray(model_b.params)

        self.assertFalse(np.allclose(params_a, params_b))


if __name__ == "__main__":
    unittest.main()
