import numpy as np
import pytest

from classifier.utils.vqcs import _su4_pauli_generator_unitary, scale_logits


def test_su4_pauli_generator_unitary():
    rng = np.random.default_rng(0)
    params = rng.normal(size=15)
    unitary = np.array(_su4_pauli_generator_unitary(params))
    identity = unitary.conj().T @ unitary
    assert np.allclose(identity, np.eye(4), atol=1e-6)


def test_su4_pauli_generator_param_count():
    with pytest.raises(ValueError):
        _su4_pauli_generator_unitary(np.zeros(14))


def test_temperature_scale_modes():
    logits = np.array([1.5, -0.5])
    temperature = 0.25
    assert np.allclose(scale_logits(logits, temperature, "multiply"), logits * temperature)
    assert np.allclose(scale_logits(logits, temperature, "divide"), logits / temperature)
