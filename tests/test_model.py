import numpy as np
import tensorflow as tf

from src.model import build_model


def test_model_output_shape():
    model = build_model(input_dim=561, num_classes=6)
    output = model(np.zeros((4, 561), dtype=np.float32))
    assert tuple(output.shape) == (4, 6)


def test_model_outputs_probabilities():
    model = build_model(input_dim=561, num_classes=6)
    output = model(np.zeros((3, 561), dtype=np.float32)).numpy()
    assert np.all(np.isfinite(output))
    assert np.all(output >= 0.0)
    np.testing.assert_allclose(output.sum(axis=1), np.ones(3), atol=1e-5)
