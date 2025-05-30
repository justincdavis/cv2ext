# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import numpy as np

from cv2ext.bboxes._kalman import (
    kalman_get_bbox,
    kalman_init,
    kalman_predict,
    kalman_update,
)

from ..helpers import wrapper, wrapper_jit


@wrapper
def test_kalman_init_bbox():
    """Test kalman_init with a bounding box tuple."""
    bbox = (100, 50, 200, 150)
    state, covariance = kalman_init(bbox)
    
    assert state.shape == (8,)
    assert covariance.shape == (8, 8)

    expected_cx = (100 + 200) / 2.0  # 150
    expected_cy = (50 + 150) / 2.0   # 100
    expected_w = 200 - 100           # 100
    expected_h = 150 - 50            # 100
    
    assert abs(state[0] - expected_cx) < 1e-6
    assert abs(state[1] - expected_cy) < 1e-6
    assert abs(state[2] - expected_w) < 1e-6
    assert abs(state[3] - expected_h) < 1e-6

    assert abs(state[4]) < 1e-6
    assert abs(state[5]) < 1e-6
    assert abs(state[6]) < 1e-6
    assert abs(state[7]) < 1e-6


@wrapper
def test_kalman_init_state_vector():
    """Test kalman_init with a state vector."""
    state_vec = np.array([150.0, 100.0, 100.0, 100.0, 5.0, 3.0, 1.0, 0.5], dtype=np.float32)
    state, covariance = kalman_init(state_vec)
    
    assert state.shape == (8,)
    assert covariance.shape == (8, 8)
    assert np.allclose(state, state_vec)


@wrapper
def test_kalman_predict_simple():
    """Test kalman_predict with basic prediction."""
    bbox = (100, 50, 200, 150)
    state, covariance = kalman_init(bbox)
    
    state_pred, covariance_pred = kalman_predict(state, covariance)
    
    assert state_pred.shape == (8,)
    assert covariance_pred.shape == (8, 8)

    assert abs(state_pred[0] - state[0]) < 1e-6
    assert abs(state_pred[1] - state[1]) < 1e-6
    assert abs(state_pred[2] - state[2]) < 1e-6
    assert abs(state_pred[3] - state[3]) < 1e-6


@wrapper
def test_kalman_predict_from_bbox():
    """Test kalman_predict using bbox directly."""
    bbox = (100, 50, 200, 150)
    _, covariance = kalman_init(bbox)

    state_pred, covariance_pred = kalman_predict(bbox, covariance)
    
    assert state_pred.shape == (8,)
    assert covariance_pred.shape == (8, 8)


@wrapper
def test_kalman_predict_with_velocity():
    """Test kalman_predict with non-zero velocities."""
    state = np.array([150.0, 100.0, 100.0, 100.0, 5.0, 3.0, 1.0, 0.5], dtype=np.float32)
    covariance = np.eye(8, dtype=np.float32) * 1000.0
    
    state_pred, covariance_pred = kalman_predict(state, covariance)
    
    assert abs(state_pred[0] - 155.0) < 1e-6
    assert abs(state_pred[1] - 103.0) < 1e-6
    assert abs(state_pred[2] - 101.0) < 1e-6
    assert abs(state_pred[3] - 100.5) < 1e-6

    assert abs(state_pred[4] - 5.0) < 1e-6
    assert abs(state_pred[5] - 3.0) < 1e-6
    assert abs(state_pred[6] - 1.0) < 1e-6
    assert abs(state_pred[7] - 0.5) < 1e-6


@wrapper
def test_kalman_update():
    """Test kalman_update with measurement."""
    bbox = (100, 50, 200, 150)
    state, covariance = kalman_init(bbox)
    state_pred, covariance_pred = kalman_predict(state, covariance)
    
    measurement = (105, 55, 205, 155)
    state_updated, covariance_updated = kalman_update(state_pred, covariance_pred, measurement)
    
    assert state_updated.shape == (8,)
    assert covariance_updated.shape == (8, 8)

    assert not np.allclose(state_updated, state_pred)


@wrapper
def test_kalman_update_with_measurement_vector():
    """Test kalman_update with measurement vector."""
    bbox = (100, 50, 200, 150)
    state, covariance = kalman_init(bbox)
    state_pred, covariance_pred = kalman_predict(state, covariance)

    measurement_vec = np.array([152.5, 105.0, 100.0, 100.0], dtype=np.float32)
    state_updated, covariance_updated = kalman_update(state_pred, covariance_pred, measurement_vec)
    
    assert state_updated.shape == (8,)
    assert covariance_updated.shape == (8, 8)


@wrapper
def test_kalman_get_bbox():
    """Test kalman_get_bbox function."""
    bbox = (100, 50, 200, 150)
    state, _ = kalman_init(bbox)

    extracted_bbox = kalman_get_bbox(state)
    
    assert isinstance(extracted_bbox, tuple)
    assert len(extracted_bbox) == 4
    assert all(isinstance(x, int) for x in extracted_bbox)
    
    assert abs(extracted_bbox[0] - bbox[0]) <= 1
    assert abs(extracted_bbox[1] - bbox[1]) <= 1
    assert abs(extracted_bbox[2] - bbox[2]) <= 1
    assert abs(extracted_bbox[3] - bbox[3]) <= 1


@wrapper
def test_kalman_integration():
    """Test full integration of init, predict, update cycle."""
    initial_bbox = (100, 50, 200, 150)
    
    state, covariance = kalman_init(initial_bbox)
    
    state_pred, covariance_pred = kalman_predict(state, covariance)
    
    measurement = (102, 52, 202, 152)
    state_updated, covariance_updated = kalman_update(state_pred, covariance_pred, measurement)
    
    final_bbox = kalman_get_bbox(state_updated)

    assert isinstance(final_bbox, tuple)
    assert len(final_bbox) == 4
    assert final_bbox[0] < final_bbox[2]
    assert final_bbox[1] < final_bbox[3]


@wrapper_jit
def test_kalman_init_bbox_jit():
    """Test kalman_init with a bounding box tuple (JIT version)."""
    bbox = (100, 50, 200, 150)
    state, covariance = kalman_init(bbox)
    
    assert state.shape == (8,)
    assert covariance.shape == (8, 8)
    
    expected_cx = (100 + 200) / 2.0  # 150
    expected_cy = (50 + 150) / 2.0   # 100
    expected_w = 200 - 100           # 100
    expected_h = 150 - 50            # 100
    
    assert abs(state[0] - expected_cx) < 1e-6
    assert abs(state[1] - expected_cy) < 1e-6
    assert abs(state[2] - expected_w) < 1e-6
    assert abs(state[3] - expected_h) < 1e-6


@wrapper_jit
def test_kalman_predict_simple_jit():
    """Test kalman_predict with basic prediction (JIT version)."""
    bbox = (100, 50, 200, 150)
    state, covariance = kalman_init(bbox)
    
    state_pred, covariance_pred = kalman_predict(state, covariance)
    
    assert state_pred.shape == (8,)
    assert covariance_pred.shape == (8, 8)
    
    assert abs(state_pred[0] - state[0]) < 1e-6
    assert abs(state_pred[1] - state[1]) < 1e-6
    assert abs(state_pred[2] - state[2]) < 1e-6
    assert abs(state_pred[3] - state[3]) < 1e-6


@wrapper_jit
def test_kalman_update_jit():
    """Test kalman_update with measurement (JIT version)."""
    bbox = (100, 50, 200, 150)
    state, covariance = kalman_init(bbox)
    state_pred, covariance_pred = kalman_predict(state, covariance)
    
    measurement = (105, 55, 205, 155)
    state_updated, covariance_updated = kalman_update(state_pred, covariance_pred, measurement)
    
    assert state_updated.shape == (8,)
    assert covariance_updated.shape == (8, 8)

    assert not np.allclose(state_updated, state_pred)


@wrapper_jit
def test_kalman_integration_jit():
    """Test full integration of init, predict, update cycle (JIT version)."""
    initial_bbox = (100, 50, 200, 150)
    
    state, covariance = kalman_init(initial_bbox)

    state_pred, covariance_pred = kalman_predict(state, covariance)

    measurement = (102, 52, 202, 152)
    state_updated, _ = kalman_update(state_pred, covariance_pred, measurement)
    
    final_bbox = kalman_get_bbox(state_updated)

    assert isinstance(final_bbox, tuple)
    assert len(final_bbox) == 4
    assert final_bbox[0] < final_bbox[2]
    assert final_bbox[1] < final_bbox[3]

