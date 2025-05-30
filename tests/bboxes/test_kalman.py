# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import numpy as np

import cv2ext
from cv2ext.bboxes._kalman import (
    kalman_init,
    kalman_predict,
    kalman_update,
    kalman_get_bbox,
    kalman_predict_bbox,
    kalman_update_bbox,
    _bbox_to_state,
    _state_to_bbox,
)

from ..helpers import wrapper, wrapper_jit


@wrapper
def test_bbox_to_state_conversion():
    """Test conversion from bbox to state vector."""
    bbox = (100, 50, 200, 150)
    state = _bbox_to_state(bbox)
    
    # Check shape and data type
    assert state.shape == (8,)
    assert state.dtype == np.float64
    
    # Check values: [cx, cy, w, h, vx, vy, vw, vh]
    assert state[0] == 150.0  # cx = (100 + 200) / 2
    assert state[1] == 100.0  # cy = (50 + 150) / 2
    assert state[2] == 100.0  # w = 200 - 100
    assert state[3] == 100.0  # h = 150 - 50
    assert state[4] == 0.0    # vx initialized to 0
    assert state[5] == 0.0    # vy initialized to 0
    assert state[6] == 0.0    # vw initialized to 0
    assert state[7] == 0.0    # vh initialized to 0


@wrapper
def test_state_to_bbox_conversion():
    """Test conversion from state vector to bbox."""
    state = np.array([150.0, 100.0, 100.0, 100.0, 5.0, 3.0, 0.0, 0.0], dtype=np.float64)
    bbox = _state_to_bbox(state)
    
    # Check conversion back to bbox
    assert bbox == (100, 50, 200, 150)


@wrapper
def test_bbox_state_roundtrip():
    """Test that bbox -> state -> bbox conversion is consistent."""
    original_bbox = (75, 25, 175, 125)
    state = _bbox_to_state(original_bbox)
    converted_bbox = _state_to_bbox(state)
    
    assert converted_bbox == original_bbox


@wrapper
def test_kalman_init():
    """Test Kalman filter initialization."""
    bbox = (100, 50, 200, 150)
    state, covariance = kalman_init(bbox)
    
    # Check state initialization
    assert state.shape == (8,)
    assert state.dtype == np.float64
    
    # Check covariance initialization
    assert covariance.shape == (8, 8)
    assert covariance.dtype == np.float64
    
    # Check that covariance is positive definite (all eigenvalues > 0)
    eigenvals = np.linalg.eigvals(covariance)
    assert np.all(eigenvals > 0)


@wrapper
def test_kalman_get_bbox():
    """Test extracting bbox from state."""
    bbox = (100, 50, 200, 150)
    state, _ = kalman_init(bbox)
    extracted_bbox = kalman_get_bbox(state)
    
    assert extracted_bbox == bbox


@wrapper
def test_kalman_predict_basic():
    """Test basic Kalman prediction."""
    bbox = (100, 50, 200, 150)
    state, covariance = kalman_init(bbox)
    
    # First prediction should be same as initial (no velocity)
    state_pred, covariance_pred = kalman_predict(state, covariance)
    
    assert state_pred.shape == (8,)
    assert covariance_pred.shape == (8, 8)
    
    # Position should be same (no initial velocity)
    assert state_pred[0] == state[0]  # cx
    assert state_pred[1] == state[1]  # cy
    assert state_pred[2] == state[2]  # w
    assert state_pred[3] == state[3]  # h
    
    # Velocities should still be zero
    assert state_pred[4] == 0.0  # vx
    assert state_pred[5] == 0.0  # vy
    assert state_pred[6] == 0.0  # vw
    assert state_pred[7] == 0.0  # vh


@wrapper
def test_kalman_predict_with_velocity():
    """Test Kalman prediction with existing velocity."""
    bbox = (100, 50, 200, 150)
    state, covariance = kalman_init(bbox)
    
    # Add some velocity
    state[4] = 5.0  # vx = 5 pixels per frame
    state[5] = 3.0  # vy = 3 pixels per frame
    
    state_pred, covariance_pred = kalman_predict(state, covariance)
    
    # Position should have moved by velocity
    assert state_pred[0] == 150.0 + 5.0  # cx + vx
    assert state_pred[1] == 100.0 + 3.0  # cy + vy
    assert state_pred[2] == 100.0        # w unchanged
    assert state_pred[3] == 100.0        # h unchanged
    
    # Velocities should be preserved
    assert state_pred[4] == 5.0
    assert state_pred[5] == 3.0


@wrapper
def test_kalman_update():
    """Test Kalman update step."""
    bbox = (100, 50, 200, 150)
    state, covariance = kalman_init(bbox)
    state_pred, covariance_pred = kalman_predict(state, covariance)
    
    # Create a measurement (slightly different position)
    measurement_bbox = (105, 55, 205, 155)
    measurement = _bbox_to_state(measurement_bbox)[:4]  # Only position and size
    
    state_updated, covariance_updated = kalman_update(
        state_pred, covariance_pred, measurement
    )
    
    assert state_updated.shape == (8,)
    assert covariance_updated.shape == (8, 8)
    
    # State should be between prediction and measurement
    updated_bbox = kalman_get_bbox(state_updated)
    assert updated_bbox[0] >= 100  # x1 between original and measurement
    assert updated_bbox[0] <= 105
    assert updated_bbox[1] >= 50   # y1 between original and measurement  
    assert updated_bbox[1] <= 55


@wrapper
def test_kalman_predict_bbox():
    """Test high-level predict bbox function."""
    bbox = (100, 50, 200, 150)
    state, covariance = kalman_init(bbox)
    
    predicted_bbox, state_pred, covariance_pred = kalman_predict_bbox(
        bbox, state, covariance
    )
    
    assert isinstance(predicted_bbox, tuple)
    assert len(predicted_bbox) == 4
    assert state_pred.shape == (8,)
    assert covariance_pred.shape == (8, 8)


@wrapper
def test_kalman_update_bbox():
    """Test high-level update bbox function."""
    bbox = (100, 50, 200, 150)
    state, covariance = kalman_init(bbox)
    state_pred, covariance_pred = kalman_predict(state, covariance)
    
    measurement_bbox = (105, 55, 205, 155)
    
    updated_bbox, state_updated, covariance_updated = kalman_update_bbox(
        measurement_bbox, state_pred, covariance_pred
    )
    
    assert isinstance(updated_bbox, tuple)
    assert len(updated_bbox) == 4
    assert state_updated.shape == (8,)
    assert covariance_updated.shape == (8, 8)


@wrapper
def test_tracking_simulation():
    """Test a complete tracking simulation over multiple frames."""
    # Initial bbox
    bbox = (100, 50, 200, 150)
    state, covariance = kalman_init(bbox)
    
    # Simulate object moving right and down
    measurements = [
        (105, 55, 205, 155),
        (110, 60, 210, 160),
        (115, 65, 215, 165),
    ]
    
    for measurement_bbox in measurements:
        # Predict
        state_pred, covariance_pred = kalman_predict(state, covariance)
        
        # Update with measurement
        measurement = _bbox_to_state(measurement_bbox)[:4]
        state, covariance = kalman_update(state_pred, covariance_pred, measurement)
    
    # After tracking, should have learned the velocity
    assert state[4] > 0  # Positive vx (moving right)
    assert state[5] > 0  # Positive vy (moving down)
    
    # Final position should be close to last measurement
    final_bbox = kalman_get_bbox(state)
    assert abs(final_bbox[0] - 115) <= 5  # Within 5 pixels
    assert abs(final_bbox[1] - 65) <= 5


@wrapper
def test_noise_parameters():
    """Test that noise parameters affect the prediction."""
    bbox = (100, 50, 200, 150)
    state, covariance = kalman_init(bbox)
    
    # Predict with different noise levels
    _, cov_low_noise = kalman_predict(state, covariance, pos_noise=0.1)
    _, cov_high_noise = kalman_predict(state, covariance, pos_noise=10.0)
    
    # Higher noise should result in higher uncertainty
    assert np.trace(cov_high_noise) > np.trace(cov_low_noise)


@wrapper
def test_edge_cases():
    """Test edge cases and boundary conditions."""
    # Very small bbox
    small_bbox = (0, 0, 1, 1)
    state, covariance = kalman_init(small_bbox)
    state_pred, covariance_pred = kalman_predict(state, covariance)
    
    assert kalman_get_bbox(state_pred) == small_bbox
    
    # Large bbox
    large_bbox = (0, 0, 1000, 1000)
    state, covariance = kalman_init(large_bbox)
    state_pred, covariance_pred = kalman_predict(state, covariance)
    
    assert kalman_get_bbox(state_pred) == large_bbox


# JIT tests
@wrapper_jit
def test_kalman_init_jit():
    """Test Kalman filter initialization with JIT."""
    bbox = (100, 50, 200, 150)
    state, covariance = kalman_init(bbox)
    
    assert state.shape == (8,)
    assert covariance.shape == (8, 8)


@wrapper_jit
def test_kalman_predict_jit():
    """Test Kalman prediction with JIT."""
    bbox = (100, 50, 200, 150)
    state, covariance = kalman_init(bbox)
    
    state_pred, covariance_pred = kalman_predict(state, covariance)
    
    assert state_pred.shape == (8,)
    assert covariance_pred.shape == (8, 8)


@wrapper_jit
def test_kalman_update_jit():
    """Test Kalman update with JIT."""
    bbox = (100, 50, 200, 150)
    state, covariance = kalman_init(bbox)
    state_pred, covariance_pred = kalman_predict(state, covariance)
    
    measurement_bbox = (105, 55, 205, 155)
    measurement = _bbox_to_state(measurement_bbox)[:4]
    
    state_updated, covariance_updated = kalman_update(
        state_pred, covariance_pred, measurement
    )
    
    assert state_updated.shape == (8,)
    assert covariance_updated.shape == (8, 8)


@wrapper_jit 
def test_tracking_simulation_jit():
    """Test complete tracking simulation with JIT."""
    bbox = (100, 50, 200, 150)
    state, covariance = kalman_init(bbox)
    
    measurements = [
        (105, 55, 205, 155),
        (110, 60, 210, 160),
        (115, 65, 215, 165),
    ]
    
    for measurement_bbox in measurements:
        state_pred, covariance_pred = kalman_predict(state, covariance)
        measurement = _bbox_to_state(measurement_bbox)[:4]
        state, covariance = kalman_update(state_pred, covariance_pred, measurement)
    
    # Should have learned positive velocity
    assert state[4] > 0  # vx
    assert state[5] > 0  # vy
