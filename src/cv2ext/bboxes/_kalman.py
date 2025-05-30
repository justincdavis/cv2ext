# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Kalman filter implementation for bounding box tracking.

This module provides a non-class-based Kalman filter implementation specifically
designed for tracking bounding boxes represented as tuple[int, int, int, int].

The filter uses an 8-dimensional state vector: [cx, cy, w, h, vx, vy, vw, vh]
where:
- cx, cy: center coordinates
- w, h: width and height
- vx, vy: velocity of center
- vw, vh: velocity of size change

Core Functions
--------------
kalman_predict(state, covariance, **kwargs) -> (state_pred, covariance_pred)
    Predicts the next state using the constant velocity motion model.

kalman_update(state_pred, covariance_pred, measurement, **kwargs) -> (state, covariance)
    Updates the state with a new measurement.

Usage Example
-------------
>>> # Initialize from bounding box
>>> bbox = (100, 50, 200, 150)  # (x1, y1, x2, y2)
>>> state, covariance = kalman_init(bbox)
>>> # Predict next state
>>> state_pred, covariance_pred = kalman_predict(state, covariance)
>>> # Update with measurement
>>> measurement = _bbox_to_state(new_bbox)[:4]  # [cx, cy, w, h]
>>> state, covariance = kalman_update(state_pred, covariance_pred, measurement)
>>> # Get bounding box from state
>>> updated_bbox = kalman_get_bbox(state)
"""

from __future__ import annotations

import numpy as np

from cv2ext._jit import register_jit


@register_jit(nogil=True, inline="always")
def _bbox_to_state(bbox: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0  # center x
    cy = (y1 + y2) / 2.0  # center y
    w = float(x2 - x1)  # width
    h = float(y2 - y1)  # height
    # Initialize velocities to zero
    return np.array([cx, cy, w, h, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)


@register_jit(nogil=True, inline="always")
def _state_to_bbox(state: np.ndarray) -> tuple[int, int, int, int]:
    cx, cy, w, h = state[0], state[1], state[2], state[3]
    x1 = int(cx - w / 2.0)
    y1 = int(cy - h / 2.0)
    x2 = int(cx + w / 2.0)
    y2 = int(cy + h / 2.0)
    return x1, y1, x2, y2


@register_jit(nogil=True, inline="always")
def _create_transition_matrix() -> np.ndarray:
    # State: [cx, cy, w, h, vx, vy, vw, vh]
    # Next state: [cx + vx, cy + vy, w + vw, h + vh, vx, vy, vw, vh]
    f = np.eye(8, dtype=np.float32)
    f[0, 4] = 1.0  # cx += vx
    f[1, 5] = 1.0  # cy += vy
    f[2, 6] = 1.0  # w += vw
    f[3, 7] = 1.0  # h += vh
    return f


@register_jit(nogil=True, inline="always")
def _create_observation_matrix() -> np.ndarray:
    # We observe [cx, cy, w, h] from state [cx, cy, w, h, vx, vy, vw, vh]
    h = np.zeros((4, 8), dtype=np.float32)
    h[0, 0] = 1.0  # observe cx
    h[1, 1] = 1.0  # observe cy
    h[2, 2] = 1.0  # observe w
    h[3, 3] = 1.0  # observe h
    return h


@register_jit(nogil=True, inline="always")
def _create_process_noise(
    pos_noise: float = 1.0,
    vel_noise: float = 0.1,
    size_noise: float = 1.0,
    size_vel_noise: float = 0.1,
) -> np.ndarray:
    q = np.eye(8, dtype=np.float32)
    q[0, 0] = pos_noise  # cx noise
    q[1, 1] = pos_noise  # cy noise
    q[2, 2] = size_noise  # w noise
    q[3, 3] = size_noise  # h noise
    q[4, 4] = vel_noise  # vx noise
    q[5, 5] = vel_noise  # vy noise
    q[6, 6] = size_vel_noise  # vw noise
    q[7, 7] = size_vel_noise  # vh noise
    return q


@register_jit(nogil=True, inline="always")
def _create_measurement_noise(noise: float = 10.0) -> np.ndarray:
    return np.eye(4, dtype=np.float32) * noise


@register_jit(nogil=True, inline="always")
def _create_initial_covariance(uncertainty: float = 1000.0) -> np.ndarray:
    p = np.eye(8, dtype=np.float32) * uncertainty
    # Higher uncertainty for velocities
    p[4:, 4:] *= 10.0
    return p


@register_jit(nogil=True)
def kalman_predict(
    state: np.ndarray,
    covariance: np.ndarray,
    pos_noise: float = 1.0,
    vel_noise: float = 0.1,
    size_noise: float = 1.0,
    size_vel_noise: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Predict the next state using Kalman filter prediction step.

    Parameters
    ----------
    state : np.ndarray
        Current state vector [cx, cy, w, h, vx, vy, vw, vh]
    covariance : np.ndarray
        Current covariance matrix (8x8)
    pos_noise : float, optional
        Process noise for position, by default 1.0
    vel_noise : float, optional
        Process noise for velocity, by default 0.1
    size_noise : float, optional
        Process noise for size, by default 1.0
    size_vel_noise : float, optional
        Process noise for size velocity, by default 0.1

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Predicted state and covariance

    """
    f = _create_transition_matrix()
    q = _create_process_noise(pos_noise, vel_noise, size_noise, size_vel_noise)

    # Predict state: x_pred = F * x
    state_pred = f @ state

    # Predict covariance: P_pred = F * P * F^T + Q
    covariance_pred = f @ covariance @ f.T + q

    return state_pred, covariance_pred


@register_jit(nogil=True)
def kalman_update(
    state_pred: np.ndarray,
    covariance_pred: np.ndarray,
    measurement: np.ndarray,
    measurement_noise: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Update the state using Kalman filter update step.

    Parameters
    ----------
    state_pred : np.ndarray
        Predicted state vector [cx, cy, w, h, vx, vy, vw, vh]
    covariance_pred : np.ndarray
        Predicted covariance matrix (8x8)
    measurement : np.ndarray
        Measurement vector [cx, cy, w, h]
    measurement_noise : float, optional
        Measurement noise, by default 10.0

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Updated state and covariance

    """
    h = _create_observation_matrix()
    r = _create_measurement_noise(measurement_noise)

    # Innovation: y = z - H * x_pred
    innovation = measurement - h @ state_pred

    # Innovation covariance: S = H * P_pred * H^T + R
    innovation_cov = h @ covariance_pred @ h.T + r

    # Kalman gain: K = P_pred * H^T * S^-1
    kalman_gain = covariance_pred @ h.T @ np.linalg.inv(innovation_cov)

    # Update state: x = x_pred + K * y
    state_updated = state_pred + kalman_gain @ innovation

    # Update covariance: P = (I - K * H) * P_pred
    identity = np.eye(8, dtype=np.float32)
    covariance_updated = (identity - kalman_gain @ h) @ covariance_pred

    return state_updated, covariance_updated


@register_jit(nogil=True)
def kalman_init(bbox: tuple[int, int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    """
    Initialize Kalman filter state from a bounding box.

    Parameters
    ----------
    bbox : tuple[int, int, int, int]
        Initial bounding box (x1, y1, x2, y2)

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Initial state and covariance

    """
    state = _bbox_to_state(bbox)
    covariance = _create_initial_covariance()
    return state, covariance


@register_jit(nogil=True)
def kalman_get_bbox(state: np.ndarray) -> tuple[int, int, int, int]:
    """
    Extract bounding box from Kalman filter state.

    Parameters
    ----------
    state : np.ndarray
        State vector [cx, cy, w, h, vx, vy, vw, vh]

    Returns
    -------
    tuple[int, int, int, int]
        Bounding box (x1, y1, x2, y2)

    """
    return _state_to_bbox(state)


@register_jit(nogil=True)
def kalman_predict_bbox(
    bbox: tuple[int, int, int, int],
    state: np.ndarray,
    covariance: np.ndarray,
    pos_noise: float = 1.0,
    vel_noise: float = 0.1,
    size_noise: float = 1.0,
    size_vel_noise: float = 0.1,
) -> tuple[tuple[int, int, int, int], np.ndarray, np.ndarray]:
    """
    Predict next bounding box position using Kalman filter.

    Parameters
    ----------
    bbox : tuple[int, int, int, int]
        Current bounding box (x1, y1, x2, y2) - not used directly
    state : np.ndarray
        Current state vector
    covariance : np.ndarray
        Current covariance matrix
    pos_noise : float, optional
        Process noise for position, by default 1.0
    vel_noise : float, optional
        Process noise for velocity, by default 0.1
    size_noise : float, optional
        Process noise for size, by default 1.0
    size_vel_noise : float, optional
        Process noise for size velocity, by default 0.1

    Returns
    -------
    tuple[tuple[int, int, int, int], np.ndarray, np.ndarray]
        Predicted bounding box, state, and covariance

    """
    state_pred, covariance_pred = kalman_predict(
        state,
        covariance,
        pos_noise,
        vel_noise,
        size_noise,
        size_vel_noise,
    )
    bbox_pred = kalman_get_bbox(state_pred)
    return bbox_pred, state_pred, covariance_pred


@register_jit(nogil=True)
def kalman_update_bbox(
    measurement_bbox: tuple[int, int, int, int],
    state_pred: np.ndarray,
    covariance_pred: np.ndarray,
    measurement_noise: float = 10.0,
) -> tuple[tuple[int, int, int, int], np.ndarray, np.ndarray]:
    """
    Update Kalman filter with a measured bounding box.

    Parameters
    ----------
    measurement_bbox : tuple[int, int, int, int]
        Measured bounding box (x1, y1, x2, y2)
    state_pred : np.ndarray
        Predicted state vector
    covariance_pred : np.ndarray
        Predicted covariance matrix
    measurement_noise : float, optional
        Measurement noise, by default 10.0

    Returns
    -------
    tuple[tuple[int, int, int, int], np.ndarray, np.ndarray]
        Updated bounding box, state, and covariance

    """
    measurement = _bbox_to_state(measurement_bbox)[:4]  # Only position and size
    state_updated, covariance_updated = kalman_update(
        state_pred,
        covariance_pred,
        measurement,
        measurement_noise,
    )
    bbox_updated = kalman_get_bbox(state_updated)
    return bbox_updated, state_updated, covariance_updated


class KalmanBBoxFilter:
    """
    Minimal Kalman filter class for bounding box tracking.
    
    This class wraps the function-based Kalman filter implementation
    with a clean object-oriented interface.
    
    Parameters
    ----------
    bbox : tuple[int, int, int, int]
        Initial bounding box (x1, y1, x2, y2)
    
    Example
    -------
    >>> filter = KalmanBBoxFilter((100, 50, 200, 150))
    >>> predicted_bbox = filter.predict()
    >>> updated_bbox = filter.update((105, 52, 205, 152))
    """
    
    __slots__ = ('_state', '_covariance')
    
    def __init__(self, bbox: tuple[int, int, int, int]) -> None:
        """Initialize the Kalman filter with a bounding box."""
        self._state, self._covariance = kalman_init(bbox)
    
    def predict(
        self,
        pos_noise: float = 1.0,
        vel_noise: float = 0.1,
        size_noise: float = 1.0,
        size_vel_noise: float = 0.1,
    ) -> tuple[int, int, int, int]:
        """
        Predict the next bounding box position.
        
        Returns
        -------
        tuple[int, int, int, int]
            Predicted bounding box (x1, y1, x2, y2)
        """
        bbox_pred, self._state, self._covariance = kalman_predict_bbox(
            self.bbox, self._state, self._covariance,
            pos_noise, vel_noise, size_noise, size_vel_noise
        )
        return bbox_pred
    
    def update(
        self,
        measurement_bbox: tuple[int, int, int, int],
        measurement_noise: float = 10.0,
    ) -> tuple[int, int, int, int]:
        """
        Update the filter with a measured bounding box.
        
        Parameters
        ----------
        measurement_bbox : tuple[int, int, int, int]
            Measured bounding box (x1, y1, x2, y2)
        measurement_noise : float, optional
            Measurement noise, by default 10.0
            
        Returns
        -------
        tuple[int, int, int, int]
            Updated bounding box (x1, y1, x2, y2)
        """
        bbox_updated, self._state, self._covariance = kalman_update_bbox(
            measurement_bbox, self._state, self._covariance, measurement_noise
        )
        return bbox_updated
    
    @property
    def bbox(self) -> tuple[int, int, int, int]:
        """Get the current bounding box estimate."""
        return kalman_get_bbox(self._state)
    
    @property
    def state(self) -> np.ndarray:
        """Get the current state vector [cx, cy, w, h, vx, vy, vw, vh]."""
        return self._state.copy()
    
    @property
    def covariance(self) -> np.ndarray:
        """Get the current covariance matrix."""
        return self._covariance.copy()
