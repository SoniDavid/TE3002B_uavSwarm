"""Unit tests for ViconKF.

Run with:  pytest test/test_kalman_filter.py -v
Or after colcon build:  colcon test --packages-select tello_vicon
"""
import sys
import os

import numpy as np
import pytest

# Allow import without installing the package (pytest from repo root)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from kalman_filter import ViconKF


DT = 0.01   # 100 Hz


def make_kf(**kwargs):
    kf = ViconKF()
    kf.init(DT, **kwargs)
    return kf


# ── Basic API ─────────────────────────────────────────────────────────────────

def test_step_returns_correct_shape():
    kf = make_kf()
    y = np.zeros(6)
    x = kf.step(y)
    assert x.shape == (12,)


def test_static_position_converges():
    """KF should settle very close to the true static position after many steps."""
    kf = make_kf(r_pos=1e-6, r_ang=1e-6, q_vel=1e-4)
    true_pos = np.array([1.0, -0.5, 1.2, 0.0, 0.0, 0.1])
    for _ in range(500):
        kf.step(true_pos)
    x = kf.state
    # Positions (even indices 0,2,4) should be within 1 mm
    assert abs(x[0] - 1.0)  < 0.001
    assert abs(x[2] + 0.5)  < 0.001
    assert abs(x[4] - 1.2)  < 0.001


def test_velocity_estimation_from_ramp():
    """KF should estimate constant velocity from linearly moving position."""
    kf = make_kf(q_vel=1e-3, r_pos=1e-8)
    true_vx = 0.3   # m/s
    pos = 0.0
    for _ in range(300):
        pos += true_vx * DT
        kf.step(np.array([pos, 0.0, 0.0, 0.0, 0.0, 0.0]))

    x = kf.state
    # Velocity index 1 (vx) should be within 5% of true velocity after 3 s
    assert abs(x[1] - true_vx) < 0.015


def test_noise_smoothing():
    """KF output should have lower position variance than noisy input."""
    rng = np.random.default_rng(42)
    kf  = make_kf(r_pos=1e-4, q_vel=1e-3)

    true_pos = 1.0
    sigma    = 0.01   # 1 cm noise
    noisy_meas, kf_out = [], []

    for _ in range(500):
        noisy = true_pos + rng.normal(0, sigma)
        noisy_meas.append(noisy)
        x = kf.step(np.array([noisy, 0.0, 0.0, 0.0, 0.0, 0.0]))
        kf_out.append(x[0])

    var_in  = float(np.var(noisy_meas[100:]))   # ignore transient
    var_out = float(np.var(kf_out[100:]))
    assert var_out < var_in, f"KF variance {var_out:.2e} not less than input {var_in:.2e}"


def test_angle_unwrap_prevents_jump():
    """Angles crossing ±pi should not cause a state explosion."""
    kf = make_kf()
    # Simulate yaw increasing past +pi (wraps to -pi)
    yaw = 3.10
    for _ in range(10):
        yaw += 0.05
        y_wrapped = np.array([0.0, 0.0, 0.0, 0.0, 0.0, (yaw + np.pi) % (2 * np.pi) - np.pi])
        x = kf.step(y_wrapped)
    # KF yaw state should remain finite and not blow up
    assert np.isfinite(x[10])
    assert abs(x[10]) < 2 * np.pi + 1.0


def test_reset_clears_state():
    kf = make_kf()
    for _ in range(50):
        kf.step(np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3]))
    kf.reset()
    assert np.allclose(kf.state, 0.0)


def test_step_without_init_raises():
    kf = ViconKF()
    with pytest.raises(RuntimeError):
        kf.step(np.zeros(6))
