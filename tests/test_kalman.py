import numpy as np

from aps.tracker.kalman import AreaKalman


def test_converges_to_center_velocity(cfg):
    kf = AreaKalman(np.array([100.0, 200.0, 400.0]), cfg["tracker"]["kalman"])
    dt = 1 / 30
    for i in range(1, 60):
        kf.predict(dt)
        kf.update(np.array([100.0 + 90 * i * dt, 200.0 - 30 * i * dt, 400.0]))
    assert abs(kf.x[3] - 90) < 3
    assert abs(kf.x[4] + 30) < 3
    assert abs(kf.x[5]) < 10


def test_predict_follows_constant_closing_speed(cfg):
    # A = k/τ² 이면 예측만으로 τ가 줄어든 시점의 면적과 변화율을 정확히 맞춰야 한다
    k, tau, dt = 1000.0, 2.0, 0.1
    kf = AreaKalman(np.array([0.0, 0.0, k / tau**2]), cfg["tracker"]["kalman"])
    kf.x[5] = 2 * k / tau**3
    kf.predict(dt)
    assert np.isclose(kf.x[2], k / (tau - dt) ** 2)
    assert np.isclose(kf.x[5], 2 * k / (tau - dt) ** 3)


def test_predict_stays_finite_near_impact(cfg):
    kf = AreaKalman(np.array([0.0, 0.0, 100.0]), cfg["tracker"]["kalman"])
    kf.x[5] = 1e6  # TTC가 dt보다 짧은 추정
    for _ in range(3):
        kf.predict(1 / 30)
    assert np.isfinite(kf.x).all() and np.isfinite(kf.P).all()
