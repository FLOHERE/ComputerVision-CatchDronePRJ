"""칼만 필터. 상태: [cx, cy, A, vx, vy, vA]

중심은 등속 모델이고, 면적은 일정 접근 속도 모델(A ∝ 1/거리²)로 예측한다.
면적에 등속 모델을 쓰면 접근할수록 vA가 뒤처져 TTC가 실제보다 크게 나온다.
"""

import numpy as np

_H = np.hstack([np.eye(3), np.zeros((3, 3))])


class AreaKalman:
    def __init__(self, z: np.ndarray, cfg: dict):
        self.cfg = cfg
        self.x = np.concatenate([z, np.zeros(3)])
        area = z[2]
        velocity_var = [cfg["init_speed_px"] ** 2] * 2 + [(cfg["init_area_rate_ratio"] * area) ** 2]
        self.P = np.diag(np.concatenate([self._meas_var(area, False), velocity_var]))

    def _meas_var(self, area: float, truncated: bool) -> np.ndarray:
        side = np.sqrt(max(area, 1.0))
        edge = max(self.cfg["meas_edge_px"], self.cfg["meas_edge_ratio"] * side)
        # A = w*h 이고 w, h가 각각 두 변의 오차를 가지므로 std(A) ≈ 2*side*edge
        area_std = 2.0 * side * edge
        if truncated:
            area_std *= self.cfg["truncated_area_scale"]
        return np.array([edge**2 / 2, edge**2 / 2, area_std**2])

    def predict(self, dt: float) -> None:
        area = max(self.x[2], 1.0)
        rate = self.x[5]
        # 일정 접근 속도일 때 A' = A/s², vA' = vA/s³ (s = 남은 거리 비율)
        # 직전 추정이 충돌 직전이면 s가 0에 가까워 발산하므로 하한을 둔다
        s = max(1.0 - dt * rate / (2 * area), self.cfg["min_closing_scale"])
        ds_da, ds_dv = dt * rate / (2 * area**2), -dt / (2 * area)

        F = np.eye(6)
        F[:2, 3:5] = np.eye(2) * dt
        F[2, 2] = s**-2 - 2 * area * s**-3 * ds_da
        F[2, 5] = -2 * area * s**-3 * ds_dv
        F[5, 2] = -3 * rate * s**-4 * ds_da
        F[5, 5] = s**-3 - 3 * rate * s**-4 * ds_dv

        self.x[:2] += self.x[3:5] * dt
        self.x[2] = area / s**2
        self.x[5] = rate / s**3

        # 모델로 설명되지 않는 기동을 면적에 비례하는 공정 잡음으로 흡수한다
        accel = np.array([self.cfg["accel_pos_px"], self.cfg["accel_pos_px"], self.cfg["accel_area_ratio"] * area])
        G = np.vstack([np.diag(accel) * dt**2 / 2, np.diag(accel) * dt])
        self.P = F @ self.P @ F.T + G @ G.T

    def update(self, z: np.ndarray, truncated: bool = False) -> None:
        R = np.diag(self._meas_var(self.x[2], truncated))
        S = _H @ self.P @ _H.T + R
        K = self.P @ _H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - _H @ self.x)
        self.P = (np.eye(6) - K @ _H) @ self.P
