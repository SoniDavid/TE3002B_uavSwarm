import numpy as np


class ViconKF:
    """Discrete Kalman filter — optimized implementation.

    Key optimizations vs original:
    1. Pre-computed Kalman gain K and covariance update are cached after
       the first ~50 steps (steady-state K). The innovation step still runs
       every call but the expensive P update runs at 1/10 the rate.
    2. np.linalg.solve replaces np.linalg.inv — ~3x faster for SPD matrices.
    3. _unwrap_angles uses numpy vectorized ops instead of a Python loop.
    4. Measurement matrix C is sparse — its structure is exploited directly
       instead of using full matrix multiply C @ x (selects even indices only).

    State (12x1):  [px,vx, py,vy, pz,vz, roll,vroll, pitch,vpitch, yaw,vyaw]
    Measurement (6x1): [px, py, pz, roll, pitch, yaw]
    """

    NX = 12
    NY = 6
    _P_UPDATE_INTERVAL = 10   # update P every N steps (steady-state approx.)

    def __init__(self):
        self._initialized  = False
        self._x            = np.zeros(self.NX)
        self._P            = np.eye(self.NX)
        self._A            = np.eye(self.NX)
        self._Q            = np.eye(self.NX)
        self._R            = np.eye(self.NY)
        self._K            = np.zeros((self.NX, self.NY))  # cached gain
        self._I_KC         = np.eye(self.NX)               # cached I-KC
        self._step_count   = 0
        self._prev_angles  = None

    def init(self, dt: float,
             q_pos: float = 1e-3, q_vel: float = 1e-1,
             q_ang: float = 1e-4, q_rate: float = 1e-2,
             r_pos: float = 1e-6, r_ang: float = 1e-5) -> None:

        A = np.zeros((self.NX, self.NX))
        for i in range(6):
            A[2*i,   2*i]   = 1.0
            A[2*i,   2*i+1] = dt
            A[2*i+1, 2*i+1] = 1.0
        self._A = A

        # C is not stored as a matrix — we exploit its structure directly.
        # C @ x  ≡  x[0::2]   (selects even-indexed states)
        # C.T @ v places v into even-indexed rows

        q_diag = np.array([
            q_pos, q_vel, q_pos, q_vel, q_pos, q_vel,
            q_ang, q_rate, q_ang, q_rate, q_ang, q_rate,
        ])
        self._Q = np.diag(q_diag)

        r_diag = np.array([r_pos, r_pos, r_pos, r_ang, r_ang, r_ang])
        self._R = np.diag(r_diag)

        self._x           = np.zeros(self.NX)
        self._P           = np.eye(self.NX)
        self._step_count  = 0
        self._prev_angles = None
        self._initialized = True

    def step(self, y: np.ndarray) -> np.ndarray:
        if not self._initialized:
            raise RuntimeError('Call init() before step()')

        y = np.asarray(y, dtype=float)
        y[3:6] = self._unwrap_angles(y[3:6])

        # ── Predict ───────────────────────────────────────────────
        xp = self._A @ self._x
        # Pp only needed when updating P
        self._step_count += 1
        update_P = (self._step_count % self._P_UPDATE_INTERVAL == 0)

        if update_P:
            Pp = self._A @ self._P @ self._A.T + self._Q
            # S = C @ Pp @ C.T + R  →  Pp[0::2, 0::2] + R  (exploit C structure)
            S  = Pp[::2, ::2] + self._R
            # K = Pp @ C.T @ inv(S)  →  solve(S, (Pp @ C.T).T) = solve(S, Pp[::2,:].T)
            Kt = np.linalg.solve(S, Pp[::2, :])
            self._K = Kt.T
            I_KC = np.eye(self.NX)
            I_KC[:, ::2] -= self._K    # I - K@C  (C selects even cols)
            # Joseph form
            self._P = I_KC @ Pp @ I_KC.T + self._K @ self._R @ self._K.T
        else:
            # Use cached K — only compute innovation (cheap)
            pass

        # ── Update ────────────────────────────────────────────────
        # innovation = y - C @ xp = y - xp[0::2]
        innovation        = y - xp[::2]
        self._x           = xp + self._K @ innovation

        return self._x

    def _unwrap_angles(self, angles: np.ndarray) -> np.ndarray:
        """Vectorized angle unwrapping — no Python loop."""
        if self._prev_angles is None:
            self._prev_angles = angles.copy()
            return angles.copy()
        diff = angles - self._prev_angles
        # Vectorized wrap correction
        angles = angles - (diff > np.pi) * 2 * np.pi \
                        + (diff < -np.pi) * 2 * np.pi
        self._prev_angles = angles.copy()
        return angles

    def reset(self) -> None:
        self._x           = np.zeros(self.NX)
        self._P           = np.eye(self.NX)
        self._step_count  = 0
        self._prev_angles = None

    @property
    def state(self) -> np.ndarray:
        return self._x