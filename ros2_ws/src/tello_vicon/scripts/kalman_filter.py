import numpy as np


class ViconKF:
    """Discrete Kalman filter for Vicon motion-capture pose data.

    State (12x1):  [px, vx, py, vy, pz, vz, roll, vroll, pitch, vpitch, yaw, vyaw]
    Measurement (6x1): [px, py, pz, roll, pitch, yaw]  — exactly what Vicon outputs.

    All states are time-derivatives of directly observable Vicon quantities.
    No control input is modelled (constant-velocity / constant-rate assumption).

    Usage
    -----
        kf = ViconKF()
        kf.init(dt, q_pos, q_vel, q_ang, q_rate, r_pos, r_ang)
        state = kf.step(measurement)   # returns 12-vector each call
    """

    NX = 12  # state dimension
    NY = 6   # measurement dimension

    def __init__(self):
        self._initialized = False
        self._x = np.zeros(self.NX)
        self._P = np.eye(self.NX)
        self._A = np.eye(self.NX)
        self._C = np.zeros((self.NY, self.NX))
        self._Q = np.eye(self.NX)
        self._R = np.eye(self.NY)
        self._prev_angles = None   # for angle unwrapping

    def init(self, dt: float,
             q_pos: float = 1e-3, q_vel: float = 1e-1,
             q_ang: float = 1e-4, q_rate: float = 1e-2,
             r_pos: float = 1e-6, r_ang: float = 1e-5) -> None:
        """Build A, C, Q, R and reset state.

        Parameters
        ----------
        dt      : time step in seconds
        q_pos   : process noise variance for position states
        q_vel   : process noise variance for velocity states
        q_ang   : process noise variance for angle states
        q_rate  : process noise variance for angular-rate states
        r_pos   : measurement noise variance for position (Vicon ~1 mm → 1e-6 m²)
        r_ang   : measurement noise variance for angles  (Vicon ~0.1° → ~3e-6 rad²)
        """
        # Constant-velocity / constant-rate transition matrix
        # Block structure: [pos, vel; 0, 1] repeated 6 times (3 position + 3 angle axes)
        A = np.zeros((self.NX, self.NX))
        for i in range(6):
            A[2*i,   2*i]   = 1.0
            A[2*i,   2*i+1] = dt
            A[2*i+1, 2*i+1] = 1.0
        self._A = A

        # Measurement matrix: selects even-indexed states (the observables)
        C = np.zeros((self.NY, self.NX))
        for i in range(6):
            C[i, 2*i] = 1.0
        self._C = C

        # Process noise — position axes use q_pos/q_vel, angle axes use q_ang/q_rate
        q_diag = np.array([
            q_pos, q_vel,   # px, vx
            q_pos, q_vel,   # py, vy
            q_pos, q_vel,   # pz, vz
            q_ang, q_rate,  # roll, vroll
            q_ang, q_rate,  # pitch, vpitch
            q_ang, q_rate,  # yaw, vyaw
        ])
        self._Q = np.diag(q_diag)

        # Measurement noise
        r_diag = np.array([r_pos, r_pos, r_pos, r_ang, r_ang, r_ang])
        self._R = np.diag(r_diag)

        # Reset state
        self._x = np.zeros(self.NX)
        self._P = np.eye(self.NX) * 1.0
        self._prev_angles = None
        self._initialized = True

    def step(self, y: np.ndarray) -> np.ndarray:
        """Run one KF predict+update step.

        Parameters
        ----------
        y : (6,) array [px, py, pz, roll, pitch, yaw]

        Returns
        -------
        x : (12,) estimated state
        """
        if not self._initialized:
            raise RuntimeError("Call init() before step()")

        y = np.asarray(y, dtype=float)

        # Unwrap angles to prevent ±pi discontinuities
        y[3:6] = self._unwrap_angles(y[3:6])

        #  Predict 
        xp = self._A @ self._x
        Pp = self._A @ self._P @ self._A.T + self._Q

        #  Update 
        S = self._C @ Pp @ self._C.T + self._R
        K = Pp @ self._C.T @ np.linalg.inv(S)

        innovation = y - self._C @ xp
        self._x = xp + K @ innovation

        # Joseph form for numerical stability
        I_KC = np.eye(self.NX) - K @ self._C
        self._P = I_KC @ Pp @ I_KC.T + K @ self._R @ K.T

        return self._x.copy()

    # 
    def reset(self) -> None:
        self._x = np.zeros(self.NX)
        self._P = np.eye(self.NX)
        self._prev_angles = None

    @property
    def state(self) -> np.ndarray:
        return self._x.copy()

    # 
    def _unwrap_angles(self, angles: np.ndarray) -> np.ndarray:
        """Unwrap angles relative to previous measurement to avoid ±pi jumps."""
        if self._prev_angles is None:
            self._prev_angles = angles.copy()
            return angles.copy()

        unwrapped = angles.copy()
        for i in range(len(angles)):
            diff = angles[i] - self._prev_angles[i]
            if diff > np.pi:
                unwrapped[i] -= 2 * np.pi
            elif diff < -np.pi:
                unwrapped[i] += 2 * np.pi
        self._prev_angles = unwrapped.copy()
        return unwrapped
