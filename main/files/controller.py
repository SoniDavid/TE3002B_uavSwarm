import time
import numpy as np

TARGET_DIST  = 1.2
DEAD_ZONE_D  = 0.08
Kp_dist      = 60
MAX_VEL_FB   = 30

FRAME_W      = 960
FRAME_H      = 720
CENTER_X     = FRAME_W // 2
CENTER_Y     = FRAME_H // 2
DEAD_ZONE_PX = 60
MAX_VEL_YAW  = 60
MAX_VEL_UD   = 60

LOST_TIMEOUT = 0.5


class RCController:
    def __init__(self, lock, get_result_fn, stop_event):
        self.lock       = lock
        self.get_result = get_result_fn
        self.stop_event = stop_event

    def _calc_fb(self, dist):
        error = dist - TARGET_DIST
        if abs(error) < DEAD_ZONE_D:
            return 0
        return int(np.clip(Kp_dist * error, -MAX_VEL_FB, MAX_VEL_FB))

    def _calc_yaw(self, cx):
        error = cx - CENTER_X
        if abs(error) < DEAD_ZONE_PX:
            return 0
        vel = int(np.interp(abs(error), [DEAD_ZONE_PX, CENTER_X], [20, MAX_VEL_YAW]))
        return vel if error > 0 else -vel

    def _calc_ud(self, cy):
        error = cy - CENTER_Y
        if abs(error) < DEAD_ZONE_PX:
            return 0
        vel = int(np.interp(abs(error), [DEAD_ZONE_PX, CENTER_Y], [20, MAX_VEL_UD]))
        return -vel if error > 0 else vel

    def run(self, tello):
        while not self.stop_event.is_set():
            with self.lock:
                res, last = self.get_result()

            lost = (time.time() - last) > LOST_TIMEOUT

            if res is None or lost:
                tello.send_rc_control(0, 0, 0, 0)
            else:
                _, _, _, dist, cx, cy = res
                tello.send_rc_control(
                    0,
                    self._calc_fb(dist),
                    self._calc_ud(cy),
                    self._calc_yaw(cx)
                )

            time.sleep(0.05)
