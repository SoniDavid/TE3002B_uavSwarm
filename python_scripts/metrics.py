#!/usr/bin/env python3
"""
metrics.py — Extrae métricas de posición y formación de un rosbag de ROS2.

Uso:
    python3 metrics.py <ruta_al_bag>

Genera:
    metrics_report.txt  — reporte de texto con todas las métricas
    metrics_plots.png   — gráficas de errores en el tiempo

Métricas calculadas:
    - Error RMS de posición XY, Z y 3D para cada drone
    - Error máximo de posición
    - Error de seguimiento de referencia (kf_state vs reference)
    - Error de formación (distancia real vs distancia deseada entre drones)
    - Frecuencia de detección ArUco
    - Tiempo de convergencia de la formación
"""

import sys
import os
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict

try:
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
except ImportError:
    print("ERROR: ROS2 environment not sourced.")
    print("Run:  source /opt/ros/humble/setup.bash && python3 metrics.py <bag>")
    sys.exit(1)

# ── Configuration ─────────────────────────────────────────────────
DRONES    = ['tello_soni1', 'tello_soni2', 'tello_soni3']
LEADER    = 'tello_soni1'

# Formation offsets (LINE, metres) — adjust to match your flight formation
FORMATION_OFFSETS = {
    'tello_soni2': {'dx': 0.0, 'dy':  0.80, 'dz': 0.0},
    'tello_soni3': {'dx': 0.0, 'dy': -0.80, 'dz': 0.0},
}

CONVERGENCE_THRESHOLD_M = 0.20   # formation error below this = converged
ARUCO_TIMEOUT_S         = 0.5

# State vector indices
IDX_PX, IDX_PY, IDX_PZ, IDX_YAW = 0, 2, 4, 10


# ── Bag reader ────────────────────────────────────────────────────
def read_bag(bag_path: str):
    """Read all messages from bag. Returns dict: topic → list of (t_sec, msg)."""
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id='sqlite3')
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr')

    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    topic_types = {}
    for info in reader.get_all_topics_and_types():
        topic_types[info.name] = info.type

    print("Topics in bag:")
    for t, typ in sorted(topic_types.items()):
        print(f"  {t:55s}  {typ}")
    print()

    data = defaultdict(list)
    type_map = {}

    while reader.has_next():
        topic, raw, stamp_ns = reader.read_next()
        if topic not in topic_types:
            continue
        msg_type_str = topic_types[topic]
        if msg_type_str not in type_map:
            try:
                type_map[msg_type_str] = get_message(msg_type_str)
            except Exception:
                continue
        try:
            msg = deserialize_message(raw, type_map[msg_type_str])
        except Exception:
            continue
        t_sec = stamp_ns * 1e-9
        data[topic].append((t_sec, msg))

    return data


# ── Extractors ────────────────────────────────────────────────────
def extract_kf_state(messages):
    """Returns (times, px, py, pz, yaw) arrays."""
    times, px, py, pz, yaw = [], [], [], [], []
    for t, msg in messages:
        d = list(msg.data)
        if len(d) < 12:
            continue
        times.append(t); px.append(d[IDX_PX]); py.append(d[IDX_PY])
        pz.append(d[IDX_PZ]); yaw.append(d[IDX_YAW])
    return (np.array(times), np.array(px), np.array(py),
            np.array(pz), np.array(yaw))


def extract_reference(messages):
    """Returns (times, rx, ry, rz) arrays from PoseStamped."""
    times, rx, ry, rz = [], [], [], []
    for t, msg in messages:
        times.append(t)
        rx.append(msg.pose.position.x)
        ry.append(msg.pose.position.y)
        rz.append(msg.pose.position.z)
    return np.array(times), np.array(rx), np.array(ry), np.array(rz)


def extract_aruco_detected(messages):
    """Returns (times, detected) from std_msgs/Bool."""
    times, det = [], []
    for t, msg in messages:
        times.append(t)
        det.append(bool(msg.data))
    return np.array(times), np.array(det)


def interp_xyz(t_query, t_src, x, y, z):
    """Interpolate xyz arrays to query times."""
    if len(t_src) < 2:
        return np.zeros_like(t_query), np.zeros_like(t_query), np.zeros_like(t_query)
    xi = np.interp(t_query, t_src, x)
    yi = np.interp(t_query, t_src, y)
    zi = np.interp(t_query, t_src, z)
    return xi, yi, zi


def rms(arr):
    return float(np.sqrt(np.mean(arr**2)))


def body_offset_to_world(lx, ly, lz, lyaw, offset):
    """Vectorized: compute world-frame target given leader pos + yaw arrays."""
    cy, sy = np.cos(lyaw), np.sin(lyaw)
    dx_w = offset['dx'] * cy - offset['dy'] * sy
    dy_w = offset['dx'] * sy + offset['dy'] * cy
    return lx + dx_w, ly + dy_w, lz + offset['dz']


# ── Main ──────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser(description='Extract flight metrics from ROS2 bag')
    parser.add_argument('bag_path', help='Path to rosbag2 directory')
    parser.add_argument('--trim-start', type=float, default=3.0,
                        help='Seconds to trim from start (removes takeoff transient, default 3)')
    parser.add_argument('--trim-end',   type=float, default=3.0,
                        help='Seconds to trim from end (removes landing transient, default 3)')
    parser.add_argument('--formation',  type=str,   default='LINE',
                        choices=['LINE','V','COLUMN','PANORAMIC','RECONSTRUCTION'],
                        help='Formation used during flight (default LINE)')
    args = parser.parse_args()

    bag_path = args.bag_path
    trim_s   = args.trim_start
    trim_e   = args.trim_end
    print(f"Reading bag: {bag_path}")
    print(f"Trimming: first {trim_s}s + last {trim_e}s removed (transient suppression)\n")
    data = read_bag(bag_path)

    # ── Align time to t=0 then trim transients ────────────────────
    all_t = [t for msgs in data.values() for t, _ in msgs]
    t0    = min(all_t) if all_t else 0.0
    t_max = max(all_t) if all_t else 0.0
    total = t_max - t0

    def shift_and_trim(msgs):
        return [(t - t0, m) for t, m in msgs
                if (t - t0) >= trim_s and (t - t0) <= (total - trim_e)]

    for k in list(data.keys()):
        data[k] = shift_and_trim(data[k])

    print(f"Flight window: {trim_s:.1f}s → {total-trim_e:.1f}s  "
          f"({total-trim_s-trim_e:.1f}s of data)\n")

    # Update formation offsets based on argument
    FORMATIONS_ALL = {
        'LINE':  {'tello_soni2': {'dx':0.0,'dy': 0.80,'dz':0.0},
                  'tello_soni3': {'dx':0.0,'dy':-0.80,'dz':0.0}},
        'V':     {'tello_soni2': {'dx':-0.50,'dy': 0.80,'dz':0.0},
                  'tello_soni3': {'dx':-0.50,'dy':-0.80,'dz':0.0}},
        'COLUMN':{'tello_soni2': {'dx':-1.2,'dy':0.0,'dz': 0.4},
                  'tello_soni3': {'dx':-2.4,'dy':0.0,'dz': 0.8}},
    }
    global FORMATION_OFFSETS
    FORMATION_OFFSETS = FORMATIONS_ALL.get(args.formation, FORMATION_OFFSETS)
    print(f"Formation: {args.formation}\n")

    # ── Extract per-drone data ────────────────────────────────────
    kf  = {}
    ref = {}
    for ns in DRONES:
        kf_topic  = f'/{ns}/kf_state'
        ref_topic = f'/{ns}/reference'
        if kf_topic in data:
            kf[ns]  = extract_kf_state(data[kf_topic])
        else:
            print(f"  WARNING: {kf_topic} not in bag")
        if ref_topic in data:
            ref[ns] = extract_reference(data[ref_topic])
        else:
            print(f"  WARNING: {ref_topic} not in bag")

    aruco_det_topic = f'/{LEADER}/aruco_detected'
    aruco_times, aruco_det = (extract_aruco_detected(data[aruco_det_topic])
                               if aruco_det_topic in data else (np.array([]), np.array([])))

    # ── Compute metrics ───────────────────────────────────────────
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("  MÉTRICAS DE VUELO — ENJAMBRE UAV TE3002B")
    report_lines.append(f"  Trim: {trim_s}s inicio / {trim_e}s final")
    report_lines.append(f"  Formación: {args.formation}")
    report_lines.append("=" * 60)

    per_drone_errors = {}

    for ns in DRONES:
        if ns not in kf or ns not in ref:
            continue

        t_kf, px, py, pz, _ = kf[ns]
        t_ref, rx, ry, rz   = ref[ns]

        # Interpolate reference to kf times
        mask = (t_kf >= t_ref[0]) & (t_kf <= t_ref[-1])
        if mask.sum() < 10:
            print(f"  WARNING: insufficient overlap for {ns}")
            continue
        t_eval = t_kf[mask]
        rx_i, ry_i, rz_i = interp_xyz(t_eval, t_ref, rx, ry, rz)

        ex = px[mask] - rx_i
        ey = py[mask] - ry_i
        ez = pz[mask] - rz_i
        e_xy = np.sqrt(ex**2 + ey**2)
        e_3d = np.sqrt(ex**2 + ey**2 + ez**2)

        per_drone_errors[ns] = {
            't': t_eval, 'ex': ex, 'ey': ey, 'ez': ez,
            'e_xy': e_xy, 'e_3d': e_3d,
        }

        report_lines.append(f"\n{'─'*60}")
        report_lines.append(f"  Drone: {ns}  ({'LÍDER' if ns == LEADER else 'SEGUIDOR'})")
        report_lines.append(f"{'─'*60}")
        report_lines.append(f"  Error RMS  XY   : {rms(e_xy)*100:6.1f} cm")
        report_lines.append(f"  Error RMS  Z    : {rms(ez)*100:6.1f} cm")
        report_lines.append(f"  Error RMS  3D   : {rms(e_3d)*100:6.1f} cm")
        report_lines.append(f"  Error MAX  XY   : {np.max(e_xy)*100:6.1f} cm")
        report_lines.append(f"  Error MAX  Z    : {np.max(np.abs(ez))*100:6.1f} cm")
        report_lines.append(f"  Error MAX  3D   : {np.max(e_3d)*100:6.1f} cm")
        report_lines.append(f"  Error P95  3D   : {np.percentile(e_3d,95)*100:6.1f} cm")
        report_lines.append(f"  Duración eval.  : {t_eval[-1]-t_eval[0]:.1f} s")

    # ── Formation error ───────────────────────────────────────────
    if LEADER in kf:
        t_l, lx, ly, lz, lyaw = kf[LEADER]

        report_lines.append(f"\n{'─'*60}")
        report_lines.append(f"  Error de Formación (vs. offset deseado LINE ±0.80m)")
        report_lines.append(f"{'─'*60}")

        formation_errors = {}
        for ns, offset in FORMATION_OFFSETS.items():
            if ns not in kf:
                continue
            t_f, fx, fy, fz, _ = kf[ns]

            # Interpolate leader to follower times
            t_eval = t_f
            mask   = (t_eval >= t_l[0]) & (t_eval <= t_l[-1])
            if mask.sum() < 10:
                continue
            t_eval = t_eval[mask]
            lxi, lyi, lzi = interp_xyz(t_eval, t_l, lx, ly, lz)
            lyawi          = np.interp(t_eval, t_l, lyaw)

            # Desired follower position
            dx, dy, dz = body_offset_to_world(lxi, lyi, lzi, lyawi, offset)

            # Actual follower position (interpolate to t_eval)
            fxi, fyi, fzi = interp_xyz(t_eval, t_f[mask], fx[mask], fy[mask], fz[mask])

            ef_xy = np.sqrt((fxi - dx)**2 + (fyi - dy)**2)
            ef_3d = np.sqrt((fxi - dx)**2 + (fyi - dy)**2 + (fzi - dz)**2)

            formation_errors[ns] = {'t': t_eval, 'e_xy': ef_xy, 'e_3d': ef_3d}

            report_lines.append(f"  {ns}:")
            report_lines.append(f"    Error RMS formación XY : {rms(ef_xy)*100:6.1f} cm")
            report_lines.append(f"    Error RMS formación 3D : {rms(ef_3d)*100:6.1f} cm")
            report_lines.append(f"    Error MAX formación 3D : {np.max(ef_3d)*100:6.1f} cm")

            # Convergence time
            converged_idx = np.where(ef_3d < CONVERGENCE_THRESHOLD_M)[0]
            if len(converged_idx) > 0:
                # first sustained convergence (10 consecutive samples)
                for i in converged_idx:
                    if i + 10 < len(ef_3d) and np.all(ef_3d[i:i+10] < CONVERGENCE_THRESHOLD_M):
                        report_lines.append(f"    Tiempo convergencia    : {t_eval[i]:.1f} s")
                        break
            else:
                report_lines.append(f"    Tiempo convergencia    : no convergió")

    # ── ArUco quality metrics ─────────────────────────────────────
    # NOTE: "detection rate" (% time ArUco is visible) is NOT reported
    # because it depends on how long the marker was in front of the camera
    # during that specific flight, not on detector quality.
    # Instead we report: latency proxy, detection stability, distance consistency.

    dist_topic = f'/{LEADER}/aruco_distance'
    dist_data  = data.get(dist_topic, [])

    report_lines.append(f"\n{'─'*60}")
    report_lines.append(f"  Calidad de Detección ArUco (/{LEADER})")
    report_lines.append(f"{'─'*60}")

    if len(aruco_times) > 1:
        # Topic frequency
        dt_arr = np.diff(aruco_times)
        freq   = 1.0 / np.median(dt_arr)
        report_lines.append(f"  Frecuencia del tópico        : {freq:.1f} Hz")

        # ── 1. Detection stability ─────────────────────────────
        # Count consecutive-detection streaks when marker IS visible.
        # A short streak (1-2 frames) = flickering detector.
        # A long streak = stable tracking.
        transitions   = np.diff(aruco_det.astype(int))
        starts        = np.where(transitions == 1)[0] + 1
        ends          = np.where(transitions == -1)[0] + 1
        if aruco_det[0]:  starts = np.concatenate([[0], starts])
        if aruco_det[-1]: ends   = np.concatenate([ends, [len(aruco_det)]])
        if len(starts) > 0 and len(ends) > 0 and len(starts) == len(ends):
            streak_durations = []
            for s, e in zip(starts, ends):
                if e > s:
                    dur = aruco_times[e-1] - aruco_times[s]
                    streak_durations.append(dur)
            if streak_durations:
                streak_arr = np.array(streak_durations)
                report_lines.append(f"  Streaks de detección         : {len(streak_arr)} episodios")
                report_lines.append(f"  Duración media streak        : {streak_arr.mean():.2f} s")
                report_lines.append(f"  Duración máx streak          : {streak_arr.max():.2f} s")
                report_lines.append(f"  Streaks < 0.5s (parpadeos)   : {(streak_arr < 0.5).sum()} ({(streak_arr < 0.5).mean()*100:.1f} %)")

    # ── 2. Distance consistency (when detected) ────────────────
    if len(dist_data) > 0:
        dist_t   = np.array([t for t, _ in dist_data])
        dist_val = np.array([float(m.data) for _, m in dist_data])

        # Only use samples where ArUco was actively detected
        if len(aruco_times) > 0:
            det_interp = np.interp(dist_t, aruco_times, aruco_det.astype(float)) > 0.5
            dist_det   = dist_val[det_interp]
        else:
            dist_det = dist_val

        if len(dist_det) > 5:
            # Compute frame-to-frame variation as proxy for stability
            d_diff = np.abs(np.diff(dist_det))
            report_lines.append(f"  Distancia media al marcador  : {dist_det.mean():.2f} m")
            report_lines.append(f"  Desv. estándar distancia     : {dist_det.std()*100:.1f} cm")
            report_lines.append(f"  Variación media frame-frame  : {d_diff.mean()*100:.1f} cm/frame")
            report_lines.append(f"  (variación baja = detector estable)")

    # ── 3. Detection latency proxy ─────────────────────────────
    # True latency needs ground truth. We estimate it as the
    # median interval between consecutive published detections.
    if len(aruco_times) > 1:
        detected_mask = aruco_det
        det_t = aruco_times[detected_mask]
        if len(det_t) > 1:
            intervals = np.diff(det_t)
            # Filter to only normal intervals (not gaps between streaks)
            normal    = intervals[intervals < 0.5]
            if len(normal) > 0:
                report_lines.append(f"  Intervalo medio publicación  : {normal.mean()*1000:.1f} ms")
                report_lines.append(f"  (≈ latencia detección + publicación)")

    report_lines.append(f"\n{'='*60}\n")
    report_text = "\n".join(report_lines)
    print(report_text)

    with open('metrics_report.txt', 'w') as f:
        f.write(report_text)
    print("Saved: metrics_report.txt")

    # ── Plots ─────────────────────────────────────────────────────
    n_drones = len(per_drone_errors)
    if n_drones == 0:
        print("No data to plot.")
        return

    colors = {'tello_soni1': '#006E6C', 'tello_soni2': '#E67E22', 'tello_soni3': '#8E44AD'}
    labels = {'tello_soni1': 'Líder (soni1)', 'tello_soni2': 'S1 (soni2)', 'tello_soni3': 'S2 (soni3)'}

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Métricas de Vuelo — Enjambre UAV TE3002B', fontsize=14, fontweight='bold')
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    # Plot 1: XY error over time
    ax1 = fig.add_subplot(gs[0, 0])
    for ns, errs in per_drone_errors.items():
        ax1.plot(errs['t'], errs['e_xy'] * 100, color=colors[ns], label=labels[ns], linewidth=0.8)
    ax1.set_title('Error de posición XY en el tiempo')
    ax1.set_xlabel('Tiempo [s]'); ax1.set_ylabel('Error XY [cm]')
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

    # Plot 2: Z error over time
    ax2 = fig.add_subplot(gs[0, 1])
    for ns, errs in per_drone_errors.items():
        ax2.plot(errs['t'], np.abs(errs['ez']) * 100, color=colors[ns], label=labels[ns], linewidth=0.8)
    ax2.set_title('Error de posición Z en el tiempo')
    ax2.set_xlabel('Tiempo [s]'); ax2.set_ylabel('|Error Z| [cm]')
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

    # Plot 3: 3D error over time
    ax3 = fig.add_subplot(gs[1, 0])
    for ns, errs in per_drone_errors.items():
        ax3.plot(errs['t'], errs['e_3d'] * 100, color=colors[ns], label=labels[ns], linewidth=0.8)
    ax3.set_title('Error de posición 3D en el tiempo')
    ax3.set_xlabel('Tiempo [s]'); ax3.set_ylabel('Error 3D [cm]')
    ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3)

    # Plot 4: RMS bar chart
    ax4 = fig.add_subplot(gs[1, 1])
    drone_names = list(per_drone_errors.keys())
    rms_xy = [rms(per_drone_errors[ns]['e_xy']) * 100 for ns in drone_names]
    rms_z  = [rms(np.abs(per_drone_errors[ns]['ez'])) * 100 for ns in drone_names]
    rms_3d = [rms(per_drone_errors[ns]['e_3d']) * 100 for ns in drone_names]
    x = np.arange(len(drone_names))
    w = 0.25
    ax4.bar(x - w, rms_xy, w, label='XY', color='#006E6C', alpha=0.8)
    ax4.bar(x,     rms_z,  w, label='Z',  color='#E67E22', alpha=0.8)
    ax4.bar(x + w, rms_3d, w, label='3D', color='#8E44AD', alpha=0.8)
    ax4.set_title('Error RMS por drone')
    ax4.set_xticks(x)
    ax4.set_xticklabels([labels.get(ns, ns) for ns in drone_names], fontsize=8)
    ax4.set_ylabel('Error RMS [cm]')
    ax4.legend(fontsize=8); ax4.grid(True, alpha=0.3, axis='y')

    # Plot 5: Formation error
    if 'formation_errors' in dir() and formation_errors:
        ax5 = fig.add_subplot(gs[2, 0])
        for ns, errs in formation_errors.items():
            ax5.plot(errs['t'], errs['e_3d'] * 100, color=colors[ns], label=labels[ns], linewidth=0.8)
        ax5.axhline(CONVERGENCE_THRESHOLD_M * 100, color='red', linestyle='--',
                    linewidth=1, label=f'Umbral {CONVERGENCE_THRESHOLD_M*100:.0f}cm')
        ax5.set_title('Error de formación en el tiempo')
        ax5.set_xlabel('Tiempo [s]'); ax5.set_ylabel('Error formación 3D [cm]')
        ax5.legend(fontsize=8); ax5.grid(True, alpha=0.3)

    # Plot 6: ArUco detection
    if len(aruco_times) > 0:
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.fill_between(aruco_times, aruco_det.astype(float),
                         alpha=0.6, color='#006E6C', label='Detectado')
        ax6.set_title('Detección ArUco en el tiempo')
        ax6.set_xlabel('Tiempo [s]'); ax6.set_ylabel('Detectado (1=sí)')
        ax6.set_ylim(-0.1, 1.3); ax6.legend(fontsize=8); ax6.grid(True, alpha=0.3)

    plt.savefig('metrics_plots.png', dpi=150, bbox_inches='tight')
    print("Saved: metrics_plots.png")
    plt.close()


if __name__ == '__main__':
    main()