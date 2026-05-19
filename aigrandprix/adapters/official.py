"""OfficialSimAdapter — MAVLink + UDP vision bridge to the AI Grand Prix simulator.

Spec: VADR-TS-002 Issue 00.02 (2026-05-08)

Threading model:
  _heartbeat_loop  — sends MAVLink HEARTBEAT at cfg.heartbeat_rate_hz (spec requires ≥2 Hz)
  _telemetry_loop  — reads ATTITUDE + HIGHRES_IMU from MAVLink UDP
  _vision_loop     — reassembles chunked JPEG frames from UDP port 5600

step() sends SET_ATTITUDE_TARGET, waits for the next fresh vision frame, then
returns an Observation built from the latest telemetry + that frame.
"""
from __future__ import annotations

import socket
import struct
import threading
import time
from typing import Optional

import cv2
import numpy as np

from aigrandprix.adapters.base import AbstractAdapter
from aigrandprix.config import Config, OfficialAdapterConfig
from aigrandprix.types import Action, Observation

# Vision header layout (spec §4.6): little-endian, 24 bytes total
# field:        frame_id  chunk_id  total_chunks  jpeg_size  payload_size  sim_time_ns
# type:         uint32    uint16    uint16         uint32     uint32        uint64
_HDR_FMT  = '<IHHIIQ'
_HDR_SIZE = struct.calcsize(_HDR_FMT)   # == 24

# SET_ATTITUDE_TARGET type_mask: bit=1 → ignore that field
# 0b10000000 = ignore quaternion attitude; use body rates (bits 0-2 = 0) and thrust (bit 6 = 0)
_TYPEMASK_RATES_AND_THRUST = 0b10000000


class OfficialSimAdapter(AbstractAdapter):
    """Live adapter for the AI Grand Prix DCL simulator."""

    def __init__(self, config: Config):
        self._cfg: OfficialAdapterConfig = config.official

        # Latest sensor data — written by threads, read by step()
        self._latest_frame:     Optional[np.ndarray] = None
        self._latest_frame_t_ns: int = 0
        self._latest_attitude   = None   # pymavlink ATTITUDE message
        self._latest_imu        = None   # pymavlink HIGHRES_IMU message

        self._telem_lock  = threading.Lock()
        self._vision_lock = threading.Lock()
        self._send_lock   = threading.Lock()   # serialises all MAVLink sends
        self._new_frame   = threading.Event()  # set when a complete frame arrives

        self._stop_evt = threading.Event()
        self._threads: list[threading.Thread] = []

        self._t0:     float = 0.0
        self._last_t: float = 0.0

        self._mav    = None   # pymavlink connection — opened in reset()
        self._mavutil = None  # module reference, set alongside _mav

    # ------------------------------------------------------------------
    # AbstractAdapter interface
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> Observation:
        """Connect to the simulator and return the first observation."""
        self._stop_threads()
        self._stop_evt.clear()

        self._latest_frame     = None
        self._latest_attitude  = None
        self._latest_imu       = None
        self._new_frame.clear()

        self._connect()
        self._start_threads()

        # Block until telemetry and the first vision frame both arrive
        deadline = time.monotonic() + self._cfg.connect_timeout_s
        while time.monotonic() < deadline:
            with self._telem_lock:
                has_telem = self._latest_imu is not None
            with self._vision_lock:
                has_frame = self._latest_frame is not None
            if has_telem and has_frame:
                break
            time.sleep(0.05)

        self._t0     = time.monotonic()
        self._last_t = 0.0
        return self._build_obs()

    def step(self, action: Action) -> tuple[Observation, dict]:
        """Send a control command and return the next observation."""
        self._send_control(action)

        # Sync to the vision stream: wait for a fresh frame (100 ms > 1/30 Hz)
        self._new_frame.clear()
        self._new_frame.wait(timeout=0.1)

        obs  = self._build_obs()
        done = obs.t >= self._cfg.max_run_s
        info = {
            "done":        done,
            "gate_passed": False,   # ProgressLobe infers this from area transitions
            "gate_index":  0,
            "lap_time":    obs.t if done else 0.0,
        }
        return obs, info

    def close(self) -> None:
        self._stop_threads()
        if self._mav is not None:
            try:
                self._mav.close()
            except Exception:
                pass
            self._mav = None

    # ------------------------------------------------------------------
    # MAVLink connection
    # ------------------------------------------------------------------

    def _connect(self) -> None:
        try:
            from pymavlink import mavutil
        except ImportError as exc:
            raise ImportError(
                "pymavlink is required for OfficialSimAdapter.\n"
                "Install with: pip install pymavlink"
            ) from exc

        if self._mav is not None:
            try:
                self._mav.close()
            except Exception:
                pass

        self._mavutil = mavutil
        self._mav = mavutil.mavlink_connection(self._cfg.mavlink_connection)
        # Block until the simulator sends its first HEARTBEAT
        self._mav.wait_heartbeat(timeout=self._cfg.connect_timeout_s)

    # ------------------------------------------------------------------
    # Threads
    # ------------------------------------------------------------------

    def _start_threads(self) -> None:
        specs = [
            ("agp-heartbeat", self._heartbeat_loop),
            ("agp-telemetry", self._telemetry_loop),
            ("agp-vision",    self._vision_loop),
        ]
        self._threads = []
        for name, target in specs:
            t = threading.Thread(target=target, daemon=True, name=name)
            t.start()
            self._threads.append(t)

    def _stop_threads(self) -> None:
        self._stop_evt.set()
        for t in self._threads:
            t.join(timeout=2.0)
        self._threads = []

    def _heartbeat_loop(self) -> None:
        """Send MAVLink HEARTBEAT at the configured rate (spec §4.4 requires ≥2 Hz)."""
        mavutil  = self._mavutil
        interval = 1.0 / max(self._cfg.heartbeat_rate_hz, 2.0)
        while not self._stop_evt.is_set():
            with self._send_lock:
                self._mav.mav.heartbeat_send(
                    mavutil.mavlink.MAV_TYPE_GCS,
                    mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                    0, 0, 0,
                )
            time.sleep(interval)

    def _telemetry_loop(self) -> None:
        """Read ATTITUDE and HIGHRES_IMU from the MAVLink stream."""
        while not self._stop_evt.is_set():
            msg = self._mav.recv_match(
                type=["ATTITUDE", "HIGHRES_IMU"],
                blocking=True,
                timeout=0.1,
            )
            if msg is None:
                continue
            with self._telem_lock:
                t = msg.get_type()
                if t == "ATTITUDE":
                    self._latest_attitude = msg
                elif t == "HIGHRES_IMU":
                    self._latest_imu = msg

    def _vision_loop(self) -> None:
        """Receive and reassemble chunked JPEG frames from UDP port 5600 (spec §4.6)."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self._cfg.vision_host, self._cfg.vision_port))
        sock.settimeout(self._cfg.vision_recv_timeout_s)

        # frame_id → {chunk_id: payload_bytes}
        frame_buf:  dict[int, dict[int, bytes]] = {}
        # frame_id → (total_chunks, jpeg_size, sim_time_ns)
        frame_meta: dict[int, tuple]            = {}

        try:
            while not self._stop_evt.is_set():
                try:
                    data = sock.recv(65535)
                except socket.timeout:
                    continue
                except OSError:
                    break

                if len(data) < _HDR_SIZE:
                    continue

                frame_id, chunk_id, total_chunks, jpeg_size, payload_size, sim_time_ns = \
                    struct.unpack_from(_HDR_FMT, data, 0)
                payload = data[_HDR_SIZE: _HDR_SIZE + payload_size]

                if frame_id not in frame_buf:
                    frame_buf[frame_id]  = {}
                    frame_meta[frame_id] = (total_chunks, jpeg_size, sim_time_ns)

                frame_buf[frame_id][chunk_id] = payload

                if len(frame_buf[frame_id]) == total_chunks:
                    jpeg_bytes = b"".join(
                        frame_buf[frame_id][i] for i in range(total_chunks)
                    )
                    img_arr   = np.frombuffer(jpeg_bytes, dtype=np.uint8)
                    frame_bgr = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

                    if frame_bgr is not None:
                        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                        with self._vision_lock:
                            self._latest_frame      = frame_rgb
                            self._latest_frame_t_ns = sim_time_ns
                        self._new_frame.set()

                    # Purge frames older than 5 behind the latest
                    for old in [k for k in frame_buf if k < frame_id - 5]:
                        del frame_buf[old]
                        frame_meta.pop(old, None)
        finally:
            sock.close()

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    def _send_control(self, action: Action) -> None:
        """Translate Action → SET_ATTITUDE_TARGET and send via MAVLink."""
        cfg     = self._cfg
        clamped = action.clamp()

        roll_rate  = clamped.roll     * cfg.max_roll_rate_rad_s
        pitch_rate = clamped.pitch    * cfg.max_pitch_rate_rad_s
        yaw_rate   = clamped.yaw     * cfg.max_yaw_rate_rad_s
        thrust     = float(clamped.throttle)

        time_boot_ms = int(time.monotonic() * 1000) & 0xFFFFFFFF

        with self._send_lock:
            self._mav.mav.set_attitude_target_send(
                time_boot_ms,
                cfg.target_system,
                cfg.target_component,
                _TYPEMASK_RATES_AND_THRUST,
                [1.0, 0.0, 0.0, 0.0],   # quaternion — ignored per type_mask
                roll_rate,
                pitch_rate,
                yaw_rate,
                thrust,
            )

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _build_obs(self) -> Observation:
        with self._telem_lock:
            imu = self._latest_imu
            att = self._latest_attitude
        with self._vision_lock:
            frame      = self._latest_frame
            frame_t_ns = self._latest_frame_t_ns

        t  = time.monotonic() - self._t0
        dt = (t - self._last_t) if self._last_t > 0 else (1.0 / 30.0)
        self._last_t = t

        # IMU: spec §3.8 uses NED — HIGHRES_IMU exposes xacc/yacc/zacc (m/s²)
        # and xgyro/ygyro/zgyro (rad/s) in body frame
        if imu is not None:
            imu_accel = np.array([imu.xacc,  imu.yacc,  imu.zacc],  dtype=np.float64)
            imu_gyro  = np.array([imu.xgyro, imu.ygyro, imu.zgyro], dtype=np.float64)
        else:
            imu_accel = np.array([0.0, 0.0, 9.81], dtype=np.float64)
            imu_gyro  = np.zeros(3, dtype=np.float64)

        if frame is None:
            # Return black frame until vision arrives; pipeline will detect nothing
            frame = np.zeros((360, 640, 3), dtype=np.uint8)

        # Camera is tilted 20° upward from body forward (spec §3.8).
        # Stored in meta so downstream lobes can apply corrections if needed.
        meta: dict = {"camera_tilt_deg": self._cfg.camera_tilt_deg}
        if att is not None:
            meta["attitude"] = (att.roll, att.pitch, att.yaw)
        if frame_t_ns:
            meta["frame_t_ns"] = frame_t_ns

        return Observation(
            t=t,
            dt=dt,
            image=frame,
            imu_accel=imu_accel,
            imu_gyro=imu_gyro,
            meta=meta,
        )
