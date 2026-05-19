# AI Grand Prix — Round 1 Qualifier: Master Plan

**Document:** PLAN.md  
**Spec ref:** VADR-TS-002 Issue 00.02 (2026-05-08)  
**Competition deadline:** Round 1 is live — 8-minute maximum run  
**Last updated:** 2026-05-18  

---

## Read this first (cold-start summary)

This project is a fully built autonomous drone racing AI pipeline. The architecture
(lobes → brain → controller) is complete and battle-tested in the mock simulator.
**The only missing piece for competition is `adapters/official.py`** — a real
implementation that talks to the DCL/AI Grand Prix simulator over MAVLink + UDP video.

Once the official adapter works, you flip one line in the config and race.

---

## Current state audit

| Component | Status | Notes |
|---|---|---|
| `adapters/mock.py` | ✅ Complete | Full mock sim, perspective + HSV modes |
| `adapters/official.py` | ❌ Stub | Raises NotImplementedError — must implement |
| `lobes/vision.py` (HSV) | ✅ Complete | Works on high-contrast gates |
| `lobes/vision_ml.py` (ML) | ✅ Complete | 0.56ms/frame on GPU, robust |
| `ml/model.py` (GateDetector) | ✅ Trained | 26K-param CNN, .pt checkpoint exists |
| `brain/fusion.py` | ✅ Complete | SEARCH→TRACK→APPROACH→COMMIT→RECOVER |
| `controller/pid.py` | ✅ Complete | Per-state PID profiles |
| `configs/r1_qualifier.yaml` | ⚠️ Partial | Tuned for mock, not yet wired to official adapter |
| `configs/base.yaml` | ✅ Complete | All defaults |

---

## Priority 1 — Implement the Official Adapter

**File:** `aigrandprix/adapters/official.py`  
**Replaces:** current stub that raises `NotImplementedError`

The adapter must implement three methods from `AbstractAdapter`:
- `reset(seed)` → `Observation`
- `step(action)` → `(Observation, info_dict)`
- `close()`

### 1a. Dependencies to install

```bash
pip install pymavlink opencv-python numpy
# If using MAVSDK instead: pip install mavsdk
```

Recommend **pymavlink** — it's lower-level but matches the spec exactly. MAVSDK wraps
it but adds abstraction overhead.

### 1b. MAVLink connection

```python
from pymavlink import mavutil

# Spec §4.2: UDP transport
# Config: official.mavlink_connection = "udpin:0.0.0.0:14551"
# "udpin" means WE listen, sim sends TO us on port 14551
self._mav = mavutil.mavlink_connection(cfg.mavlink_connection)
self._mav.wait_heartbeat(timeout=cfg.connect_timeout_s)
```

### 1c. Heartbeat thread (MANDATORY — spec §4.4 requires ≥2Hz)

Run this in a daemon thread. Config default is 4Hz.

```python
def _heartbeat_loop(self):
    interval = 1.0 / self._cfg.heartbeat_rate_hz
    while not self._stop_evt.is_set():
        self._mav.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_GCS,
            mavutil.mavlink.MAV_AUTOPILOT_INVALID,
            0, 0, 0
        )
        time.sleep(interval)
```

### 1d. Telemetry receive thread

Listen for ATTITUDE and HIGHRES_IMU messages. Store latest in thread-safe variables.

```python
# ATTITUDE message fields: roll, pitch, yaw (rad), rollspeed, pitchspeed, yawspeed (rad/s)
# HIGHRES_IMU fields: xacc, yacc, zacc (m/s^2), xgyro, ygyro, zgyro (rad/s)

def _telemetry_loop(self):
    while not self._stop_evt.is_set():
        msg = self._mav.recv_match(
            type=['ATTITUDE', 'HIGHRES_IMU'], blocking=True, timeout=0.1
        )
        if msg is None:
            continue
        with self._telem_lock:
            if msg.get_type() == 'ATTITUDE':
                self._latest_attitude = msg
            elif msg.get_type() == 'HIGHRES_IMU':
                self._latest_imu = msg
```

### 1e. Vision stream receiver thread

**Spec §4.6:** UDP port 5600, chunked JPEG frames, 30Hz.

Header structure (24 bytes, little-endian):
```
frame_id      uint32  4B   unique sequence ID
chunk_id      uint16  2B   index within frame (0..total_chunks-1)
total_chunks  uint16  2B   total packets for this frame
jpeg_size     uint32  4B   total reconstructed JPEG size
payload_size  uint32  4B   JPEG data in this packet
sim_time_ns   uint64  8B   nanosecond timestamp
```

```python
import socket
import struct

HEADER_FMT = '<IHHIIQ'   # little-endian: uint32 uint16 uint16 uint32 uint32 uint64
HEADER_SIZE = 24         # struct.calcsize(HEADER_FMT) == 24

def _vision_loop(self):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((self._cfg.vision_host, self._cfg.vision_port))
    sock.settimeout(self._cfg.vision_recv_timeout_s)

    frame_buffer = {}   # frame_id → {chunk_id: bytes}
    frame_meta = {}     # frame_id → (total_chunks, jpeg_size, sim_time_ns)

    while not self._stop_evt.is_set():
        try:
            data = sock.recv(65535)
        except socket.timeout:
            continue

        # Parse header
        hdr = struct.unpack_from(HEADER_FMT, data, 0)
        frame_id, chunk_id, total_chunks, jpeg_size, payload_size, sim_time_ns = hdr
        payload = data[HEADER_SIZE : HEADER_SIZE + payload_size]

        if frame_id not in frame_buffer:
            frame_buffer[frame_id] = {}
            frame_meta[frame_id] = (total_chunks, jpeg_size, sim_time_ns)

        frame_buffer[frame_id][chunk_id] = payload

        # Check if all chunks arrived
        if len(frame_buffer[frame_id]) == total_chunks:
            jpeg_bytes = b''.join(
                frame_buffer[frame_id][i] for i in range(total_chunks)
            )
            img_array = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            frame_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            with self._vision_lock:
                self._latest_frame = frame_rgb
                self._latest_frame_t_ns = sim_time_ns

            # Clean up old frames (prevent memory leak)
            for old_id in list(frame_buffer.keys()):
                if old_id < frame_id - 5:
                    del frame_buffer[old_id]
                    frame_meta.pop(old_id, None)
```

### 1f. Control command sender

**Spec §4.3:** Use `SET_ATTITUDE_TARGET` for control.  
Config scales Action [-1,1] → rad/s using `max_*_rate_rad_s`.

```python
import struct

def _send_control(self, action: Action):
    cfg = self._cfg
    # Roll/pitch/yaw rates in rad/s (body frame, NED convention)
    roll_rate  = action.roll     * cfg.max_roll_rate_rad_s
    pitch_rate = action.pitch    * cfg.max_pitch_rate_rad_s
    yaw_rate   = action.yaw     * cfg.max_yaw_rate_rad_s
    thrust     = float(np.clip(action.throttle, 0.0, 1.0))

    # Quaternion: use identity (0,0,0,1) — rates are what matters here
    # type_mask = 0b00000111 → ignore roll/pitch/yaw (use rates only) + body rates
    # type_mask bits: bit0=body_roll_rate, bit1=body_pitch_rate, bit2=body_yaw_rate
    # Setting 0 means USE this field. Check pymavlink SET_ATTITUDE_TARGET docs.
    # Typically type_mask=0b10000000 to use thrust + body rates.
    self._mav.mav.set_attitude_target_send(
        int(time.monotonic() * 1000) & 0xFFFFFFFF,  # time_boot_ms
        cfg.target_system,
        cfg.target_component,
        0b00000000,   # type_mask: use all fields
        [1.0, 0.0, 0.0, 0.0],  # quaternion (identity — rates override)
        roll_rate,
        pitch_rate,
        yaw_rate,
        thrust,
    )
```

> ⚠️ **IMPORTANT:** The exact `type_mask` bits need verification against the
> simulator's MAVLink implementation. Test with type_mask=0b00000000 first,
> then tune if the sim doesn't respond. The simulator may only read body rates.

### 1g. Build Observation from telemetry

```python
def _build_obs(self) -> Observation:
    with self._telem_lock:
        imu = self._latest_imu
        att = self._latest_attitude
    with self._vision_lock:
        frame = self._latest_frame
        frame_t_ns = self._latest_frame_t_ns

    t = time.monotonic() - self._t0

    # IMU: NED convention — HIGHRES_IMU gives (xacc, yacc, zacc) in m/s^2
    imu_accel = np.array([imu.xacc, imu.yacc, imu.zacc], dtype=np.float64)
    imu_gyro  = np.array([imu.xgyro, imu.ygyro, imu.zgyro], dtype=np.float64)

    # Camera tilt: 20° upward from body forward (spec §3.8)
    # Store in meta so VisionLobe can optionally correct cy offset
    meta = {
        'camera_tilt_deg': self._cfg.camera_tilt_deg,
        'frame_t_ns': frame_t_ns,
        'attitude': (att.roll, att.pitch, att.yaw) if att else None,
    }

    return Observation(
        t=t,
        dt=t - self._last_t if self._last_t > 0 else 1.0 / 30.0,
        image=frame if frame is not None else np.zeros((360, 640, 3), dtype=np.uint8),
        imu_accel=imu_accel,
        imu_gyro=imu_gyro,
        meta=meta,
    )
```

### 1h. `reset()` and `step()` implementation

```python
def reset(self, seed=None) -> Observation:
    # Start threads if not already running
    self._start_threads()
    # Wait for first telemetry and frame
    deadline = time.monotonic() + self._cfg.connect_timeout_s
    while time.monotonic() < deadline:
        if self._latest_frame is not None and self._latest_imu is not None:
            break
        time.sleep(0.05)
    self._t0 = time.monotonic()
    self._last_t = 0.0
    return self._build_obs()

def step(self, action: Action) -> tuple[Observation, dict]:
    self._send_control(action)
    # Sleep to match ~30Hz step rate (vision stream rate)
    time.sleep(1.0 / 30.0)
    obs = self._build_obs()
    self._last_t = obs.t

    # Check timeout
    done = obs.t >= self._cfg.max_run_s
    info = {
        'done': done,
        'gate_passed': False,   # official sim doesn't expose this; inferred by ProgressLobe
        'gate_index': 0,        # ditto
        'lap_time': obs.t if done else 0.0,
    }
    return obs, info
```

---

## Priority 2 — Config for Official Sim

Create `configs/official_r1.yaml`:

```yaml
adapter:
  type: "official"

official:
  mavlink_connection: "udpin:0.0.0.0:14551"
  vision_host: "0.0.0.0"
  vision_port: 5600
  connect_timeout_s: 20.0
  heartbeat_rate_hz: 4.0
  max_run_s: 480.0
  max_roll_rate_rad_s: 6.0
  max_pitch_rate_rad_s: 6.0
  max_yaw_rate_rad_s: 4.0
  camera_tilt_deg: 20.0

vision:
  backend: "ml"              # ML is more robust — use it for competition
  model_path: "models/gate_detector.pt"
  ml_input_h: 128
  ml_input_w: 160
  ml_conf_threshold: 0.45    # slightly lower to catch gates early
  resize_h: 180              # half of 360px camera height
  resize_w: 320              # half of 640px camera width

state_machine:
  track_confidence_min: 0.35
  commit_area_threshold: 35000.0   # re-tune for real gate size in camera

sim:
  fps: 30                    # real camera is 30Hz
```

---

## Priority 3 — Camera Geometry Notes

These are fixed by the spec (§3.7, §3.8) — critical for tuning thresholds:

| Property | Value |
|---|---|
| Camera resolution | 640 × 360 px |
| Principal point [cx, cy] | [320, 180] |
| Focal length [fx, fy] | [320, 320] |
| VFoV | 90° |
| Camera tilt | 20° upward from body |
| Gate outer | 2700 × 2700 mm |
| Gate inner opening | 1500 × 1500 mm |
| Drone body | 280 × 280 × 160 mm |

**Useful derived fact:** At 5m distance, gate inner (1500mm) subtends:
`pixel_width = fx * (1500 / 5000) = 320 * 0.3 = 96px`

So at 5m the gate fills ~15% of frame width. At 2m it fills ~37%. Use this to
calibrate `commit_area_threshold` for when to fire COMMIT state.

**Camera tilt correction:** The 20° upward tilt means gates appear LOWER in
the frame than their true vertical position. If the drone is level, the gate
center will be at ~`cy + fx * tan(20°) = 180 + 320 * 0.364 ≈ 297px` (in NED
terms, forward-down tilts the image up). Account for this in `cy` offsets if
centering is off.

---

## Priority 4 — Competition Entry Script

Create `scripts/compete.py`:

```python
"""Entry point for official competition run."""
from aigrandprix.config import load_config
from aigrandprix.runner import PipelineRunner

cfg = load_config("configs/base.yaml", "configs/official_r1.yaml")
runner = PipelineRunner(cfg)
result = runner.run(track_id="r1_qualifier")
print(f"Lap time: {result['lap_time']:.2f}s | Gates: {result['gate_count']}")
```

Run with: `python scripts/compete.py`

---

## Testing Plan

### Phase 1: Adapter unit tests (no sim required)

Test threading and packet parsing in isolation:

1. **Vision packet parser test** — build fake UDP packets with the 24-byte header,
   send to a local socket, verify the reassembled JPEG decodes to correct shape.
   File: `tests/unit/test_official_adapter.py`

2. **MAVLink mock test** — use pymavlink's `mavutil.mavlink_connection('udpout:...')`
   to inject fake ATTITUDE + HIGHRES_IMU messages, verify `_build_obs()` returns
   correct `imu_accel` / `imu_gyro` arrays.

3. **Heartbeat rate test** — start adapter, count heartbeats over 2 seconds, assert ≥ 4.

### Phase 2: Integration with simulator (when sim is running)

1. **Connection smoke test** — run `scripts/compete.py` with logging on, verify
   first frame arrives within `connect_timeout_s`, check log for non-zero `imu_accel`.

2. **Vision stream latency test** — log `sim_time_ns` vs. `time.monotonic()` for
   100 frames, check p95 latency < 50ms.

3. **Control response test** — send constant pitch=0.2 for 2 seconds, verify
   drone moves forward via attitude change in telemetry.

4. **End-to-end smoke** — run full pipeline for 60 seconds, verify state machine
   cycles through SEARCH → TRACK → APPROACH, no crashes.

### Phase 3: Tuning

1. **Gate color check** — screenshot the first frame. If HSV backend is used,
   tune `hsv_lower/upper` for actual gate color. ML backend should handle this
   automatically.

2. **Commit threshold** — watch logs for `COMMIT` state entries. If triggering
   too early (drone clips gate) raise `commit_area_threshold`. Too late → slow.

3. **PID tuning** — monitor `pipeline_ms` per step. If oscillating on approach,
   lower `pitch_kp` in APPROACH profile. If too slow to center, raise `yaw_kp`.

4. **Camera tilt compensation** — if drone consistently flies below gates, the
   20° tilt means true gate center is above `cy`. Adjust `ProgressLobe` `dy`
   calculation to subtract tilt offset.

---

## Competition Day Checklist

### Before connecting to sim
- [ ] `pip install pymavlink opencv-python numpy torch` verified on competition machine
- [ ] ML model checkpoint at path matching `configs/official_r1.yaml` `model_path`
- [ ] `adapter.type = "official"` in config (not "mock")
- [ ] Firewall allows UDP ports 14551 and 5600 inbound
- [ ] Sim machine IP known — update `mavlink_connection` if sim sends from specific IP

### First connection
- [ ] `python scripts/compete.py` — watch for "heartbeat received" log line
- [ ] Check first frame is not black (vision socket working)
- [ ] Check `imu_accel` is non-zero (telemetry flowing)

### During run
- [ ] Logs going to `logs/` directory
- [ ] `pipeline_ms` staying under 15ms (watch terminal output)
- [ ] No "RECOVER" state flooding (would indicate gate loss)

### If something breaks
- Common: vision socket blocked → check port 5600, try `configs/official.vision_port`
- Common: no telemetry → check port 14551, verify `udpin` vs `udpout` direction
- Common: drone not moving → wrong `type_mask` in SET_ATTITUDE_TARGET, check §1f above
- Common: gates not detected → use HSV backend + screenshot gate color + tune HSV range

---

## Key Spec Facts (quick reference)

| Item | Value | Source |
|---|---|---|
| MAVLink transport | UDP | §4.2 |
| MAVLink listen port | 14551 (config default) | OfficialAdapterConfig |
| Control message | SET_ATTITUDE_TARGET | §4.3 |
| Heartbeat rate | ≥ 2Hz (use 4Hz) | §4.4 |
| Physics rate | 120 Hz | §4.4 |
| Command rate | < 100 Hz | §4.4 |
| Vision port | 5600 | §4.6 |
| Vision rate | 30 Hz | §4.6 |
| Vision resolution | 640 × 360 | §4.6 |
| Vision header size | 24 bytes | §4.6 |
| Coordinate frame | NED (North-East-Down) | §3.8 |
| Camera tilt | 20° upward from body | §3.8 |
| Max run time | 8 minutes (480s) | §8.3 |
| Gate outer | 2700 × 2700 mm | §3.7 |
| Gate inner | 1500 × 1500 mm | §3.7 |
| No GPS | confirmed | §3.3 |
| Platform | Windows 11, Python 3.14.2 | §5.1 |

---

## Issues Found and Fixed (2026-05-18)

| # | Severity | Issue | Fix |
|---|---|---|---|
| 1 | **Critical** | Camera 20° tilt → gate appears at cy≈0.82, not 0.5 → `aligned_score≈0.55` → NEVER reaches APPROACH threshold (0.65) | Added `cy_tilt_offset: 0.16` to `ProgressConfig`; lobes/progress.py uses it for dy |
| 2 | **Critical** | MLVisionLobe ran on CPU only despite 8GB VRAM GPU | `vision_ml.py` now auto-detects CUDA, moves model+tensors to GPU |
| 3 | **Critical** | Missing model file → silent `gate_detected=False` every frame → drone spins in SEARCH forever | `compete.py` validates model path + loads it before connecting |
| 4 | **Performance** | TRACK throttle 0.5, APPROACH 0.65 too slow for competition | `official_r1.yaml`: TRACK→0.60, APPROACH→0.75, COMMIT→0.92 |
| 5 | **Performance** | RECOVER throttle 0.35 → drone loses altitude during recovery → misses next gate | RECOVER→0.50 in `official_r1.yaml` |
| 6 | **Performance** | push_throttle_step 0.03 → max push boost only +9% | Raised to 0.05 → +15% at max push |

## Open Roadmap (lower priority)

- Velocity estimation using IMU integration + gate area growth rate (§roadmap)
- Fine-tune ML model on actual sim screenshots (run `scripts/train_detector.py`)
- `ProgressLobe` camera tilt correction for `dy` centering
- ML bbox size calibration (currently over-predicts gate size by ~15%)
- Webcam adapter for local camera testing

---

## File Map (most important files)

```
aigrandprix/
  adapters/
    official.py     ← IMPLEMENT THIS (Priority 1)
    mock.py         ← reference implementation, patterns to copy
    base.py         ← AbstractAdapter interface
  config.py         ← OfficialAdapterConfig already defined (line 174)
  types.py          ← Observation, Action, VisionResult contracts
  runner.py         ← PipelineRunner, wires everything together
  brain/fusion.py   ← state machine (do not touch unless tuning)
  lobes/vision_ml.py ← ML vision backend (preferred for competition)

configs/
  base.yaml         ← all defaults
  r1_qualifier.yaml ← mock-tuned R1 overrides (needs official adapter version)
  official_r1.yaml  ← CREATE THIS (Priority 2)

scripts/
  compete.py        ← CREATE THIS (Priority 4)
  demo.py           ← reference for how to call PipelineRunner
```
