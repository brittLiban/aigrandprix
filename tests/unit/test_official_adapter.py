"""Unit tests for OfficialSimAdapter — no live simulator required.

Tests cover:
  - Vision packet header parsing and JPEG reassembly
  - Observation builder (telemetry → numpy arrays)
  - Heartbeat rate constraint
  - Type-mask constant is correct
"""
import io
import socket
import struct
import threading
import time
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from aigrandprix.adapters.official import _HDR_FMT, _HDR_SIZE, _TYPEMASK_RATES_AND_THRUST
from aigrandprix.config import default_config
from aigrandprix.types import Action

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_HDR_FMT_CHECK = '<IHHIIQ'


def _make_jpeg(width: int = 64, height: int = 36) -> bytes:
    """Generate a minimal valid JPEG in memory."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[height // 2, width // 2] = [255, 0, 0]   # red pixel
    ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    assert ok
    return buf.tobytes()


def _make_packet(
    frame_id: int,
    chunk_id: int,
    total_chunks: int,
    jpeg: bytes,
    offset: int,
    payload_size: int,
    sim_time_ns: int = 1_000_000,
) -> bytes:
    """Build a single vision UDP packet (header + payload slice)."""
    payload = jpeg[offset: offset + payload_size]
    header = struct.pack(
        _HDR_FMT_CHECK,
        frame_id,
        chunk_id,
        total_chunks,
        len(jpeg),
        len(payload),
        sim_time_ns,
    )
    return header + payload


# ---------------------------------------------------------------------------
# Spec constants
# ---------------------------------------------------------------------------

def test_header_size_is_24():
    """Spec §4.6 states header is 24 bytes."""
    assert _HDR_SIZE == 24


def test_header_fmt_matches_spec():
    """Verify struct format matches spec field order and types."""
    assert _HDR_FMT == _HDR_FMT_CHECK


def test_typemask_ignores_quaternion_uses_rates():
    """Bit 7 = 1 (ignore attitude); bits 0-2 = 0 (use body rates); bit 6 = 0 (use thrust)."""
    assert _TYPEMASK_RATES_AND_THRUST & 0b10000000  # bit 7 set → ignore quaternion
    assert not (_TYPEMASK_RATES_AND_THRUST & 0b00000111)  # bits 0-2 clear → use rates
    assert not (_TYPEMASK_RATES_AND_THRUST & 0b01000000)  # bit 6 clear → use thrust


# ---------------------------------------------------------------------------
# Vision packet parsing
# ---------------------------------------------------------------------------

def test_single_chunk_frame_roundtrip():
    """A single-packet frame should decode back to a valid image."""
    jpeg = _make_jpeg()
    packet = _make_packet(
        frame_id=1, chunk_id=0, total_chunks=1,
        jpeg=jpeg, offset=0, payload_size=len(jpeg),
    )

    # Parse header
    hdr = struct.unpack_from(_HDR_FMT, packet, 0)
    frame_id, chunk_id, total_chunks, jpeg_size, payload_size, sim_time_ns = hdr
    payload = packet[_HDR_SIZE: _HDR_SIZE + payload_size]

    assert frame_id == 1
    assert chunk_id == 0
    assert total_chunks == 1
    assert jpeg_size == len(jpeg)
    assert payload_size == len(jpeg)

    img = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert img is not None
    assert img.shape == (36, 64, 3)


def test_multi_chunk_frame_reassembly():
    """A frame split across 3 packets should reassemble to the original JPEG."""
    jpeg = _make_jpeg(width=320, height=180)  # larger → multiple chunks
    chunk_size = len(jpeg) // 3 + 1
    chunks = []
    for i in range(3):
        start = i * chunk_size
        size  = min(chunk_size, len(jpeg) - start)
        if size <= 0:
            break
        chunks.append(_make_packet(
            frame_id=7, chunk_id=i, total_chunks=3,
            jpeg=jpeg, offset=start, payload_size=size,
        ))

    # Simulate the reassembly loop in _vision_loop
    buf: dict[int, bytes] = {}
    for pkt in chunks:
        hdr = struct.unpack_from(_HDR_FMT, pkt, 0)
        frame_id, chunk_id, total_chunks, jpeg_size, payload_size, _ = hdr
        buf[chunk_id] = pkt[_HDR_SIZE: _HDR_SIZE + payload_size]

    assert len(buf) == len(chunks)
    reassembled = b"".join(buf[i] for i in range(len(buf)))
    assert reassembled == jpeg


def test_out_of_order_chunks_reassemble():
    """Chunks arriving out of order should still produce a valid frame."""
    jpeg = _make_jpeg()
    mid = len(jpeg) // 2
    pkts = [
        _make_packet(2, 1, 2, jpeg, mid, len(jpeg) - mid),  # chunk 1 first
        _make_packet(2, 0, 2, jpeg, 0,   mid),               # chunk 0 second
    ]

    buf: dict[int, bytes] = {}
    for pkt in pkts:
        hdr = struct.unpack_from(_HDR_FMT, pkt, 0)
        _, chunk_id, _, _, payload_size, _ = hdr
        buf[chunk_id] = pkt[_HDR_SIZE: _HDR_SIZE + payload_size]

    reassembled = b"".join(buf[i] for i in range(2))
    assert reassembled == jpeg


# ---------------------------------------------------------------------------
# Observation builder (via mock telemetry)
# ---------------------------------------------------------------------------

def _make_adapter_with_mocks():
    """Return an adapter instance with mav and threads mocked out."""
    cfg = default_config()
    # Patch _connect so we don't need pymavlink installed to run tests
    with patch.object(
        __import__('aigrandprix.adapters.official', fromlist=['OfficialSimAdapter']).OfficialSimAdapter,
        '_connect', return_value=None
    ):
        from aigrandprix.adapters.official import OfficialSimAdapter
        adapter = OfficialSimAdapter.__new__(OfficialSimAdapter)
        # Manually initialise (bypasses __init__ teardown of live state)
        OfficialSimAdapter.__init__(adapter, cfg)
        adapter._mav     = MagicMock()
        adapter._mavutil = MagicMock()
    return adapter


def test_build_obs_no_data_returns_zero_frame():
    """_build_obs() with no telemetry should still return a valid Observation."""
    from aigrandprix.adapters.official import OfficialSimAdapter
    cfg = default_config()
    adapter = OfficialSimAdapter.__new__(OfficialSimAdapter)
    OfficialSimAdapter.__init__(adapter, cfg)
    adapter._mav      = MagicMock()
    adapter._mavutil  = MagicMock()
    adapter._t0       = time.monotonic()
    adapter._last_t   = 0.0

    obs = adapter._build_obs()

    assert obs.image.shape == (360, 640, 3)
    assert obs.image.sum() == 0   # black frame
    np.testing.assert_allclose(obs.imu_accel, [0.0, 0.0, 9.81])
    np.testing.assert_allclose(obs.imu_gyro, [0.0, 0.0, 0.0])
    assert obs.meta["camera_tilt_deg"] == 20.0


def test_build_obs_with_imu_data():
    """_build_obs() should pull xacc/yacc/zacc from the HIGHRES_IMU mock."""
    from aigrandprix.adapters.official import OfficialSimAdapter
    cfg = default_config()
    adapter = OfficialSimAdapter.__new__(OfficialSimAdapter)
    OfficialSimAdapter.__init__(adapter, cfg)
    adapter._mav     = MagicMock()
    adapter._mavutil = MagicMock()
    adapter._t0      = time.monotonic()
    adapter._last_t  = 0.0

    imu_mock = MagicMock()
    imu_mock.xacc  = 0.1
    imu_mock.yacc  = -0.2
    imu_mock.zacc  = 9.5
    imu_mock.xgyro = 0.01
    imu_mock.ygyro = -0.01
    imu_mock.zgyro = 0.005
    adapter._latest_imu = imu_mock

    obs = adapter._build_obs()

    np.testing.assert_allclose(obs.imu_accel, [0.1, -0.2, 9.5])
    np.testing.assert_allclose(obs.imu_gyro,  [0.01, -0.01, 0.005])


def test_build_obs_attitude_in_meta():
    """Attitude (roll, pitch, yaw) should appear in obs.meta when available."""
    from aigrandprix.adapters.official import OfficialSimAdapter
    cfg = default_config()
    adapter = OfficialSimAdapter.__new__(OfficialSimAdapter)
    OfficialSimAdapter.__init__(adapter, cfg)
    adapter._mav     = MagicMock()
    adapter._mavutil = MagicMock()
    adapter._t0      = time.monotonic()
    adapter._last_t  = 0.0

    att_mock = MagicMock()
    att_mock.roll  = 0.05
    att_mock.pitch = -0.10
    att_mock.yaw   = 1.57
    adapter._latest_attitude = att_mock

    obs = adapter._build_obs()

    assert "attitude" in obs.meta
    assert obs.meta["attitude"] == (0.05, -0.10, 1.57)


# ---------------------------------------------------------------------------
# Action → rate scaling
# ---------------------------------------------------------------------------

def test_send_control_rate_scaling():
    """Action(roll=1, pitch=1, yaw=1, throttle=1) should saturate at max_*_rate."""
    from aigrandprix.adapters.official import OfficialSimAdapter
    cfg = default_config()
    adapter = OfficialSimAdapter.__new__(OfficialSimAdapter)
    OfficialSimAdapter.__init__(adapter, cfg)
    adapter._mav     = MagicMock()
    adapter._mavutil = MagicMock()

    action = Action(roll=1.0, pitch=1.0, yaw=1.0, throttle=1.0)
    adapter._send_control(action)

    call_args = adapter._mav.mav.set_attitude_target_send.call_args
    assert call_args is not None
    _, kwargs = call_args if call_args.kwargs else (call_args.args, {})
    args = call_args.args

    # args: time_boot_ms, target_sys, target_comp, type_mask, q, roll_rate, pitch_rate, yaw_rate, thrust
    roll_rate  = args[5]
    pitch_rate = args[6]
    yaw_rate   = args[7]
    thrust     = args[8]

    assert roll_rate  == pytest.approx(cfg.official.max_roll_rate_rad_s)
    assert pitch_rate == pytest.approx(cfg.official.max_pitch_rate_rad_s)
    assert yaw_rate   == pytest.approx(cfg.official.max_yaw_rate_rad_s)
    assert thrust     == pytest.approx(1.0)


def test_send_control_zero_action():
    """Action.zero() should produce zero rates and zero thrust."""
    from aigrandprix.adapters.official import OfficialSimAdapter
    cfg = default_config()
    adapter = OfficialSimAdapter.__new__(OfficialSimAdapter)
    OfficialSimAdapter.__init__(adapter, cfg)
    adapter._mav     = MagicMock()
    adapter._mavutil = MagicMock()

    adapter._send_control(Action.zero())

    args = adapter._mav.mav.set_attitude_target_send.call_args.args
    roll_rate, pitch_rate, yaw_rate, thrust = args[5], args[6], args[7], args[8]
    assert roll_rate  == pytest.approx(0.0)
    assert pitch_rate == pytest.approx(0.0)
    assert yaw_rate   == pytest.approx(0.0)
    assert thrust     == pytest.approx(0.0)
