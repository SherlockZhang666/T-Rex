"""tactile_convert pinned against the data converter it mirrors.

    .venv/bin/python -m pytest hardware_code/openarm/tactile_convert_test.py -q

The rollout client cannot import utils/gen_json_openarm_sharpa_left.py (its venv has no
cv2 / h5py), so tactile_convert re-states the constants and the two small functions. These
tests import BOTH and assert they agree, so an edit to the converter that is not mirrored
here fails a test instead of shipping a client whose fingers are in the wrong order.
"""

import pathlib
import sys

import numpy as np
import pytest

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
for p in (HERE, ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import tactile_convert as tc  # noqa: E402

gen = pytest.importorskip("utils.gen_json_openarm_sharpa_left")


# ------------------------------------------------------------------ constants agree
def test_finger_order_constants_match_the_converter():
    assert list(tc.COLLECTOR_FINGER_ORDER) == list(gen.COLLECTOR_FINGER_ORDER)
    assert list(tc.TREX_FINGER_ORDER) == list(gen.TREX_FINGER_ORDER)
    assert tc.COLLECTOR_TO_TREX == gen.COLLECTOR_TO_TREX
    assert tc.N_FINGERS_PER_HAND == gen.N_FINGERS_PER_HAND
    assert tc.F6_PER_FINGER == gen.F6_PER_FINGER
    assert tc.DEFORM_HW == gen.DEFORM_HW


def test_thumb_is_collector_channel_index_4_and_trex_index_0():
    assert tc.COLLECTOR_TO_TREX[0] == 4
    assert tc.COLLECTOR_TO_TREX == [4, 3, 2, 1, 0]


# ------------------------------------------------------------------ functions agree
def test_deform_floor_matches_the_converter_on_random_frames():
    rng = np.random.default_rng(1)
    frames = rng.integers(0, 255, size=(120, 5, 240, 240), dtype=np.uint8)
    frames[:, :, :100] = 2                      # a dominant floor
    assert tc.deform_floor(frames) == gen.deform_floor(frames) == 2


def test_f6_bias_is_the_mean_of_all_six_channels_as_the_converter_takes_it():
    """The converter: `bias = f6[:n_head].mean(axis=0)` over (5, 6) -- torques included."""
    rng = np.random.default_rng(2)
    head = rng.normal(size=(60, 5, 6)).astype(np.float32)
    attrs = {}                                  # no archived offset -> n_head_default
    bias_gen, n_head, _ = gen.f6_zero_offset(head, attrs, n_head_default=60)
    assert n_head == 60
    assert np.allclose(tc.f6_bias(head), bias_gen, atol=1e-6)


def test_convert_f6_window_reproduces_the_converters_line():
    """`f6 = (f6_raw - bias[None])[:, COLLECTOR_TO_TREX, :]`, verbatim."""
    rng = np.random.default_rng(3)
    raw = rng.normal(size=(16, 5, 6)).astype(np.float32)
    bias = rng.normal(size=(5, 6)).astype(np.float32)
    expected = (raw - bias[None])[:, gen.COLLECTOR_TO_TREX, :]
    assert np.allclose(tc.convert_f6_window(raw, bias), expected)


def test_convert_deform_reproduces_the_converters_lines():
    """`clip(int16 - floor, 0, 255).astype(uint8)` then `[:, COLLECTOR_TO_TREX]`."""
    rng = np.random.default_rng(4)
    raw = rng.integers(0, 20, size=(5, 240, 240), dtype=np.uint8)
    floor = 2
    expected = np.clip(raw.astype(np.int16) - floor, 0, 255).astype(np.uint8)[gen.COLLECTOR_TO_TREX]
    out = tc.convert_deform(raw, floor)
    assert out.dtype == np.uint8
    assert np.array_equal(out, expected)


def test_right_hand_padding_matches_the_converters_zero_blocks():
    f6 = np.ones((16, 5, 6), np.float32)
    padded = tc.pad_right_hand_f6(f6)
    assert padded.shape == (16, 10, 6)
    assert np.all(padded[:, :5] == 1) and np.all(padded[:, 5:] == 0)
    d = tc.pad_right_hand_deform(np.ones((5, 240, 240), np.uint8))
    assert d.shape == (10, 240, 240) and d[5:].max() == 0


# ------------------------------------------------------------------ refusals
def test_shapes_and_dtypes_are_enforced():
    with pytest.raises(ValueError):
        tc.convert_f6_window(np.zeros((16, 6, 6)), np.zeros((5, 6)))
    with pytest.raises(ValueError):
        tc.convert_deform(np.zeros((5, 240, 240), np.float32), 2)
    with pytest.raises(ValueError, match="10 frames"):
        tc.f6_bias(np.zeros((3, 5, 6)))
    with pytest.raises(ValueError, match="non-finite"):
        w = np.zeros((16, 5, 6)); w[0, 0, 0] = np.nan
        tc.convert_f6_window(w, np.zeros((5, 6)))
