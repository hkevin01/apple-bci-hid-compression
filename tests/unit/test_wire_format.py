# pylint: disable=missing-function-docstring, import-error
import pytest

from src.core.wire_format import (
    MAGIC,
    VERSION,
    decode_header,
    encode_header,
    validate_crc,
)


def test_encode_decode_header_roundtrip() -> None:
    payload = b"hello world"
    hdr_bytes = encode_header(
        frames=128,
        channels=4,
        level=3,
        top_k_scaled=1234,
        wavelet_name="db4",
        payload=payload,
    )
    hdr, size = decode_header(hdr_bytes)
    assert hdr.magic == MAGIC
    assert hdr.version == VERSION
    assert size == len(hdr_bytes)
    assert hdr.frames == 128
    assert hdr.channels == 4
    assert hdr.level == 3
    assert hdr.top_k_scaled == 1234
    assert hdr.wavelet_name == "db4"
    assert hdr.payload_len == len(payload)
    # crc validates
    validate_crc(payload, hdr.crc32)


def test_decode_invalid_magic() -> None:
    payload = b"x" * 10
    good = encode_header(
        frames=1,
        channels=1,
        level=1,
        top_k_scaled=0,
        wavelet_name="haar",
        payload=payload,
    )
    bad = b"XXXX" + good[4:]
    with pytest.raises(ValueError):
        decode_header(bad)


def test_crc_mismatch_raises() -> None:
    payload = b"abcdef"
    hdr = encode_header(
        frames=2,
        channels=2,
        level=2,
        top_k_scaled=2000,
        wavelet_name="sym2",
        payload=payload,
    )
    parsed, _ = decode_header(hdr)
    with pytest.raises(ValueError):
        validate_crc(b"wrong", parsed.crc32)
