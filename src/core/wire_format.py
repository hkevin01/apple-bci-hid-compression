"""Versioned wire format for compressed BCI frames.

This module defines a simple, extensible header format with a magic tag,
versioning, and CRC32 over the payload. The payload is opaque to this module
and interpreted by the compressor.

Layout (little-endian):
    - magic: 4 bytes = b"BCIW"
    - version: uint8
    - flags: uint8 (reserved for future use; 0 for now)
    - header_size: uint16 (bytes, including wavelet name section)
    - frames: uint32
    - channels: uint16
    - level: uint16
    - top_k_scaled: uint16  # stores ratio * 1e4
    - wavelet_name_len: uint16
    - payload_len: uint32
    - crc32: uint32  # CRC32 of payload only
    - wavelet_name: bytes[wavelet_name_len] (utf-8)

Followed by payload bytes of length payload_len.

This header aims to be compact yet flexible. "header_size" allows appending
fields after the crc32 in future versions while remaining parsable.
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass

MAGIC = b"BCIW"
VERSION = 1
_BASE_FMT = "<4sBBH I H H H H I I"  # see fields in WireHeader below
_BASE_SIZE = struct.calcsize(_BASE_FMT)


@dataclass(frozen=True)
class WireHeader:
    magic: bytes
    version: int
    flags: int
    header_size: int
    frames: int
    channels: int
    level: int
    top_k_scaled: int
    wavelet_name_len: int
    payload_len: int
    crc32: int
    wavelet_name: str


def encode_header(
    *,
    frames: int,
    channels: int,
    level: int,
    top_k_scaled: int,
    wavelet_name: str,
    payload: bytes,
    flags: int = 0,
) -> bytes:
    """Encode a wire header for the given metadata and payload.

    Returns the complete header bytes (excluding the payload).
    """
    wname_bytes = wavelet_name.encode("utf-8")
    wavelet_name_len = len(wname_bytes)
    crc32 = zlib.crc32(payload) & 0xFFFFFFFF
    # header_size includes base + wavelet name tail
    header_size = _BASE_SIZE + wavelet_name_len
    packed = struct.pack(
        _BASE_FMT,
        MAGIC,
        VERSION,
        flags,
        header_size,
        frames,
        channels,
        level,
        top_k_scaled,
        wavelet_name_len,
        len(payload),
        crc32,
    )
    return packed + wname_bytes


def decode_header(buffer: bytes) -> tuple[WireHeader, int]:
    """Decode header from buffer.

    Returns (header, header_total_size). Raises ValueError on parse/validation
    errors. Does not validate payload CRC; use validate_crc for that.
    """
    if len(buffer) < _BASE_SIZE:
        raise ValueError("buffer too small for header")
    (
        magic,
        version,
        flags,
        header_size,
        frames,
        channels,
        level,
        top_k_scaled,
        wavelet_name_len,
        payload_len,
        crc32,
    ) = struct.unpack(_BASE_FMT, buffer[:_BASE_SIZE])

    if magic != MAGIC:
        raise ValueError("invalid magic")
    if version != VERSION:
        raise ValueError(f"unsupported version: {version}")

    total_size = _BASE_SIZE + wavelet_name_len
    if header_size != total_size:
        # allow forward-compat larger header but must be at least base + name
        if header_size < total_size:
            raise ValueError("invalid header_size")
        total_size = header_size

    if len(buffer) < total_size:
        raise ValueError("buffer too small for full header")

    wname_end = _BASE_SIZE + wavelet_name_len
    wavelet_name_bytes = buffer[_BASE_SIZE:wname_end]
    try:
        wavelet_name = wavelet_name_bytes.decode("utf-8")
    except UnicodeDecodeError as e:
        raise ValueError("invalid wavelet_name encoding") from e

    hdr = WireHeader(
        magic=magic,
        version=version,
        flags=flags,
        header_size=header_size,
        frames=frames,
        channels=channels,
        level=level,
        top_k_scaled=top_k_scaled,
        wavelet_name_len=wavelet_name_len,
        payload_len=payload_len,
        crc32=crc32,
        wavelet_name=wavelet_name,
    )
    return hdr, total_size


def validate_crc(payload: bytes, expected_crc32: int) -> None:
    """Validate CRC32 of payload; raise ValueError if mismatch."""
    calc = zlib.crc32(payload) & 0xFFFFFFFF
    if calc != expected_crc32:
        raise ValueError("CRC32 mismatch")
