# Wire format

A compact, versioned header precedes all compressed payloads. Fields are little-endian.

Header fields:

- magic: bytes(4) = BCIW
- version: u8 (current 1)
- flags: u8 (reserved)
- header_size: u16 (bytes; includes wavelet name)
- frames: u32
- channels: u16
- level: u16
- top_k_scaled: u16 (top_k_ratio * 1e4)
- wavelet_name_len: u16
- payload_len: u32
- crc32: u32 (payload only)
- wavelet_name: utf-8 string

Followed by payload bytes.

Validation:

- magic and version are checked on decode.
- header_size allows future growth.
- crc32 validated against payload.
