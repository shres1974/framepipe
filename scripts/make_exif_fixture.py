#!/usr/bin/env python3
"""Inserts an EXIF APP1 segment with Orientation=<n> into an existing JPEG.

Used once to create tests/fixtures/exif_orientation6.jpg (a 4x2 JPEG tagged
Orientation=6, i.e. "rotate 90 CW to display"). FramePipe must ignore the
tag and decode 4x2 in stored order (spec 6.1).

Usage: python3 scripts/make_exif_fixture.py in.jpg out.jpg [orientation]
"""

import struct
import sys


def exif_app1(orientation: int) -> bytes:
    # TIFF header (little endian), one IFD0 entry: tag 0x0112 SHORT count 1.
    tiff = b"II" + struct.pack("<HI", 42, 8)
    ifd = struct.pack("<H", 1) + struct.pack("<HHIHH", 0x0112, 3, 1, orientation, 0) + struct.pack("<I", 0)
    payload = b"Exif\x00\x00" + tiff + ifd
    return b"\xff\xe1" + struct.pack(">H", len(payload) + 2) + payload


def main():
    src, dst = sys.argv[1], sys.argv[2]
    orientation = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    data = open(src, "rb").read()
    if data[:2] != b"\xff\xd8":
        sys.exit("input is not a JPEG")
    # Drop an existing JFIF APP0 so APP1 EXIF is the first application segment.
    rest = data[2:]
    if rest[:2] == b"\xff\xe0":
        (n,) = struct.unpack(">H", rest[2:4])
        rest = rest[2 + n:]
    open(dst, "wb").write(b"\xff\xd8" + exif_app1(orientation) + rest)


if __name__ == "__main__":
    main()
