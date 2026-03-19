import os
import gzip
import struct
import numpy as np


def ply_to_spz(ply_path, spz_path=None):
    """
    Convert a PLY gaussian splat file to SPZ format.

    Uses the spz package if available, otherwise falls back to
    float16 quantization + gzip compression.

    Args:
        ply_path: path to input .ply file
        spz_path: path to output .spz file (default: same name with .spz extension)

    Returns:
        path to the .spz file
    """
    if spz_path is None:
        spz_path = os.path.splitext(ply_path)[0] + ".spz"

    os.makedirs(os.path.dirname(os.path.abspath(spz_path)), exist_ok=True)

    try:
        import spz

        with open(ply_path, "rb") as f:
            ply_data = f.read()
        compressed = spz.compress(ply_data)
        with open(spz_path, "wb") as f:
            f.write(compressed)
        return spz_path
    except ImportError:
        pass

    # Fallback: quantize to float16 and gzip
    return _fallback_compress(ply_path, spz_path)


def _fallback_compress(ply_path, spz_path):
    """Quantize floats to float16 and gzip the PLY data."""
    with open(ply_path, "rb") as f:
        raw = f.read()

    # Find end of PLY header
    header_end = raw.find(b"end_header\n")
    if header_end == -1:
        header_end = raw.find(b"end_header\r\n")
        if header_end == -1:
            raise ValueError("Not a valid PLY file: missing end_header")
        header_end += len(b"end_header\r\n")
    else:
        header_end += len(b"end_header\n")

    header = raw[:header_end]
    body = raw[header_end:]

    # Parse vertex count and properties from header
    header_text = header.decode("ascii", errors="replace")
    vertex_count = 0
    properties = []
    for line in header_text.split("\n"):
        line = line.strip()
        if line.startswith("element vertex"):
            vertex_count = int(line.split()[-1])
        elif line.startswith("property float"):
            properties.append(line.split()[-1])
        elif line.startswith("property"):
            properties.append(line.split()[-1])

    if vertex_count == 0 or not properties:
        # Cannot parse, just gzip the raw file
        with gzip.open(spz_path, "wb", compresslevel=6) as gz:
            gz.write(raw)
        return spz_path

    # Quantize float32 body to float16
    num_floats = len(body) // 4
    if num_floats == vertex_count * len(properties):
        floats = np.frombuffer(body, dtype=np.float32)
        quantized = floats.astype(np.float16)
        compressed_body = quantized.tobytes()
    else:
        compressed_body = body

    # Write header (modified to note float16) + quantized body, all gzipped
    modified_header = header_text.replace("property float", "property float16")
    modified_header_bytes = modified_header.encode("ascii")

    with gzip.open(spz_path, "wb", compresslevel=6) as gz:
        gz.write(modified_header_bytes)
        gz.write(compressed_body)

    return spz_path
