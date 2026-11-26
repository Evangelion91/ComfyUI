from pathlib import Path
import struct
import tempfile
import zlib

import numpy as np
from PIL.PngImagePlugin import putchunk


def save_unfiltered_rgb_png(path, pixels, metadata):
    """Write uint8 RGB pixels and PNG text chunks without filtering or compression."""
    height, width, _ = pixels.shape
    rows = np.empty((height, 1 + width * 3), dtype=np.uint8)
    rows[:, 0] = 0
    rows[:, 1:] = pixels.reshape(height, width * 3)

    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as file:
            temporary = Path(file.name)
            file.write(b"\x89PNG\r\n\x1a\n")
            putchunk(file, b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            if metadata is not None:
                for kind, data, *_ in metadata.chunks:
                    putchunk(file, kind, data)
            putchunk(file, b"IDAT", zlib.compress(rows, level=0))
            putchunk(file, b"IEND")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
