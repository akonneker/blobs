"""Seed bounded topology, mask, and virtual-world boundary cases."""

from pathlib import Path

corpus = Path(__file__).resolve().parent / "corpus" / "neighborhood"
corpus.mkdir(parents=True, exist_ok=True)
for width in (0, 1, 2, 8):
    for height in (0, 1, 3, 8):
        for flags in (0, 1, 8, 9, 128, 129):
            for count in (0, 1, 8, 32, 33):
                data = bytearray([255] * 181)
                data[:5] = bytes((width, height, flags, 3, count))
                (corpus / f"geometry-{width}-{height}-{flags}-{count}").write_bytes(data)
for flags in (2, 3, 6, 7):
    data = bytearray([0] * 181)
    data[:5] = bytes((3, 5, flags, 128, 2))
    data[49:57] = bytes((128, 127, 1, 0, 127, 128, 255, 255))
    (corpus / f"extreme-offsets-{flags}").write_bytes(data)

for label, radius, offsets in (
    ("zero-offset", 3, (0, 0, 1, 0, 1, 0, 1, 0)),
    ("zero-cost", 3, (1, 0, 0, 0, 0, 1, 1, 0)),
    ("duplicate-offset", 3, (1, 0, 1, 0, 1, 0, 2, 0)),
    ("outside-radius", 1, (2, 0, 1, 0, 0, 1, 1, 0)),
):
    data = bytearray([0] * 181)
    data[:5] = bytes((3, 3, 6, radius, 2))
    data[49:57] = bytes(offsets)
    (corpus / label).write_bytes(data)
