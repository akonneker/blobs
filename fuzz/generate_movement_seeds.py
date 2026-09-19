"""Seed corner occupancy, thin wrapped worlds, rejected slots and elevation."""

from pathlib import Path

corpus = Path(__file__).resolve().parent / "corpus" / "movement"
corpus.mkdir(parents=True, exist_ok=True)
for width, height, origin in ((2, 2, 4), (0, 0, 0), (0, 2, 1), (2, 0, 1)):
    for boundary_corner in range(6):
        for occupied in (0, 1 << 5, 1 << 7, (1 << 5) | (1 << 7), 511):
            for slot in (4, 7, 10):
                data = bytearray(64)
                data[:7] = bytes((width, height, boundary_corner, origin, 8, slot, 255))
                data[7:11] = occupied.to_bytes(4, "little")
                (corpus / f"move-{width}-{height}-{boundary_corner}-{occupied}-{slot}").write_bytes(data)
