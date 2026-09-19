"""Seed visibility, all activities, progress boundaries and rejected actions."""

from pathlib import Path

corpus = Path(__file__).resolve().parent / "corpus" / "observations"
corpus.mkdir(parents=True, exist_ok=True)
for fields in range(128):
    for width, height, wrap, count in ((0, 0, 1, 8), (4, 4, 0, 8), (2, 2, 1, 8), (1, 3, 0, 0)):
        data = bytearray([255] * 256)
        data[:5] = bytes((width, height, wrap, fields, count))
        for field in range(7):
            data[5 + field] = 255 if fields & (1 << field) else 0
        (corpus / f"visibility-{fields}-{width}-{height}-{wrap}-{count}").write_bytes(data)

# Keep these fixtures aligned with the explicit coverage assertions in the
# shared Rust harness. One northern neighbor acts toward a separate east tile.
data = bytearray(256)
data[:5] = bytes((2, 2, 1, 4, 8))
data[5:16] = bytes([255] * 11)
start = 17 + 8
data[start:start + 8] = bytes((5, 19, 4, 0, 16, 32, 20, 0))
for kind in range(10):
    data[start] = 5 | (kind << 4)
    for phase in range(6):
        data[217] = phase
        (corpus / f"activity-{kind}-phase-{phase}").write_bytes(data)
for guarded in (False, True):
    data[start] = 3 if guarded else 1
    (corpus / f"idle-guarded-{int(guarded)}").write_bytes(data)
for kind, mask_index in ((1, 12), (2, 13), (5, 14), (6, 15)):
    data[start] = 5 | (kind << 4)
    for phase in range(6):
        data[217] = phase
        for invalid_slot in (False, True):
            data[start + 2] = 11 if invalid_slot else 4
            data[mask_index] = 255 if invalid_slot else 0
            (corpus / f"rejected-{kind}-phase-{phase}-invalid-{int(invalid_slot)}").write_bytes(data)
        data[mask_index] = 255
