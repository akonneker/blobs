"""Write bounded, reproducible action-family seeds without replacing fuzz discoveries."""

from pathlib import Path


def main():
    corpus = Path(__file__).resolve().parent / "corpus" / "resolver_commands"
    corpus.mkdir(parents=True, exist_ok=True)
    for profile in (0, 7, 15, 31, 32, 39, 47, 63, 64, 71, 79, 95, 96, 103, 111, 127):
        for action in range(10):
            data = bytearray((2, 2, profile, 100))  # 4x4 world, eight initial cells.
            for slot in range(8):
                for operation in (1, 4, 2, 4, 3):
                    data.extend((operation, 0, action, slot, slot % 3, 32, 8, slot))
            (corpus / f"action-{action}-profile-{profile}").write_bytes(data)
            if profile & 64:
                data = bytearray((3, 3, profile, 100))  # 5x5, 13 mixed actors.
                for mode in range(8):
                    for operation in (5, 4, 2, 4):
                        # Later modes include sidecar validation and oversized
                        # memory, along with ordering/unknown/busy actors.
                        data.extend((operation, mode, action, 4, 129 if mode & 4 else 1,
                                     16, 17 if mode == 7 else 8, mode))
                (corpus / f"atomic-{action}-profile-{profile}").write_bytes(data)
    # Minimal formerly rejected idle-cell checkpoint after a clock advance.
    (corpus / "idle-checkpoint").write_bytes(
        bytes((0, 0, 0, 100, 3, 0, 0, 0, 0, 8, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0))
    )


if __name__ == "__main__":
    main()
