#!/usr/bin/env python3
"""DISABLED: 104-dim → 188-dim migration is not possible from stored vectors.

The face-offset block ([104:188]) encodes per-hand-point offsets to the face
center, which cannot be recovered from a saved 104-dim vector — the face
position was never stored. Migrating would fill 84 channels with zeros, and
a model trained on such data would learn "face-offset always zero" as a
spurious pattern that breaks on real data.

To upgrade: re-run scripts/collect_data.py for every action and retrain from
scratch. Do not mix old and new data. See CHANGELOG for the breaking change.
"""

import sys


def main():
    print(__doc__)
    sys.exit(1)


if __name__ == "__main__":
    main()
