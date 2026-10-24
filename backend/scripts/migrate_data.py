#!/usr/bin/env python3
"""Migrate legacy training data to the aspect-corrected normalized convention.

Legacy data problem: collect_data.py passed frame_shape in pixels but no
neutral_lens, so arm_depth channels [102:104] hold pixel_len / 2.2 — saturated
garbage (observed mean ~1.99/1.28, clipped at 2.0). The canonical pose/hands /
shoulder-offset channels are already resolution-free and need no change.

Migration (in place, with backup):
  - For each data/<action>/<seq>/*.npy with shape (356,):
      - all-zero vector (missing landmarks) → untouched (Masking compat)
      - otherwise → channels [102:104] reset to neutral 1.0 (no depth info,
        but honest neutral instead of saturated noise)
  - Writes meta.json {"device": "legacy"} per sequence dir if missing.

Usage:
    python scripts/migrate_data.py --dry-run
    python scripts/migrate_data.py --yes
"""

import sys, os, shutil, argparse, json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import config as cfg


def main():
    ap = argparse.ArgumentParser(description="Migrate legacy arm_depth channels to neutral 1.0.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()

    seq_dirs = []
    for action in cfg.ACTIONS:
        adir = os.path.join(cfg.DATA_DIR, action)
        if not os.path.isdir(adir):
            continue
        for s in sorted(os.listdir(adir)):
            sdir = os.path.join(adir, s)
            if os.path.isdir(sdir):
                seq_dirs.append((action, s, sdir))
    print(f"Found {len(seq_dirs)} sequences.")

    if not args.dry_run and not args.yes:
        ans = input("Reset arm_depth channels to 1.0 in place (backup first)? (y/n) [n]: ").strip().lower()
        if ans not in ("y", "yes"):
            print("Huỷ.")
            return

    backup_root = os.path.join(cfg.DATA_DIR, "..", "data_pre_migrate_bak")
    if not args.dry_run:
        os.makedirs(backup_root, exist_ok=True)

    n_files, n_touched, n_meta = 0, 0, 0
    for action, s, sdir in seq_dirs:
        if not args.dry_run:
            bdir = os.path.join(backup_root, action, s)
            if not os.path.exists(bdir):
                shutil.copytree(sdir, bdir)
        meta_path = os.path.join(sdir, "meta.json")
        if not os.path.exists(meta_path) and not args.dry_run:
            with open(meta_path, "w") as f:
                json.dump({"device": "legacy", "src_shape": None,
                           "std_size": cfg.FRAME_STD_SIZE}, f)
            n_meta += 1
        for n in range(cfg.SEQUENCE_LEN):
            fp = os.path.join(sdir, f"{n}.npy")
            if not os.path.isfile(fp):
                continue
            v = np.load(fp)
            n_files += 1
            if v.shape == (356,) and np.any(v):
                if not np.allclose(v[102:104], 1.0):
                    n_touched += 1
                    if not args.dry_run:
                        v = v.copy()
                        v[102:104] = 1.0
                        np.save(fp, v)

    tag = "[dry-run] " if args.dry_run else ""
    print(f"{tag}Xong — {n_files} files, {n_touched} arm_depth reset, "
          f"{n_meta} meta.json written. Backup: {backup_root}")


if __name__ == "__main__":
    main()
