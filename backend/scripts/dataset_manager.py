#!/usr/bin/env python3
"""Trim sign-language datasets (sequence folders) per action.

Layout: data/<action>/<seq_id>/*.npy  (seq_id = 0, 1, 2, ...)

Usage:
    # Giữ lại N sequence đầu tiên của mỗi action
    python scripts/dataset_manager.py --keep-first 50

    # Giữ lại N% sequence đầu tiên của mỗi action
    python scripts/dataset_manager.py --keep-percent 30

    # Giữ lại COUNT sequence bắt đầu từ vị trí START (0-based)
    python scripts/dataset_manager.py --keep-range 10 20

Common options:
    --action camon toi      chỉ áp dụng cho các action này (mặc định: tất cả)
    --dry-run               chỉ in ra kế hoạch, không xoá
    --yes                   bỏ qua xác nhận

Sau khi trim phải retrain: python scripts/train.py
"""

import sys, os, shutil, argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import config as cfg


# ── Helpers ──────────────────────────────────────────────────────────────────

def _seq_dirs(action_dir):
    """Sorted sequence folder names (numeric sort when possible)."""
    names = [x for x in os.listdir(action_dir)
             if os.path.isdir(os.path.join(action_dir, x))]
    try:
        return sorted(names, key=int)
    except ValueError:
        return sorted(names)


def _apply(action, keep_set, dry_run):
    d = os.path.join(cfg.DATA_DIR, action)
    deleted = 0
    for name in _seq_dirs(d):
        if name not in keep_set:
            p = os.path.join(d, name)
            print(f"  xoá {action}/{name}")
            if not dry_run:
                shutil.rmtree(p)
            deleted += 1
    return deleted


# ── Modes ────────────────────────────────────────────────────────────────────

def _mode_keep_first(n, actions, dry_run):
    total_del = 0
    for action in actions:
        names = _seq_dirs(os.path.join(cfg.DATA_DIR, action))
        keep = set(names[:n])
        span = f"{names[0]}..{names[len(keep) - 1]}" if keep else "rỗng"
        print(f"{action}: {len(names)} → giữ {len(keep)} ({span})")
        total_del += _apply(action, keep, dry_run)
    return total_del


def _mode_keep_percent(pct, actions, dry_run):
    total_del = 0
    for action in actions:
        names = _seq_dirs(os.path.join(cfg.DATA_DIR, action))
        n = int(len(names) * pct / 100)
        keep = set(names[:n])
        print(f"{action}: {len(names)} → giữ {len(keep)} ({pct}% đầu)")
        total_del += _apply(action, keep, dry_run)
    return total_del


def _mode_keep_range(start, count, actions, dry_run):
    total_del = 0
    for action in actions:
        names = _seq_dirs(os.path.join(cfg.DATA_DIR, action))
        keep = set(names[start:start + count])
        span = f"{names[start]}..{names[start + len(keep) - 1]}" if keep else "rỗng"
        print(f"{action}: {len(names)} → giữ {len(keep)} ({span})")
        total_del += _apply(action, keep, dry_run)
    return total_del


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Trim dataset sequences per action.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--keep-first", type=int, metavar="N",
                   help="giữ lại N sequence đầu tiên của mỗi action")
    g.add_argument("--keep-percent", type=float, metavar="P",
                   help="giữ lại P%% sequence đầu tiên của mỗi action")
    g.add_argument("--keep-range", nargs=2, type=int, metavar=("START", "COUNT"),
                   help="giữ lại COUNT sequence từ vị trí START (0-based)")
    ap.add_argument("--action", nargs="+", default=None,
                    help="chỉ áp dụng cho các action này (mặc định: tất cả)")
    ap.add_argument("--dry-run", action="store_true",
                    help="chỉ in kế hoạch, không xoá")
    ap.add_argument("--yes", action="store_true",
                    help="bỏ qua xác nhận")
    args = ap.parse_args()

    if args.keep_first is not None and args.keep_first < 0:
        ap.error("--keep-first N phải >= 0")
    if args.keep_percent is not None and not (0 <= args.keep_percent <= 100):
        ap.error("--keep-percent P phải trong 0..100")
    if args.keep_range is not None:
        s, c = args.keep_range
        if s < 0 or c < 0:
            ap.error("--keep-range START COUNT phải >= 0")

    actions = args.action or list(cfg.ACTIONS)
    for a in actions:
        if not os.path.isdir(os.path.join(cfg.DATA_DIR, a)):
            ap.error(f"action '{a}' không có thư mục data")

    if not args.dry_run and not args.yes:
        ans = input("Xoá các sequence ngoài phạm vi giữ lại? (y/n) [n]: ").strip().lower()
        if ans not in ("y", "yes"):
            print("Huỷ.")
            return

    if args.keep_first is not None:
        n = _mode_keep_first(args.keep_first, actions, args.dry_run)
    elif args.keep_percent is not None:
        n = _mode_keep_percent(args.keep_percent, actions, args.dry_run)
    else:
        s, c = args.keep_range
        n = _mode_keep_range(s, c, actions, args.dry_run)

    tag = "[dry-run] " if args.dry_run else ""
    print(f"\n{tag}Xong — đã xoá {n} sequence.")
    if not args.dry_run:
        print("Nhớ retrain: python scripts/train.py")


if __name__ == "__main__":
    main()
