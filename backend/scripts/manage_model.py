#!/usr/bin/env python3
"""Manage model backups: list, restore, and clean.

Usage:
    python scripts/manage_model.py list
    python scripts/manage_model.py restore <backup_name>
    python scripts/manage_model.py clean
"""

import sys, os, json, shutil, argparse, glob

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config as cfg


def _get_backups():
    """Return list of backup dirs sorted oldest → newest."""
    if not os.path.isdir(cfg.BACKUP_DIR):
        return []
    return sorted(glob.glob(os.path.join(cfg.BACKUP_DIR, "backup_*")))


def cmd_list(_args):
    """List all available model backups."""
    backups = _get_backups()
    if not backups:
        print("\n  Chưa có backup nào.")
        print("  Backup được tạo tự động mỗi khi chạy train.py\n")
        return

    print(f"\n  📦 Model Backups ({len(backups)}/{cfg.MAX_BACKUPS} tối đa):\n")
    print(f"  {'#':<4} {'Tên backup':<28} {'Thời gian':<22} {'Classes':<30} {'Files'}")
    print(f"  {'─'*4} {'─'*28} {'─'*22} {'─'*30} {'─'*20}")

    for i, bp in enumerate(backups):
        name = os.path.basename(bp)
        info_path = os.path.join(bp, "backup_info.json")

        # Read metadata
        timestamp = "?"
        actions = "?"
        if os.path.exists(info_path):
            with open(info_path) as f:
                info = json.load(f)
            timestamp = info.get("timestamp", "?")
            actions_list = info.get("actions", [])
            actions = ", ".join(actions_list)
            if len(actions) > 28:
                actions = actions[:25] + "..."

        # Format timestamp nicely
        if timestamp != "?" and len(timestamp) == 15:
            ts = f"{timestamp[:4]}-{timestamp[4:6]}-{timestamp[6:8]} {timestamp[9:11]}:{timestamp[11:13]}:{timestamp[13:15]}"
        else:
            ts = timestamp

        # Count files
        files = [f for f in os.listdir(bp) if os.path.isfile(os.path.join(bp, f))]
        file_list = ", ".join(f for f in files if f != "backup_info.json")

        marker = " ← mới nhất" if i == len(backups) - 1 else ""
        print(f"  {i:<4} {name:<28} {ts:<22} {actions:<30} {file_list}{marker}")

    print()

    # Show current model info
    model_path = os.path.join(cfg.MODELS_DIR, cfg.MODEL_FILENAME)
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"  Model hiện tại: {cfg.MODEL_FILENAME} ({size_mb:.1f} MB)")
    else:
        print("  ⚠️  Không có model hiện tại")
    print()


def cmd_restore(args):
    """Restore a model from a backup."""
    backups = _get_backups()
    if not backups:
        print("Không có backup nào để restore.")
        return

    target = args.name

    # Allow restore by index number
    if target.isdigit():
        idx = int(target)
        if 0 <= idx < len(backups):
            target = os.path.basename(backups[idx])
        else:
            print(f"Index {idx} không hợp lệ. Có {len(backups)} backups (0-{len(backups)-1}).")
            return

    # Find the backup
    backup_path = os.path.join(cfg.BACKUP_DIR, target)
    if not os.path.isdir(backup_path):
        print(f"Không tìm thấy backup '{target}'.")
        print("Dùng 'list' để xem danh sách backups.")
        return

    # Show what will be restored
    info_path = os.path.join(backup_path, "backup_info.json")
    if os.path.exists(info_path):
        with open(info_path) as f:
            info = json.load(f)
        print(f"\n  Restore backup: {target}")
        print(f"  Thời gian   : {info.get('timestamp', '?')}")
        print(f"  Classes     : {', '.join(info.get('actions', []))}")
    else:
        print(f"\n  Restore backup: {target}")

    # Backup current model first (so we don't lose it)
    current_model = os.path.join(cfg.MODELS_DIR, cfg.MODEL_FILENAME)
    if os.path.exists(current_model):
        ans = input("\n  Backup model hiện tại trước khi restore? (y/n) [y]: ").strip().lower()
        if ans in ("", "y", "yes"):
            from datetime import datetime
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            pre_restore = os.path.join(cfg.BACKUP_DIR, f"backup_{ts}_pre_restore")
            os.makedirs(pre_restore, exist_ok=True)
            shutil.copy2(current_model, os.path.join(pre_restore, cfg.MODEL_FILENAME))
            stats = os.path.join(cfg.MODELS_DIR, "feature_stats.npz")
            if os.path.exists(stats):
                shutil.copy2(stats, os.path.join(pre_restore, "feature_stats.npz"))
            print(f"  💾 Model hiện tại đã backup → {os.path.basename(pre_restore)}")

    # Restore files
    files_restored = []
    for fname in os.listdir(backup_path):
        if fname == "backup_info.json":
            continue
        src = os.path.join(backup_path, fname)
        dst = os.path.join(cfg.MODELS_DIR, fname)
        if os.path.isfile(src):
            shutil.copy2(src, dst)
            files_restored.append(fname)

    print(f"\n  ✅ Đã restore: {', '.join(files_restored)}")
    print(f"  → Model hiện tại đã được thay thế bằng backup '{target}'\n")


def cmd_clean(_args):
    """Delete all backups."""
    backups = _get_backups()
    if not backups:
        print("Không có backup nào để xoá.")
        return

    total_size = 0
    for bp in backups:
        for f in os.listdir(bp):
            fp = os.path.join(bp, f)
            if os.path.isfile(fp):
                total_size += os.path.getsize(fp)

    size_mb = total_size / (1024 * 1024)
    print(f"\n  Sẽ xoá {len(backups)} backups ({size_mb:.1f} MB)")
    ans = input("  Xác nhận? (y/n) [n]: ").strip().lower()
    if ans not in ("y", "yes"):
        print("  → Huỷ bỏ.")
        return

    for bp in backups:
        shutil.rmtree(bp)

    print(f"  ✅ Đã xoá {len(backups)} backups.\n")


def main():
    parser = argparse.ArgumentParser(
        description="Quản lý model backups cho VSL Translator.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  python scripts/manage_model.py list
  python scripts/manage_model.py restore 0
  python scripts/manage_model.py restore backup_20261018_011800
  python scripts/manage_model.py clean
        """,
    )
    sub = parser.add_subparsers(dest="command")

    p_list = sub.add_parser("list", help="Liệt kê tất cả backups")
    p_list.set_defaults(func=cmd_list)

    p_restore = sub.add_parser("restore", help="Khôi phục model từ backup")
    p_restore.add_argument("name", help="Tên backup hoặc số thứ tự (từ 'list')")
    p_restore.set_defaults(func=cmd_restore)

    p_clean = sub.add_parser("clean", help="Xoá tất cả backups")
    p_clean.set_defaults(func=cmd_clean)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()
