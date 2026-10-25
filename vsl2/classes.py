#!/usr/bin/env python3
"""Manage sign-language classes (actions): add, delete, and list.

Usage:
    python scripts/manage_classes.py list
    python scripts/manage_classes.py add <class_name> [<class_name2> ...]
    python scripts/manage_classes.py delete <class_name> [<class_name2> ...]

After adding a class you should collect data for it:
    python scripts/collect_data.py --action <class_name>

After any class change you must retrain:
    python scripts/train.py
"""

import sys, os, json, shutil, argparse


from . import defaults as cfg


# ── Helpers ──────────────────────────────────────────────────────────────────

def _read_actions():
    """Read current actions from actions.json."""
    if os.path.exists(cfg.ACTIONS_FILE):
        with open(cfg.ACTIONS_FILE, "r") as f:
            return json.load(f)
    return list(cfg.ACTIONS)


def _write_actions(actions):
    """Write the actions list to actions.json."""
    with open(cfg.ACTIONS_FILE, "w") as f:
        json.dump(actions, f, indent=2, ensure_ascii=False)


def _data_dir_for(action):
    return os.path.join(cfg.DATA_DIR, action)


def _count_sequences(action):
    """Count how many valid sequence folders exist for an action."""
    d = _data_dir_for(action)
    if not os.path.isdir(d):
        return 0
    return sum(1 for x in os.listdir(d) if os.path.isdir(os.path.join(d, x)))


def _invalidate_model():
    """Warn (and optionally remove) the old model since classes changed."""
    model_path = os.path.join(cfg.MODELS_DIR, cfg.MODEL_FILENAME)
    stats_path = os.path.join(cfg.MODELS_DIR, "feature_stats.npz")
    if os.path.exists(model_path) or os.path.exists(stats_path):
        print("\n⚠️  Các class đã thay đổi — model cũ không còn hợp lệ.")
        ans = input("   Xoá model cũ? (y/n) [y]: ").strip().lower()
        if ans in ("", "y", "yes"):
            for p in (model_path, stats_path):
                if os.path.exists(p):
                    os.remove(p)
                    print(f"   ✓ Đã xoá {os.path.basename(p)}")
        else:
            print("   → Giữ lại model cũ (sẽ lỗi nếu dùng với số class mới).")


# ── Commands ─────────────────────────────────────────────────────────────────

def cmd_list(_args):
    """List all current classes with their data status."""
    actions = _read_actions()
    if not actions:
        print("Chưa có class nào. Dùng 'add' để thêm.")
        return

    print(f"\n📋 Danh sách class ({len(actions)}):\n")
    print(f"  {'#':<4} {'Class':<20} {'Sequences':<12} {'Data'}")
    print(f"  {'─'*4} {'─'*20} {'─'*12} {'─'*20}")
    for i, action in enumerate(actions):
        n_seq = _count_sequences(action)
        data_path = _data_dir_for(action)
        has_data = "✓" if n_seq > 0 else "✗ (chưa thu thập)"
        print(f"  {i:<4} {action:<20} {n_seq:<12} {has_data}")
    print()


def cmd_add(args):
    """Add one or more new classes."""
    actions = _read_actions()
    added = []

    for name in args.names:
        name = name.strip().lower()
        if not name:
            continue
        if name in actions:
            print(f"⚠️  Class '{name}' đã tồn tại — bỏ qua.")
            continue

        actions.append(name)
        # Create data directory
        os.makedirs(_data_dir_for(name), exist_ok=True)
        added.append(name)

    if not added:
        print("Không có class mới nào được thêm.")
        return

    _write_actions(actions)
    print(f"\n✅ Đã thêm {len(added)} class: {', '.join(added)}")
    print(f"   Tổng số class: {len(actions)}\n")

    # Remind to collect data
    print("📹 Tiếp theo, thu thập dữ liệu cho các class mới:")
    for name in added:
        print(f"   python scripts/collect_data.py --action {name}")
    print(f"\n🏋️  Sau đó train lại model:")
    print(f"   python scripts/train.py")

    if any(os.path.exists(os.path.join(cfg.MODELS_DIR, f))
           for f in (cfg.MODEL_FILENAME, "feature_stats.npz")):
        _invalidate_model()


def cmd_delete(args):
    """Delete one or more classes."""
    actions = _read_actions()
    deleted = []

    for name in args.names:
        name = name.strip().lower()
        if name not in actions:
            print(f"⚠️  Class '{name}' không tồn tại — bỏ qua.")
            continue

        n_seq = _count_sequences(name)
        data_path = _data_dir_for(name)

        # Confirmation
        if n_seq > 0:
            print(f"\n🗑️  Class '{name}' có {n_seq} sequences trong {data_path}")
            ans = input(f"   Xoá class VÀ dữ liệu? (y/n) [n]: ").strip().lower()
        else:
            ans = input(f"   Xoá class '{name}'? (y/n) [y]: ").strip().lower()
            if ans == "":
                ans = "y"

        if ans not in ("y", "yes"):
            print(f"   → Bỏ qua '{name}'.")
            continue

        # Remove from list
        actions.remove(name)

        # Remove data directory
        if os.path.isdir(data_path):
            shutil.rmtree(data_path)
            print(f"   ✓ Đã xoá thư mục dữ liệu: {data_path}")

        deleted.append(name)

    if not deleted:
        print("Không có class nào bị xoá.")
        return

    _write_actions(actions)
    print(f"\n✅ Đã xoá {len(deleted)} class: {', '.join(deleted)}")
    print(f"   Còn lại {len(actions)} class: {', '.join(actions) if actions else '(trống)'}\n")

    if actions:
        print("🏋️  Cần train lại model:")
        print("   python scripts/train.py")
        _invalidate_model()
    else:
        print("⚠️  Không còn class nào! Thêm class mới bằng:")
        print("   python scripts/manage_classes.py add <tên_class>")
        # Clean up model since there are no classes
        for f in (cfg.MODEL_FILENAME, "feature_stats.npz"):
            p = os.path.join(cfg.MODELS_DIR, f)
            if os.path.exists(p):
                os.remove(p)


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Quản lý class (hành động ký hiệu) cho VSL Translator.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  python scripts/manage_classes.py list
  python scripts/manage_classes.py add goodbye please
  python scripts/manage_classes.py delete goodbye
        """,
    )
    sub = parser.add_subparsers(dest="command")

    # list
    p_list = sub.add_parser("list", help="Liệt kê tất cả class hiện có")
    p_list.set_defaults(func=cmd_list)

    # add
    p_add = sub.add_parser("add", help="Thêm class mới")
    p_add.add_argument("names", nargs="+", help="Tên các class cần thêm")
    p_add.set_defaults(func=cmd_add)

    # delete
    p_del = sub.add_parser("delete", help="Xoá class")
    p_del.add_argument("names", nargs="+", help="Tên các class cần xoá")
    p_del.set_defaults(func=cmd_delete)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()
