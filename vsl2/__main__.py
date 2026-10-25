"""CLI: python -m vsl2 train|predict|collect|noise|classes ..."""

import argparse
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(prog="vsl2")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Train the classifier")
    p_train.add_argument("--data", default=None)
    p_train.add_argument("--epochs", type=int, default=None)
    p_train.add_argument("--batch-size", type=int, default=None)
    p_train.add_argument("--augment-factor", type=int, default=None)

    p_pred = sub.add_parser("predict", help="Batch predict over video/folder")
    p_pred.add_argument("source")
    p_pred.add_argument("--weights", default=None)

    p_live = sub.add_parser("live", help="Webcam realtime (q to quit)")
    p_live.add_argument("--weights", default=None)

    p_col = sub.add_parser("collect", help="Collect keypoint sequences")
    p_col.add_argument("--action", default="hello")
    p_col.add_argument("--sequences", type=int, default=None)
    p_col.add_argument("--device", default="macbook")

    p_noise = sub.add_parser("noise", help="Generate noisy copies")
    p_noise.add_argument("--action", default="hello")
    p_noise.add_argument("--copies", type=int, default=1)

    p_ing = sub.add_parser("ingest", help="Video -> featurized data (12 .npy)")
    p_ing.add_argument("video")
    p_ing.add_argument("--action", default="hello")
    p_ing.add_argument("--device", default="video")

    p_cls = sub.add_parser("classes", help="Manage classes")
    p_cls.add_argument("cmd", nargs="?", default="list")
    p_cls.add_argument("names", nargs="*")

    args = parser.parse_args(argv)
    from .api import VSL2
    m = VSL2(getattr(args, "weights", None))

    if args.command == "train":
        kw = {k: v for k, v in vars(args).items()
              if k not in ("command", "data") and v is not None}
        m.train(data=args.data, **kw)
    elif args.command == "predict":
        import json
        print(json.dumps(m.predict(args.source), ensure_ascii=False))
    elif args.command == "live":
        m.predict_live()
    elif args.command == "collect":
        m.collect(action=args.action, sequences=args.sequences,
                  device=args.device)
    elif args.command == "noise":
        m.noise(action=args.action, copies=args.copies)
    elif args.command == "ingest":
        print(m.ingest(args.video, action=args.action, device=args.device))
    elif args.command == "classes":
        m.classes(command=args.cmd, names=args.names)


if __name__ == "__main__":
    main()
