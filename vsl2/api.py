"""Style-ultralytics API: VSL2(weights).train() / .predict() / .collect() / ...

Usage:
    from vsl2 import VSL2
    m = VSL2("vsl2/weights/model.keras")
    m.train(data="vsl2/data", epochs=50)
    m.predict("video.mp4")
    m.predict(source=0)          # webcam realtime
    m.collect(action="xinchao")
"""

import os
import sys

from . import defaults as cfg


class VSL2:
    def __init__(self, weights=None):
        self.weights = weights

    # ── Train ────────────────────────────────────────────────────────────
    def train(self, data=None, epochs=None, batch_size=None,
              augment_factor=None, **overrides):
        """Train the classifier. Overrides map 1:1 to vsl2/defaults.py names.

        e.g. m.train(epochs=50, lr_initial=5e-4, augment_preset="strong")
        """
        from . import train as _train
        if data is not None:
            cfg.DATA_DIR = data
        if epochs is not None:
            cfg.EPOCHS = epochs
        if batch_size is not None:
            cfg.BATCH_SIZE = batch_size
        if augment_factor is not None:
            cfg.AUGMENT_FACTOR = augment_factor
        for k, v in overrides.items():
            if not hasattr(cfg, k.upper()) and not hasattr(cfg, k):
                raise ValueError(f"Unknown config override: {k!r}")
            setattr(cfg, k if hasattr(cfg, k) else k.upper(), v)
        _train.main()

    # ── Predict ──────────────────────────────────────────────────────────
    def predict(self, source, **kwargs):
        """Batch predict: video path, image folder, seq dir, or 12 vectors.

        Feature input: list of 12 (356,) arrays or a (12, 356) ndarray.
        Returns {"raw": word, "probs": [...]}.
        """
        from .predict import predict as _predict
        return _predict(source, model_path=self.weights, **kwargs)

    def predict_features(self, vectors, **kwargs):
        """Predict directly over 12 featurized vectors. No MediaPipe run."""
        from .predict import predict_features as _pf
        return _pf(vectors, model_path=self.weights, **kwargs)

    def predict_live(self, **kwargs):
        """Webcam realtime loop (blocks until 'q'). Returns sentence list."""
        from .realtime import run_realtime
        return run_realtime(model_path=self.weights, **kwargs)

    # ── Data ─────────────────────────────────────────────────────────────
    def collect(self, action="hello", sequences=None, device="macbook"):
        from . import collect as _collect
        argv = ["collect", "--action", action, "--device", device]
        if sequences is not None:
            argv += ["--sequences", str(sequences)]
        _run_with_argv(_collect.main, argv)

    def ingest(self, video, action="hello", device="video"):
        """Convert one video into data/<action>/<seq>/ (12 .npy frames)."""
        from .ingest import ingest_video
        return ingest_video(video, action, device=device)

    def noise(self, action="hello", copies=1, **kwargs):
        from . import noise as _noise
        argv = ["noise", "--action", action, "--copies", str(copies)]
        for k, v in kwargs.items():
            flag = "--" + k.replace("_", "-")
            if isinstance(v, bool):
                if v:
                    argv.append(flag)
            elif isinstance(v, (list, tuple)):
                argv += [flag] + [str(x) for x in v]
            else:
                argv += [flag, str(v)]
        _run_with_argv(_noise.main, argv)

    def classes(self, command="list", names=()):
        from . import classes as _classes
        _run_with_argv(_classes.main, ["classes", command, *names])


def _run_with_argv(fn, argv):
    old = sys.argv
    sys.argv = argv
    try:
        return fn()
    finally:
        sys.argv = old
