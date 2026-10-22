"""
GRU model definition for sign language classification.
Includes Masking and CategoricalFocalCrossentropy to combat class bias.
"""

from keras import ops
from keras.models import Model
from keras.saving import register_keras_serializable
from keras.layers import (
    GRU, Dense, Input, Dropout, LayerNormalization, Bidirectional, Masking,
    Concatenate, Layer,
)
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.losses import CategoricalFocalCrossentropy

import config as cfg


@register_keras_serializable(package="vsl")
class WeightedMean(Layer):
    """Attention weighted mean over the time axis (replaces last-state).

    Softmax is computed with ops.softmax (mask-free) instead of a Softmax
    layer, because the 2D padding mask is not broadcastable to the (B, T, 1)
    score tensor.
    """

    def call(self, inputs):
        x, scores = inputs
        weights = ops.softmax(scores, axis=1)
        return ops.sum(x * weights, axis=1)

    def compute_mask(self, inputs, mask=None):
        return None


@register_keras_serializable(package="vsl")
class Slice(Layer):
    """Feature-group slice that passes the Masking mask through untouched."""

    def __init__(self, start, end, **kwargs):
        super().__init__(**kwargs)
        self.start = start
        self.end = end

    def call(self, inputs):
        return inputs[..., self.start:self.end]

    def compute_mask(self, inputs, mask=None):
        return mask

    def get_config(self):
        config = super().get_config()
        config.update({"start": self.start, "end": self.end})
        return config


def build_model(n_actions, sequence_len=cfg.SEQUENCE_LEN, n_features=None):
    """Build and compile an improved GRU classifier.

    Key architectural choices:
      - Masking(mask_value=0.0): skips entirely missing/dropped frames in recurrent steps.
      - Bidirectional GRU layers capture both forward and backward temporal
        context for gesture recognition.
      - LayerNormalization after each GRU and Dense layer stabilises training
        without depending on batch statistics (BatchNormalization is biased by
        masked zero-frames and unstable on small datasets).
      - Dropout (0.3-0.4) combat overfitting on small sign-language datasets.
      - L2 regularisation on Dense layers adds a further overfitting penalty.
      - CategoricalFocalCrossentropy with label smoothing penalizes hard samples
        and prevents overconfident probability spikes on dominant/easy classes.
        NOTE: alpha as a scalar only scales the whole loss — it does NOT
        rebalance classes. Class balance is handled by class_weight in train.py,
        so alpha stays 1.0 on balanced data. Pass a per-class list only when
        the data is truly imbalanced (and then drop class_weight to avoid
        double-counting).
      - Gradient clipping via clipnorm prevents exploding gradients.

    Args:
        n_actions:    Number of output classes.
        sequence_len: Number of frames per sequence (default cfg.SEQUENCE_LEN).
        n_features:   Number of keypoint features per frame (default: from
                       landmarks.n_features() — 356 canonical features).

    Returns:
        A compiled Keras functional Model.
    """
    from vsl_translator.landmarks import (
        POSE_DIM, HAND_DIM, ARM_DEPTH_DIM, BODY_ONLY_DIM,
    )
    if n_features is None:
        n_features = BODY_ONLY_DIM
    assert n_features == BODY_ONLY_DIM, (
        f"build_model expects the 356-dim canonical layout, got {n_features}"
    )
    pose_end = POSE_DIM                      # 18
    hands_end = POSE_DIM + HAND_DIM * 2      # 102
    depths_end = hands_end + ARM_DEPTH_DIM   # 104

    inp = Input(shape=(sequence_len, n_features))
    x = Masking(mask_value=0.0)(inp)

    # ── Per-group input norm ───────────────────────────────────────────
    pose_n = LayerNormalization()(Slice(0, pose_end)(x))
    hands_n = LayerNormalization()(Slice(pose_end, hands_end)(x))
    depths_n = LayerNormalization()(Slice(hands_end, depths_end)(x))
    sh_n = LayerNormalization()(Slice(depths_end, n_features)(x))
    x = Concatenate()([pose_n, hands_n, depths_n, sh_n])

    # ── Recurrent backbone ─────────────────────────────────────────────
    x = Bidirectional(GRU(48, return_sequences=True, activation='tanh'))(x)
    x = LayerNormalization()(x)
    x = Dropout(0.4)(x)

    x = Bidirectional(GRU(48, return_sequences=True, activation='tanh'))(x)
    x = LayerNormalization()(x)
    x = Dropout(0.4)(x)
    scores = Dense(1, activation='tanh')(x)      # (B, T, 1)
    x = WeightedMean()([x, scores])  # (B, 96)

    # ── Classifier head ────────────────────────────────────────────────
    x = Dense(64, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = LayerNormalization()(x)
    x = Dropout(0.4)(x)

    x = Dense(64, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = LayerNormalization()(x)
    x = Dropout(0.3)(x)

    out = Dense(n_actions, activation='softmax')(x)

    loss = CategoricalFocalCrossentropy(
        alpha=cfg.FOCAL_ALPHA,
        gamma=cfg.FOCAL_GAMMA,
        label_smoothing=cfg.LABEL_SMOOTHING
    )
    optimizer = Adam(learning_rate=cfg.LR_INITIAL, clipnorm=1.0)
    model = Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['categorical_accuracy']
    )
    return model
