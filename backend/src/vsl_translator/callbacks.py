"""
Custom Keras callbacks for fine-grained training control.

Provides:
  - CosineAnnealingWarmup: LR warmup then cosine decay
  - TrainingMonitor: per-epoch dashboard with overfitting detection
  - SmartEarlyStopping: stops on target accuracy/loss OR overfitting gap
"""

import math
import numpy as np
import keras


class CosineAnnealingWarmup(keras.callbacks.Callback):
    """Linear warmup for `warmup_epochs`, then cosine annealing to `lr_min`.

    This schedule helps the model:
      - Start training gently (warmup avoids early divergence)
      - Gradually reduce LR following a smooth cosine curve
      - Avoid getting stuck in sharp local minima
    """

    def __init__(self, lr_max, lr_min, warmup_epochs, total_epochs):
        super().__init__()
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            # Linear warmup: lr_min → lr_max
            lr = self.lr_min + (self.lr_max - self.lr_min) * (epoch / self.warmup_epochs)
        else:
            # Cosine annealing: lr_max → lr_min
            progress = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1 + math.cos(math.pi * progress))
        self.model.optimizer.learning_rate.assign(lr)


class TrainingMonitor(keras.callbacks.Callback):
    """Rich per-epoch dashboard that tracks loss, accuracy, LR, and overfitting.

    Prints a formatted table row each epoch showing:
      - Train/Val loss and accuracy
      - Current learning rate
      - Overfitting gap (train_acc - val_acc)
      - Warnings when overfitting is detected
      - Best val_loss / val_acc tracking
    """

    def __init__(self, overfit_gap_threshold=0.25):
        super().__init__()
        self.overfit_gap_threshold = overfit_gap_threshold
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.best_val_loss_epoch = 0
        self.best_val_acc_epoch = 0
        self.history = {
            'lr': [], 'loss': [], 'acc': [],
            'val_loss': [], 'val_acc': [], 'overfit_gap': [],
        }

    def on_train_begin(self, logs=None):
        print()
        print("┌─────┬────────────┬────────────┬────────────┬────────────┬────────────┬───────────┬────────┐")
        print("│ Ep  │ Train Loss │ Train Acc  │ Val Loss   │ Val Acc    │ LR         │ OvFit Gap │ Status │")
        print("├─────┼────────────┼────────────┼────────────┼────────────┼────────────┼───────────┼────────┤")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        train_loss = logs.get('loss', 0)
        train_acc = logs.get('categorical_accuracy', 0)
        val_loss = logs.get('val_loss', 0)
        val_acc = logs.get('val_categorical_accuracy', 0)
        lr = float(self.model.optimizer.learning_rate)

        # Track overfitting gap
        gap = train_acc - val_acc
        self.history['lr'].append(lr)
        self.history['loss'].append(train_loss)
        self.history['acc'].append(train_acc)
        self.history['val_loss'].append(val_loss)
        self.history['val_acc'].append(val_acc)
        self.history['overfit_gap'].append(gap)

        # Track bests
        status_parts = []
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_val_loss_epoch = epoch + 1
            status_parts.append("↓L")
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.best_val_acc_epoch = epoch + 1
            status_parts.append("↑A")

        # Overfitting warning
        if gap > self.overfit_gap_threshold:
            status_parts.append("⚠OF")
        elif gap > self.overfit_gap_threshold * 0.6:
            status_parts.append("~OF")

        status = " ".join(status_parts) if status_parts else "  ──"

        # Print every epoch (compact)
        ep = epoch + 1
        print(f"│ {ep:3d} │ {train_loss:10.6f} │ {train_acc:10.4f} │ {val_loss:10.6f} │ {val_acc:10.4f} │ {lr:10.2e} │ {gap:+9.4f} │ {status:<6s} │")

    def on_train_end(self, logs=None):
        print("└─────┴────────────┴────────────┴────────────┴────────────┴────────────┴───────────┴────────┘")
        print()
        print(f"  📊 Best val_loss : {self.best_val_loss:.6f} (epoch {self.best_val_loss_epoch})")
        print(f"  📊 Best val_acc  : {self.best_val_acc:.4f}   (epoch {self.best_val_acc_epoch})")

        # Summarise overfitting trend
        gaps = self.history['overfit_gap']
        if len(gaps) > 10:
            recent_gap = np.mean(gaps[-10:])
            early_gap = np.mean(gaps[:10])
            if recent_gap > self.overfit_gap_threshold:
                print(f"  ⚠️  Overfitting detected: recent gap={recent_gap:.4f} (threshold={self.overfit_gap_threshold})")
                print(f"      → Consider: more data, stronger augmentation, or more dropout")
            elif recent_gap > early_gap + 0.1:
                print(f"  ℹ️  Overfitting trend: gap grew from {early_gap:.4f} → {recent_gap:.4f}")
            else:
                print(f"  ✅ No significant overfitting (gap ≈ {recent_gap:.4f})")
        print()


class SmartEarlyStopping(keras.callbacks.Callback):
    """Multi-criteria early stopping with timely-stop signals.

    Stops training when ANY of these conditions are met:
      1. val_loss hasn't improved for `patience` epochs (standard)
      2. val_acc reaches `target_acc` AND val_loss <= `target_loss`
      3. Severe overfitting persists (gap > threshold for many epochs)
      4. TIMELY: val_loss rises consecutively for `loss_spike_patience`
         epochs (diverging — further training only wastes time)
      5. TIMELY: neither val_loss nor val_acc improves for
         `plateau_patience` epochs (converged — training has stalled)

    Always restores best weights, and ignores stopping criteria before `start_from_epoch`.
    """

    def __init__(self, patience=50, min_delta=1e-4,
                 target_acc=0.98, target_loss=0.05,
                 overfit_gap_threshold=0.25, overfit_patience=30,
                 start_from_epoch=15,
                 loss_spike_patience=8,
                 plateau_patience=15,
                 plateau_min_delta=1e-4):
        super().__init__()
        self.patience = patience
        self.min_delta = min_delta
        self.target_acc = target_acc
        self.target_loss = target_loss
        self.overfit_gap_threshold = overfit_gap_threshold
        self.overfit_patience = overfit_patience
        self.start_from_epoch = start_from_epoch
        self.loss_spike_patience = loss_spike_patience
        self.plateau_patience = plateau_patience
        self.plateau_min_delta = plateau_min_delta

        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.best_weights = None
        self.wait = 0
        self.spike_count = 0
        self.plateau_count = 0
        self.prev_val_loss = None
        self.overfit_count = 0
        self.stopped_epoch = 0
        self.stop_reason = ""

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_loss = logs.get('val_loss', float('inf'))
        val_acc = logs.get('val_categorical_accuracy', 0)
        train_acc = logs.get('categorical_accuracy', 0)
        gap = train_acc - val_acc

        # Track best weights whenever val_loss improves
        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.best_weights = self.model.get_weights()
            self.wait = 0
        else:
            self.wait += 1

        # Track best val_acc as well (for plateau detection)
        acc_improved = val_acc > self.best_val_acc + self.plateau_min_delta
        if acc_improved:
            self.best_val_acc = val_acc

        # Timely-stop signal 1: val_loss rising consecutively (diverging)
        if self.prev_val_loss is not None and val_loss > self.prev_val_loss + self.min_delta:
            self.spike_count += 1
        else:
            self.spike_count = 0
        self.prev_val_loss = val_loss

        # Timely-stop signal 2: plateau — neither loss nor acc improves
        if (val_loss < self.best_val_loss - self.min_delta) or acc_improved:
            self.plateau_count = 0
        else:
            self.plateau_count += 1

        # Track overfitting streak
        if gap > self.overfit_gap_threshold:
            self.overfit_count += 1
        else:
            self.overfit_count = 0

        # Do not stop training during warmup / before start_from_epoch
        if (epoch + 1) < self.start_from_epoch:
            return

        # Condition 1: Target reached
        if val_acc >= self.target_acc and val_loss <= self.target_loss:
            self.stop_reason = f"🎯 Đạt mục tiêu: val_acc={val_acc:.4f} ≥ {self.target_acc}, val_loss={val_loss:.6f} ≤ {self.target_loss}"
            self.stopped_epoch = epoch
            self.best_weights = self.model.get_weights()  # current is great
            self.model.stop_training = True
            return

        # Condition 2: No improvement for `patience` epochs
        if self.wait >= self.patience:
            self.stop_reason = f"⏸️  val_loss không cải thiện sau {self.patience} epochs (best={self.best_val_loss:.6f})"
            self.stopped_epoch = epoch
            self.model.stop_training = True
            return

        # Condition 4 (TIMELY): val_loss diverging — rising consecutively
        if self.spike_count >= self.loss_spike_patience:
            self.stop_reason = (
                f"📈 val_loss tăng liên tiếp {self.spike_count} epochs "
                f"(best={self.best_val_loss:.6f}) → dừng kịp lúc, tránh train thừa"
            )
            self.stopped_epoch = epoch
            self.model.stop_training = True
            return

        # Condition 5 (TIMELY): plateau — loss và acc cùng giậm chân
        if self.plateau_count >= self.plateau_patience:
            self.stop_reason = (
                f"➖ Hội tụ: val_loss/val_acc giậm chân {self.plateau_count} epochs "
                f"(best_loss={self.best_val_loss:.6f}, best_acc={self.best_val_acc:.4f})"
            )
            self.stopped_epoch = epoch
            self.model.stop_training = True
            return

        # Condition 3: Severe overfitting persists
        if self.overfit_count >= self.overfit_patience:
            self.stop_reason = (
                f"⚠️  Overfitting liên tục {self.overfit_count} epochs "
                f"(gap={gap:.4f} > {self.overfit_gap_threshold})"
            )
            self.stopped_epoch = epoch
            self.model.stop_training = True
            return

    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)
            print(f"  ✅ Đã khôi phục weights tốt nhất (val_loss={self.best_val_loss:.6f})")
        if self.stop_reason:
            print(f"  🛑 Dừng tại epoch {self.stopped_epoch + 1}: {self.stop_reason}")
        print()
