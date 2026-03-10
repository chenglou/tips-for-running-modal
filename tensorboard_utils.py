# TensorBoard logging utilities for experiments
# Usage:
#   from tensorboard_utils import TBLogger
#   logger = TBLogger(output_dir, "exp_name")
#   logger.log(step, loss=0.5, train_acc=0.8, test_acc=0.7)
#   logger.close()

import os

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False


class TBLogger:
    """Simple TensorBoard logger that gracefully handles missing tensorboard."""

    def __init__(self, output_dir, experiment_name):
        self.writer = None
        if HAS_TENSORBOARD:
            log_dir = os.path.join(output_dir, "runs", experiment_name)
            self.writer = SummaryWriter(log_dir=log_dir)
            print(f"TensorBoard logging to: {log_dir}")
        else:
            print("TensorBoard not installed, skipping TB logging")

    def log(self, step, **kwargs):
        """Log metrics to TensorBoard.

        Common kwargs:
            loss: training loss
            train_acc: training accuracy (accepts 0-1 or 0-100, stores percent)
            test_acc: test accuracy (accepts 0-1 or 0-100, stores percent)

        Any kwarg is logged as a scalar with name based on key.
        """
        if self.writer is None:
            return

        for key, value in kwargs.items():
            if value is not None:
                # Convert common names to TensorBoard conventions
                if key == "loss":
                    self.writer.add_scalar("Loss/train", value, step)
                elif key == "train_acc":
                    self.writer.add_scalar("Accuracy/train", self._normalize_accuracy(value), step)
                elif key == "test_acc":
                    self.writer.add_scalar("Accuracy/test", self._normalize_accuracy(value), step)
                else:
                    self.writer.add_scalar(key, value, step)

    def close(self):
        if self.writer is not None:
            self.writer.close()

    @staticmethod
    def _normalize_accuracy(value):
        if 0.0 <= value <= 1.0:
            return value * 100.0
        return value
