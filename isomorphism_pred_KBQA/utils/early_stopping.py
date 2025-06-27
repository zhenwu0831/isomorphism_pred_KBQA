# utils/early_stopping.py

class EarlyStopping:
    """
    Early stopping to terminate training when a monitored metric stops improving.

    Args:
        patience (int): Number of epochs with no improvement after which training is stopped.
        delta (float): Minimum change in the monitored metric to qualify as improvement.
    """
    def __init__(self, patience: int = 5, delta: float = 0.0):
        self.patience = patience
        self.delta = delta
        self.best_score: float = None
        self.counter: int = 0
        self.early_stop: bool = False

    def __call__(self, metric: float) -> None:
        """
        Call with the new metric value. Updates internal counter and stop flag.
        """
        if self.best_score is None or metric > self.best_score + self.delta:
            self.best_score = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def should_stop(self) -> bool:
        """
        Returns True if training should be stopped early.
        """
        return self.early_stop