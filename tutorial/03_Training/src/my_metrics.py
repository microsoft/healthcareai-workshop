class CompositeClassificationMetric:
    """
    Similar to transforms.Compose, but for multiple classification metrics,
    plus optional average loss counting.
    """
    def __init__(self, metrics_dict, device='cpu'):
        """
        Args:
            metrics_dict (dict): A dict mapping names -> TorchMetrics objects.
            device (str): The device to move metrics to.
            track_loss (bool): Whether to track average loss or not.
        """
        self.metrics_dict = {}
        for name, metric in metrics_dict.items():
            self.metrics_dict[name] = metric.to(device)
        self.loss_sum = 0.0
        self.num_samples = 0
        self.device = device

    def update(self, preds, targets, loss_val=None):
        preds = preds.to(self.device)
        targets = targets.to(self.device)

                # If tracking loss, aggregate here
        if loss_val is not None:
            # loss_val = loss_val.to(self.device)
            # We can infer batch size from preds
            batch_size = preds.size(0)
            self.loss_sum += loss_val * batch_size
            self.num_samples += batch_size

        # Update classification metrics
        for metric in self.metrics_dict.values():
            metric.update(preds, targets)

    def compute(self):
        # Compute classification metrics
        results = {}
        for name, metric in self.metrics_dict.items():
            results[name] = metric.compute().item()

        # Compute average loss if requested
        if self.num_samples > 0:
            results["loss"] = self.loss_sum / self.num_samples
        return results

    def reset(self):
        # Reset classification metrics
        for metric in self.metrics_dict.values():
            metric.reset()

        # Reset loss counters
        self.loss_sum = 0.0
        self.num_samples = 0