import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torchxrayvision as xrv
from my_metrics import CompositeClassificationMetric
from torchmetrics.classification import (BinaryAccuracy, BinaryAUROC,
                                         BinaryF1Score, BinaryPrecision,
                                         BinaryRecall)


class XRVDenseNetLightning(pl.LightningModule):
    def __init__(self, weights="chexpert", learning_rate=1e-3, goal_metric="auc", run=None):
        super().__init__()
        weight_mapping = {
            "chexpert": "densenet121-res224-chex",
            "all": "densenet121-res224-all",
            "rsna": "densenet121-res224-rsna",
            "nih": "densenet121-res224-nih",
            "pc": "densenet121-res224-pc",
            "mimic_nb": "densenet121-res224-mimic_nb",
            "mimic_ch": "densenet121-res224-mimic_ch"
        }
        if weights in weight_mapping:
            weights = weight_mapping[weights]

        self.goal_metric = goal_metric
        self.model = xrv.models.DenseNet(num_classes=1, weights=None)
        self.criterion = nn.BCEWithLogitsLoss()
        self.learning_rate = learning_rate

        # Metrics on CPU
        self.train_metrics = CompositeClassificationMetric({
            "acc": BinaryAccuracy(threshold=0.5),
        }, device="cpu")

        self.val_metrics = CompositeClassificationMetric({
            "acc": BinaryAccuracy(threshold=0.5),
            "f1": BinaryF1Score(threshold=0.5),
            "precision": BinaryPrecision(threshold=0.5),
            "recall": BinaryRecall(threshold=0.5),
            "auc": BinaryAUROC()
        }, device="cpu")

        self.run = run

    def forward(self, x):
        logits = self.model(x)
        return torch.sigmoid(logits)
    
    # This method overrides PyTorch Lightning's log method
    # It ensures all metrics logged by Lightning are also sent to AzureML
    def log(self, name, value, *args, **kwargs):
        # Log to AzureML as well as using Lightning's logging mechanism
        if self.run is not None:
            self.run.log(name, value)
        return super().log(name, value, *args, **kwargs)

    def training_step(self, batch, batch_idx):
        images, labels = _extract_batch(batch)
        outputs = self(images).squeeze(1)
        loss = self.criterion(outputs, labels)
        preds = torch.sigmoid(outputs)
        self.train_metrics.update(preds, labels.int(), loss_val=loss.item())
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = _extract_batch(batch)
        outputs = self(images).squeeze(1)
        loss = self.criterion(outputs, labels)
        preds = torch.sigmoid(outputs)
        self.val_metrics.update(preds, labels.int(), loss_val=loss.item())

    def on_train_epoch_end(self):
        results = self.train_metrics.compute()
        self.log_dict({f"train_{k}": v for k, v in results.items()})
        self.train_metrics.reset()

    def on_validation_epoch_end(self):
        results = self.val_metrics.compute()
        for k, v in results.items():
            self.log(f"val_{k}", v)
        # Track best metric
        goal_metric_value = results.get(self.goal_metric)
        if goal_metric_value is not None:
            self.log(f"val_goal_metric_val", goal_metric_value)
        self.val_metrics.reset()

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)


def _extract_batch(batch):
    return batch["img"], batch["lab"][:, 0]