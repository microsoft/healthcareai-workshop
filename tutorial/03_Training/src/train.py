import argparse
import glob
import os

import mlflow
import pytorch_lightning as pl
import torchxrayvision as xrv
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from my_model import XRVDenseNetLightning
from utils import build_transforms


def save_model(model_dir, output_dir):
    import shutil

    from my_mlflow_wraper import MyMLFlowWraper, build_signature
    shutil.rmtree(output_dir, ignore_errors=True)

    mlflow_wrapper_paths = list(glob.glob('*.py'))

    mlflow.pyfunc.save_model(
        output_dir,
        python_model=MyMLFlowWraper(),
        code_path=mlflow_wrapper_paths,
        conda_env="environment.yml",
        signature=build_signature(),
        artifacts={
            "model_dir": model_dir
        }
    )

def main(args):

    # This section handles the AzureML run context
    # If running in AzureML, it gets the current run for logging
    # Otherwise, it creates a dummy run for local execution
    if "AZUREML_RUN_ID" in os.environ:
        from azureml.core.run import Run
        run = Run.get_context()
        
    else:
        print("Not running in AzureML, using dummy run.")
        class DummyRun:
            def log(self, *args, **kwargs):
                print(f"Log: {args}, {kwargs}")
        run = DummyRun()


    # Define transform for training and validation
    transform = build_transforms()

    images_dir = os.path.join(args.data_dir, "stage_2_train_images")

    # Load training dataset
    train_csv = os.path.join(args.data_dir, args.train_csv)
    train_dataset = xrv.datasets.RSNA_Pneumonia_Dataset(
        imgpath=images_dir,
        csvpath=train_csv,
        dicomcsvpath='USE_INCLUDED_FILE',
        views=['PA'],
        transform=transform,
        extension='.dcm'
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # Load validation set
    val_csv = os.path.join(args.data_dir, args.val_csv)
    val_dataset = xrv.datasets.RSNA_Pneumonia_Dataset(
        imgpath=images_dir,
        csvpath=val_csv,
        dicomcsvpath='USE_INCLUDED_FILE',
        views=['PA'],
        transform=transform,
        extension='.dcm'
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = XRVDenseNetLightning(weights=args.weights, learning_rate=args.learning_rate, goal_metric=args.goal_metric, run=run)

    # Define checkpoint callback to save the best model based on validation metric
    checkpoint_callback = ModelCheckpoint(
        monitor="val_goal_metric_val",
        mode="max",
        dirpath="outputs/best",
        filename="best_model",
        save_top_k=1,
    )

    args_list = [
        "data_dir",
        "train_csv",
        "val_csv",
        "batch_size",
        "learning_rate",
        "weights",
        "goal_metric",
    ]

    kwargs = vars(args)

    mlflow_model_dir = kwargs.pop("mlflow_model_dir")
    kwargs["default_root_dir"] = "outputs"
    for k in args_list:
        kwargs.pop(k)

    kwargs["accelerator"] = kwargs["accelerator"] or "auto"
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback],
        **kwargs
    )

    # Logs initial validation metrics to AzureML before training starts
    initial_metrics = trainer.validate(model, dataloaders=val_loader, verbose=True)[0]
    for k, v in initial_metrics.items():
        run.log(k, v)

    # Now start training
    trainer.fit(model, train_loader, val_loader)

    # Save the best model
    best_model_path = checkpoint_callback.best_model_path
    save_model(best_model_path, mlflow_model_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="./data", help="Directory with CSV files and images.")
    parser.add_argument("--mlflow_model_dir", type=str, default="outputs/mlflow_model_dir", help="Directory to save the MLflow model.")
    parser.add_argument("--train_csv", type=str, default="train.csv", help="Path to the training CSV file.")
    parser.add_argument("--val_csv", type=str, default="val.csv", help="Path to the validation CSV file.")

    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for the optimizer.")
    parser.add_argument("--weights", type=str, default="chexpert", help="Pretrained weights short name for XRV DenseNet (e.g., 'chexpert', 'all').")
    parser.add_argument("--goal_metric", type=str, default="auc", help="Metric to optimize during training (e.g., 'auc', 'f1').")

    pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()
    main(args)