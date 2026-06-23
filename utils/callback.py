import os
import time
import torch
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)


class ResearchCallbacks:
    def __init__(
        self,
        monitor="val_f1",
        mode="max",
        patience=4,
        min_delta=1e-3,
        best_model_path="best_model.pth",
        last_checkpoint_path="last_checkpoint.pth",
        log_path="training_log.csv",
        scheduler=None
    ):

        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta

        self.best_model_path = best_model_path
        self.last_checkpoint_path = last_checkpoint_path
        self.log_path = log_path

        self.scheduler = scheduler

        self.best_metric = (
            -float("inf")
            if mode == "max"
            else float("inf")
        )

        self.counter = 0
        self.stop_training = False
        self.start_time = None

        self.history = []

    ##################################################
    # Timer
    ##################################################

    def on_epoch_begin(self):
        self.start_time = time.time()

    ##################################################
    # Compute Metrics
    ##################################################

    def compute_metrics(
        self,
        y_true,
        y_pred
    ):

        metrics = {
            "acc": accuracy_score(y_true, y_pred),
            "precision": precision_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0
            ),
            "recall": recall_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0
            ),
            "f1": f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0
            ),
            "balanced_acc": balanced_accuracy_score(
                y_true,
                y_pred
            )
        }

        return metrics

    ##################################################
    # End Epoch
    ##################################################

    def on_epoch_end(
        self,
        epoch,
        model,
        optimizer,
        train_loss,
        val_loss,
        train_metrics,
        val_metrics
    ):

        metric_name = self.monitor.replace(
            "val_",
            ""
        )

        current_metric = val_metrics[
            metric_name
        ]

        ##################################################
        # Best Model + Early Stopping
        ##################################################

        if self.mode == "max":
            improved = (
                current_metric >
                self.best_metric +
                self.min_delta
            )
        else:
            improved = (
                current_metric <
                self.best_metric -
                self.min_delta
            )

        if improved:

            self.best_metric = current_metric
            self.counter = 0

            torch.save(
                model.state_dict(),
                self.best_model_path
            )

            print(
                f"Best model saved "
                f"(epoch {epoch+1})"
            )

        else:

            self.counter += 1

            print(
                f"No improvement "
                f"({self.counter}/"
                f"{self.patience})"
            )

            if self.counter >= self.patience:
                self.stop_training = True

        ##################################################
        # Save Last Checkpoint
        ##################################################

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict":
                    model.state_dict(),
                "optimizer_state_dict":
                    optimizer.state_dict()
            },
            self.last_checkpoint_path
        )

        ##################################################
        # Scheduler
        ##################################################

        if self.scheduler is not None:
            self.scheduler.step(
                current_metric
            )

        ##################################################
        # Timer + LR
        ##################################################

        epoch_time = (
            time.time() -
            self.start_time
        )

        lr = optimizer.param_groups[0]["lr"]

        ##################################################
        # Logging
        ##################################################

        row = {
            "epoch": epoch + 1,

            "train_loss": train_loss,
            "val_loss": val_loss,

            "train_acc":
                train_metrics["acc"],
            "val_acc":
                val_metrics["acc"],

            "train_precision":
                train_metrics["precision"],
            "val_precision":
                val_metrics["precision"],

            "train_recall":
                train_metrics["recall"],
            "val_recall":
                val_metrics["recall"],

            "train_f1":
                train_metrics["f1"],
            "val_f1":
                val_metrics["f1"],

            "train_bal_acc":
                train_metrics[
                    "balanced_acc"
                ],
            "val_bal_acc":
                val_metrics[
                    "balanced_acc"
                ],

            "lr": lr,
            "time_sec": epoch_time
        }

        self.history.append(row)

        pd.DataFrame(
            self.history
        ).to_csv(
            self.log_path,
            index=False
        )

        ##################################################
        # Print
        ##################################################

        print(
            f"Epoch {epoch+1} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_metrics['acc']:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f} | "
            f"LR: {lr:.2e} | "
            f"Time: {epoch_time:.1f}s"
        )

    ##################################################
    # Resume
    ##################################################

    def resume(
        self,
        model,
        optimizer
    ):

        if not os.path.exists(
            self.last_checkpoint_path
        ):
            return 0

        checkpoint = torch.load(
            self.last_checkpoint_path
        )

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        optimizer.load_state_dict(
            checkpoint[
                "optimizer_state_dict"
            ]
        )

        start_epoch = (
            checkpoint["epoch"] + 1
        )

        print(
            f"Resuming from "
            f"epoch {start_epoch}"
        )

        return start_epoch

    ##################################################
    # Plot Curves
    ##################################################

    def plot_curves(self):

        df = pd.DataFrame(
            self.history
        )

        # Loss
        plt.figure(figsize=(8, 5))
        plt.plot(df["train_loss"])
        plt.plot(df["val_loss"])
        plt.legend(["Train", "Val"])
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.tight_layout()
        plt.savefig("loss_curve.png")
        plt.close()

        # Accuracy
        plt.figure(figsize=(8, 5))
        plt.plot(df["train_acc"])
        plt.plot(df["val_acc"])
        plt.legend(["Train", "Val"])
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.tight_layout()
        plt.savefig("accuracy_curve.png")
        plt.close()

        # F1
        plt.figure(figsize=(8, 5))
        plt.plot(df["train_f1"])
        plt.plot(df["val_f1"])
        plt.legend(["Train", "Val"])
        plt.xlabel("Epoch")
        plt.ylabel("F1")
        plt.tight_layout()
        plt.savefig("f1_curve.png")
        plt.close()

    ##################################################
    # Confusion Matrix
    ##################################################

    def save_confusion_matrix(
        self,
        y_true,
        y_pred,
        class_names=None
    ):

        cm = confusion_matrix(
            y_true,
            y_pred
        )

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=class_names
        )

        fig, ax = plt.subplots(
            figsize=(8, 8)
        )

        disp.plot(ax=ax)
        plt.tight_layout()
        plt.savefig(
            "confusion_matrix.png"
        )
        plt.close()

        report = classification_report(
            y_true,
            y_pred
        )

        with open(
            "classification_report.txt",
            "w"
        ) as f:
            f.write(report)
