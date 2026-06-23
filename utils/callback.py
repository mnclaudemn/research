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

    def on_epoch_begin(self):
        self.start_time = time.time()

    def compute_metrics(self, y_true, y_pred):
        return {
            "acc": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
            "balanced_acc": balanced_accuracy_score(y_true, y_pred)
        }

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
        # FIX 1: Safely handle 'val_loss' or metrics from the dictionary
        if self.monitor == "val_loss":
            current_metric = val_loss
        elif self.monitor == "train_loss":
            current_metric = train_loss
        else:
            metric_name = self.monitor.replace("val_", "").replace("train_", "")
            current_metric = val_metrics[metric_name] if "val_" in self.monitor else train_metrics[metric_name]

        # Early Stopping & Best Model Checkpoint
        if self.mode == "max":
            improved = current_metric > self.best_metric + self.min_delta
        else:
            improved = current_metric < self.best_metric - self.min_delta

        if improved:
            self.best_metric = current_metric
            self.counter = 0
            torch.save(model.state_dict(), self.best_model_path)
            print(f"Best model saved (epoch {epoch+1})")
        else:
            self.counter += 1
            print(f"No improvement ({self.counter}/{self.patience})")
            if self.counter >= self.patience:
                self.stop_training = True

        # Save Last Checkpoint (including early stopping state)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_metric": self.best_metric,
                "counter": self.counter
            },
            self.last_checkpoint_path
        )

        # FIX 2: Adaptable Learning Rate Scheduler
        if self.scheduler is not None:
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(current_metric)
            else:
                self.scheduler.step()

        epoch_time = time.time() - self.start_time
        lr = optimizer.param_groups[0]["lr"]

        # Logging
        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_acc": train_metrics["acc"],
            "val_acc": val_metrics["acc"],
            "train_precision": train_metrics["precision"],
            "val_precision": val_metrics["precision"],
            "train_recall": train_metrics["recall"],
            "val_recall": val_metrics["recall"],
            "train_f1": train_metrics["f1"],
            "val_f1": val_metrics["f1"],
            "train_bal_acc": train_metrics["balanced_acc"],
            "val_bal_acc": val_metrics["balanced_acc"],
            "lr": lr,
            "time_sec": epoch_time
        }

        self.history.append(row)
        pd.DataFrame(self.history).to_csv(self.log_path, index=False)

        print(
            f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_metrics['acc']:.4f} | Val F1: {val_metrics['f1']:.4f} | "
            f"LR: {lr:.2e} | Time: {epoch_time:.1f}s"
        )

    def resume(self, model, optimizer):
        if not os.path.exists(self.last_checkpoint_path):
            return 0

        checkpoint = torch.load(self.last_checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        # FIX 3: Retain historical bests and thresholds upon resume
        self.best_metric = checkpoint.get("best_metric", self.best_metric)
        self.counter = checkpoint.get("counter", 0)

        # FIX 4: Reload CSV logs so we append instead of overwriting history
        if os.path.exists(self.log_path):
            try:
                self.history = pd.read_csv(self.log_path).to_dict(orient="records")
            except Exception:
                self.history = []

        start_epoch = checkpoint["epoch"] + 1
        print(f"Resuming from epoch {start_epoch}")
        return start_epoch

    def plot_curves(self):
        if not self.history:
            return
        df = pd.DataFrame(self.history)

        for metric in ["loss", "acc", "f1"]:
            plt.figure(figsize=(8, 5))
            plt.plot(df[f"train_{metric}"], label="Train")
            plt.plot(df[f"val_{metric}"], label="Val")
            plt.title(f"{metric.capitalize()} Curves")
            plt.xlabel("Epoch")
            plt.ylabel(metric.capitalize())
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{metric}_curve.png")
            plt.close()

    def save_confusion_matrix(self, y_true, y_pred, class_names=None):
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        disp.plot(ax=ax, cmap="Blues") # Added a clean default color map
        plt.tight_layout()
        plt.savefig("confusion_matrix.png")
        plt.close()

        report = classification_report(y_true, y_pred, target_names=class_names)
        with open("classification_report.txt", "w") as f:
            f.write(report)
