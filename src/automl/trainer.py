from typing import Tuple

import tqdm
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch import distributed as TorchDistributed

from automl.utils import log as print, MAIN_NODE, RANK, DISTRIBUTED, log_prefix


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer = None,
        lr_scheduler: optim.lr_scheduler._LRScheduler = None,
        loss_fn: nn.Module = None,
        device: torch.device = None,
        output_device: torch.device = None,
        results_file: str = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_fn = loss_fn
        self.device = device
        self.output_device = output_device if output_device else device
        self.results_file = results_file

        self.model.to(self.device)
        self.model.train()

        if self.optimizer is None:
            self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-3)

        if self.lr_scheduler is None:
            self.lr_scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=1000, gamma=0.1
            )

        if self.loss_fn is None:
            self.loss_fn = nn.CrossEntropyLoss()

    @property
    def pb_desc_template(self):
        return f"{log_prefix()} " + "Epoch {}/{} - {}"

    def train_step(
        self, data: torch.Tensor, target: torch.Tensor
    ) -> Tuple[float, float]:
        self.optimizer.zero_grad()
        data, target = data.to(self.device), target.to(self.output_device)
        output = self.model(data)
        loss = self.loss_fn(output, target)
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()

        accuracy = (output.argmax(dim=1) == target).float().mean()
        return loss.item(), accuracy.item()

    def train_epoch(
        self,
        data_loader: DataLoader,
        epoch: int,
        epochs: int,
    ) -> Tuple[float, float]:
        cumulative_loss = 0.0
        cumulative_accuracy = 0.0

        for data, target in tqdm.tqdm(
            data_loader,
            desc=self.pb_desc_template.format(epoch, epochs, "Train"),
            position=RANK,
        ):
            loss, accuracy = self.train_step(data, target)
            cumulative_loss += loss
            cumulative_accuracy += accuracy

        return cumulative_loss / len(data_loader), cumulative_accuracy / len(
            data_loader
        )

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
    ):
        losses, accuracies = [], []
        val_losses, val_accuracies = [], []

        for epoch in range(1, epochs + 1):
            train_loss, train_accuracy = self.train_epoch(train_loader, epoch, epochs)

            if DISTRIBUTED:
                TorchDistributed.barrier()

            val_loss, val_accuracy, _ = self.eval(val_loader, epoch, epochs)

            if DISTRIBUTED:
                TorchDistributed.barrier()
                print(
                    f"Epoch {epoch}/{epochs} - "
                    f"Train Loss: {train_loss:.4f}, "
                    f"Train Accuracy: {train_accuracy:.4f}, "
                    f"Val Loss: {val_loss:.4f}, "
                    f"Val Accuracy: {val_accuracy:.4f}"
                )
                metrics = torch.tensor(
                    [train_loss, train_accuracy, val_loss, val_accuracy],
                    device=self.device,
                )
                TorchDistributed.all_reduce(metrics)
                metrics /= TorchDistributed.get_world_size()
                train_loss, train_accuracy, val_loss, val_accuracy = metrics.tolist()
                TorchDistributed.barrier()

            if MAIN_NODE:
                print(
                    f"Epoch {epoch}/{epochs} (Overall Values) - "
                    f"Train Loss: {train_loss:.4f}, "
                    f"Train Accuracy: {train_accuracy:.4f}, "
                    f"Val Loss: {val_loss:.4f}, "
                    f"Val Accuracy: {val_accuracy:.4f}"
                )
                print("-" * 80, no_prefix=True)

                losses.append(train_loss)
                accuracies.append(train_accuracy)
                val_losses.append(val_loss)
                val_accuracies.append(val_accuracy)

            if TorchDistributed.is_initialized():
                TorchDistributed.barrier()

        if self.results_file is not None and MAIN_NODE:
            data = {
                "epoch": range(1, epochs + 1),
                "train_loss": losses,
                "train_accuracy": accuracies,
                "val_loss": val_losses,
                "val_accuracy": val_accuracies,
            }
            with open(self.results_file, "w") as f:
                f.write("epoch,train_loss,train_accuracy,val_loss,val_accuracy\n")
                for i in range(epochs):
                    f.write(
                        f"{data['epoch'][i]},{data['train_loss'][i]},{data['train_accuracy'][i]},{data['val_loss'][i]},{data['val_accuracy'][i]}\n"
                    )

    def eval_step(
        self,
        data: torch.Tensor,
        target: torch.Tensor,
        return_predictions: bool = False,
    ) -> Tuple[float, float, torch.Tensor | None]:
        data, target = data.to(self.device), target.to(self.output_device)
        output = self.model(data)
        loss = self.loss_fn(output, target)

        predictions = output.argmax(dim=1)
        accuracy = (predictions == target).float().mean()
        return loss.item(), accuracy.item(), predictions if return_predictions else None

    def eval(
        self,
        data_loader: DataLoader,
        epoch: int = None,
        epochs: int = None,
        return_predictions: bool = False,
    ) -> Tuple[float, float, torch.Tensor | None]:
        with torch.no_grad():
            self.model.eval()

            cumulative_loss = 0.0
            cumulative_accuracy = 0.0

            if return_predictions:
                all_predictions = torch.tensor([], device=self.device)

            for data, target in tqdm.tqdm(
                data_loader,
                desc=(
                    self.pb_desc_template.format(epoch, epochs, "Val")
                    if epoch is not None and epochs is not None
                    else f"{log_prefix()} " + "Val"
                ),
                position=RANK,
            ):
                loss, accuracy, predictions = self.eval_step(
                    data, target, return_predictions
                )
                cumulative_loss += loss
                cumulative_accuracy += accuracy

                if return_predictions:
                    all_predictions = torch.cat([all_predictions, predictions])

            self.model.train()

            return (
                cumulative_loss / len(data_loader),
                cumulative_accuracy / len(data_loader),
                predictions if return_predictions else None,
            )

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path), map_location=self.device)
        self.model.train()
