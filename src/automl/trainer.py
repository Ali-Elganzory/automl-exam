from time import time
from enum import Enum
from pathlib import Path
from typing import Tuple, Type, List, Union

import tqdm
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch import distributed as TorchDistributed

from automl.model import Model
from automl import distributed as dist


class Optimizer(Enum):
    adamw = "adamw"
    adam = "adam"
    sgd = "sgd"
    rmsprop = "rmsprop"

    @property
    def factory(self):
        return OPTIMIZERS[self]


OPTIMIZERS = {
    Optimizer.adamw: optim.AdamW,
    Optimizer.adam: optim.Adam,
    Optimizer.sgd: optim.SGD,
    Optimizer.rmsprop: optim.RMSprop,
}


class LR_Scheduler(Enum):
    step = "step"
    multi_step = "multi_step"
    exponential = "exponential"

    @property
    def factory(self):
        return LR_SCHEDULERS[self]


LR_SCHEDULERS = {
    LR_Scheduler.step: optim.lr_scheduler.StepLR,
    LR_Scheduler.multi_step: optim.lr_scheduler.MultiStepLR,
    LR_Scheduler.exponential: optim.lr_scheduler.ExponentialLR,
}


class LossFn(Enum):
    cross_entropy = "cross_entropy"

    @property
    def factory(self):
        return LOSS_FNS[self]


LOSS_FNS = {
    LossFn.cross_entropy: lambda: lambda x, y, weight=None: nn.CrossEntropyLoss(weight)(
        x, y
    ),
}


class Trainer:
    def __init__(
        self,
        model: Model,
        optimizer: Union[
            optim.Optimizer, Type[optim.Optimizer], Optimizer
        ] = Optimizer.adamw,
        lr: float = 1e-3,
        weight_decay: float = 1e-2,
        lr_scheduler: Union[
            optim.lr_scheduler._LRScheduler,
            Type[optim.lr_scheduler._LRScheduler],
            LR_Scheduler,
        ] = LR_Scheduler.step,
        scheduler_step_size: int = 1000,
        scheduler_gamma: float = 0.1,
        scheduler_step_every_epoch: bool = False,
        loss_fn: Union[nn.Module, Type[nn.Module], LossFn] = LossFn.cross_entropy,
        results_file: str = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_scheduler = lr_scheduler
        self.scheduler_step_size = scheduler_step_size
        self.scheduler_gamma = scheduler_gamma
        self.scheduler_step_every_epoch = scheduler_step_every_epoch
        self.loss_fn = loss_fn
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.output_device = self.device
        self.results_file = results_file

        self.epochs_already_trained: int = 0

        self.model.to(self.device)
        if dist.is_enabled():
            self.model = nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[dist.get_local_rank()],
                output_device=dist.get_local_rank(),
            )
        self.model.train()

        if not isinstance(self.optimizer, optim.Optimizer):
            if isinstance(self.optimizer, Optimizer):
                self.optimizer = self.optimizer.factory
            self.optimizer = self.optimizer(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )

        if not isinstance(self.lr_scheduler, optim.lr_scheduler._LRScheduler):
            if isinstance(self.lr_scheduler, LR_Scheduler):
                self.lr_scheduler = self.lr_scheduler.factory
            self.lr_scheduler = self.lr_scheduler(
                self.optimizer,
                step_size=self.scheduler_step_size,
                gamma=self.scheduler_gamma,
            )

        if not isinstance(self.loss_fn, nn.Module):
            if isinstance(self.loss_fn, LossFn):
                self.loss_fn = self.loss_fn.factory()

    @property
    def pb_desc_template(self):
        return "Epoch {}/{} - {}"

    def train_step(
        self,
        data: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[float, float]:
        self.optimizer.zero_grad()
        data, target = data.to(self.device), target.to(self.output_device)
        output = self.model(data)

        labels = list(range(self.model.num_classes))
        loss_weight = (
            None
            if labels is None
            else torch.tensor(
                data.size(0) * data.size(2) * data.size(3),
                device=self.device,
            )
            / (
                torch.tensor(
                    list(map(lambda label: (target == label).sum(), labels)),
                    device=self.device,
                )
                + 1e-6
            )
        )

        loss = self.loss_fn(output, target, loss_weight)
        loss.backward()
        self.optimizer.step()
        if not self.scheduler_step_every_epoch:
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
        ):
            loss, accuracy = self.train_step(data, target)
            cumulative_loss += loss
            cumulative_accuracy += accuracy

        if self.scheduler_step_every_epoch:
            self.lr_scheduler.step()

        return cumulative_loss / len(data_loader), cumulative_accuracy / len(
            data_loader
        )

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        save_best_to: Union[str, Path, None] = None,
    ) -> Tuple[List[float], List[float], List[float], List[float], float]:
        """Train the model.

        Args:
            train_loader (DataLoader): The training data loader.
            val_loader (DataLoader): The validation data loader.
            epochs (int): The number of epochs to train for.

        Returns:
            Tuple[List[float], List[float], List[float], List[float], float]: The training losses, training accuracies, validation losses, validation accuracies, and the time taken to train.
        """
        start_time = time()

        losses, accuracies = [], []
        val_losses, val_accuracies = [], []
        f1s = []

        best_val_loss = float("inf")

        for epoch in range(self.epochs_already_trained + 1, epochs + 1):
            train_loss, train_accuracy = self.train_epoch(train_loader, epoch, epochs)
            self.epochs_already_trained = epochs

            if dist.is_enabled():
                TorchDistributed.barrier()

            val_loss, val_accuracy, f1, confusion_matrix, _ = self.eval(
                val_loader, epoch, epochs
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if save_best_to is not None:
                    self.save_model(save_best_to)

            if dist.is_enabled():
                TorchDistributed.barrier()
                print(
                    f"Epoch {epoch}/{epochs} - "
                    f"Train Loss: {train_loss:.4f}, "
                    f"Train Accuracy: {train_accuracy:.4f}, "
                    f"Val Loss: {val_loss:.4f}, "
                    f"Val Accuracy: {val_accuracy:.4f}",
                    f"F1 Score: {f1:.4f}",
                    force=True,
                )
                metrics = torch.tensor(
                    [train_loss, train_accuracy, val_loss, val_accuracy, f1],
                    device=self.device,
                )
                TorchDistributed.all_reduce(metrics)
                metrics /= TorchDistributed.get_world_size()
                train_loss, train_accuracy, val_loss, val_accuracy, f1 = (
                    metrics.tolist()
                )
                TorchDistributed.barrier()

            print(
                f"Epoch {epoch}/{epochs} (Overall Values) - "
                f"Train Loss: {train_loss:.4f}, "
                f"Train Accuracy: {train_accuracy:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Val Accuracy: {val_accuracy:.4f}, "
                f"F1 Score: {f1:.4f}",
            )

            self.print_confusion_matrix(confusion_matrix)

            print("-" * 80)

            if dist.is_main_process():
                losses.append(train_loss)
                accuracies.append(train_accuracy)
                val_losses.append(val_loss)
                val_accuracies.append(val_accuracy)
                f1s.append(f1)

            if dist.is_enabled():
                TorchDistributed.barrier()

        if self.results_file is not None and dist.is_main_process():
            data = {
                "epoch": range(1, epochs + 1),
                "train_loss": losses,
                "train_accuracy": accuracies,
                "val_loss": val_losses,
                "val_accuracy": val_accuracies,
                "f1": f1s,
            }
            with open(self.results_file, "w") as f:
                f.write("epoch,train_loss,train_accuracy,val_loss,val_accuracy\n")
                for i in range(epochs):
                    f.write(
                        f"{data['epoch'][i]},{data['train_loss'][i]},{data['train_accuracy'][i]},{data['val_loss'][i]},{data['val_accuracy'][i]},{data['f1'][i]}\n"
                    )

        return losses, accuracies, val_losses, val_accuracies, time() - start_time

    def eval_step(
        self,
        data: torch.Tensor,
        target: torch.Tensor,
        return_predictions: bool = False,
    ) -> Tuple[float, float, Union[torch.Tensor, None]]:
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
    ) -> Tuple[float, float, float, torch.Tensor, Union[torch.Tensor, None]]:
        with torch.no_grad():
            self.model.eval()

            cumulative_loss = 0.0
            cumulative_accuracy = 0.0

            num_classes = self._inner_model.num_classes
            confusion_matrix = torch.zeros(
                num_classes,
                num_classes,
                device=self.device,
                dtype=torch.int32,
            )

            if return_predictions:
                all_predictions = torch.tensor([], device=self.device)

            for data, target in tqdm.tqdm(
                data_loader,
                desc=(
                    self.pb_desc_template.format(epoch, epochs, "Val")
                    if epoch is not None and epochs is not None
                    else "Val"
                ),
            ):
                loss, accuracy, predictions = self.eval_step(data, target, True)
                cumulative_loss += loss
                cumulative_accuracy += accuracy

                # Calculate confusion matrix
                confusion_matrix += torch.bincount(
                    torch.flatten(target.to(self.device) * num_classes + predictions),
                    minlength=num_classes**2,
                ).reshape(num_classes, num_classes)

                # Normalize confusion matrix
                confusion_matrix = (
                    confusion_matrix.float()
                    / confusion_matrix.sum(dim=1, keepdim=True).clamp(min=1e-6)
                ).round(decimals=2)

                # Calculate F1 Score
                precision = confusion_matrix.diag() / confusion_matrix.sum(dim=0).clamp(
                    min=1e-6
                )
                recall = confusion_matrix.diag() / confusion_matrix.sum(dim=1).clamp(
                    min=1e-6
                )
                f1 = 2 * (precision * recall) / (precision + recall).clamp(min=1e-6)
                f1 = f1.mean()

                if return_predictions:
                    all_predictions = torch.cat([all_predictions, predictions])

            self.model.train()

            return (
                cumulative_loss / len(data_loader),
                cumulative_accuracy / len(data_loader),
                f1.item(),
                confusion_matrix,
                predictions if return_predictions else None,
            )

    def predict(self, x: Union[torch.Tensor, DataLoader]) -> torch.Tensor:
        with torch.no_grad():
            self.model.eval()

            all_predictions = self.model.predict(x)

            if dist.is_enabled():
                TorchDistributed.barrier()
                # Gather the size of local predictions
                size = torch.tensor([all_predictions.size(0)], device=self.device)
                sizes = [
                    torch.zeros_like(size)
                    for _ in range(TorchDistributed.get_world_size())
                ]
                TorchDistributed.all_gather(sizes, size)
                # Gather all predictions
                local_predictions = all_predictions
                all_predictions = [
                    torch.zeros(s.item(), device=self.device) for s in sizes
                ]
                TorchDistributed.all_gather(all_predictions, local_predictions)
                all_predictions = torch.cat(all_predictions)
                TorchDistributed.barrier()

            self.model.train()

            return all_predictions.type(torch.int32)

    @property
    def _inner_model(self) -> Model:
        return (
            self.model.module
            if isinstance(self.model, nn.parallel.DistributedDataParallel)
            else self.model
        )

    @property
    def _model_name(self) -> str:
        return self._inner_model.__class__.__name__

    def save(self, path: Union[str, Path, None] = None):
        if not dist.is_main_process():
            return

        path = Path(path or f"checkpoints/{self._model_name}.pt")
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "lr_scheduler": self.lr_scheduler.state_dict(),
                "epochs_already_trained": self.epochs_already_trained,
            },
            path,
        )

    def load(self, path: Union[str, Path, None] = None):
        path = Path(path or f"checkpoints/{self._model_name}.pt")

        if not path.exists():
            raise FileNotFoundError(f"File {path} does not exist")

        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        self.epochs_already_trained = checkpoint["epochs_already_trained"]

    def save_model(self, path: Union[str, Path, None] = None):
        if not dist.is_main_process():
            return

        path = Path(path or f"models/{self._model_name}.pt")
        path.parent.mkdir(parents=True, exist_ok=True)

        self._inner_model.save(path)

    def load_model(self, path: Union[str, Path, None] = None):
        print("load_model", force=True)
        path = Path(path or f"models/{self._model_name}.pt")

        if not path.exists():
            raise FileNotFoundError(f"File {path} does not exist")

        state_dict = torch.load(path, map_location=self.device)
        self._inner_model.load_state_dict(state_dict)

    def print_confusion_matrix(self, confusion_matrix: torch.Tensor):
        confusion_matrix_string = f"Confusion Matrix:\n"
        bg_color_code = lambda score: f"\033[48;2;0;{255 - int(score * 255)};0m"
        for i in range(confusion_matrix.size(0)):
            for j in range(confusion_matrix.size(1)):
                confusion_matrix_string += (
                    bg_color_code(confusion_matrix[i, i]) if i == j else "\033[0m"
                ) + f"{confusion_matrix[i, j]:.2f} "
            confusion_matrix_string += "\033[0m\n"
        print(confusion_matrix_string, end="")
