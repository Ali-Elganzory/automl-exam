"""
AutoML class for vision classification tasks.
"""

from __future__ import annotations
from time import time
from pathlib import Path
from typing import Tuple, Type

import neps
import torch
import random
import numpy as np
from torch import nn
from torchvision.transforms.v2 import RandomChoice, AugMix, TrivialAugmentWide

from automl.model import ResNet50
from automl.trainer import Trainer, Optimizer, LR_Scheduler, LossFn

from automl.utils import log as print
from automl.data import DataLoaders, BaseVisionDataset


class AutoML:
    augmentations = RandomChoice(
        [
            AugMix(),
            TrivialAugmentWide(),
        ]
    )

    def __init__(
        self,
        dataset_class: Type[BaseVisionDataset],
        seed: int,
    ) -> None:
        self.dataset_class = dataset_class
        self.seed = seed
        self.dataloaders = None
        self.trainer = None

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    @property
    def model(self) -> nn.Module:
        return self.trainer.model

    def _generate_plots(self, train_losses, train_accuracies, val_losses, val_accuracies):
        epochs = range(1, len(train_losses) + 1)
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_losses, label='Training loss')
        plt.plot(epochs, val_losses, label='Validation loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_accuracies, label='Training accuracy')
        plt.plot(epochs, val_accuracies, label='Validation accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_validation_plots.png')
        plt.show()
        
    def run_pipeline(
        self,
        epochs: int,
        batch_size: int,
        optimizer: str | Optimizer,
        lr: float,
        weight_decay: float,
        lr_scheduler: str | LR_Scheduler,
        scheduler_step_size: int,
        scheduler_gamma: float,
        schedular_step_every_epoch: bool,
        loss_fn: str | LossFn,
        device: str | None = None,
        output_device: str | None = None,
        pipeline_directory: Path | None = None,
        previous_pipeline_directory: Path | None = None,
        results_file: str | None = None,
    ) -> None:
        if isinstance(optimizer, str):
            optimizer = Optimizer(optimizer)
        if isinstance(lr_scheduler, str):
            lr_scheduler = LR_Scheduler(lr_scheduler)
        if isinstance(loss_fn, str):
            loss_fn = LossFn(loss_fn)

        start = time()

        self.dataloaders = DataLoaders(
            batch_size=batch_size,
            num_workers=8,
            augmentations=self.augmentations,
            transform=ResNet50.transform,
            dataset_class=self.dataset_class,
        )

        self.trainer = Trainer(
            model=ResNet50(self.dataset_class.num_classes),
            device=torch.device(device) if device else None,
            output_device=torch.device(output_device) if output_device else None,
            optimizer=optimizer,
            lr=lr,
            weight_decay=weight_decay,
            lr_scheduler=lr_scheduler,
            scheduler_step_size=scheduler_step_size,
            scheduler_gamma=scheduler_gamma,
            scheduler_step_every_epoch=schedular_step_every_epoch,
            loss_fn=loss_fn,
            results_file=results_file,
        )

        if previous_pipeline_directory and previous_pipeline_directory.exists():
            self.trainer.load(previous_pipeline_directory / "checkpoint.pth")

        start_epoch = self.trainer.epochs_already_trained

        train_losses, train_accuracies, val_losses, val_accuracies = self.trainer.train(
            self.dataloaders.train,
            self.dataloaders.val,
            epochs=epochs,
        )

        if pipeline_directory and pipeline_directory.exists():
            self.trainer.save(pipeline_directory / "checkpoint.pth")

        end = time()

        self._generate_plots(train_losses, train_accuracies, val_losses, val_accuracies)
        
        return {
            "loss": val_losses[-1],
            "cost": end - start,
            "info_dict": {
                "train_losses": train_losses,
                "train_accuracies": train_accuracies,
                "val_losses": val_losses,
                "val_accuracy": val_accuracies,
                "train_time": end - start,
                "cost": epochs - start_epoch,
            },
        }

    def fit(self, budget: int) -> None:
        neps.run(
            lambda pipeline_directory, previous_pipeline_directory, **kwargs: self.run_pipeline(
                pipeline_directory=pipeline_directory,
                previous_pipeline_directory=previous_pipeline_directory,
                **{
                    **kwargs,
                    "lr_scheduler": LR_Scheduler.step,
                    "scheduler_step_every_epoch": False,
                    "loss_fn": LossFn.cross_entropy,
                    "device": "cuda:0",
                    "output_device": "cuda:0",
                    "results_file": None,
                },
            ),
            root_directory="./results/" + self.dataset_class.__name__,
            pipeline_space="./pipeline_space.yaml",
            searcher="priorband_bo",
            max_cost_total=budget,
            post_run_summary=True,
            overwrite_working_directory=True,
        )

    def predict(self) -> Tuple[float, float, np.ndarray]:
        loss, accuracy, preds = self.trainer.eval(
            self.dataloaders.test,
            return_predictions=True,
        )
        predictions = preds.cpu().numpy()
        
        df = pd.DataFrame(predictions, columns=["Prediction"])
        df.to_csv(output_file, index=False)
        return loss, accuracy, predictions
