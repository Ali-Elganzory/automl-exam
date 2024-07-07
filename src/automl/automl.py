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
from automl.trainer import Trainer

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

    def run_pipeline(
        self,
        pipeline_directory: Path,
        previous_pipeline_directory: Path,
        epochs: int,
        batch_size: int,
        optimizer: str,
        learning_rate: float,
        weight_decay: float,
        scheduler_step_size: int,
        scheduler_gamma: float,
    ) -> None:
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
            device=torch.device("cuda:0"),
            optimizer=optimizer,
            lr=learning_rate,
            weight_decay=weight_decay,
            lr_scheduler="step",
            scheduler_step_size=scheduler_step_size,
            scheduler_gamma=scheduler_gamma,
            loss_fn=nn.CrossEntropyLoss(),
            scheduler_step_every_epoch=True,
        )

        if (
            previous_pipeline_directory
            and previous_pipeline_directory / "checkpoint.pth"
        ):
            self.trainer.load(previous_pipeline_directory / "checkpoint.pth")

        start_epoch = self.trainer.epochs_already_trained

        train_losses, train_accuracies, val_losses, val_accuracies = self.trainer.train(
            self.dataloaders.train,
            self.dataloaders.test,
            epochs=epochs,
        )

        self.trainer.save(pipeline_directory / "checkpoint.pth")
        end = time()

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

    def fit(self) -> None:
        neps.run(
            lambda pipeline_directory, previous_pipeline_directory, **kwargs: self.run_pipeline(
                pipeline_directory=pipeline_directory,
                previous_pipeline_directory=previous_pipeline_directory,
                **kwargs,
            ),
            root_directory="./results/" + self.dataset_class.__name__,
            pipeline_space="./pipeline_space.yaml",
            searcher="priorband_bo",
            max_cost_total=60 * 60 * 12,
            post_run_summary=True,
            overwrite_working_directory=True,
        )

    def predict(self) -> Tuple[float, float, np.ndarray]:
        loss, accuracy, preds = self.trainer.eval(
            self.dataloaders.test,
            return_predictions=True,
        )
        return loss, accuracy, preds.cpu().numpy()
