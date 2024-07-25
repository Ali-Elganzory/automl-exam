"""
AutoML class for vision classification tasks.
"""

from __future__ import annotations
from time import time
from pathlib import Path
from typing import Tuple, Type, Dict, Union

import neps
import torch
import random
import numpy as np
from torchvision.transforms.v2 import RandomChoice, AugMix, TrivialAugmentWide

from automl.model import Models
from automl.dataset import DataLoaders, BaseVisionDataset
from automl.trainer import Trainer, Optimizer, LR_Scheduler, LossFn


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

    def run_pipeline(
        self,
        epochs: int,
        batch_size: int,
        optimizer: str | Optimizer,
        learning_rate: float,
        weight_decay: float,
        lr_scheduler: str | LR_Scheduler,
        scheduler_step_size: int,
        scheduler_gamma: float,
        schedular_step_every_epoch: bool,
        loss_fn: str | LossFn,
        pipeline_directory: Path | None = None,
        previous_pipeline_directory: Path | None = None,
        results_file: str | None = None,
    ) -> Dict[str, any]:
        if isinstance(optimizer, str):
            optimizer = Optimizer(optimizer)
        if isinstance(lr_scheduler, str):
            lr_scheduler = LR_Scheduler(lr_scheduler)
        if isinstance(loss_fn, str):
            loss_fn = LossFn(loss_fn)

        # Default model
        model = Models.ResNet50.factory(self.dataset_class.num_classes)

        start = time()

        # Dataset
        self.dataloaders = DataLoaders(
            batch_size=batch_size,
            num_workers=16,
            augmentations=self.augmentations,
            transform=model.transform,
            dataset_class=self.dataset_class,
        )

        # Trainer
        self.trainer = Trainer(
            model=model,
            optimizer=optimizer,
            lr=learning_rate,
            weight_decay=weight_decay,
            lr_scheduler=lr_scheduler,
            scheduler_step_size=scheduler_step_size,
            scheduler_gamma=scheduler_gamma,
            scheduler_step_every_epoch=schedular_step_every_epoch,
            loss_fn=loss_fn,
            results_file=results_file,
        )

        # Resume training if previous pipeline exists
        if previous_pipeline_directory and previous_pipeline_directory.exists():
            self.trainer.load(previous_pipeline_directory / "checkpoint.pth")

        start_epoch = self.trainer.epochs_already_trained

        # Train
        train_losses, train_accuracies, val_losses, val_accuracies, _ = (
            self.trainer.train(
                self.dataloaders.train,
                self.dataloaders.val,
                epochs=epochs,
            )
        )

        # Save checkpoint
        if pipeline_directory and pipeline_directory.exists():
            self.trainer.save(pipeline_directory / "checkpoint.pth")

        end = time()

        return {
            "loss": val_losses[-1],
            "cost": end - start,
            "info_dict": {
                "train_losses": train_losses,
                "train_accuracies": train_accuracies,
                "val_losses": val_losses,
                "val_accuracies": val_accuracies,
                "train_time": end - start,
                "cost": epochs - start_epoch,
            },
        }

    def fit(self, budget: int) -> Dict[str, Union[float, str]]:
        """
        Run the AutoML pipeline, optimizing for the given budget.

        Args:
            budget: The maximum time to spend on the optimization.

        Returns:
            A dictionary containing the best configuration found by the optimizer.
        """
        root_directory = (
            "./results/"
            + f"benchmark={self.dataset_class.__name__.replace('Dataset', '')}/"
            + f"algorithm=PriorBand-BO/"
            + f"seed={self.seed}/"
        )

        # HPO
        print(f"Running AutoML pipeline on dataset {self.dataset_class.__name__}")
        start = time()
        neps.run(
            lambda pipeline_directory, previous_pipeline_directory, **kwargs: self.run_pipeline(
                pipeline_directory=pipeline_directory,
                previous_pipeline_directory=previous_pipeline_directory,
                **{
                    **kwargs,
                    "lr_scheduler": LR_Scheduler.step,
                    "loss_fn": LossFn.cross_entropy,
                    "schedular_step_every_epoch": False,
                    "results_file": None,
                },
            ),
            root_directory=root_directory,
            pipeline_space="./pipeline_space.yaml",
            searcher="priorband_bo",
            max_cost_total=budget,
            post_run_summary=True,
            overwrite_working_directory=True,
        )
        end = time()
        with open(f"{root_directory}/time.txt", "w") as f:
            f.write(str(end - start))

        # Load best configuration
        best_config = neps.get_summary_dict(root_directory)["best_config"]
        print("-" * 80)
        print(f"Best configuration: {best_config}")
        print("-" * 80)

        # Train with best configuration
        print(f"Training with best configuration (final model)")
        results = self.run_pipeline(
            pipeline_directory=None,
            previous_pipeline_directory=None,
            epochs=100,
            lr_scheduler=LR_Scheduler.step,
            schedular_step_every_epoch=False,
            loss_fn=LossFn.cross_entropy,
            results_file=f"{root_directory}best_config_results.csv",
            **(best_config.pop("epochs") and best_config),
        )
        print("-" * 80)
        print(f"Results: {results}")
        print("-" * 80)

        # Save the model
        self.trainer.save_model(f"{root_directory}best_config_model.pth")

    def evaluate(self) -> Tuple[float, float]:
        loss, accuracy, _ = self.trainer.eval(self.dataloaders.test)
        return loss, accuracy

    def predict(self) -> np.ndarray:
        return self.trainer.predict(self.dataloaders.test).cpu().numpy()
