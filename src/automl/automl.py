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
        all_run_performances: list = None,
    ) -> None:
        if all_run_performances is None:
            all_run_performances = []
            
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

        val_loss, val_accuracy, _ = self.trainer.eval(self.dataloaders.val)
        all_run_performances.append(val_accuracy)
        
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
            "all_run_performances": all_run_performances
        }

    def fit(self, budget: int, num_runs: int = 5) -> None:
        all_run_performances = []
        for _ in range(num_runs):
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
            
            self._generate_plots(all_run_performances)

    def predict(self) -> Tuple[float, float, np.ndarray]:
        loss, accuracy, preds = self.trainer.eval(
            self.dataloaders.test,
            return_predictions=True,
        )
        predictions = preds.cpu().numpy()
        
        df = pd.DataFrame(predictions, columns=["Prediction"])
        df.to_csv(output_file, index=False)
        return loss, accuracy, predictions
        
def _generate_plots(self, all_run_performances):
        num_iterations = len(all_run_performances[0])
        iterations = range(1, num_iterations + 1)

        all_run_performances = np.array(all_run_performances)

        mean_performance = np.mean(all_run_performances, axis=0)
        std_performance = np.std(all_run_performances, axis=0)

        plt.figure(figsize=(10, 6))
        for run in all_run_performances:
            plt.plot(iterations, run, alpha=0.3)
        plt.plot(iterations, mean_performance, label='Mean Incumbent Performance')
        plt.fill_between(iterations, mean_performance - std_performance, mean_performance + std_performance, alpha=0.2)
        plt.title('Incumbent Performance Over Time')
        plt.xlabel('Iterations')
        plt.ylabel('Performance')
        plt.legend()
        plt.tight_layout()
        plt.savefig('incumbent_performance_over_time.png')
        plt.show()
