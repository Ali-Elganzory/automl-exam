"""
AutoML class for vision classification tasks.
"""

from __future__ import annotations
from typing import Tuple, Type

import torch
import random
import numpy as np
from torch import nn
from torch.utils.data import DataLoader

from automl.model import ResNet50
from automl.trainer import Trainer
from automl.data import DataLoaders, BaseVisionDataset


class AutoML:

    def __init__(
        self,
        seed: int,
    ) -> None:
        self.seed = seed
        self.trainer = None

    @property
    def model(self) -> nn.Module:
        return self._trainer.model

    def fit(
        self,
        dataset_class: Type[BaseVisionDataset],
    ) -> AutoML:
        # Ensure deterministic behavior for reproducibility
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.dataloaders = DataLoaders(
            batch_size=64,
            num_workers=2,
            transform=ResNet50.transform,
            dataset_class=dataset_class,
        )

        self.trainer = Trainer(
            model=ResNet50(dataset_class.num_classes),
            device=torch.device("cuda:0"),
            results_file="logs/resnet50_results.csv",
        )

        self.trainer.train(
            self.dataloaders.train,
            self.dataloaders.test,
            epochs=2,
        )

        return self

    def predict(self) -> Tuple[float, float, np.ndarray]:
        loss, accuracy, preds = self.trainer.eval(
            self.dataloaders.test,
            return_predictions=True,
        )
        return loss, accuracy, preds.cpu().numpy()
