"""
AutoML class for vision classification tasks.
"""

from __future__ import annotations
from typing import Tuple, Type

import torch
import random
import numpy as np
from torch import nn
from torchvision.transforms.v2 import Compose, RandomChoice, AugMix, TrivialAugmentWide

from automl.model import ResNet50
from automl.trainer import Trainer
from automl.utils import log as print
from automl.data import DataLoaders, BaseVisionDataset


class AutoML:
    augmentations = RandomChoice(
        [
            TrivialAugmentWide(),
            AugMix(),
            Compose(
                [
                    TrivialAugmentWide(),
                    AugMix(),
                ]
            ),
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

        self.dataloaders = DataLoaders(
            batch_size=64,
            num_workers=2,
            augmentations=self.augmentations,
            transform=ResNet50.transform,
            dataset_class=self.dataset_class,
        )

        self.trainer = Trainer(
            model=ResNet50(self.dataset_class.num_classes),
            device=torch.device("cuda:0"),
            results_file="logs/resnet50_results.csv",
        )

    @property
    def model(self) -> nn.Module:
        return self.trainer.model

    def fit(self) -> None:
        self.trainer.train(
            self.dataloaders.train,
            self.dataloaders.test,
            epochs=2,
        )

    def predict(self) -> Tuple[float, float, np.ndarray]:
        loss, accuracy, preds = self.trainer.eval(
            self.dataloaders.test,
            return_predictions=True,
        )
        return loss, accuracy, preds.cpu().numpy()
