from abc import ABC
from enum import Enum
from collections import OrderedDict
from typing import Union, Optional, Callable

import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.models import ResNet50_Weights, resnet50


class Models(Enum):
    ConvNet = "convnet"
    ResNet50 = "resnet50"

    @property
    def factory(self):
        return {
            Models.ConvNet: ConvNet,
            Models.ResNet50: ResNet50,
        }[self]


class Model(ABC, nn.Module):
    transform: Optional[Callable[..., torch.Tensor]] = None

    def __init__(self):
        super(Model, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @torch.inference_mode()
    def predict(
        self,
        x: Union[torch.Tensor, DataLoader],
        module: Optional[nn.Module] = None,
        return_logits: bool = False,
    ) -> torch.Tensor:
        model = module or self

        preds = []
        if isinstance(x, DataLoader):
            for data, _ in tqdm(x, desc="Predicting"):
                preds.append(model(data.to(self.device)))
        else:
            preds.append(model(x.to(self.device)))

        if return_logits:
            return torch.cat(preds)

        return torch.cat(preds).argmax(dim=1).type(torch.int32)

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> "Model":
        self.load_state_dict(torch.load(path))
        return self


class ConvNet(Model):
    transform = ResNet50_Weights.IMAGENET1K_V2.transforms()

    def __init__(self, num_classes: int):
        super(ConvNet, self).__init__()

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.bn2 = nn.BatchNorm2d(64)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 54 * 54, 128)
        self.fc2 = nn.Linear(128, num_classes)

        self.groups = OrderedDict(
            {
                "backbone": nn.Sequential(
                    self.conv1,
                    self.bn1,
                    self.relu,
                    self.pool,
                    self.conv2,
                    self.bn2,
                    self.relu,
                    self.pool,
                ),
                "head": nn.Sequential(
                    self.flatten,
                    self.fc1,
                    self.relu,
                    self.fc2,
                ),
            }
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.groups["backbone"](x)
        x = self.groups["head"](x)
        return x


class ResNet50(Model):
    transform = ResNet50_Weights.IMAGENET1K_V2.transforms()

    def __init__(self, num_classes: int):
        super(ResNet50, self).__init__()

        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, num_classes),
        )

        # Freeze the backbone
        for param in self.resnet.parameters():
            param.requires_grad = False

        # Replace the head
        self.resnet.fc = self.head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resnet(x)
