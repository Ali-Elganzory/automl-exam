from torchvision.models import ResNet50_Weights
from torchvision.models.resnet import ResNet, Bottleneck


class ResNet50(ResNet):
    transform = ResNet50_Weights.IMAGENET1K_V2.transforms()

    def __init__(self, num_classes: int):
        super(ResNet50, self).__init__(
            Bottleneck, [3, 4, 6, 3], num_classes=num_classes
        )
