import torch
import torch.nn as nn
from typing import Optional


class MRIClassifierCNN(nn.Module):

    def __init__(self, num_classes: int = 2, dropout_rate: float = 0.5):
        super(MRIClassifierCNN, self).__init__()

        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        self.conv_output = None
        self.gradients = None

        self.conv1 = self._make_conv_block(3, 32)
        self.conv2 = self._make_conv_block(32, 64)
        self.conv3 = self._make_conv_block(64, 128)
        self.conv4 = self._make_conv_block(128, 256)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes)
        )

        self._initialize_weights()

    def _make_conv_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def _save_gradient(self, grad):
        self.gradients = grad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        if x.requires_grad:
            x.register_hook(self._save_gradient)

        self.conv_output = x

        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x

    def get_conv_features(self) -> Optional[torch.Tensor]:
        return self.conv_output

    def get_gradients(self) -> Optional[torch.Tensor]:
        return self.gradients


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = MRIClassifierCNN(num_classes=2)

    dummy_input = torch.randn(4, 3, 224, 224)

    output = model(dummy_input)

    print("Model Architecture:")
    print("=" * 70)
    print(model)
    print("=" * 70)
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Total trainable parameters: {count_parameters(model):,}")
    print(f"Convolutional feature map shape: {model.get_conv_features().shape}")