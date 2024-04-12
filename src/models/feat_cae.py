import torch
import torch.nn as nn
from src.models.non_local import NLBlockND


class FeatCAE(nn.Module):
    """Autoencoder."""

    def __init__(
        self, width=64, height=64, in_channels=1000, latent_dim=50, is_bn=True
    ):
        super(FeatCAE, self).__init__()

        self.width = width
        self.height = height
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.is_bn = is_bn

        self.relu = nn.ReLU()

        # Encoder
        self.conv1 = nn.Conv2d(
            self.in_channels,
            (self.in_channels + 2 * self.latent_dim) // 2,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.bn_1 = nn.BatchNorm2d(
            num_features=(self.in_channels + 2 * self.latent_dim) // 2
        )
        self.non_local_block1 = NLBlockND(
            in_channels=(self.in_channels + 2 * self.latent_dim) // 2,
            mode="embedded",
            dimension=2,
            bn_layer=True,
        )

        self.conv2 = nn.Conv2d(
            (self.in_channels + 2 * self.latent_dim) // 2,
            2 * self.latent_dim,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.bn_2 = nn.BatchNorm2d(num_features=2 * self.latent_dim)
        self.non_local_block2 = NLBlockND(
            in_channels=2 * self.latent_dim, mode="embedded", dimension=2, bn_layer=True
        )

        # Embedding
        self.conv3 = nn.Conv2d(
            2 * self.latent_dim, self.latent_dim, kernel_size=1, stride=1, padding=0
        )

        # Decoder 1x1 conv to reconstruct the rgb values
        self.conv4 = nn.Conv2d(
            latent_dim, 2 * latent_dim, kernel_size=1, stride=1, padding=0
        )
        self.bn_4 = nn.BatchNorm2d(num_features=2 * latent_dim)
        self.non_local_block3 = NLBlockND(
            in_channels=2 * self.latent_dim, mode="embedded", dimension=2, bn_layer=True
        )

        self.conv5 = nn.Conv2d(
            2 * latent_dim,
            (in_channels + 2 * latent_dim) // 2,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.bn_5 = nn.BatchNorm2d(num_features=(in_channels + 2 * latent_dim) // 2)
        self.non_local_block4 = NLBlockND(
            in_channels=(in_channels + 2 * latent_dim) // 2,
            mode="embedded",
            dimension=2,
            bn_layer=True,
        )

        self.conv6 = nn.Conv2d(
            (in_channels + 2 * latent_dim) // 2,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, x):
        # encoder
        x = self.relu(self.bn_1(self.conv1(x)))
        x = self.non_local_block1(x)

        x = self.relu(self.bn_2(self.conv2(x)))
        x = self.non_local_block2(x)

        x = self.conv3(x)

        # decoder
        x = self.relu(self.bn_4(self.conv4(x)))
        x = self.non_local_block3(x)

        x = self.relu(self.bn_5(self.conv5(x)))
        x = self.non_local_block4(x)

        x = self.conv6(x)

        return x


if __name__ == "__main__":
    import numpy as np
    import time

    from torchsummary import summary

    device = "cuda" if torch.cuda.is_available() else "cpu"

    x = torch.Tensor(np.random.randn(1, 2048, 56, 56)).to(device)
    feat_ae = FeatCAE(width=56, height=56, in_channels=2048, latent_dim=148).to(device)

    summary(feat_ae, (2048, 56, 56), 1, device)

    time_s = time.time()
    for i in range(10):
        time_ss = time.time()
        out = feat_ae(x)
        print("Time cost:", (time.time() - time_ss), "s")

    print("Time cost:", (time.time() - time_s), "s")
    print("Feature (n_samples, n_features):", out.shape)
