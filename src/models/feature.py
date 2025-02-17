import torch
import torch.nn as nn
import numpy as np

from src.models.vgg19 import VGG19

# backbone nets
backbone_nets = {"vgg19": VGG19}


# aggregation
class AvgFeatAGG2d(nn.Module):
    """
    Aggregating features on feat maps: avg
    """

    def __init__(
        self,
        kernel_size,
        output_size=None,
        dilation=1,
        stride=1,
        device=torch.device("cpu"),
    ):
        super(AvgFeatAGG2d, self).__init__()
        self.device = device
        self.kernel_size = kernel_size
        self.unfold = nn.Unfold(
            kernel_size=kernel_size, dilation=dilation, stride=stride
        )
        self.fold = nn.Fold(
            output_size=output_size, kernel_size=1, dilation=1, stride=1
        )
        self.output_size = output_size

    def forward(self, input):
        N, C, H, W = input.shape
        output = self.unfold(input)
        output = torch.reshape(
            output,
            (
                N,
                C,
                int(self.kernel_size[0] * self.kernel_size[1]),
                int(self.output_size[0] * self.output_size[1]),
            ),
        )
        output = torch.mean(output, dim=2)
        return output


class Extractor(nn.Module):
    r"""
    Build muti-scale regional feature based on VGG-feature maps.
    """

    def __init__(
        self,
        backbone="vgg19",
        cnn_layers=("relu1_1",),
        upsample="nearest",
        is_agg=True,
        kernel_size=(4, 4),
        stride=(4, 4),
        dilation=1,
        featmap_size=(256, 256),
        device="cpu",
    ):
        super(Extractor, self).__init__()
        self.device = torch.device(device)
        self.feature = backbone_nets[backbone]()  # build backbone net
        self.feat_layers = cnn_layers
        self.is_agg = is_agg
        self.map_size = featmap_size
        self.upsample = upsample
        self.patch_size = kernel_size
        self.stride = stride
        self.dilation = dilation

        # feature processing
        padding_h = (self.patch_size[0] - self.stride[0]) // 2
        padding_w = (self.patch_size[1] - self.stride[1]) // 2
        self.padding = (padding_h, padding_w)
        self.replicationpad = nn.ReplicationPad2d(
            (padding_w, padding_w, padding_h, padding_h)
        )

        self.out_h = int(
            (
                self.map_size[0]
                + 2 * self.padding[0]
                - (self.dilation * (self.patch_size[0] - 1) + 1)
            )
            / self.stride[0]
            + 1
        )
        self.out_w = int(
            (
                self.map_size[1]
                + 2 * self.padding[1]
                - (self.dilation * (self.patch_size[1] - 1) + 1)
            )
            / self.stride[1]
            + 1
        )
        self.out_size = (self.out_h, self.out_w)

        self.feat_agg = AvgFeatAGG2d(
            kernel_size=self.patch_size,
            output_size=self.out_size,
            dilation=self.dilation,
            stride=self.stride,
            device=self.device,
        )

    def forward(self, input):
        feat_maps = self.feature(input, feature_layers=self.feat_layers)

        features = torch.Tensor().to(self.device)
        # extracting features
        for _, feat_map in feat_maps.items():
            if self.is_agg:
                # allignment by resizing all feature maps to original input resolution
                feat_map = nn.functional.interpolate(
                    feat_map,
                    size=self.map_size,
                    mode=self.upsample,
                    align_corners=True if self.upsample == "bilinear" else None,
                )
                feat_map = self.replicationpad(feat_map)

                # aggregating features for every pixel
                feat_map = self.feat_agg(feat_map)

                # concatenating
                features = torch.cat([features, feat_map], dim=1)

            else:
                # allignment by resizing all feature maps to original input resolution
                feat_map = nn.functional.interpolate(
                    feat_map, size=self.out_size, mode=self.upsample
                )

                # concatenating
                features = torch.cat([features, feat_map], dim=1)

        b, c, _ = features.shape
        features = torch.reshape(features, (b, c, self.out_size[0], self.out_size[1]))

        return features

    def feat_vec(self, input):
        feat_maps = self.feature(input, feature_layers=self.feat_layers)
        features = torch.Tensor().to(self.device)
        # extracting features
        for name, feat_map in feat_maps.items():
            # resizing
            feat_map = nn.functional.interpolate(
                feat_map,
                size=self.map_size,
                mode=self.upsample,
                align_corners=True if self.upsample == "bilinear" else None,
            )
            feat_map = self.replicationpad(feat_map)

            # aggregating features for every pixel
            feat_map = self.feat_agg(feat_map)

            # concatenating
            features = torch.cat([features, feat_map], dim=1)

        # reshaping features
        features = features.permute(0, 2, 1)
        features = torch.unbind(features, dim=0)
        features = torch.cat(features, dim=0)
        return features


if __name__ == "__main__":
    import time

    vgg19_layers = (
        "relu1_1",
        "relu1_2",
        "relu2_1",
        "relu2_2",
        "relu3_1",
        "relu3_2",
        "relu3_3",
        "relu3_4",
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = Extractor(
        backbone="vgg19",
        cnn_layers=vgg19_layers,
        featmap_size=(256, 256),
        device=device,
    )

    time_s = time.time()
    extractor.to(device)
    batch_size = 1
    input = torch.Tensor(np.random.randn(batch_size, 3, 256, 256)).to(device)
    feats = extractor(input)

    print("Feature (n_samples, n_features):", feats.shape)
    print("Time cost:", (time.time() - time_s) / batch_size, "s")
