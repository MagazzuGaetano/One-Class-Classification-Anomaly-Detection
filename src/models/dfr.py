import torch
import torch.nn as nn
from src.models.feature import Extractor
from src.models.feat_cae import FeatCAE
from sklearn.decomposition import PCA


class DFR(nn.Module):
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
        in_channels=1000,
        latent_dim=50,
        is_bn=True,
        data_loader=None,
    ):
        super(DFR, self).__init__()

        self.backbone = backbone
        self.cnn_layers = cnn_layers
        self.upsample = upsample
        self.is_agg = is_agg
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.featmap_size = featmap_size
        self.device = device

        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.is_bn = is_bn
        self.data_loader = data_loader

        # feature extractor
        self.extractor = Extractor(
            backbone=self.backbone,
            cnn_layers=self.cnn_layers,
            upsample=self.upsample,
            is_agg=self.is_agg,
            kernel_size=self.kernel_size,
            stride=self.stride,
            dilation=self.dilation,
            featmap_size=self.featmap_size,
            device=self.device,
        ).to(self.device)

        # Freeze Extractor layers
        for param in self.extractor.feature.parameters():
            param.requires_grad = False

        # estimate autoencoder parameters
        if self.in_channels is None or self.latent_dim is None:
            if self.data_loader is None:
                raise Exception(
                    "if in_channels or latent_dim are None data_loader must be defined!!!"
                )
            self.estimate_parameters()

        # autoencoder
        self.autoencoder = FeatCAE(
            width=featmap_size[0] // 4,
            height=featmap_size[0] // 4,
            in_channels=self.in_channels,
            latent_dim=self.latent_dim,
            is_bn=self.is_bn,
        ).to(self.device)

    def forward(self, x):
        extracted_feat = self.extractor(x)
        reconstructed_feat = self.autoencoder(extracted_feat)
        return extracted_feat, reconstructed_feat

    def get_hotmap(self, x, x_hat):
        # compute heatmap B x (W // 4) x (H // 4)
        # loss = torch.sum((x - x_hat) ** 2, dim=1)

        loss = torch.mean((x - x_hat) ** 2, dim=1)
        return loss

    def loss_fn(self, x, x_hat, dim=None):
        # B x 2048 x 56 x 56
        # loss = self.get_hotmap(x, x_hat)
        # B x 56 x 56
        # loss = loss.mean(axis=(1, 2)) / (x.shape[0] * self.in_channels)
        # B x 1

        loss = torch.mean((x - x_hat) ** 2, dim=dim)
        return loss

    def estimate_parameters(self):
        print("Estimating one class classifier AE parameter...")

        feats = torch.Tensor()
        for batch_x, _ in self.data_loader:
            normal_img = batch_x[0].unsqueeze(0)  # first image of the batch
            normal_img = normal_img.to(self.device)
            feat = self.extractor.feat_vec(normal_img)
            feats = torch.cat([feats, feat.cpu()], dim=0)
        feats = feats.detach().numpy()

        # estimate parameters for ae
        pca = PCA(n_components=0.8)  # 0.9 here try 0.8
        pca.fit(feats)
        n_dim, in_feat = pca.components_.shape
        print("AE Parameter (in_feat, n_dim): ({}, {})".format(in_feat, n_dim))

        # set ae parameters
        self.in_channels = in_feat
        self.latent_dim = n_dim
