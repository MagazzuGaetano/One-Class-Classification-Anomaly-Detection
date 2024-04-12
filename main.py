from src.models.dfr import DFR
from opencv_transforms import transforms as v2

import torch
import numpy as np

from src.config import DATA_FOLDER
from src.datasets.anomaly_dataset import AnomalyDataset
from src.train import train_loop

import matplotlib.pyplot as plt


def plot_loss(train_metric, test_metric):
    # plt.plot(test_metric, label="test")
    plt.plot(train_metric, label="train")
    plt.title("Loss per epoch")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend()
    plt.show()


# configs
lr = 1e-4
wd = 5e-4
max_epochs = 25
channels = 3
image_size = 224
train_print_freq = 5
val_print_freq = 1
num_workers = 0
batch_size = 2
seed = 42

backbone = "vgg19"
cnn_layers = ("relu4_1", "relu4_2", "relu4_3", "relu4_4")
upsample = "bilinear"
is_agg = True
kernel_size = (4, 4)
stride = (4, 4)
dilation = 1
featmap_size = (224, 224)

in_channels = 2048
latent_dim = 148
is_bn = True

# set reproducibility
random_seed = seed
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

# set device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# load data files
folder = DATA_FOLDER / "MVTecAD" / "wood"
train_npy = np.load(folder / "train.npz")
X_train, y_train = train_npy["x"], train_npy["y"]


# data augmentation
transforms = v2.Compose(
    [
        v2.RandomCrop(224),
        v2.RandomVerticalFlip(p=0.5),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomGrayscale(p=0.2),
        v2.ToTensor(),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

transforms_test = v2.Compose(
    [
        v2.Resize(224),
        v2.ToTensor(),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

train_data = AnomalyDataset(y_train, X_train, transforms)
train_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=True,
)

# load model
model = DFR(
    backbone=backbone,
    cnn_layers=cnn_layers,
    upsample=upsample,
    is_agg=is_agg,
    kernel_size=kernel_size,
    stride=stride,
    dilation=dilation,
    featmap_size=featmap_size,
    device=DEVICE,
    in_channels=in_channels,
    latent_dim=latent_dim,
    is_bn=is_bn,
    data_loader=train_loader,
).to(DEVICE)


# training
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

train_results = {}
train_results["train_loss"] = []

best_epoch_idx = 0
best_epoch_val = 10000
best_model = None

for epoch in range(max_epochs):
    print("Starting Epoch: {}".format(epoch + 1))

    train_loss = train_loop(
        train_loader,
        model,
        optimizer,
        DEVICE,
        train_print_freq,
    )
    train_results["train_loss"].append(train_loss)

    if train_loss < best_epoch_val:
        best_epoch_idx = epoch
        best_epoch_val = train_loss
        best_model = model.state_dict()
        torch.save({"model": best_model}, "model.pth")


torch.save({"model": best_model}, "model.pth")
#######################################################################################################

plot_loss(train_results["train_loss"], None)
