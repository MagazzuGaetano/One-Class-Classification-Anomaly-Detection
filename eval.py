import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from src.config import DATA_FOLDER
from src.datasets.anomaly_dataset import AnomalyDataset

from torchmetrics import ConfusionMatrix, MetricCollection
from torchmetrics.classification import BinaryF1Score
from opencv_transforms import transforms as v2
from src.models.dfr import DFR


def anomaly_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    colors = [
        (0.5, 0.5, 0.5, 1.0),
        "yellow",
    ]  # Gray for non-anomalies, Yellow for anomalies
    cmap_name = "anomaly_cmap"
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=2)

    return cm


def estimate_segmentation_threshold(data_loader, model, fpr=0.05):
    errors = []
    with torch.no_grad():
        model.eval()

        for x in data_loader:
            images, labels = x
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            extracted_feat, reconstructed_feat = model(images)
            error = model.get_hotmap(extracted_feat, reconstructed_feat)
            errors.append(error.detach().cpu().numpy())

    print("\nTheshold Estimation:")

    errors = np.asarray(errors)
    T = np.percentile(errors, 100 - fpr)

    print(
        "class: {:<19} \t classification reconstraction error: {} \t classification threshold: {}".format(
            "good", round(errors.mean(), 6), round(T, 6)
        )
    )

    return T


def display_anomalies(test_loader, model, classes, normal_seg_t, device):
    for x_batch, y_batch in test_loader:
        if y_batch[0].item() == 1:
            extracted_feat, reconstructed_feat = model(x_batch.to(device))

            hotmaps = model.get_hotmap(extracted_feat, reconstructed_feat)
            hotmaps = hotmaps.detach().cpu().numpy()

            for x, y, hotmap in zip(x_batch, y_batch, hotmaps):
                prediction = np.any(hotmap > normal_seg_t)

                hotmap = cv2.resize(hotmap, (x.shape[1], x.shape[2]))
                hotmap = np.expand_dims(hotmap, axis=0)
                mask = np.where(hotmap > normal_seg_t, 1, 0)
                mask = np.transpose(mask, axes=(1, 2, 0))

                x = np.transpose(x, axes=(1, 2, 0)).detach().cpu().numpy()
                x = (x - np.min(x)) / np.ptp(x)
                x = (x * 255).astype(np.uint8)

                f, axarr = plt.subplots(1, 2, figsize=(12, 4))
                axarr[0].imshow(x)
                axarr[0].set_title(
                    "original image: "
                    + f"GT: {classes[int(y)]}, anomaly detected: {prediction}"
                )

                axarr[1].imshow(mask)
                # axarr[1].imshow(np.transpose(hotmap, axes=(1, 2, 0)))
                axarr[1].imshow(x, alpha=0.75)
                axarr[1].set_title(
                    "output image: "
                    + f"Threshold: {normal_seg_t} MaxValue: {hotmap.max():.6}"
                )

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


# load train data files
folder = DATA_FOLDER / "MVTecAD" / "wood"
train_npy = np.load(folder / "train.npz")
X_train, y_train = train_npy["x"], train_npy["y"]

# load test data files
folder = DATA_FOLDER / "MVTecAD" / "wood"
test_npy = np.load(folder / "test.npz")
X_test, y_test = test_npy["x"], test_npy["y"]

# data augmentation
transforms = v2.Compose(
    [
        v2.Resize(512),
        v2.RandomCrop(224),
        v2.RandomVerticalFlip(p=0.5),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomGrayscale(p=0.2),
        # v2.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.1),
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

train_data = AnomalyDataset(labels=y_train, imgs=X_train, transform=transforms)
train_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=True,
)

test_data = AnomalyDataset(labels=y_test, imgs=X_test, transform=transforms_test)
test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=batch_size, shuffle=True, num_workers=num_workers
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
    data_loader=None,
).to(DEVICE)

# load the saved state dictionaries for the encoder and decoder
checkpoint = torch.load("model.pth")
model.load_state_dict(checkpoint["model"])

metric_collection = MetricCollection(
    {
        "f1": BinaryF1Score(),
        "cm": ConfusionMatrix(task="binary"),
    }
).to(DEVICE)

classes = ["good", "color", "hole", "liquid", "scratch", "combined"]
normal_seg_t = estimate_segmentation_threshold(train_loader, model, fpr=0.05)

with torch.no_grad():
    model.eval()

    for x in test_loader:
        images, labels = x
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        extracted_feat, reconstructed_feat = model(images)
        hotmaps = model.get_hotmap(extracted_feat, reconstructed_feat)

        # predict anomalous images
        preds = torch.any(hotmaps > normal_seg_t, dim=(1, 2)).type(torch.int64)

        # from multiclass to binary problem
        truth_values = labels
        truth_values[truth_values > 0] = 1

        metric_collection.update(preds, truth_values)

    cl_metrics = metric_collection.compute()

    print("\nTest Evaluation:")
    print(f"AVG F1: {cl_metrics['f1']}, AVG Reconstraction Error: {hotmaps.mean():.6f}")
    fig_, ax_ = metric_collection["cm"].plot()
    plt.show()

    metric_collection.reset()


display_anomalies(test_loader, model, classes, normal_seg_t, DEVICE)
