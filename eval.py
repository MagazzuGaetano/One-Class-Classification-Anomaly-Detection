import torch
import numpy as np
import matplotlib.pyplot as plt

from src.config import DATA_FOLDER
from src.datasets.anomaly_dataset import AnomalyDataset
from src.utils import (
    display_anomalies,
    estimate_segmentation_threshold,
    evaluate_classification,
    evaluate_localization,
)


from opencv_transforms import transforms as v2
from src.models.dfr import DFR


# configs
lr = 1e-4
wd = 5e-4
max_epochs = 250
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

classes = ["good", "crack", "cut", "hole", "print"]

# set reproducibility
random_seed = seed
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

# set device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# load train data files
folder = DATA_FOLDER / "MVTecAD" / "hazelnut"
train_npy = np.load(folder / "train.npz")
X_train, y_train = train_npy["x"], train_npy["y"]

# load test data files
folder = DATA_FOLDER / "MVTecAD" / "hazelnut"
test_npy = np.load(folder / "test.npz")
X_test, y_test, y_mask_test = test_npy["x"], test_npy["y"], test_npy["z"]

# data augmentation
transforms = v2.Compose(
    [
        v2.Resize(224),
        v2.RandomVerticalFlip(p=0.5),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomGrayscale(p=0.5),
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
checkpoint = torch.load("non_local_dfr_model.pth")
model.load_state_dict(checkpoint["model"])


# Evaluate Anomaly Localization on Test Set
original_size = (1024, 1024)
auc_score, fpr, tpr, pixel_auc_score, pixel_fpr, pixel_tpr = evaluate_localization(
    X_test, y_mask_test, model, original_size, transforms_test, DEVICE
)

print("\nTest Anomaly Classification & Localizzation:")
print(f"AUC_ROC: {auc_score}, Pixel AUC_ROC: {pixel_auc_score}")

plt.plot(fpr, tpr, marker=".")
plt.title(f"AUC_ROC: {auc_score}")
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
plt.show()

plt.plot(pixel_fpr, pixel_tpr, marker=".")
plt.title(f"Pixel AUC_ROC: {pixel_auc_score}")
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
plt.show()

# Estimate Threshold on Train Set
T = estimate_segmentation_threshold(train_loader, model, DEVICE, fpr=0.05)
print("class: {:<19} \t threshold: {}".format("good", round(T, 6)))

# Evaluate Anomaly Classification on Test Set with Threshold
f1, cm, rec_error = evaluate_classification(test_loader, model, DEVICE, T)
print("\nTest Anomaly Classification:")
print(f"AVG F1: {f1}, AVG Reconstraction Error: {rec_error:.6f}")
fig_, ax_ = cm.plot()
plt.show()

# Anomaly Localization Examples on Test Set
display_anomalies(test_loader, model, classes, T, DEVICE)
