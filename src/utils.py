import cv2
import torch
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from sklearn.metrics import roc_auc_score, roc_curve
from torchmetrics import ConfusionMatrix, MetricCollection
from torchmetrics.classification import BinaryF1Score


def estimate_segmentation_threshold(data_loader, model, device, fpr=0.05):
    errors = []
    with torch.no_grad():
        model.eval()

        for x in data_loader:
            images, labels = x
            images = images.to(device)
            labels = labels.to(device)

            extracted_feat, reconstructed_feat = model(images)
            error = model.get_hotmap(extracted_feat, reconstructed_feat)
            errors.append(error.detach().cpu().numpy())

    errors = np.asarray(errors)
    T = np.percentile(errors, 100 - fpr)
    return T


def evaluate_classification(data_loader, model, DEVICE, T):
    metric_collection = MetricCollection(
        {
            "f1": BinaryF1Score(),
            "cm": ConfusionMatrix(task="binary"),
        }
    ).to(DEVICE)

    with torch.no_grad():
        model.eval()

        for x in data_loader:
            images, labels = x
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            extracted_feat, reconstructed_feat = model(images)
            hotmaps = model.get_hotmap(extracted_feat, reconstructed_feat)

            # predict anomalous images
            preds = torch.any(hotmaps > T, dim=(1, 2)).type(torch.int64)

            # from multiclass to binary problem
            truth_values = labels
            truth_values[truth_values > 0] = 1

            metric_collection.update(preds, truth_values)

        cl_metrics = metric_collection.compute()
        f1 = cl_metrics["f1"]
        cm = metric_collection["cm"]
        rec_error = hotmaps.mean() / (
            reconstructed_feat.shape[0] * reconstructed_feat.shape[1]
        )

        return f1, cm, rec_error


def evaluate_localization(X, y, model, original_size, transforms_test, device):
    preds = []
    labels = []
    for i in range(X.shape[0]):
        image = X[i]
        label = y[i]

        # preprocess label
        label = label.astype(np.uint8)
        label[label == 255] = 1

        # preprocess image
        image = transforms_test(image).unsqueeze(0).to(device)

        extracted_feat, reconstructed_feat = model(image)
        hotmap = model.get_hotmap(extracted_feat, reconstructed_feat)
        hotmap = hotmap.detach().cpu().numpy().squeeze(0)

        pred_mask = cv2.resize(hotmap, original_size)

        preds.append(pred_mask)
        labels.append(label)

    preds = np.asarray(preds)
    labels = np.asarray(labels)

    auc_score = roc_auc_score(labels.max(axis=(1, 2)), preds.max(axis=(1, 2)))
    fpr, tpr, _ = roc_curve(
        labels.max(axis=(1, 2)), preds.max(axis=(1, 2)), pos_label=1
    )

    pixel_auc_score = roc_auc_score(labels.ravel(), preds.ravel())
    pixel_fpr, pixel_tpr, _ = roc_curve(
        labels.ravel(), preds.ravel(), pos_label=1, drop_intermediate=True
    )

    return auc_score, fpr, tpr, pixel_auc_score, pixel_fpr, pixel_tpr


def display_anomalies(test_loader, model, classes, T, device, N=10):
    for i in range(N):
        x_batch, y_batch = next(iter(test_loader))

        extracted_feat, reconstructed_feat = model(x_batch.to(device))
        hotmaps = model.get_hotmap(extracted_feat, reconstructed_feat)

        hotmaps = hotmaps.detach().cpu().numpy()
        x_batch = x_batch.detach().cpu().numpy()

        for x, y, hotmap in zip(x_batch, y_batch, hotmaps):
            prediction = np.any(hotmap > T)
            mask = get_segmentation_mask(hotmap, (x.shape[1], x.shape[2]), T)
            x = normalize_image(np.transpose(x, axes=(1, 2, 0)))  # for visualization

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
                "output image: " + f"Threshold: {T} MaxValue: {hotmap.max():.6}"
            )

            plt.show()


def get_segmentation_mask(hotmap, original_size, T):
    hotmap = cv2.resize(hotmap, original_size)
    hotmap = np.expand_dims(hotmap, axis=0)
    mask = np.where(hotmap > T, 1, 0)
    mask = np.transpose(mask, axes=(1, 2, 0))
    return mask


def normalize_image(image):
    image = (image - np.min(image)) / np.ptp(image)
    image = (image * 255).astype(np.uint8)
    return image


def anomaly_cmap():
    colors = [
        (0.5, 0.5, 0.5, 1.0),
        "yellow",
    ]  # Gray for non-anomalies, Yellow for anomalies
    cmap_name = "anomaly_cmap"
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=2)

    return cm
