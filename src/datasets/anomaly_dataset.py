from torch.utils.data import Dataset

class AnomalyDataset(Dataset):
    def __init__(self, labels, imgs, transform=None):
        self.imgs = imgs
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        x = self.imgs[index]
        y = self.labels[index]

        if self.transform:
            x = self.transform(x)

        x = x.float()
        return x, y
