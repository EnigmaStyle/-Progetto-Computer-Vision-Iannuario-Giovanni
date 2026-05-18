import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset, random_split
import torch
from PIL import Image as PILImage
from src.utils import DATA_DIR, IMG_SIZE, BATCH_SIZE, NUM_WORKERS, set_seed

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_transforms(split: str) -> T.Compose:
    if split == "train":
        return T.Compose([
            T.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
            T.RandomCrop(IMG_SIZE),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(30),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class AugDataset(Dataset):
    """Wraps a Subset and applies augmentation transforms, picklable for multiprocessing."""
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        path = self.subset.dataset.samples[self.subset.indices[idx]][0]
        label = self.subset.dataset.targets[self.subset.indices[idx]]
        img = PILImage.open(path).convert("RGB")
        return self.transform(img), label


def get_dataloaders(data_dir: str = DATA_DIR, seed: int = 42):
    """Return (train_loader, val_loader, test_loader, class_names)."""
    set_seed(seed)
    full_dataset = ImageFolder(data_dir, transform=get_transforms("val"))
    class_names = full_dataset.classes

    n = len(full_dataset)
    n_train = int(0.70 * n)
    n_val   = int(0.15 * n)
    n_test  = n - n_train - n_val

    train_ds, val_ds, test_ds = random_split(
        full_dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(seed)
    )

    aug_train = AugDataset(train_ds, get_transforms("train"))

    def _loader(ds, shuffle):
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle,
                          num_workers=NUM_WORKERS, pin_memory=True)

    return (
        _loader(aug_train, True),
        _loader(val_ds, False),
        _loader(test_ds, False),
        class_names,
    )
