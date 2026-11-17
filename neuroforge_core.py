import hashlib
import copy
import csv
import json
import os
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import audio_pipeline

# 1. Device Configuration (Apple Silicon MPS or standard CPU/CUDA)
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"NEUROFORGE Engine initialized. Using device: {device}")

SUPPORTED_ACTIVATIONS = {"ReLU", "GELU", "SiLU"}
SUPPORTED_KERNELS = {3, 5}
SUPPORTED_IMAGE_EXTENSIONS = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def set_random_seed(seed):
    """Make candidate initialization and dataset shuffling repeatable."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_config(config):
    """Validate an architecture config before it reaches PyTorch."""
    if not isinstance(config, dict):
        raise ValueError("Architecture config must be a dictionary")

    num_layers = config.get("num_layers")
    filters = config.get("filters")
    kernel_sizes = config.get("kernel_sizes")
    if not isinstance(num_layers, int) or not 1 <= num_layers <= 8:
        raise ValueError("num_layers must be an integer between 1 and 8")
    if not isinstance(filters, list) or len(filters) != num_layers:
        raise ValueError("filters must contain one entry per layer")
    if not isinstance(kernel_sizes, list) or len(kernel_sizes) != num_layers:
        raise ValueError("kernel_sizes must contain one entry per layer")
    if any(not isinstance(value, int) or value <= 0 for value in filters):
        raise ValueError("filters must contain positive integers")
    if any(value not in SUPPORTED_KERNELS for value in kernel_sizes):
        raise ValueError("kernel_sizes must contain only 3 or 5")
    if config.get("activation", "ReLU") not in SUPPORTED_ACTIVATIONS:
        raise ValueError("activation must be ReLU, GELU, or SiLU")

    dropout = config.get("dropout", 0)
    if not isinstance(dropout, (int, float)) or not 0 <= dropout <= 0.5:
        raise ValueError("dropout must be between 0 and 0.5")
    if config.get("block_type", "plain") not in SUPPORTED_BLOCK_TYPES:
        raise ValueError("block_type must be plain, residual, or depthwise")
    if config.get("input_size", DEFAULT_INPUT_SIZE) not in SUPPORTED_INPUT_SIZES:
        raise ValueError("input_size must be 64, 96, or 128")
    return config


def config_hash(config):
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


SPLIT_SCHEMA_VERSION = 2
VALID_FRACTION = 0.15
TEST_FRACTION = 0.15


def get_split_indices(dataset_size, seed=42, labels=None, valid_fraction=VALID_FRACTION, test_fraction=TEST_FRACTION):
    """Create one deterministic train/valid/test split.

    Validation is used for every development decision (candidate ranking,
    checkpoint choice, early stopping). Test is touched exactly once, after
    the model is locked. Classes with fewer than three samples stay in train
    because they cannot be represented in all three partitions.
    """
    if dataset_size < 3:
        raise ValueError("Dataset must contain at least three samples")
    generator = torch.Generator().manual_seed(seed)

    def carve(indices):
        count = len(indices)
        if count < 3:
            return list(indices), [], []
        test_count = max(1, round(count * test_fraction))
        valid_count = max(1, round(count * valid_fraction))
        return (
            indices[test_count + valid_count:],
            indices[test_count:test_count + valid_count],
            indices[:test_count],
        )

    train_indices, valid_indices, test_indices = [], [], []
    if labels is not None:
        if len(labels) != dataset_size:
            raise ValueError("labels must have one entry per dataset sample")
        grouped_indices = {}
        for index, label in enumerate(labels):
            grouped_indices.setdefault(label, []).append(index)
        for label in sorted(grouped_indices):
            class_indices = grouped_indices[label]
            order = torch.randperm(len(class_indices), generator=generator).tolist()
            train, valid, test = carve([class_indices[index] for index in order])
            train_indices.extend(train)
            valid_indices.extend(valid)
            test_indices.extend(test)
    else:
        train_indices, valid_indices, test_indices = carve(torch.randperm(dataset_size, generator=generator).tolist())

    return {
        "schema": SPLIT_SCHEMA_VERSION,
        "strategy": "stratified_imagefolder" if labels is not None else "random_imagefolder",
        "train": sorted(train_indices),
        "valid": sorted(valid_indices),
        "test": sorted(test_indices),
        "seed": seed,
        "stratified": labels is not None,
    }


def split_is_current(split_indices):
    """Older runs stored a two-way train/test split that leaked test into selection."""
    return bool(split_indices) and "valid" in split_indices and "test" in split_indices


def find_roboflow_dataset_root(dataset_dir):
    """Find a Roboflow export containing train/valid/test CSV labels."""
    requested_root = Path(dataset_dir).expanduser().resolve()
    candidates = [requested_root]
    candidates.extend(path.parent.parent for path in requested_root.rglob("_classes.csv"))
    for candidate in candidates:
        if all((candidate / split / "_classes.csv").is_file() for split in ("train", "valid", "test")):
            return candidate
    return None


def _is_ignored_dataset_entry(name):
    return name.startswith(".") or name == "__MACOSX"


def _is_usable_image_file(path):
    """Reject archive metadata and non-image files before PIL tries to open them."""
    image_path = Path(path)
    if image_path.name.startswith("._") or image_path.name == ".DS_Store":
        return False
    if any(part == "__MACOSX" or part.startswith("._") for part in image_path.parts):
        return False
    return image_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS


def find_imagefolder_dataset_root(dataset_dir):
    """Find the meaningful ImageFolder root inside an extracted archive."""
    current_root = Path(dataset_dir).expanduser().resolve()
    while True:
        child_directories = [
            entry
            for entry in current_root.iterdir()
            if entry.is_dir() and not _is_ignored_dataset_entry(entry.name)
        ]
        has_class_images = any(
            any(_is_usable_image_file(file) for file in child.iterdir() if file.is_file())
            for child in child_directories
        )
        if has_class_images or len(child_directories) != 1:
            return current_root
        current_root = child_directories[0]


class CleanImageFolder(torchvision.datasets.ImageFolder):
    """ImageFolder variant that ignores hidden macOS archive entries."""

    def find_classes(self, directory):
        classes = sorted(
            entry.name
            for entry in os.scandir(directory)
            if entry.is_dir() and not _is_ignored_dataset_entry(entry.name)
        )
        if not classes:
            raise FileNotFoundError(f"Couldn't find class folders in {directory}")
        return classes, {class_name: index for index, class_name in enumerate(classes)}


def _read_roboflow_labels(split_dir):
    csv_path = Path(split_dir) / "_classes.csv"
    with csv_path.open(newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows or len(rows[0]) < 2:
        raise ValueError(f"Invalid Roboflow label file: {csv_path}")

    class_names = [name.strip() for name in rows[0][1:]]
    samples = []
    skipped_rows = []
    for row in rows[1:]:
        if len(row) != len(class_names) + 1:
            raise ValueError(f"Malformed label row in {csv_path}: {row}")
        positive_classes = [index for index, value in enumerate(row[1:]) if value.strip() == "1"]
        if len(positive_classes) != 1:
            skipped_rows.append({
                "filename": row[0].strip(),
                "positive_class_count": len(positive_classes),
                "reason": "no_label" if not positive_classes else "multiple_labels",
            })
            continue
        image_path = Path(split_dir) / row[0].strip()
        if not image_path.is_file():
            raise ValueError(f"Image listed in {csv_path} does not exist: {image_path}")
        samples.append((str(image_path), positive_classes[0]))
    return class_names, samples, {
        "source_rows": len(rows) - 1,
        "usable_rows": len(samples),
        "skipped_rows": len(skipped_rows),
        "skipped_examples": skipped_rows[:10],
    }


class RoboflowClassificationDataset(torch.utils.data.Dataset):
    """Dataset adapter for Roboflow's image + _classes.csv export."""

    def __init__(self, split_dir, transform=None):
        self.split_dir = Path(split_dir)
        self.class_names, self.samples, self.report = _read_roboflow_labels(self.split_dir)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        image = torchvision.datasets.folder.default_loader(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
TRANSFER_INPUT_SIZE = 224
DEFAULT_INPUT_SIZE = 64
SUPPORTED_INPUT_SIZES = {64, 96, 128}
SUPPORTED_BLOCK_TYPES = {"plain", "residual", "depthwise"}


def input_size_for(config):
    return int((config or {}).get("input_size", DEFAULT_INPUT_SIZE))


def preprocessing_descriptor(config=None, dataset_dir=None, transfer=False):
    """Describe exactly how inference inputs must be prepared for a checkpoint."""
    if _dataset_modality(dataset_dir) == "audio":
        return {
            "modality": "audio",
            "sample_rate": audio_pipeline.TARGET_SAMPLE_RATE,
            "clip_seconds": audio_pipeline.TARGET_SAMPLES / audio_pipeline.TARGET_SAMPLE_RATE,
            "n_fft": audio_pipeline.N_FFT,
            "n_mels": audio_pipeline.N_MELS,
            "features": "log-mel, per-clip normalised",
        }
    if transfer:
        return {
            "modality": "image",
            "resize": 256,
            "center_crop": TRANSFER_INPUT_SIZE,
            "mean": list(IMAGENET_MEAN),
            "std": list(IMAGENET_STD),
        }
    size = input_size_for(config)
    return {"modality": "image", "resize": [size, size], "mean": [0.5] * 3, "std": [0.5] * 3}


def _image_transform(dataset_dir=None, train=False, transfer=False, image_size=DEFAULT_INPUT_SIZE):
    if transfer:
        # Pretrained backbones expect ImageNet resolution and statistics;
        # feeding them 64px images with 0.5/0.5 normalisation cripples them.
        if train:
            operations = [
                transforms.RandomResizedCrop(TRANSFER_INPUT_SIZE, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            ]
        else:
            operations = [transforms.Resize(256), transforms.CenterCrop(TRANSFER_INPUT_SIZE)]
        return transforms.Compose(operations + [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

    if dataset_dir is None:
        if train:
            return transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        return transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    if train:
        operations = [
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(8),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        ]
    else:
        operations = [transforms.Resize((image_size, image_size))]
    operations.extend([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    return transforms.Compose(operations)


def get_image_split_indices(dataset_dir, seed=42):
    """Build a deterministic stratified train/valid/test split for an image dataset."""
    if find_roboflow_dataset_root(dataset_dir):
        metadata = get_dataset_metadata(dataset_dir)
        return {
            "strategy": "predefined",
            "train_size": metadata["splits"]["train"],
            "validation_size": metadata["splits"]["valid"],
            "test_size": metadata["splits"]["test"],
            "seed": seed,
        }
    imagefolder_root = find_imagefolder_dataset_root(dataset_dir)
    dataset = CleanImageFolder(root=imagefolder_root, is_valid_file=_is_usable_image_file)
    return get_split_indices(len(dataset), seed=seed, labels=dataset.targets)


def get_dataset_split_metadata(dataset_dir, seed=42):
    """Return the stable split descriptor for either image or audio data."""
    audio_metadata = audio_pipeline.get_split_metadata(dataset_dir, seed=seed)
    if audio_metadata:
        return audio_metadata
    return get_image_split_indices(dataset_dir, seed=seed)


def get_dataset_metadata(dataset_dir=None):
    """Return stable class metadata for the current supported image datasets."""
    if dataset_dir:
        if audio_pipeline.detect_audio_dataset(dataset_dir):
            return audio_pipeline.get_audio_metadata(dataset_dir)
        roboflow_root = find_roboflow_dataset_root(dataset_dir)
        if roboflow_root:
            class_names, train_samples, train_report = _read_roboflow_labels(roboflow_root / "train")
            _, valid_samples, valid_report = _read_roboflow_labels(roboflow_root / "valid")
            _, test_samples, test_report = _read_roboflow_labels(roboflow_root / "test")
            return {
                "format": "roboflow_csv_classification",
                "modality": "image",
                "dataset_root": str(roboflow_root),
                "class_names": class_names,
                "class_to_idx": {name: index for index, name in enumerate(class_names)},
                "num_classes": len(class_names),
                "num_samples": len(train_samples) + len(valid_samples) + len(test_samples),
                "splits": {
                    "train": len(train_samples),
                    "valid": len(valid_samples),
                    "test": len(test_samples),
                },
                "quality_report": {
                    "train": train_report,
                    "valid": valid_report,
                    "test": test_report,
                    "policy": "use single-label rows; skip rows with no label or multiple labels",
                },
            }
        imagefolder_root = find_imagefolder_dataset_root(dataset_dir)
        dataset = CleanImageFolder(root=imagefolder_root, is_valid_file=_is_usable_image_file)
        split = get_split_indices(len(dataset), labels=dataset.targets) if len(dataset) >= 3 else None
        counts = torch.bincount(torch.tensor(dataset.targets, dtype=torch.long), minlength=len(dataset.classes)).tolist()
        return {
            "format": "imagefolder_classification",
            "modality": "image",
            "dataset_root": str(imagefolder_root),
            "class_names": list(dataset.classes),
            "class_to_idx": dict(dataset.class_to_idx),
            "class_counts": dict(zip(dataset.classes, counts)),
            "num_classes": len(dataset.classes),
            "num_samples": len(dataset),
            "splits": {key: len(split[key]) for key in ("train", "valid", "test")} if split else None,
        }

    classes = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck",
    ]
    return {
        "format": "cifar10_classification",
        "modality": "image",
        "dataset_root": "./data",
        "class_names": classes,
        "class_to_idx": {name: index for index, name in enumerate(classes)},
        "num_classes": len(classes),
        "num_samples": 60000,
        "splits": {"train": 50000, "valid": 0, "test": 10000},
    }


# 2. Dataset & Preprocessing (Supports CIFAR-10 Fallback or Custom ImageFolder)
def get_dataloaders(
    batch_size=64,
    dataset_dir=None,
    split_indices=None,
    seed=42,
    evaluation_split="valid",
    transfer=False,
    image_size=DEFAULT_INPUT_SIZE,
):
    """Return (train loader, evaluation loader, class count).

    ``evaluation_split`` is "valid" for every development decision and "test"
    only for the single final measurement of a locked model.
    """
    if evaluation_split not in {"valid", "test"}:
        raise ValueError("evaluation_split must be 'valid' or 'test'")
    train_transform = _image_transform(dataset_dir, train=True, transfer=transfer, image_size=image_size)
    eval_transform = _image_transform(dataset_dir, train=False, transfer=transfer, image_size=image_size)
    if dataset_dir:
        if audio_pipeline.detect_audio_dataset(dataset_dir):
            return audio_pipeline.get_audio_dataloaders(
                batch_size=batch_size,
                dataset_dir=dataset_dir,
                split_metadata=split_indices,
                seed=seed,
                evaluation_split=evaluation_split,
            )
        roboflow_root = find_roboflow_dataset_root(dataset_dir)
        if roboflow_root:
            trainset = RoboflowClassificationDataset(roboflow_root / "train", transform=train_transform)
            evaluation_set = RoboflowClassificationDataset(roboflow_root / evaluation_split, transform=eval_transform)
            loader_generator = torch.Generator().manual_seed(seed)
            trainloader = torch.utils.data.DataLoader(
                trainset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                generator=loader_generator,
            )
            testloader = torch.utils.data.DataLoader(
                evaluation_set,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
            )
            return trainloader, testloader, len(trainset.class_names)

        # Custom Dataset Pipeline. Keep separate dataset objects so only the
        # training subset receives augmentation; evaluation must remain stable.
        imagefolder_root = find_imagefolder_dataset_root(dataset_dir)
        train_dataset = CleanImageFolder(root=imagefolder_root, transform=train_transform, is_valid_file=_is_usable_image_file)
        evaluation_dataset = CleanImageFolder(root=imagefolder_root, transform=eval_transform, is_valid_file=_is_usable_image_file)
        num_classes = len(train_dataset.classes)

        # Reuse one deterministic split for every candidate in a search run.
        # Legacy two-way splits are regenerated so test never drives selection.
        if not split_is_current(split_indices):
            split_indices = get_split_indices(len(train_dataset), seed=seed, labels=train_dataset.targets)
        trainset = torch.utils.data.Subset(train_dataset, split_indices["train"])
        testset = torch.utils.data.Subset(evaluation_dataset, split_indices[evaluation_split])

        loader_generator = torch.Generator().manual_seed(seed)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=0,
            generator=loader_generator,
        )
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=0,
        )

        return trainloader, testloader, num_classes
        
    else:
        # Fallback to CIFAR-10: carve a fixed 5,000-image validation set out of
        # the official training split so the official test set stays sealed.
        cifar_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
        full_eval = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=eval_transform)
        order = torch.randperm(len(cifar_train), generator=torch.Generator().manual_seed(seed)).tolist()
        trainset = torch.utils.data.Subset(cifar_train, order[5000:])
        if evaluation_split == "valid":
            evalset = torch.utils.data.Subset(full_eval, order[:5000])
        else:
            evalset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=eval_transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
        testloader = torch.utils.data.DataLoader(evalset, batch_size=batch_size, shuffle=False, num_workers=0)
        return trainloader, testloader, 10


def _dataset_labels(dataset):
    """Extract labels from ImageFolder, Roboflow, or Subset datasets."""
    if isinstance(dataset, torch.utils.data.Subset):
        base_labels = _dataset_labels(dataset.dataset)
        return [base_labels[index] for index in dataset.indices]
    if hasattr(dataset, "targets"):
        return list(dataset.targets)
    if hasattr(dataset, "samples"):
        return [label for _, label in dataset.samples]
    return []


def _class_weights(dataset, num_classes):
    """Return normalized inverse-frequency weights for macro metrics."""
    labels = _dataset_labels(dataset)
    counts = torch.bincount(torch.tensor(labels, dtype=torch.long), minlength=num_classes).float()
    weights = torch.zeros(num_classes, dtype=torch.float32)
    present = counts > 0
    weights[present] = counts[present].sum() / (present.sum() * counts[present])
    if present.any():
        weights[present] /= weights[present].mean()
    return weights

# 3. Dynamic Candidate Generator (The Forge)
def _activation(name):
    return {"ReLU": nn.ReLU, "GELU": nn.GELU, "SiLU": nn.SiLU}[name]()


class ResidualBlock(nn.Module):
    """Two convolutions with an identity (or 1x1 projection) shortcut."""

    def __init__(self, in_channels, out_channels, kernel, batch_norm, activation):
        super().__init__()
        padding = kernel // 2
        norm = (lambda c: nn.BatchNorm2d(c)) if batch_norm else (lambda c: nn.Identity())
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel, padding=padding)
        self.bn1 = norm(out_channels)
        self.act1 = _activation(activation)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel, padding=padding)
        self.bn2 = norm(out_channels)
        self.shortcut = (
            nn.Identity() if in_channels == out_channels
            else nn.Sequential(nn.Conv2d(in_channels, out_channels, 1), norm(out_channels))
        )
        self.act2 = _activation(activation)

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act2(out + self.shortcut(x))


class DepthwiseSeparableBlock(nn.Module):
    """Depthwise k x k convolution followed by a 1x1 pointwise mix (edge-friendly)."""

    def __init__(self, in_channels, out_channels, kernel, batch_norm, activation):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel, padding=kernel // 2, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)
        self.bn = nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity()
        self.act = _activation(activation)

    def forward(self, x):
        return self.act(self.bn(self.pointwise(self.depthwise(x))))


class DynamicCNN(nn.Module):
    def __init__(self, config, num_classes=10, in_channels=3):
        super(DynamicCNN, self).__init__()
        validate_config(config)
        layers = []
        current_channels = in_channels
        block_type = config.get("block_type", "plain")

        if block_type != "plain":
            block_class = ResidualBlock if block_type == "residual" else DepthwiseSeparableBlock
            for i in range(config["num_layers"]):
                filters = config["filters"][i]
                layers.append(block_class(
                    current_channels, filters, config["kernel_sizes"][i],
                    config.get("batch_norm", True), config.get("activation", "ReLU"),
                ))
                if config.get("pooling", True):
                    layers.append(nn.MaxPool2d(2, 2))
                if config.get("dropout", 0) > 0:
                    layers.append(nn.Dropout2d(config["dropout"]))
                current_channels = filters
            self.features = nn.Sequential(*layers)
            self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = nn.Linear(current_channels, num_classes)
            return

        # Plain blocks keep the original module layout so older checkpoints load unchanged.
        for i in range(config.get('num_layers', 2)):
            filters = config['filters'][i]
            kernel = config['kernel_sizes'][i]
            padding = kernel // 2 # Keep spatial dimensions intact
            
            layers.append(nn.Conv2d(current_channels, filters, kernel_size=kernel, padding=padding))
            
            if config.get('batch_norm', True):
                layers.append(nn.BatchNorm2d(filters))
                
            # Activation selection
            activation = config.get('activation', 'ReLU')
            if activation == 'ReLU': layers.append(nn.ReLU())
            elif activation == 'GELU': layers.append(nn.GELU())
            elif activation == 'SiLU': layers.append(nn.SiLU())
                
            if config.get('pooling', True):
                layers.append(nn.MaxPool2d(2, 2))
                
            if config.get('dropout', 0) > 0:
                layers.append(nn.Dropout2d(config['dropout']))
                
            current_channels = filters

        self.features = nn.Sequential(*layers)
        
        # Adaptive pooling ensures output is exactly 1x1 per filter, eliminating shape math for the linear layer
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(current_channels, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def _dataset_modality(dataset_dir=None):
    if dataset_dir and audio_pipeline.detect_audio_dataset(dataset_dir):
        return "audio"
    return "image"


def _build_transfer_model(transfer_model, num_classes, pretrained=False):
    t_model = (transfer_model or "resnet18").lower()
    if "mobilenet" in t_model:
        weights = torchvision.models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = torchvision.models.mobilenet_v3_small(weights=weights)
        for p in model.parameters():
            p.requires_grad = False
        in_feat = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_feat, num_classes)
        return model
    elif "shuffle" in t_model:
        weights = torchvision.models.ShuffleNet_V2_X0_5_Weights.DEFAULT if pretrained else None
        model = torchvision.models.shufflenet_v2_x0_5(weights=weights)
        for p in model.parameters():
            p.requires_grad = False
        in_feat = model.fc.in_features
        model.fc = nn.Linear(in_feat, num_classes)
        return model
    elif "efficient" in t_model:
        weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = torchvision.models.efficientnet_b0(weights=weights)
        for p in model.parameters():
            p.requires_grad = False
        in_feat = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_feat, num_classes)
        return model
    else:  # default to resnet18
        weights = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
        model = torchvision.models.resnet18(weights=weights)
        for p in model.parameters():
            p.requires_grad = False
        in_feat = model.fc.in_features
        model.fc = nn.Linear(in_feat, num_classes)
        return model


def _build_model(config, num_classes, dataset_dir=None, use_transfer_learning=False, transfer_model="resnet18", pretrained=False):
    modality = _dataset_modality(dataset_dir)
    if use_transfer_learning:
        if modality != "image":
            raise ValueError(f"The transfer learning baselines are available for image datasets only")
        return _build_transfer_model(transfer_model, num_classes, pretrained=pretrained)
    return DynamicCNN(config, num_classes=num_classes, in_channels=1 if modality == "audio" else 3)

# 4. Proxy Trainer & Judge
def classification_metrics(labels, predictions, num_classes, probabilities=None):
    """Calculate classification metrics plus probability-vector MSE/RMSE.

    MSE and RMSE compare each predicted probability vector with a one-hot
    target vector. They are auxiliary classification diagnostics, not
    regression scores, and remain in raw 0-1 units while the traditional
    classification metrics are percentages.
    """
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)
    for actual, predicted in zip(labels, predictions):
        if 0 <= actual < num_classes and 0 <= predicted < num_classes:
            confusion[actual, predicted] += 1

    true_positive = confusion.diag().float()
    predicted_count = confusion.sum(dim=0).float()
    actual_count = confusion.sum(dim=1).float()
    precision = torch.where(predicted_count > 0, true_positive / predicted_count, torch.zeros_like(true_positive))
    recall = torch.where(actual_count > 0, true_positive / actual_count, torch.zeros_like(true_positive))
    f1_denominator = precision + recall
    f1 = torch.where(
        f1_denominator > 0,
        2 * precision * recall / f1_denominator,
        torch.zeros_like(true_positive),
    )
    present_classes = actual_count > 0
    macro_f1_present = f1[present_classes].mean().item() * 100 if present_classes.any() else 0.0
    macro_precision_present = precision[present_classes].mean().item() * 100 if present_classes.any() else 0.0
    macro_recall_present = recall[present_classes].mean().item() * 100 if present_classes.any() else 0.0
    total = confusion.sum().item()
    accuracy = true_positive.sum().item() / total if total else 0.0
    mse = None
    rmse = None
    if probabilities is not None and len(probabilities) == len(labels):
        probability_tensor = torch.as_tensor(probabilities, dtype=torch.float32)
        if probability_tensor.ndim == 2 and probability_tensor.shape == (len(labels), num_classes):
            target_tensor = torch.nn.functional.one_hot(
                torch.as_tensor(labels, dtype=torch.long),
                num_classes=num_classes,
            ).float()
            mse = torch.mean((probability_tensor - target_tensor) ** 2).item()
            rmse = float(torch.sqrt(torch.tensor(mse)).item())

    return {
        "accuracy": accuracy * 100,
        "precision": precision.mean().item() * 100,
        "recall": recall.mean().item() * 100,
        "f1": f1.mean().item() * 100,
        "f1_present_classes": macro_f1_present,
        "precision_present_classes": macro_precision_present,
        "recall_present_classes": macro_recall_present,
        "mse": mse,
        "rmse": rmse,
        "evaluated_class_count": int(present_classes.sum().item()),
        "metric_average": "macro",
        "metric_scale": "percent",
        "confusion_matrix": confusion.tolist(),
    }


def evaluate_candidate(config, epochs=3, dataset_dir=None, seed=42, split_indices=None):
    print(f"\n--- Testing Candidate Architecture ---")
    validate_config(config)
    set_random_seed(seed)
    # Candidates are ranked on the validation split only; test stays sealed.
    trainloader, testloader, num_classes = get_dataloaders(
        dataset_dir=dataset_dir,
        split_indices=split_indices,
        seed=seed,
        evaluation_split="valid",
        image_size=input_size_for(config),
    )
    model = _build_model(config, num_classes, dataset_dir=dataset_dir).to(device)
    
    criterion = nn.CrossEntropyLoss(
        weight=_class_weights(trainloader.dataset, num_classes).to(device),
    )
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Calculate Parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Training Loop (Proxy Budget)
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    training_time = time.time() - start_time
    
    # Evaluation
    model.eval()
    all_labels = []
    predictions = []
    probabilities = []
    inference_start = time.time()
    with torch.no_grad():
        for inputs, batch_labels in testloader:
            inputs, batch_labels = inputs.to(device), batch_labels.to(device)
            outputs = model(inputs)
            batch_probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(batch_probabilities.data, 1)
            all_labels.extend(batch_labels.detach().cpu().tolist())
            predictions.extend(predicted.detach().cpu().tolist())
            probabilities.extend(batch_probabilities.detach().cpu().tolist())
    inference_time = time.time() - inference_start
    metrics = classification_metrics(all_labels, predictions, num_classes, probabilities)

    print(f"Params: {total_params:,} | Accuracy: {metrics['accuracy']:.2f}% | F1: {metrics['f1']:.2f}%")
    print(f"Training Time: {training_time:.2f}s | Inference Time: {inference_time:.2f}s")

    return {
        **metrics,
        "parameters": total_params,
        "training_time_seconds": training_time,
        "inference_latency_seconds": inference_time,
        "inference_latency_per_sample_seconds": inference_time / max(1, len(all_labels)),
        "num_classes": num_classes,
        "seed": seed,
        "config_hash": config_hash(config),
    }

# 5. Full Trainer
def _run_eval(model, loader, num_classes):
    model.eval()
    labels, predictions, probabilities = [], [], []
    with torch.no_grad():
        for inputs, batch_labels in loader:
            outputs = model(inputs.to(device))
            batch_probabilities = torch.softmax(outputs, dim=1)
            labels.extend(batch_labels.tolist())
            predictions.extend(batch_probabilities.argmax(dim=1).cpu().tolist())
            probabilities.extend(batch_probabilities.cpu().tolist())
    return classification_metrics(labels, predictions, num_classes, probabilities), len(labels)


def full_train(
    config,
    epochs=30,
    st_callback=None,
    dataset_dir=None,
    use_transfer_learning=False,
    transfer_model="resnet18",
    seed=42,
    weights_path="best_model_weights.pth",
    metadata_path="best_model_metadata.json",
    patience=8,
    split_indices=None,
):
    """Train from fresh weights with validation-only model selection.

    Protocol (matches docs/neuroforge-ultimate-plan section 08):
      * train split  -> weight updates
      * valid split  -> checkpoint choice (macro F1, accuracy tie-break) and early stopping
      * test split   -> evaluated exactly once on the locked checkpoint
    """
    print(f"\n--- Starting full training for up to {epochs} epochs ---")
    set_random_seed(seed)
    image_size = input_size_for(config)
    trainloader, validloader, num_classes = get_dataloaders(
        batch_size=64,
        dataset_dir=dataset_dir,
        split_indices=split_indices,
        seed=seed,
        evaluation_split="valid",
        transfer=use_transfer_learning,
        image_size=image_size,
    )
    model = _build_model(
        config,
        num_classes,
        dataset_dir=dataset_dir,
        use_transfer_learning=use_transfer_learning,
        transfer_model=transfer_model,
        pretrained=use_transfer_learning,
    ).to(device)

    class_weights = _class_weights(trainloader.dataset, num_classes)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    learning_rate = 0.003
    optimizer = optim.AdamW(trainable_params, lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_key = None
    best_epoch = 0
    best_valid_metrics = None
    curve = []
    epochs_without_improvement = 0
    started = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        running_loss, train_correct, train_total = 0.0, 0, 0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_total += labels.size(0)
            train_correct += (outputs.argmax(dim=1) == labels).sum().item()
        scheduler.step()

        valid_metrics, _ = _run_eval(model, validloader, num_classes)
        train_accuracy = 100 * train_correct / max(1, train_total)
        key = (round(valid_metrics["f1"], 4), round(valid_metrics["accuracy"], 4))
        improved = best_key is None or key > best_key
        curve.append({
            "epoch": epoch,
            "train_loss": running_loss / max(1, len(trainloader)),
            "train_accuracy": train_accuracy,
            "valid_accuracy": valid_metrics["accuracy"],
            "valid_f1": valid_metrics["f1"],
        })
        eta = int((time.time() - epoch_start) * (epochs - epoch))
        print(
            f"Epoch [{epoch}/{epochs}] loss {curve[-1]['train_loss']:.4f} | train {train_accuracy:.1f}% "
            f"| valid acc {valid_metrics['accuracy']:.1f}% f1 {valid_metrics['f1']:.1f}%"
            + ("  <- best" if improved else "")
        )
        if st_callback:
            st_callback({
                "epoch": epoch,
                "total_epochs": epochs,
                "train_acc": train_accuracy,
                "val_acc": valid_metrics["accuracy"],
                "val_precision": valid_metrics["precision"],
                "val_recall": valid_metrics["recall"],
                "val_f1": valid_metrics["f1"],
                "eta": f"{eta // 60}m {eta % 60}s",
            })
        if improved:
            best_key, best_epoch, best_valid_metrics = key, epoch, valid_metrics
            epochs_without_improvement = 0
            torch.save(model.state_dict(), weights_path)
        else:
            epochs_without_improvement += 1
            if patience and epochs_without_improvement >= patience:
                print(f"Early stopping: no validation F1 gain for {patience} epochs.")
                break

    # Lock the best validation checkpoint, then open the sealed test split once.
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    _, testloader, _ = get_dataloaders(
        batch_size=64,
        dataset_dir=dataset_dir,
        split_indices=split_indices,
        seed=seed,
        evaluation_split="test",
        transfer=use_transfer_learning,
        image_size=image_size,
    )
    test_start = time.time()
    test_metrics, test_count = _run_eval(model, testloader, num_classes)
    test_elapsed = time.time() - test_start
    test_metrics.update({
        "split": "test",
        "sample_count": test_count,
        "inference_latency_seconds": test_elapsed,
        "inference_latency_per_sample_seconds": test_elapsed / max(1, test_count),
    })

    dataset_metadata = get_dataset_metadata(dataset_dir)
    metadata = {
        "schema_version": 2,
        "num_classes": num_classes,
        "class_names": dataset_metadata["class_names"],
        "config": copy.deepcopy(config),
        "trained_config": copy.deepcopy(config),
        "config_hash": config_hash(config),
        "training_mode": f"transfer_learning_{transfer_model}" if use_transfer_learning else "dynamic_cnn",
        "transfer_model": transfer_model if use_transfer_learning else None,
        "seed": seed,
        "epochs": epochs,
        "epochs_run": len(curve),
        "best_epoch": best_epoch,
        "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "training_seconds": time.time() - started,
        "device": str(device),
        "training": {
            "optimizer": "AdamW",
            "learning_rate": learning_rate,
            "weight_decay": 1e-4,
            "scheduler": "CosineAnnealingLR",
            "loss": "class-weighted cross-entropy",
            "class_weights": [round(w, 4) for w in class_weights.tolist()],
            "selection_rule": "best validation macro F1 (accuracy tie-break)",
            "early_stopping_patience": patience,
        },
        "evaluation_protocol": "train -> weights; valid -> checkpoint + early stopping; test -> evaluated once on the locked checkpoint",
        "preprocessing": preprocessing_descriptor(config, dataset_dir, transfer=use_transfer_learning),
        "best_metrics": {k: best_valid_metrics[k] for k in ("accuracy", "precision", "recall", "f1", "mse", "rmse")},
        "validation_metrics": best_valid_metrics,
        "held_out_test_metrics": test_metrics,
        "learning_curve": curve,
        "dataset": dataset_metadata,
        "weights_path": str(weights_path),
        "training_complete": True,
    }
    temporary_path = f"{metadata_path}.tmp"
    with open(temporary_path, "w") as handle:
        json.dump(metadata, handle, indent=2)
    os.replace(temporary_path, metadata_path)

    print(
        f"\nTraining complete. Best epoch {best_epoch}: valid acc {best_valid_metrics['accuracy']:.2f}% / F1 {best_valid_metrics['f1']:.2f}%"
        f" | test acc {test_metrics['accuracy']:.2f}% / F1 {test_metrics['f1']:.2f}% (n={test_count})"
    )
    return best_valid_metrics["accuracy"]

# 6. Live Inference
def evaluate_checkpoint(
    config,
    num_classes,
    dataset_dir=None,
    evaluation_split="test",
    use_transfer_learning=False,
    transfer_model="resnet18",
    weights_path="best_model_weights.pth",
    seed=42,
):
    """Evaluate a saved model on a named dataset split without updating weights."""
    if use_transfer_learning:
        model = _build_model(
            config,
            num_classes,
            dataset_dir=dataset_dir,
            use_transfer_learning=True,
            transfer_model=transfer_model,
            pretrained=False,
        )
    else:
        model = _build_model(config, num_classes, dataset_dir=dataset_dir)

    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    _, evaluation_loader, detected_classes = get_dataloaders(
        batch_size=128,
        dataset_dir=dataset_dir,
        seed=seed,
        evaluation_split=evaluation_split,
        transfer=use_transfer_learning,
        image_size=input_size_for(config),
    )
    if detected_classes != num_classes:
        raise ValueError("Dataset class count does not match the saved model")

    all_labels = []
    predictions = []
    probabilities = []
    inference_start = time.time()
    with torch.no_grad():
        for inputs, batch_labels in evaluation_loader:
            outputs = model(inputs.to(device))
            batch_probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(batch_probabilities.data, 1)
            all_labels.extend(batch_labels.tolist())
            predictions.extend(predicted.cpu().tolist())
            probabilities.extend(batch_probabilities.cpu().tolist())
    elapsed = time.time() - inference_start
    metrics = classification_metrics(all_labels, predictions, num_classes, probabilities)
    metrics.update({
        "split": evaluation_split,
        "sample_count": len(all_labels),
        "inference_latency_seconds": elapsed,
        "inference_latency_per_sample_seconds": elapsed / max(1, len(all_labels)),
    })
    return metrics


def predict_image(
    image_tensor,
    config,
    num_classes,
    use_transfer_learning=False,
    weights_path="best_model_weights.pth",
    dataset_dir=None,
):
    if use_transfer_learning:
        model = _build_model(
            config,
            num_classes,
            dataset_dir=dataset_dir,
            use_transfer_learning=True,
            pretrained=False,
        )
    else:
        model = _build_model(config, num_classes, dataset_dir=dataset_dir)
        
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    expected_key = "fc.weight" if use_transfer_learning else "classifier.weight"
    if expected_key not in state_dict:
        expected_model = "ResNet-18" if use_transfer_learning else "DynamicCNN"
        raise ValueError(
            f"Checkpoint is incompatible with {expected_model}. "
            f"Expected '{expected_key}' in {weights_path}. Retrain the selected model first."
        )
    output_classes = state_dict[expected_key].shape[0]
    if output_classes != num_classes:
        raise ValueError(
            f"Checkpoint has {output_classes} output classes but the dataset has {num_classes}. "
            "Retrain the model for this dataset."
        )
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).to(device) # Add batch dimension
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)
        
    return predicted_idx.item(), confidence.item()


def predict_audio(audio_bytes, config, num_classes, weights_path="best_model_weights.pth", dataset_dir=None):
    """Predict a 16-bit PCM WAV upload with the saved audio checkpoint."""
    probabilities, _ = predict_audio_distribution(
        audio_bytes,
        config,
        num_classes,
        weights_path=weights_path,
        dataset_dir=dataset_dir,
    )
    confidence, predicted_idx = torch.max(probabilities, 0)
    return predicted_idx.item(), confidence.item()


def predict_audio_distribution(audio_bytes, config, num_classes, weights_path="best_model_weights.pth", dataset_dir=None):
    """Return all class probabilities and feature shape for an audio upload."""
    if _dataset_modality(dataset_dir) != "audio":
        raise ValueError("The selected dataset is not an audio dataset")
    model = _build_model(config, num_classes, dataset_dir=dataset_dir).to(device)
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    if "classifier.weight" not in state_dict:
        raise ValueError("The saved checkpoint is not a DynamicCNN audio checkpoint")
    model.load_state_dict(state_dict)
    model.eval()
    feature_tensor = audio_pipeline.audio_sample_from_bytes(audio_bytes)
    with torch.no_grad():
        probabilities = torch.nn.functional.softmax(model(feature_tensor.unsqueeze(0).to(device))[0], dim=0)
    return probabilities.cpu(), list(feature_tensor.shape)

# Quick test if run directly
if __name__ == "__main__":
    import os
    config_path = "best_model_config.json"
    
    if os.path.exists(config_path):
        print(f"Found {config_path}. Loading architecture...")
        with open(config_path, "r") as f:
            best_config = json.load(f)
        
        # You can toggle this to evaluate_candidate for quick tests, 
        # or full_train to go for 90%+ accuracy.
        # full_train(best_config, epochs=50)
        evaluate_candidate(best_config, epochs=1)
    else:
        print("No best_model_config.json found. Run architect.py first to generate one!")
