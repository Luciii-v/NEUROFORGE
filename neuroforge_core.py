import hashlib
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
    return config


def config_hash(config):
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def get_split_indices(dataset_size, seed=42, train_fraction=0.8, labels=None):
    """Create one deterministic, class-aware train/evaluation split."""
    if dataset_size < 2:
        raise ValueError("Dataset must contain at least two samples")
    generator = torch.Generator().manual_seed(seed)

    if labels is not None:
        if len(labels) != dataset_size:
            raise ValueError("labels must have one entry per dataset sample")
        grouped_indices = {}
        for index, label in enumerate(labels):
            grouped_indices.setdefault(label, []).append(index)
        train_indices = []
        test_indices = []
        for class_indices in grouped_indices.values():
            order = torch.randperm(len(class_indices), generator=generator).tolist()
            shuffled = [class_indices[index] for index in order]
            if len(shuffled) == 1:
                train_indices.extend(shuffled)
                continue
            train_size = max(1, min(len(shuffled) - 1, int(train_fraction * len(shuffled))))
            train_indices.extend(shuffled[:train_size])
            test_indices.extend(shuffled[train_size:])
        return {
            "train": train_indices,
            "test": test_indices,
            "seed": seed,
            "stratified": True,
        }

    train_size = max(1, min(dataset_size - 1, int(train_fraction * dataset_size)))
    indices = torch.randperm(dataset_size, generator=generator).tolist()
    return {
        "train": indices[:train_size],
        "test": indices[train_size:],
        "seed": seed,
        "stratified": False,
    }


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


def _image_transform(dataset_dir=None, train=False):
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

    operations = [transforms.Resize((64, 64))]
    if train:
        operations.extend([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(8),
        ])
    operations.extend([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    return transforms.Compose(operations)


def get_image_split_indices(dataset_dir, seed=42, train_fraction=0.8):
    """Build a deterministic stratified split for an ImageFolder dataset."""
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
    return get_split_indices(
        len(dataset),
        seed=seed,
        train_fraction=train_fraction,
        labels=dataset.targets,
    )


def get_dataset_metadata(dataset_dir=None):
    """Return stable class metadata for the current supported image datasets."""
    if dataset_dir:
        roboflow_root = find_roboflow_dataset_root(dataset_dir)
        if roboflow_root:
            class_names, train_samples, train_report = _read_roboflow_labels(roboflow_root / "train")
            _, valid_samples, valid_report = _read_roboflow_labels(roboflow_root / "valid")
            _, test_samples, test_report = _read_roboflow_labels(roboflow_root / "test")
            return {
                "format": "roboflow_csv_classification",
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
        return {
            "format": "imagefolder_classification",
            "dataset_root": str(imagefolder_root),
            "class_names": list(dataset.classes),
            "class_to_idx": dict(dataset.class_to_idx),
            "num_classes": len(dataset.classes),
            "num_samples": len(dataset),
            "splits": None,
        }

    classes = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck",
    ]
    return {
        "format": "cifar10_classification",
        "dataset_root": "./data",
        "class_names": classes,
        "class_to_idx": {name: index for index, name in enumerate(classes)},
        "num_classes": len(classes),
        "num_samples": 60000,
        "splits": {"train": 50000, "valid": 0, "test": 10000},
    }


# 2. Dataset & Preprocessing (Supports CIFAR-10 Fallback or Custom ImageFolder)
def get_dataloaders(batch_size=64, dataset_dir=None, split_indices=None, seed=42, evaluation_split="valid"):
    if dataset_dir:
        if evaluation_split not in {"valid", "test"}:
            raise ValueError("evaluation_split must be 'valid' or 'test'")
        roboflow_root = find_roboflow_dataset_root(dataset_dir)
        if roboflow_root:
            trainset = RoboflowClassificationDataset(
                roboflow_root / "train",
                transform=_image_transform(dataset_dir, train=True),
            )
            evaluation_set = RoboflowClassificationDataset(
                roboflow_root / evaluation_split,
                transform=_image_transform(dataset_dir, train=False),
            )
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
        train_dataset = CleanImageFolder(
            root=imagefolder_root,
            transform=_image_transform(dataset_dir, train=True),
            is_valid_file=_is_usable_image_file,
        )
        evaluation_dataset = CleanImageFolder(
            root=imagefolder_root,
            transform=_image_transform(dataset_dir, train=False),
            is_valid_file=_is_usable_image_file,
        )
        num_classes = len(train_dataset.classes)

        # Reuse one deterministic split for every candidate in a search run.
        split_indices = split_indices or get_split_indices(
            len(train_dataset),
            seed=seed,
            labels=train_dataset.targets,
        )
        trainset = torch.utils.data.Subset(train_dataset, split_indices["train"])
        testset = torch.utils.data.Subset(evaluation_dataset, split_indices["test"])

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
        # Fallback to CIFAR-10
        transform_train = _image_transform(train=True)
        transform_test = _image_transform(train=False)
    
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)
    
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
class DynamicCNN(nn.Module):
    def __init__(self, config, num_classes=10):
        super(DynamicCNN, self).__init__()
        validate_config(config)
        layers = []
        in_channels = 3 # RGB Images
        
        # Build Convolutional Layers dynamically based on config
        for i in range(config.get('num_layers', 2)):
            filters = config['filters'][i]
            kernel = config['kernel_sizes'][i]
            padding = kernel // 2 # Keep spatial dimensions intact
            
            layers.append(nn.Conv2d(in_channels, filters, kernel_size=kernel, padding=padding))
            
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
                
            in_channels = filters

        self.features = nn.Sequential(*layers)
        
        # Adaptive pooling ensures output is exactly 1x1 per filter, eliminating shape math for the linear layer
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# 4. Proxy Trainer & Judge
def classification_metrics(labels, predictions, num_classes):
    """Calculate accuracy and macro precision/recall/F1 in percentage units."""
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
    total = confusion.sum().item()
    accuracy = true_positive.sum().item() / total if total else 0.0
    return {
        "accuracy": accuracy * 100,
        "precision": precision.mean().item() * 100,
        "recall": recall.mean().item() * 100,
        "f1": f1.mean().item() * 100,
        "metric_average": "macro",
        "metric_scale": "percent",
        "confusion_matrix": confusion.tolist(),
    }


def evaluate_candidate(config, epochs=3, dataset_dir=None, seed=42, split_indices=None):
    print(f"\n--- Testing Candidate Architecture ---")
    validate_config(config)
    set_random_seed(seed)
    trainloader, testloader, num_classes = get_dataloaders(
        dataset_dir=dataset_dir,
        split_indices=split_indices,
        seed=seed,
    )
    model = DynamicCNN(config, num_classes=num_classes).to(device)
    
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
    inference_start = time.time()
    with torch.no_grad():
        for inputs, batch_labels in testloader:
            inputs, batch_labels = inputs.to(device), batch_labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(batch_labels.detach().cpu().tolist())
            predictions.extend(predicted.detach().cpu().tolist())
    inference_time = time.time() - inference_start
    metrics = classification_metrics(all_labels, predictions, num_classes)

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

# 5. Full Trainer (To achieve >90% accuracy)
def full_train(
    config,
    epochs=50,
    st_callback=None,
    dataset_dir=None,
    use_transfer_learning=False,
    seed=42,
    weights_path="best_model_weights.pth",
    metadata_path="best_model_metadata.json",
):
    print(f"\n--- Starting Full Deep Training for {epochs} Epochs ---")
    set_random_seed(seed)
    trainloader, testloader, num_classes = get_dataloaders(
        batch_size=128,
        dataset_dir=dataset_dir,
        seed=seed,
    )
    
    if use_transfer_learning:
        print("Using Pretrained ResNet18 for Transfer Learning...")
        model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        # Freeze early layers for speed
        for param in model.parameters():
            param.requires_grad = False
        # Replace the final classification layer (this will have requires_grad=True by default)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model = model.to(device)
    else:
        model = DynamicCNN(config, num_classes=num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss(
        weight=_class_weights(trainloader.dataset, num_classes).to(device),
    )
    # Using AdamW with weight decay for better regularization
    optimizer = optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
    # Cosine Annealing Learning Rate Scheduler helps push accuracy in later epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_accuracy = -1.0
    best_metrics = None
    
    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
        train_accuracy = 100 * train_correct / train_total
        scheduler.step()
        
        # Validation Phase
        model.eval()
        validation_labels = []
        validation_predictions = []
        with torch.no_grad():
            for inputs, batch_labels in testloader:
                inputs, batch_labels = inputs.to(device), batch_labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                validation_labels.extend(batch_labels.detach().cpu().tolist())
                validation_predictions.extend(predicted.detach().cpu().tolist())

        validation_metrics = classification_metrics(
            validation_labels,
            validation_predictions,
            num_classes,
        )
        val_accuracy = validation_metrics["accuracy"]
        
        epoch_time = time.time() - epoch_start
        eta_seconds = epoch_time * (epochs - epoch - 1)
        mins, secs = divmod(int(eta_seconds), 60)
        eta_str = f"{mins}m {secs}s"
        
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {running_loss/len(trainloader):.4f} | Train Acc: {train_accuracy:.2f}% | Val Acc: {val_accuracy:.2f}% | ETA: {eta_str}")
        
        if st_callback:
            st_callback({
                "epoch": epoch + 1,
                "total_epochs": epochs,
                "train_acc": train_accuracy,
                "val_acc": val_accuracy,
                "val_precision": validation_metrics["precision"],
                "val_recall": validation_metrics["recall"],
                "val_f1": validation_metrics["f1"],
                "eta": eta_str,
            })
        
        # Checkpoint the best model weights
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_metrics = validation_metrics
            torch.save(model.state_dict(), weights_path)
            print(f"  -> New best model saved! ({best_accuracy:.2f}%)")

            existing_metadata = {}
            if os.path.exists(metadata_path):
                with open(metadata_path) as handle:
                    existing_metadata = json.load(handle)
            existing_metadata.update({
                "schema_version": 1,
                "num_classes": num_classes,
                "class_names": get_dataset_metadata(dataset_dir)["class_names"],
                "config_hash": config_hash(config),
                "training_mode": "transfer_learning_resnet18" if use_transfer_learning else "dynamic_cnn",
                "seed": seed,
                "epochs": epochs,
                "best_metrics": {
                    key: best_metrics[key]
                    for key in ("accuracy", "precision", "recall", "f1")
                },
                "weights_path": weights_path,
            })
            with open(metadata_path, "w") as handle:
                json.dump(existing_metadata, handle, indent=2)
            
    print(f"\n✅ Full Training Complete. Best Validation Accuracy: {best_accuracy:.2f}%")

    # Evaluate the best checkpoint on the held-out test split and persist it
    # separately from the validation metrics shown during training.
    held_out_test_metrics = evaluate_checkpoint(
        config,
        num_classes,
        dataset_dir=dataset_dir,
        evaluation_split="test",
        use_transfer_learning=use_transfer_learning,
        weights_path=weights_path,
        seed=seed,
    )
    existing_metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path) as handle:
            existing_metadata = json.load(handle)
    existing_metadata["held_out_test_metrics"] = held_out_test_metrics
    existing_metadata["training_complete"] = True
    with open(metadata_path, "w") as handle:
        json.dump(existing_metadata, handle, indent=2)

    return best_accuracy

# 6. Live Inference
def evaluate_checkpoint(
    config,
    num_classes,
    dataset_dir=None,
    evaluation_split="test",
    use_transfer_learning=False,
    weights_path="best_model_weights.pth",
    seed=42,
):
    """Evaluate a saved model on a named dataset split without updating weights."""
    if use_transfer_learning:
        model = torchvision.models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        expected_key = "fc.weight"
    else:
        model = DynamicCNN(config, num_classes=num_classes)
        expected_key = "classifier.weight"

    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    if expected_key not in state_dict:
        raise ValueError(f"Checkpoint is incompatible with the selected model in {weights_path}")
    if state_dict[expected_key].shape[0] != num_classes:
        raise ValueError(f"Checkpoint output classes do not match num_classes={num_classes}")
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    _, evaluation_loader, detected_classes = get_dataloaders(
        batch_size=128,
        dataset_dir=dataset_dir,
        seed=seed,
        evaluation_split=evaluation_split,
    )
    if detected_classes != num_classes:
        raise ValueError("Dataset class count does not match the saved model")

    all_labels = []
    predictions = []
    inference_start = time.time()
    with torch.no_grad():
        for inputs, batch_labels in evaluation_loader:
            outputs = model(inputs.to(device))
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(batch_labels.tolist())
            predictions.extend(predicted.cpu().tolist())
    elapsed = time.time() - inference_start
    metrics = classification_metrics(all_labels, predictions, num_classes)
    metrics.update({
        "split": evaluation_split,
        "sample_count": len(all_labels),
        "inference_latency_seconds": elapsed,
        "inference_latency_per_sample_seconds": elapsed / max(1, len(all_labels)),
    })
    return metrics


def predict_image(image_tensor, config, num_classes, use_transfer_learning=False, weights_path="best_model_weights.pth"):
    if use_transfer_learning:
        model = torchvision.models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        model = DynamicCNN(config, num_classes=num_classes)
        
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
