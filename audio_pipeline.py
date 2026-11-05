"""Audio classification data utilities for the NEUROFORGE MVP.

The first audio path intentionally uses Python's standard-library ``wave``
reader and PyTorch's STFT implementation. That keeps the project portable and
avoids coupling the app to a second audio framework while the modality is
being validated.
"""

import csv
import math
import random
import wave
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset


SUPPORTED_AUDIO_EXTENSIONS = {".wav"}
TARGET_SAMPLE_RATE = 16_000
CLIP_SECONDS = 4.0
TARGET_SAMPLES = int(TARGET_SAMPLE_RATE * CLIP_SECONDS)
N_FFT = 512
HOP_LENGTH = 256
N_MELS = 64


def _normalise_name(value):
    return str(value).strip().lower().replace(" ", "_")


def _is_audio_file(path):
    return Path(path).suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS


def _read_wav(path):
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        frame_count = handle.getnframes()
        raw = handle.readframes(frame_count)

    if sample_width == 1:
        waveform = torch.tensor(list(raw), dtype=torch.float32)
        waveform = (waveform - 128.0) / 128.0
    elif sample_width == 2:
        waveform = torch.frombuffer(bytearray(raw), dtype=torch.int16).float() / 32768.0
    elif sample_width == 4:
        waveform = torch.frombuffer(bytearray(raw), dtype=torch.int32).float() / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width {sample_width} bytes: {path}")

    if channels > 1:
        waveform = waveform[: waveform.numel() - (waveform.numel() % channels)]
        waveform = waveform.reshape(-1, channels).mean(dim=1)
    return waveform.clamp(-1.0, 1.0), sample_rate


def _resample(waveform, source_rate, target_rate=TARGET_SAMPLE_RATE):
    if source_rate == target_rate:
        return waveform
    if waveform.numel() == 0:
        return torch.zeros(target_rate // 10)
    target_length = max(1, int(round(waveform.numel() * target_rate / source_rate)))
    return F.interpolate(
        waveform.reshape(1, 1, -1),
        size=target_length,
        mode="linear",
        align_corners=False,
    ).reshape(-1)


def _fit_clip(waveform, target_samples=TARGET_SAMPLES, train=False):
    if waveform.numel() > target_samples:
        if train:
            start = random.randint(0, waveform.numel() - target_samples)
        else:
            start = (waveform.numel() - target_samples) // 2
        waveform = waveform[start : start + target_samples]
    elif waveform.numel() < target_samples:
        waveform = F.pad(waveform, (0, target_samples - waveform.numel()))
    return waveform


def _hz_to_mel(frequency):
    return 2595.0 * math.log10(1.0 + frequency / 700.0)


def _mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _mel_filterbank(sample_rate=TARGET_SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS):
    low_mel = _hz_to_mel(0.0)
    high_mel = _hz_to_mel(sample_rate / 2.0)
    mel_points = torch.linspace(low_mel, high_mel, n_mels + 2)
    hz_points = torch.tensor([_mel_to_hz(value.item()) for value in mel_points])
    bins = torch.floor((n_fft + 1) * hz_points / sample_rate).long()
    filters = torch.zeros(n_mels, n_fft // 2 + 1)
    for index in range(n_mels):
        left, centre, right = bins[index : index + 3].tolist()
        centre = max(centre, left + 1)
        right = max(right, centre + 1)
        right = min(right, n_fft // 2)
        if centre > left:
            filters[index, left:centre] = torch.linspace(0.0, 1.0, centre - left)
        if right > centre:
            filters[index, centre:right] = torch.linspace(1.0, 0.0, right - centre)
    return filters


MEL_FILTERBANK = _mel_filterbank()
_FEATURE_CACHE = {}


def waveform_to_log_mel(waveform, sample_rate=TARGET_SAMPLE_RATE):
    window = torch.hann_window(N_FFT)
    spectrum = torch.stft(
        waveform,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=N_FFT,
        window=window,
        return_complex=True,
        center=True,
    )
    power = spectrum.abs().pow(2)
    mel_energy = MEL_FILTERBANK.to(power.device).matmul(power)
    return torch.log(mel_energy.clamp_min(1e-6)).unsqueeze(0)


def _folder_samples(root):
    root = Path(root).expanduser().resolve()
    class_dirs = sorted(path for path in root.iterdir() if path.is_dir() and not path.name.startswith("."))
    class_names = [path.name for path in class_dirs if any(_is_audio_file(item) for item in path.iterdir())]
    class_to_idx = {name: index for index, name in enumerate(class_names)}
    samples = []
    for class_name in class_names:
        for path in sorted((root / class_name).iterdir()):
            if _is_audio_file(path):
                samples.append((path, class_to_idx[class_name], class_name, None))
    return class_names, samples


def _read_esc50(root):
    root = Path(root).expanduser().resolve()
    metadata_path = root / "meta" / "esc50.csv"
    audio_dir = root / "audio"
    if not metadata_path.is_file() or not audio_dir.is_dir():
        return None
    rows = list(csv.DictReader(metadata_path.open(newline="")))
    class_names = sorted({_normalise_name(row["category"]) for row in rows})
    class_to_idx = {name: index for index, name in enumerate(class_names)}
    samples = []
    for row in rows:
        path = audio_dir / row["filename"]
        if path.is_file():
            class_name = _normalise_name(row["category"])
            samples.append((path, class_to_idx[class_name], class_name, int(row["fold"])))
    return class_names, samples, "esc50"


def _read_urbansound8k(root):
    root = Path(root).expanduser().resolve()
    metadata_path = root / "metadata" / "UrbanSound8K.csv"
    audio_dir = root / "audio"
    if not metadata_path.is_file() or not audio_dir.is_dir():
        return None
    rows = list(csv.DictReader(metadata_path.open(newline="")))
    class_names = sorted({_normalise_name(row["class"]) for row in rows})
    class_to_idx = {name: index for index, name in enumerate(class_names)}
    samples = []
    for row in rows:
        fold = int(row["fold"])
        path = audio_dir / f"fold{fold}" / row["slice_file_name"]
        if path.is_file():
            class_name = _normalise_name(row["class"])
            samples.append((path, class_to_idx[class_name], class_name, fold))
    return class_names, samples, "urbansound8k"


def detect_audio_dataset(root):
    """Return an audio dataset descriptor or ``None`` for image datasets."""
    root = Path(root).expanduser().resolve()
    candidates = [root]
    if root.is_dir():
        candidates.extend(path for path in root.iterdir() if path.is_dir() and not path.name.startswith("."))
    for candidate in candidates:
        for reader in (_read_esc50, _read_urbansound8k):
            descriptor = reader(candidate)
            if descriptor:
                class_names, samples, kind = descriptor
                return {
                    "kind": kind,
                    "root": str(candidate),
                    "class_names": class_names,
                    "samples": samples,
                    "num_classes": len(class_names),
                }
        if candidate.is_dir():
            class_names, samples = _folder_samples(candidate)
            if class_names and samples:
                return {
                    "kind": "audio_folder",
                    "root": str(candidate),
                    "class_names": class_names,
                    "samples": samples,
                    "num_classes": len(class_names),
                }
    return None


def _stratified_folder_splits(samples, seed=42):
    rng = random.Random(seed)
    by_class = {}
    for index, sample in enumerate(samples):
        by_class.setdefault(sample[1], []).append(index)
    split_indices = {"train": [], "valid": [], "test": []}
    for class_indices in by_class.values():
        rng.shuffle(class_indices)
        count = len(class_indices)
        if count < 3:
            split_indices["train"].extend(class_indices)
            continue
        valid_count = max(1, round(count * 0.1))
        test_count = max(1, round(count * 0.1))
        split_indices["test"].extend(class_indices[:test_count])
        split_indices["valid"].extend(class_indices[test_count : test_count + valid_count])
        split_indices["train"].extend(class_indices[test_count + valid_count :])
    return {
        "strategy": "stratified_audio_folder",
        "seed": seed,
        "train": sorted(split_indices["train"]),
        "valid": sorted(split_indices["valid"]),
        "test": sorted(split_indices["test"]),
    }


def get_split_metadata(dataset_dir, seed=42):
    descriptor = detect_audio_dataset(dataset_dir)
    if not descriptor:
        return None
    if descriptor["kind"] == "esc50":
        return {
            "strategy": "esc50_folds",
            "seed": seed,
            "train_folds": [1, 2, 3],
            "valid_folds": [4],
            "test_folds": [5],
        }
    if descriptor["kind"] == "urbansound8k":
        return {
            "strategy": "urbansound8k_folds",
            "seed": seed,
            "train_folds": list(range(1, 9)),
            "valid_folds": [9],
            "test_folds": [10],
        }
    return _stratified_folder_splits(descriptor["samples"], seed=seed)


class AudioClassificationDataset(Dataset):
    def __init__(self, samples, transform=None, train=False):
        self.samples = samples
        self.targets = [sample[1] for sample in samples]
        self.transform = transform
        self.train = train

    def __len__(self):
        return len(self.samples)

    def _load_features(self, path):
        key = str(path)
        if key not in _FEATURE_CACHE:
            waveform, sample_rate = _read_wav(path)
            waveform = _resample(waveform, sample_rate)
            waveform = _fit_clip(waveform, train=False)
            features = waveform_to_log_mel(waveform)
            _FEATURE_CACHE[key] = (features - features.mean()) / features.std().clamp_min(1e-5)
        return _FEATURE_CACHE[key].clone()

    def __getitem__(self, index):
        path, label, _, _ = self.samples[index]
        features = self._load_features(path)
        if self.train:
            if random.random() < 0.5:
                features = features * random.uniform(0.85, 1.15)
            if random.random() < 0.25:
                features = features + torch.randn_like(features) * 0.02
            if random.random() < 0.25:
                width = random.randint(4, 18)
                start = random.randint(0, max(0, features.shape[-1] - width))
                features[:, :, start : start + width] = 0
        if self.transform:
            features = self.transform(features)
        return features, label


def _split_samples(descriptor, split_metadata, evaluation_split):
    samples = descriptor["samples"]
    strategy = split_metadata["strategy"]
    if strategy == "stratified_audio_folder":
        train_samples = [samples[index] for index in split_metadata["train"]]
        valid_samples = [samples[index] for index in split_metadata["valid"]]
        test_samples = [samples[index] for index in split_metadata["test"]]
    else:
        train_folds = set(split_metadata["train_folds"])
        valid_folds = set(split_metadata["valid_folds"])
        test_folds = set(split_metadata["test_folds"])
        train_samples = [sample for sample in samples if sample[3] in train_folds]
        valid_samples = [sample for sample in samples if sample[3] in valid_folds]
        test_samples = [sample for sample in samples if sample[3] in test_folds]
    evaluation_samples = valid_samples if evaluation_split == "valid" else test_samples
    return train_samples, evaluation_samples


def get_audio_metadata(dataset_dir, seed=42):
    descriptor = detect_audio_dataset(dataset_dir)
    if not descriptor:
        raise ValueError("No supported audio dataset found")
    split_metadata = get_split_metadata(dataset_dir, seed=seed)
    train_samples, valid_samples = _split_samples(descriptor, split_metadata, "valid")
    _, test_samples = _split_samples(descriptor, split_metadata, "test")
    return {
        "format": f"audio_{descriptor['kind']}",
        "modality": "audio",
        "dataset_root": descriptor["root"],
        "class_names": descriptor["class_names"],
        "class_to_idx": {name: index for index, name in enumerate(descriptor["class_names"])},
        "num_classes": descriptor["num_classes"],
        "num_samples": len(descriptor["samples"]),
        "sample_rate": TARGET_SAMPLE_RATE,
        "clip_seconds": CLIP_SECONDS,
        "splits": {
            "train": len(train_samples),
            "valid": len(valid_samples),
            "test": len(test_samples),
        },
        "split_strategy": split_metadata,
    }


def get_audio_dataloaders(
    batch_size=32,
    dataset_dir=None,
    split_metadata=None,
    seed=42,
    evaluation_split="valid",
):
    if evaluation_split not in {"valid", "test"}:
        raise ValueError("evaluation_split must be 'valid' or 'test'")
    descriptor = detect_audio_dataset(dataset_dir)
    if not descriptor:
        raise ValueError("No supported audio dataset found")
    split_metadata = split_metadata or get_split_metadata(dataset_dir, seed=seed)
    train_samples, evaluation_samples = _split_samples(descriptor, split_metadata, evaluation_split)
    trainset = AudioClassificationDataset(train_samples, train=True)
    evaluationset = AudioClassificationDataset(evaluation_samples, train=False)
    generator = torch.Generator().manual_seed(seed)
    trainloader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    evaluation_loader = DataLoader(
        evaluationset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    return trainloader, evaluation_loader, descriptor["num_classes"]


def audio_sample_from_bytes(data):
    """Decode an uploaded WAV payload into the same log-mel tensor as training."""
    import io

    with wave.open(io.BytesIO(data), "rb") as handle:
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        raw = handle.readframes(handle.getnframes())
    if sample_width != 2:
        raise ValueError("Uploaded audio must be a 16-bit PCM WAV file")
    waveform = torch.frombuffer(bytearray(raw), dtype=torch.int16).float() / 32768.0
    if channels > 1:
        waveform = waveform[: waveform.numel() - (waveform.numel() % channels)]
        waveform = waveform.reshape(-1, channels).mean(dim=1)
    waveform = _resample(waveform, sample_rate)
    waveform = _fit_clip(waveform, train=False)
    features = waveform_to_log_mel(waveform)
    return (features - features.mean()) / features.std().clamp_min(1e-5)
