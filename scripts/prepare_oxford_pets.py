"""Convert the Oxford-IIIT Pet archive into an ImageFolder NEUROFORGE can mount.

Source: https://www.robots.ox.ac.uk/~vgg/data/pets/ (37 cat and dog breeds,
~200 images each). Filenames encode the breed: ``Abyssinian_12.jpg`` and
``american_bulldog_7.jpg``. Images are resized so the short side is 256px,
which keeps every model's input resolution available (DynamicCNN uses 64-128,
pretrained backbones 224) while making each epoch far cheaper to decode.

    python scripts/prepare_oxford_pets.py datasets/_downloads/oxford-pets/images.tar.gz datasets/oxford-pets
"""

import argparse
import io
import re
import sys
import tarfile
from collections import Counter
from pathlib import Path

from PIL import Image, UnidentifiedImageError

NAME = re.compile(r"^(?P<breed>[A-Za-z_]+?)_(?P<index>\d+)\.jpe?g$")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--short-side", type=int, default=256)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    counts, skipped = Counter(), []
    with tarfile.open(args.archive, "r:gz") as archive:
        for member in archive:
            if not member.isfile():
                continue
            name = Path(member.name).name  # never trust archive paths
            match = NAME.match(name)
            if not match:
                continue
            breed = match["breed"]
            data = archive.extractfile(member).read()
            try:
                image = Image.open(io.BytesIO(data))
                image.load()
                image = image.convert("RGB")  # a few files are CMYK, greyscale or RGBA PNGs named .jpg
            except (UnidentifiedImageError, OSError) as error:
                skipped.append(f"{name}: {error}")
                continue
            scale = args.short_side / min(image.size)
            if scale < 1:
                image = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)
            folder = args.output / breed
            folder.mkdir(exist_ok=True)
            image.save(folder / name, quality=92)
            counts[breed] += 1

    print(f"{sum(counts.values())} images in {len(counts)} breeds -> {args.output}")
    print(f"smallest class {min(counts.values())}, largest {max(counts.values())}")
    if skipped:
        print(f"skipped {len(skipped)} unreadable files:", *skipped[:10], sep="\n  ")
    return 0 if counts else 1


if __name__ == "__main__":
    sys.exit(main())
