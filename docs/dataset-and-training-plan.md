# NEUROFORGE dataset and training plan

## Recommendation

Use two clean, well-known public benchmarks instead of the current small fruit export for the next presentation/research run:

### Image: Oxford-IIIT Pet

- 37 cat and dog breeds.
- Roughly 200 images per breed.
- Good for a visually clear classification demo without the very large compute cost of a 100-class dataset.
- Best path: use a pretrained image backbone for the benchmark, then compare it with the generated DynamicCNN.

Official source: <https://www.robots.ox.ac.uk/~vgg/data/pets/>

### Audio: ESC-50

- 2,000 environmental recordings.
- 50 classes.
- Five predefined folds, so evaluation can be reproducible instead of relying on a random split.
- WAV files are directly compatible with the current audio path.

Official source: <https://github.com/karolpiczak/ESC-50>

For a smaller first audio demo, export only the ESC-10 subset listed in the ESC-50 metadata into class folders, then move to all 50 classes after the pipeline is stable.

UrbanSound8K is also supported by the adapter and is a good 10-class alternative, but its official protocol requires 10-fold evaluation over the predefined folds. Use it for a benchmark only after adding the full cross-validation runner; do not headline a single fold-10 result as the final score.

## Full image workflow

1. Download Oxford-IIIT Pet from the official source.
2. Convert the dataset to the supported class-folder layout:

   ```text
   oxford-pets/
   ├── Abyssinian/
   │   ├── image-001.jpg
   │   └── image-002.jpg
   ├── Bengal/
   └── ...
   ```

3. Keep a fixed train/validation/test split and record the seed.
4. In NEUROFORGE, choose **Data → Images**.
5. Enter the absolute dataset path and select **Connect dataset**.
6. Run a small search first: 2 generations × 4 candidates × 2 proxy epochs.
7. Retrain the selected DynamicCNN for 15–25 epochs.
8. Run the image sample test and export the weights, config, and report.
9. Repeat with the image transfer-learning baseline and report both results separately.

## Full audio workflow

1. Download or clone ESC-50:

   ```bash
   git clone https://github.com/karolpiczak/ESC-50.git data/ESC-50
   ```

2. In NEUROFORGE, choose **Data → Audio**.
3. Enter the path to the extracted `ESC-50` folder.
4. The audio adapter reads `meta/esc50.csv`, `audio/*.wav`, and the predefined folds.
5. The current model converts each clip to a normalized log-mel spectrogram.
6. Run the small search first to confirm the pipeline.
7. Retrain the selected DynamicCNN with 15–30 epochs.
8. Use **Try a sample** to upload a 16-bit PCM WAV clip.
9. Export the audio checkpoint and metadata separately from image runs.

## How to increase accuracy without weakening evaluation

1. Keep a fixed, untouched test fold. Never tune against it.
2. Use the validation fold for search and early stopping.
3. Report accuracy and macro F1 together.
4. Report `macro F1 over present test classes` when a split has no example for a class, and show the evaluated class count.
5. Use multiple seeds for the final benchmark, not only one lucky run.
6. Use transfer learning only as a separately labelled baseline; do not mix its result with the generated DynamicCNN result.
7. Search training settings as well as architecture settings: learning rate, weight decay, augmentation strength, and scheduler.
8. Increase input resolution only after checking that the extra detail improves validation performance.

## How to reduce runtime without compromising the final check

- Cache duplicate candidate evaluations during a search run.
- Reuse decoded audio features across candidate models.
- Use a small proxy budget for exploration, then train only the selected candidates deeply.
- Use early stopping on validation macro F1.
- Keep the final held-out test evaluation unchanged.
- Do not parallelize candidates on the same MPS device unless a benchmark proves it helps; simultaneous models can compete for the same accelerator memory.

## Suggested experiment matrix

| Run | Dataset | Search | Final training | Purpose |
| --- | --- | --- | --- | --- |
| A | Oxford-IIIT Pet | 2 × 4 × 2 | 20 epochs | Fast image pipeline check |
| B | Oxford-IIIT Pet | 5 × 8 × 3 | 20–30 epochs | Image search result |
| C | ESC-10 subset | 2 × 4 × 2 | 15 epochs | Fast audio pipeline check |
| D | ESC-50 | 5 × 8 × 3 | 20–30 epochs | Full audio benchmark |

The goal is not to promise a fixed percentage before measuring. The goal is to make the comparison fair, repeatable, and fast enough to iterate.
