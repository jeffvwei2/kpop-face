# KPop Classifier

Face classifier for K-pop idols. Upload a photo; it finds the face, crops it, and returns the top-5 most likely idols.

## Data

Training uses two merged datasets, with labels in `data/combined_train.csv` / `combined_test.csv` (`file_name` relative to `data/`):

- `kid_f_*`: the original 512×512 dataset (`data/images/HQ_512x512`, `data/test_final_with_degrad`). The test images are deliberately degraded.
- `Kpop_Faces`: organized as `<group>/<person>/`. `build_kpop_faces_csv.py` turns it into CSVs. A label is just the person's name unless two groups share that stage name, in which case it becomes `<group>_<person>`.

People who appear in both datasets are merged into one class. **Only classes present in both train and test are used**, currently 98 of 127. A class missing from the test CSV is silently dropped from training too.

Class balance is the main difficulty. The 98 classes range from **1 to 1,292 images (median 13)**. `curation_agent.py --list-underrepresented` lists the worst ones.

## Model

`KpopClassifier.py`: an ImageNet-pretrained EfficientNet-B0 backbone, then an embedding head (1280 → 640 → 512), then L2 normalization.

It's trained with an **ArcFace loss** (`index.py`), not plain softmax. ArcFace has its own per-class weight vectors and adds an angular margin to the correct class. This pushes embeddings of the same person tightly together on the unit sphere, which suits faces: the differences between people are small, and many classes have very few examples.

Inference doesn't use a separate classifier. A face is scored by cosine similarity between its embedding and each class's ArcFace weight vector (without the margin), scaled and softmaxed. `api.py` does the same. Because of this:

- The checkpoint stores both `model` and `criterion` weights. Without the ArcFace weights, the model can't classify.
- The small MLP classifier head is only built for the cross-entropy fallback (`use_arcface = False`). With ArcFace it's left out. The model's weights then don't depend on how many classes there are, so after adding classes, `index.py` still resumes the trained backbone and embedding, and only the ArcFace class weights start fresh. Older checkpoints that include the head still load; its weights are skipped.

## Training

```bash
python index.py
```

Expects an Apple Silicon GPU (`device = torch.device("mps")` in `index.py`); change it to `"cuda"` or `"cpu"` elsewhere. Set `QUICK_RUN = True` for a 5-epoch run with a capped test set.

What `index.py` does about the class imbalance and the small dataset:

- **Per-class validation split.** 20% of each class is held out, not 20% overall, so rare classes still appear in validation. A class with a single image can't be split, so it's trained on but never validated.
- **Balanced sampling.** A `WeightedRandomSampler` draws classes evenly. Otherwise the few classes with 1,000+ images would dominate the gradients.
- **Macro validation accuracy** is printed next to overall accuracy. Overall accuracy is dominated by the big classes. Macro accuracy (the average of per-class accuracies) shows whether the small classes are being learned.
- **Frozen backbone first.** For the first 3 epochs only the head trains (fast, and it doesn't disturb the pretrained features). Then everything trains at 1e-4. The head is kept at the same learning rate so it doesn't jump at the moment the backbone unfreezes.
- Learning rate halves after 5 epochs without improvement; training stops early after 15. There's light augmentation (crop, flip, ±10° rotation, color jitter, random erasing).

### Checkpoints and resuming

The best model is saved to `best_model.pth`, and its score to `metrics.json`. Re-running `index.py` **continues from `best_model.pth`**, and only overwrites it when validation accuracy beats the score in `metrics.json`.

`metrics.json` also records which dataset the score came from: the class list and the number of training images. If either has changed since (for example, after adding curated images), or the checkpoint couldn't be loaded, the old score is ignored and this run's best is saved. The old score was measured on a different validation set, so it isn't a fair comparison.

Past results (overall validation accuracy):

| Classes | Best val acc | File |
|---|---|---|
| 23 | 93.5% | `metrics_23class.json` |
| 98 | 87.9% | `metrics.json` |

## Web interface

A React + Vite frontend (`web/`) with a small Flask API (`api.py`, port 5001). The API uses MediaPipe to find the face, crops it with 25% padding, and returns the top-5 predictions. It runs on CPU and needs `best_model.pth`, `models/blaze_face_short_range.tflite`, and the two combined CSVs (it reads the class list from them).

```bash
pip install torch torchvision flask flask-cors mediapipe pandas pillow
cd web
npm install
npm run dev   # starts api.py and the Vite dev server together
```

Open the URL Vite prints. The frontend sends `/predict` to the API through the Vite proxy.

## Growing the training data

Many classes have only a handful of images. Two scripts help find more, without adding mislabeled ones.

### `wiki_search.py`: find candidate images (free)

Searches Wikimedia Commons and the Kpop Fandom wiki, where images are already labeled by person (Commons person categories, Fandom `/Gallery` pages). No API key or credits needed.

```bash
python wiki_search.py yujin --group ive
python wiki_search.py liz --group ive --source fandom --limit 50
python wiki_search.py yujin --group ive --refresh   # ignore the cache and re-query
```

- **Always pass `--group`.** It's used to tell apart idols with the same stage name (e.g. IVE's Yujin vs Kep1er's Yujin). Check the page/category the script prints to confirm it found the right person.
- Use `--person "An Yujin"` if the search name differs from the training label.
- Results are cached in `data/candidates/<label>.json` with each image's URL, size and source page. Re-runs read the cache.
- Requests are rate-limited to one per second to stay within Wikimedia's limits.

These are **candidates only**. Some will be group shots or unusable crops, so filter them before training (see below).

### `curation_agent.py`: download and crop new training images

```bash
python curation_agent.py --list-underrepresented --min-images 30   # classes that need more data
python curation_agent.py --label miyeon                            # curate until the class has 20 new images
python curation_agent.py --label yujin --group ive --target 10     # --group when it can't be inferred
```

For one class, it gets candidates from `wiki_search.py` (cached, free), then for each candidate:

1. Downloads it **into memory only**. The full-size image is never written to disk.
2. Keeps it only if MediaPipe finds **exactly one face** at least 80px across. Group shots, signatures and faces that are too small or far away are rejected.
3. Saves a square face crop, resized to **256×256 JPEG (~15–25 KB)**, as `data/Kpop_Faces/<group>/<person>/curated_<hash>.jpg`. Training resizes to 256 anyway, so nothing is lost.

It stops once the class has `--target` curated images in total (default 20; earlier runs count, so re-running is safe). Candidates are shuffled first, so the picks aren't all from one photoshoot.

- **Group:** taken from the `data/Kpop_Faces/<group>/<person>/` folder. Pass `--group` when there's no folder or the name is in more than one group. Group renames and alternate spellings (gidle → i-dle, kepler → Kep1er, snsd → Girls' Generation) are mapped in `GROUP_ALIASES` in `wiki_search.py`.
- **Output:** accepted images are listed in `data/curated_train.csv` with their source URL and page. Every URL tried, and why it was rejected, goes to `data/curation_log.csv`, so re-runs skip images already tried.

**New images are not used for training automatically.** `index.py` only reads `combined_train.csv`. To use them, review the crops, delete any bad ones, and add the remaining rows (`file_name,name`) to `combined_train.csv`.

**Not done yet:** checking each crop against hand-picked reference images (`data/gold_anchors/<label>/`) with a pretrained face model, to catch wrong-person images. Until then, rows are saved with `verified=False`, and manual review is the only check on identity.

#### Optional: LLM search (uses API credits)

```bash
python curation_agent.py --label yujin --search-only
```

Prints candidate pages found by web search, for idols the wikis don't cover. Works with OpenAI or Anthropic. Choose with `--provider openai|anthropic` or `CURATION_PROVIDER`; otherwise whichever of `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` is set is used (OpenAI first). Override the model with `--model` or `CURATION_MODEL`. Keys can go in `.env`. It isn't wired into the download step.
