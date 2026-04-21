# Project Scripts Guide (`src/`)

This document explains what each script in `src/` does and how they connect in the end-to-end pipeline.

## Big Picture

The workflow has three major phases:

1. Data preparation
2. Model training and checkpoint inspection
3. Inpainting evaluation (generation + OCR scoring)

### High-level flow

```text
 download_data.py
        |
        v
 generate_imgs_paralel.py  ---> stripe_text_dataset/*.png
        |
        v
 create_splits.py ---------> splits/train.txt, val.txt, test.txt
        |
        +--> train.py ------> checkpoints/latest,best + ddpm_text_model[/best_model]
        |
        +--> reconstruct_manifest.py OR build_manifest_via_ocr.py
                   |
                   v
            splits/test_manifest.jsonl
                   |
                   v
          generate_test_set.py ------> test_outputs/*.png + metadata_log.jsonl
                   |
                   v
             evaluate_ocr.py -------> evaluation_results.csv
```

---

## Script-by-Script

## 1) Data download and stripe generation

### `download_data.py`
- Downloads `VisionTheta/fineweb-1B` (`train` split) from Hugging Face.
- Saves it locally to `./local_fineweb` using `save_to_disk`.

Inputs:
- Hugging Face network dataset

Outputs:
- `local_fineweb/`

---

### `generate_imgs_paralel.py`
- Main production stripe generator.
- Loads local dataset from `./local_fineweb`.
- Uses `pixel_renderer` + local mono font to render text lines.
- Performs pixel-aware whole-word chunking to ensure chunks fit `16x1024`.
- Saves one or more images per dataset row using naming pattern:
  - `stripe_<rowidx:06d>_<chunkidx:02d>.png`
- Multiprocessing with spawn mode for stability.

Inputs:
- `local_fineweb/`
- `MONO_FONT_PATH`

Outputs:
- `stripe_text_dataset/*.png`

Notes:
- This is the canonical generator for the dataset used by training/evaluation.
- `NUM_IMAGES_TO_GENERATE` is currently very small in file config and should be adjusted for real runs.

---

### `generate_imgs.py`
- Older/simple streaming generator.
- Uses simpler truncation strategy (`MAX_CHARS`) rather than the newer pixel-aware chunking.

Recommendation:
- Prefer `generate_imgs_paralel.py` for dataset quality/consistency.

---

## 2) Splits and manifests

### `create_splits.py`
- Creates row-safe train/val/test splits based on filename pattern.
- Ensures chunks from the same source row stay in the same split by splitting on `row_id`.
- Writes:
  - `splits/train.txt`
  - `splits/val.txt`
  - `splits/test.txt`

Inputs:
- `stripe_text_dataset/*.png`

Outputs:
- split manifest text files in `splits/`

---

### `reconstruct_manifest.py`
- Reconstructs exact `image_path -> text` mapping for the test split by replaying chunking on original dataset text.
- Parses test filenames to `(row_idx, chunk_idx)` and rebuilds chunk text via renderer.
- Writes `splits/test_manifest.jsonl`.
- Logs failures to `splits/reconstruct_manifest_errors.jsonl`.

Inputs:
- `splits/test.txt`
- `local_fineweb/`

Outputs:
- `splits/test_manifest.jsonl`
- `splits/reconstruct_manifest_errors.jsonl` (if errors)

When to use:
- Use this when you want true reconstructed ground-truth text (preferred over OCR-derived pseudo labels).

---

### `build_manifest_via_ocr.py`
- Alternative manifest builder that OCRs existing stripe images to generate `text` values.
- Resume-safe append mode with skip logic.

Inputs:
- `splits/test.txt`
- `stripe_text_dataset/*.png`

Outputs:
- `splits/test_manifest.jsonl`

Caution:
- This produces pseudo ground truth (from OCR), not source-truth text.
- For rigorous evaluation/reporting, prefer `reconstruct_manifest.py` when possible.

---

### `verify_manifest.py`
- Quick visual sanity check utility.
- Samples a random row from `test_manifest.jsonl`, renders stripe + text on a canvas, and saves preview image.

Inputs:
- `splits/test_manifest.jsonl`
- `stripe_text_dataset/`

Outputs:
- `verification_outputs/verify_*.png`

---

## 3) Training and checkpoints

### `train.py`
- Trains diffusion model (`UNet2DModel` + `DDPMScheduler`) on folded stripes (`16x1024 -> 128x128`).
- Uses split manifests for train/val.
- Supports resume with Accelerator state:
  - `checkpoints/latest`
  - `checkpoints/best`
- Saves inference-ready models via `save_pretrained`:
  - `ddpm_text_model/`
  - `ddpm_text_model/best_model/` on best val improvement.

Inputs:
- `stripe_text_dataset/`
- `splits/train.txt`
- `splits/val.txt`

Outputs:
- `checkpoints/latest`, `checkpoints/best`
- `ddpm_text_model/`, `ddpm_text_model/best_model/`

Key distinction:
- `checkpoints/*` = resumable training state
- `ddpm_text_model*` = inference loading directories

---

### `inspect_checkpoint.py`
- Inspects either checkpoint type.
- Prints file inventory, trainer state, and attempts Diffusers load.

Use cases:
- Confirm whether directory is resume-state only or Diffusers-loadable.

---

## 4) Inference and inpainting utilities

### `test_checkpoint.py`
- Single-image denoising test (add noise at `NOISE_TIMESTEP`, denoise back to 0).
- Saves denoised image and a review panel (`original | noisy | denoised`).

Inputs:
- `ddpm_text_model` or `ddpm_text_model/best_model`
- image path or split-based selection

Outputs:
- `review_denoised.png`
- `review_panel.png`

---

### `repaint_inpaint.py`
- Single-image RePaint-style inpainting demo.
- Builds mask from character location and word length.
- Uses jump resampling loop and outputs panel (`original | masked input | inpainted | mask`).

Inputs:
- model checkpoint dir
- selected image
- manual mask config (`CHAR_START`, `WORD_LENGTH`)

Outputs:
- `repaint_inpainted.png`
- `repaint_panel.png`

---

### `generate_test_set.py`
- Main batch Script A for evaluation set generation.
- Reads `test_manifest` rows containing `image_path` and `text`.
- Deterministically selects a target word per sample.
- Runs RePaint inpainting and saves generated outputs.
- Crash-resilient resume behavior:
  - skips already existing outputs
  - appends metadata log
  - writes error log for failed samples

Inputs:
- `splits/test_manifest.jsonl` (or `.csv` with `image_path,text`)
- `stripe_text_dataset/`
- inference-ready checkpoint directory

Outputs:
- `test_outputs/*.png`
- `test_outputs/metadata_log.jsonl`
- `test_outputs/generate_test_set_errors.jsonl`

---

## 5) OCR evaluation

### `evaluate_ocr.py`
- Main Script B for OCR-based quantitative evaluation.
- Runs EasyOCR on generated outputs.
- Extracts predicted masked word using anchor-based alignment + difflib fallback.
- Computes:
  - word CER
  - binary word WER
  - global sentence CER
- Optional baseline OCR calibration on original source images (`--calibrate`).

Inputs:
- `test_outputs/metadata_log.jsonl`
- generated outputs from `generate_test_set.py`
- optional source dataset for baseline calibration

Outputs:
- `test_outputs/evaluation_results.csv`

---

## Recommended End-to-End Sequence

1. `python src/download_data.py`
2. `python src/generate_imgs_paralel.py`
3. `python src/create_splits.py`
4. `python src/train.py`
5. Build test manifest (prefer reconstruction):
   - `python src/reconstruct_manifest.py`
   - (alternative) `python src/build_manifest_via_ocr.py`
6. Optional sanity check:
   - `python src/verify_manifest.py`
7. Generate inpainted test set:
   - `python src/generate_test_set.py`
8. Evaluate OCR metrics:
   - `python src/evaluate_ocr.py --calibrate`

---

## Common Pitfalls

- Using wrong checkpoint type:
  - Resume training from `checkpoints/latest|best`.
  - Inference with `ddpm_text_model` or `ddpm_text_model/best_model`.

- Manifest quality:
  - `build_manifest_via_ocr.py` is convenient but may bias evaluation.
  - `reconstruct_manifest.py` is better for source-truth alignment.

- Environment imports in editor:
  - Some scripts may show unresolved imports unless VS Code interpreter points to the environment with required packages (`diffusers`, `accelerate`, `datasets`, `easyocr`, project local packages).
