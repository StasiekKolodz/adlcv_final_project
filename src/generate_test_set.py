import csv
import hashlib
import json
import random
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from diffusers import DDPMScheduler, UNet2DModel
from torchvision import transforms
from tqdm.auto import tqdm

# --- CONFIGURATION ---
CHECKPOINT_DIR = "ddpm_text_model/best_model"
DATASET_DIR = "stripe_text_dataset"
TEST_MANIFEST = "splits/test_manifest.jsonl"  # JSONL or CSV; must include image_path and text
OUTPUT_DIR = "test_outputs"
METADATA_LOG_PATH = "test_outputs/metadata_log.jsonl"

# RePaint parameters
START_TIMESTEP = 250
RESAMPLING_NUMBER = 3
SEED = 42
PIXELS_PER_CHAR = 9.6
WORD_PADDING_PX = 2  # Left/right whitespace padding to prevent glyph-edge leakage

# Image geometry
STRIPE_WIDTH = 1024
STRIPE_HEIGHT = 16
FOLDED_SIZE = 128

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ---------------------


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stable_u32_seed(key: str, base_seed: int) -> int:
    digest = hashlib.sha256(f"{base_seed}|{key}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def load_manifest_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")

    rows: list[dict] = []
    suffix = path.suffix.lower()

    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSONL at line {i} in {path}: {e}") from e
                image_path = item.get("image_path")
                text = item.get("text")
                if not image_path or text is None:
                    raise ValueError(
                        f"Manifest line {i} missing required keys image_path/text: {item}"
                    )
                rows.append({"image_path": str(image_path), "text": str(text)})
        return rows

    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            required = {"image_path", "text"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"CSV manifest missing required columns: {sorted(missing)}")
            for i, row in enumerate(reader, start=2):
                image_path = (row.get("image_path") or "").strip()
                text = row.get("text")
                if not image_path or text is None:
                    raise ValueError(f"CSV line {i} missing image_path/text")
                rows.append({"image_path": image_path, "text": str(text)})
        return rows

    raise ValueError(
        f"Unsupported manifest extension for {path}. Use .jsonl or .csv with image_path,text"
    )


def select_target_word(full_text: str, rng: random.Random) -> dict | None:
    words = [m for m in re.finditer(r"[A-Za-z]+", full_text)]
    if not words:
        return None

    target = rng.choice(words)
    return {
        "word": target.group(),
        "char_start": target.start(),
        "length": len(target.group()),
    }


def to_folded(tensor_stripe: torch.Tensor) -> torch.Tensor:
    # (1, 1, 16, 1024) -> (1, 1, 128, 128)
    return tensor_stripe.view(1, 1, FOLDED_SIZE, FOLDED_SIZE)


def to_stripe(tensor_folded: torch.Tensor) -> torch.Tensor:
    # (1, 1, 128, 128) -> (1, 1, 16, 1024)
    return tensor_folded.view(1, 1, STRIPE_HEIGHT, STRIPE_WIDTH)


def build_word_mask_folded(
    device: str,
    char_start: int,
    word_length: int,
    pixels_per_char: float,
    padding_px: int,
) -> torch.Tensor:
    if word_length <= 0:
        raise ValueError(f"word_length must be positive, got {word_length}")

    mask_stripe = torch.ones((1, 1, STRIPE_HEIGHT, STRIPE_WIDTH), device=device)

    pixel_start = int(char_start * pixels_per_char) - padding_px
    pixel_end = int((char_start + word_length) * pixels_per_char) + padding_px

    pixel_start = max(0, pixel_start)
    pixel_end = min(STRIPE_WIDTH, pixel_end)

    if pixel_start >= pixel_end:
        raise ValueError(
            f"Computed empty mask range [{pixel_start}, {pixel_end}) for char_start={char_start}, "
            f"word_length={word_length}"
        )

    mask_stripe[:, :, :, pixel_start:pixel_end] = 0.0
    return to_folded(mask_stripe)


def save_folded_as_stripe_png(tensor_folded: torch.Tensor, output_path: Path) -> None:
    stripe = to_stripe(tensor_folded)
    vis = (stripe / 2 + 0.5).clamp(0, 1)
    pil = transforms.ToPILImage()(vis[0].cpu()).convert("L")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pil.save(output_path, format="PNG", optimize=True)


def append_jsonl(log_path: Path, metadata: dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(metadata, ensure_ascii=True) + "\n")


def load_logged_output_paths(log_path: Path) -> set[str]:
    if not log_path.exists():
        return set()

    seen: set[str] = set()
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                # Keep going; JSONL robustness means one bad line should not block resume.
                continue
            out = item.get("output_path")
            if isinstance(out, str) and out:
                seen.add(out)
    return seen


def make_output_name(image_path: str, char_start: int, word: str) -> str:
    stem = Path(image_path).stem
    safe_word = re.sub(r"[^A-Za-z0-9]+", "_", word.lower()).strip("_") or "word"
    return f"{stem}__c{char_start}__{safe_word}.png"


@torch.no_grad()
def run_repaint(
    model: UNet2DModel,
    scheduler: DDPMScheduler,
    clean_folded: torch.Tensor,
    mask_folded: torch.Tensor,
    start_timestep: int,
    resampling_number: int,
    generator: torch.Generator,
) -> torch.Tensor:
    if start_timestep < 1 or start_timestep >= scheduler.config.num_train_timesteps:
        raise ValueError(
            f"START_TIMESTEP must be in [1, {scheduler.config.num_train_timesteps - 1}], got {start_timestep}"
        )

    # Pre-blank unknown region with white pixels before initial noising.
    white = torch.ones_like(clean_folded)
    masked_clean = (mask_folded * clean_folded) + ((1.0 - mask_folded) * white)

    t0 = torch.tensor([start_timestep], device=clean_folded.device, dtype=torch.long)
    x_t = scheduler.add_noise(masked_clean, torch.randn_like(masked_clean, generator=generator), t0)

    for step_t in range(start_timestep, 0, -1):
        t = torch.tensor([step_t], device=clean_folded.device, dtype=torch.long)
        t_prev = torch.tensor([step_t - 1], device=clean_folded.device, dtype=torch.long)

        for rep in range(resampling_number):
            noise_pred = model(x_t, t, return_dict=False)[0]
            x_t_minus_1_unknown = scheduler.step(noise_pred, step_t, x_t).prev_sample

            known_noise = torch.randn_like(clean_folded, generator=generator)
            x_t_minus_1_known = scheduler.add_noise(clean_folded, known_noise, t_prev)

            x_t_minus_1 = (mask_folded * x_t_minus_1_known) + ((1.0 - mask_folded) * x_t_minus_1_unknown)

            if rep < resampling_number - 1:
                beta_t = scheduler.betas[step_t].to(
                    device=clean_folded.device, dtype=x_t_minus_1.dtype
                ).view(1, 1, 1, 1)
                jump_noise = torch.randn_like(x_t_minus_1, generator=generator)
                x_t = torch.sqrt(1.0 - beta_t) * x_t_minus_1 + torch.sqrt(beta_t) * jump_noise
            else:
                x_t = x_t_minus_1

    return x_t


def main() -> None:
    set_global_seeds(SEED)

    checkpoint_dir = Path(CHECKPOINT_DIR)
    dataset_dir = Path(DATASET_DIR)
    manifest_path = Path(TEST_MANIFEST)
    output_dir = Path(OUTPUT_DIR)
    log_path = Path(METADATA_LOG_PATH)

    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"CHECKPOINT_DIR not found: {checkpoint_dir}")
    if not dataset_dir.exists():
        raise FileNotFoundError(f"DATASET_DIR not found: {dataset_dir}")

    rows = load_manifest_rows(manifest_path)
    if not rows:
        raise RuntimeError(f"No rows loaded from manifest: {manifest_path}")

    print(f"Loading checkpoint from: {checkpoint_dir}")
    model = UNet2DModel.from_pretrained(str(checkpoint_dir)).to(DEVICE)
    scheduler = DDPMScheduler.from_pretrained(str(checkpoint_dir))
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    logged_outputs = load_logged_output_paths(log_path)

    produced = 0
    skipped_existing = 0
    skipped_no_word = 0
    recovered_log_entries = 0
    errors = 0

    pbar = tqdm(rows, desc="Generating test set", total=len(rows))
    for row in pbar:
        image_rel = row["image_path"]
        full_text = row["text"]
        image_path = dataset_dir / image_rel

        if not image_path.exists():
            errors += 1
            pbar.set_postfix({"errors": errors})
            continue

        # Deterministic target-word selection per sample independent of processing order.
        sample_seed = stable_u32_seed(f"select|{image_rel}|{full_text}", SEED)
        sample_rng = random.Random(sample_seed)
        target = select_target_word(full_text, sample_rng)
        if target is None:
            skipped_no_word += 1
            pbar.set_postfix({"skip_no_word": skipped_no_word})
            continue

        output_name = make_output_name(image_rel, target["char_start"], target["word"])
        output_path = output_dir / output_name

        metadata = {
            "image_id": Path(image_rel).stem,
            "original_text": full_text,
            "masked_word": target["word"],
            "masked_word_length": target["length"],
            "char_start_idx": target["char_start"],
            "output_path": str(output_path),
        }

        # Resume-safe mode: skip existing outputs, but recover missing metadata if needed.
        if output_path.exists():
            skipped_existing += 1
            if metadata["output_path"] not in logged_outputs:
                append_jsonl(log_path, metadata)
                logged_outputs.add(metadata["output_path"])
                recovered_log_entries += 1
            pbar.set_postfix({"skip_existing": skipped_existing, "recovered": recovered_log_entries})
            continue

        try:
            img = Image.open(image_path).convert("L")
            clean_stripe = transform(img).unsqueeze(0).to(DEVICE)  # (1, 1, 16, 1024)

            expected = (1, 1, STRIPE_HEIGHT, STRIPE_WIDTH)
            if tuple(clean_stripe.shape) != expected:
                raise ValueError(f"Image shape mismatch for {image_path}: {tuple(clean_stripe.shape)} != {expected}")

            clean_folded = to_folded(clean_stripe)
            mask_folded = build_word_mask_folded(
                device=DEVICE,
                char_start=target["char_start"],
                word_length=target["length"],
                pixels_per_char=PIXELS_PER_CHAR,
                padding_px=WORD_PADDING_PX,
            )

            infer_seed = stable_u32_seed(f"infer|{image_rel}|{target['char_start']}|{target['word']}", SEED)
            infer_gen = torch.Generator(device=DEVICE).manual_seed(infer_seed)

            inpainted_folded = run_repaint(
                model=model,
                scheduler=scheduler,
                clean_folded=clean_folded,
                mask_folded=mask_folded,
                start_timestep=START_TIMESTEP,
                resampling_number=RESAMPLING_NUMBER,
                generator=infer_gen,
            )

            save_folded_as_stripe_png(inpainted_folded, output_path)
            append_jsonl(log_path, metadata)
            logged_outputs.add(metadata["output_path"])
            produced += 1

            # Avoid flushing allocator cache every iteration; do occasional cleanup only.
            if DEVICE == "cuda" and produced % 100 == 0:
                torch.cuda.empty_cache()

            pbar.set_postfix({"produced": produced, "errors": errors})
        except Exception:
            errors += 1
            pbar.set_postfix({"errors": errors})
            continue

    print("Done.")
    print(f"Manifest rows          : {len(rows)}")
    print(f"Produced new outputs   : {produced}")
    print(f"Skipped existing output: {skipped_existing}")
    print(f"Recovered log entries  : {recovered_log_entries}")
    print(f"Skipped no-word rows   : {skipped_no_word}")
    print(f"Errors                 : {errors}")
    print(f"Output dir             : {output_dir}")
    print(f"Metadata log           : {log_path}")


if __name__ == "__main__":
    main()
