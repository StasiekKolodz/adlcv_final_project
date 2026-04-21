import csv
import hashlib
import json
import random
import re
from contextlib import nullcontext
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
ERROR_LOG_PATH = "test_outputs/generate_test_set_errors.jsonl"
MAX_GENERATED_IMAGES = 100  # Set int (e.g. 1000) to cap newly generated outputs per run

# RePaint parameters
START_TIMESTEP = 500
RESAMPLING_NUMBER = 3
FAST_DEBUG_MODE = False       # True -> quick sanity settings below
DEBUG_START_TIMESTEP = 50
DEBUG_RESAMPLING_NUMBER = 1
SHOW_REPAINT_PROGRESS = True  # Shows inner diffusion progress so first item doesn't look stuck
SEED = 42
WORD_PADDING_PX = 1  # Left/right whitespace padding to prevent glyph-edge leakage

# Image geometry
STRIPE_WIDTH = 1024
STRIPE_HEIGHT = 16
FOLDED_SIZE = 128

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REQUIRE_CUDA = False  # Set True to fail fast instead of silently running very slow CPU inference

# Must match generate_imgs_paralel.py: the TTF and size that drove dataset rendering.
# The renderer uses Pango/Cairo with fontconfig aliasing "sans" → DejaVu Sans Mono.
# DejaVu Sans Mono at 16px has exactly 10px advance in Pango — PIL getlength() gives
# ~9.64px/char, causing a ~0.36px/char accumulation error (~31px at 86 chars = 3 chars off).
MONO_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
MONO_FONT_SIZE_PX = 16
RENDERER_LEFT_PAD_PX = 5  # pixel-renderer/pixel_renderer/renderer.py hardcodes x=5 before drawing
# ---------------------


# Use Pango (same library the renderer uses) for pixel-accurate measurements.
# PIL getlength() accumulates ~0.36px/char error vs Pango, causing 3-4 char drift
# over long prefixes (86 chars → ~31px off = ~3 chars too early).
_pango_layout = None
_pango_surface = None  # keep alive; Cairo surfaces are GC'd if not referenced

try:
    import cairo as _cairo
    import gi as _gi
    _gi.require_version("Pango", "1.0")
    _gi.require_version("PangoCairo", "1.0")
    _gi.require_foreign("cairo")
    from gi.repository import Pango as _Pango, PangoCairo as _PangoCairo
    _PANGO_AVAILABLE = True
except Exception:
    _PANGO_AVAILABLE = False

from PIL import ImageFont  # noqa: E402

_mono_font = None


def _ensure_pango_layout():
    global _pango_layout, _pango_surface
    if _pango_layout is None:
        _pango_surface = _cairo.ImageSurface(_cairo.FORMAT_RGB24, 1, 1)
        ctx = _cairo.Context(_pango_surface)
        _pango_layout = _PangoCairo.create_layout(ctx)
        font_desc = _Pango.font_description_from_string(f"DejaVu Sans Mono {MONO_FONT_SIZE_PX}px")
        _pango_layout.set_font_description(font_desc)


def _prefix_width_px(text: str) -> int:
    if not text:
        return 0
    if _PANGO_AVAILABLE:
        _ensure_pango_layout()
        _pango_layout.set_text(text, -1)
        width, _ = _pango_layout.get_pixel_size()
        return width
    # Fallback: PIL (may drift ~0.36px/char for long prefixes)
    global _mono_font
    if _mono_font is None:
        _mono_font = ImageFont.truetype(MONO_FONT_PATH, MONO_FONT_SIZE_PX)
    return int(round(_mono_font.getlength(text)))
# ---------------------


def resolve_checkpoint_dir(configured_path: str) -> Path:
    configured = Path(configured_path)
    if configured.exists():
        return configured

    # Common fallback: if best_model is missing, use the base save_pretrained directory.
    if configured.name == "best_model" and configured.parent.exists():
        print(
            f"Configured checkpoint '{configured}' was not found. "
            f"Falling back to '{configured.parent}'."
        )
        return configured.parent

    raise FileNotFoundError(f"CHECKPOINT_DIR not found: {configured}")


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
    text: str,
    char_start: int,
    word_length: int,
    padding_px: int,
) -> torch.Tensor:
    pixel_start, pixel_end = compute_mask_pixel_range(
        text=text,
        char_start=char_start,
        word_length=word_length,
        padding_px=padding_px,
    )

    mask_stripe = torch.ones((1, 1, STRIPE_HEIGHT, STRIPE_WIDTH), device=device)
    mask_stripe[:, :, :, pixel_start:pixel_end] = 0.0
    return to_folded(mask_stripe)


def compute_mask_pixel_range(
    text: str,
    char_start: int,
    word_length: int,
    padding_px: int,
) -> tuple[int, int]:
    if word_length <= 0:
        raise ValueError(f"word_length must be positive, got {word_length}")
    if char_start < 0 or char_start + word_length > len(text):
        raise ValueError(
            f"Word range [{char_start}, {char_start + word_length}) out of bounds for text of length {len(text)}"
        )

    # Measure real pixel span via the same Pango layout the renderer uses.
    prefix_w = _prefix_width_px(text[:char_start])
    end_w = _prefix_width_px(text[:char_start + word_length])

    pixel_start = RENDERER_LEFT_PAD_PX + prefix_w - padding_px
    pixel_end = RENDERER_LEFT_PAD_PX + end_w + padding_px

    pixel_start = max(0, pixel_start)
    pixel_end = min(STRIPE_WIDTH, pixel_end)

    if pixel_start >= pixel_end:
        raise ValueError(
            f"Computed empty mask range [{pixel_start}, {pixel_end}) for char_start={char_start}, "
            f"word_length={word_length}"
        )

    return pixel_start, pixel_end


def save_folded_as_stripe_png(tensor_folded: torch.Tensor, output_path: Path) -> None:
    stripe = to_stripe(tensor_folded)
    vis = (stripe / 2 + 0.5).clamp(0, 1)
    # Output must be cast back to float32 for PIL conversion
    pil = transforms.ToPILImage()(vis[0].cpu().to(torch.float32)).convert("L")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pil.save(output_path, format="PNG", optimize=True)


def append_jsonl(log_path: Path, metadata: dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(metadata, ensure_ascii=True) + "\n")


def append_error_jsonl(error_log_path: Path, error_row: dict) -> None:
    error_log_path.parent.mkdir(parents=True, exist_ok=True)
    with error_log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(error_row, ensure_ascii=True) + "\n")


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
                continue
            out = item.get("output_path")
            if isinstance(out, str) and out:
                seen.add(out)
    return seen


def make_output_name(image_path: str, char_start: int, word: str) -> str:
    stem = Path(image_path).stem
    safe_word = re.sub(r"[^A-Za-z0-9]+", "_", word.lower()).strip("_") or "word"
    return f"{stem}__c{char_start}__{safe_word}.png"

# OPTIMIZATION 3: Using inference_mode instead of no_grad (Slightly faster)
@torch.inference_mode()
def run_repaint(
    model: UNet2DModel,
    scheduler: DDPMScheduler,
    clean_folded: torch.Tensor,
    mask_folded: torch.Tensor,
    start_timestep: int,
    resampling_number: int,
    generator: torch.Generator,
    show_progress: bool = False,
    use_amp: bool = False,
) -> torch.Tensor:
    if start_timestep < 1 or start_timestep >= scheduler.config.num_train_timesteps:
        raise ValueError(
            f"START_TIMESTEP must be in [1, {scheduler.config.num_train_timesteps - 1}], got {start_timestep}"
        )

    white = torch.ones_like(clean_folded)
    masked_clean = (mask_folded * clean_folded) + ((1.0 - mask_folded) * white)

    t0 = torch.tensor([start_timestep], device=clean_folded.device, dtype=torch.long)
    init_noise = torch.randn(
        masked_clean.shape,
        generator=generator,
        device=masked_clean.device,
        dtype=masked_clean.dtype,
    )
    x_t = scheduler.add_noise(masked_clean, init_noise, t0)

    step_iter = range(start_timestep, 0, -1)
    if show_progress:
        step_iter = tqdm(step_iter, desc="RePaint steps", leave=False)

    for step_t in step_iter:
        t = torch.tensor([step_t], device=clean_folded.device, dtype=torch.long)
        t_prev = torch.tensor([step_t - 1], device=clean_folded.device, dtype=torch.long)

        for rep in range(resampling_number):
            amp_ctx = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if use_amp
                else nullcontext()
            )
            with amp_ctx:
                noise_pred = model(x_t, t, return_dict=False)[0]
            x_t_minus_1_unknown = scheduler.step(noise_pred, step_t, x_t).prev_sample

            known_noise = torch.randn(
                clean_folded.shape,
                generator=generator,
                device=clean_folded.device,
                dtype=clean_folded.dtype,
            )
            x_t_minus_1_known = scheduler.add_noise(clean_folded, known_noise, t_prev)

            x_t_minus_1 = (mask_folded * x_t_minus_1_known) + ((1.0 - mask_folded) * x_t_minus_1_unknown)

            if rep < resampling_number - 1:
                beta_t = scheduler.betas[step_t].to(
                    device=clean_folded.device, dtype=x_t_minus_1.dtype
                ).view(1, 1, 1, 1)
                jump_noise = torch.randn(
                    x_t_minus_1.shape,
                    generator=generator,
                    device=x_t_minus_1.device,
                    dtype=x_t_minus_1.dtype,
                )
                x_t = torch.sqrt(1.0 - beta_t) * x_t_minus_1 + torch.sqrt(beta_t) * jump_noise
            else:
                x_t = x_t_minus_1

    return x_t


def main() -> None:
    set_global_seeds(SEED)
    torch.set_grad_enabled(False)  # Disables gradient tracking globally

    start_timestep = DEBUG_START_TIMESTEP if FAST_DEBUG_MODE else START_TIMESTEP
    resampling_number = DEBUG_RESAMPLING_NUMBER if FAST_DEBUG_MODE else RESAMPLING_NUMBER

    if REQUIRE_CUDA and DEVICE != "cuda":
        raise RuntimeError(
            "CUDA is required but not available. "
            "Activate an environment with GPU-enabled PyTorch or set REQUIRE_CUDA=False."
        )

    # OPTIMIZATION 1: cuDNN Benchmarking
    # Since our images are always exactly 1x1x128x128, this tells your GPU to 
    # run a quick profile on the first loop to find the fastest convolution algorithm.
    if DEVICE == "cuda":
        torch.backends.cudnn.benchmark = True

    checkpoint_dir = resolve_checkpoint_dir(CHECKPOINT_DIR)
    dataset_dir = Path(DATASET_DIR)
    manifest_path = Path(TEST_MANIFEST)
    output_dir = Path(OUTPUT_DIR)
    log_path = Path(METADATA_LOG_PATH)
    error_log_path = Path(ERROR_LOG_PATH)

    if MAX_GENERATED_IMAGES is not None and MAX_GENERATED_IMAGES <= 0:
        raise ValueError(f"MAX_GENERATED_IMAGES must be positive or None, got {MAX_GENERATED_IMAGES}")

    if not dataset_dir.exists():
        raise FileNotFoundError(f"DATASET_DIR not found: {dataset_dir}")

    rows = load_manifest_rows(manifest_path)
    if not rows:
        raise RuntimeError(f"No rows loaded from manifest: {manifest_path}")

    print(f"Loading checkpoint from: {checkpoint_dir}")
    use_amp = False

    steps_per_image = start_timestep * resampling_number
    print(f"Runtime device         : {DEVICE}")
    print(f"AMP enabled            : {use_amp}")
    print(f"RePaint steps/image    : {steps_per_image} ({start_timestep} timesteps x {resampling_number} resamples)")
    print(f"Fast debug mode        : {FAST_DEBUG_MODE}")
    if DEVICE != "cuda":
        print(
            "Warning: running on CPU. RePaint is very slow on CPU and may appear to hang at 0% for a long time."
        )
    
    model = UNet2DModel.from_pretrained(str(checkpoint_dir)).to(
        device=DEVICE,
        memory_format=torch.channels_last,
    )
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
        if MAX_GENERATED_IMAGES is not None and produced >= MAX_GENERATED_IMAGES:
            break

        image_rel = row["image_path"]
        full_text = row["text"]
        image_path = dataset_dir / image_rel

        if not image_path.exists():
            errors += 1
            pbar.set_postfix({"errors": errors})
            continue

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
            "start_timestep": start_timestep,
            "resampling_number": resampling_number,
            "output_path": str(output_path),
        }

        if output_path.exists():
            skipped_existing += 1
            if metadata["output_path"] not in logged_outputs:
                append_jsonl(log_path, metadata)
                logged_outputs.add(metadata["output_path"])
                recovered_log_entries += 1
            pbar.set_postfix({"skip_existing": skipped_existing, "recovered": recovered_log_entries})
            continue

        try:
            with Image.open(image_path) as img:
                clean_stripe = transform(img.convert("L")).unsqueeze(0).to(
                    device=DEVICE,
                    memory_format=torch.channels_last,
                )

            expected = (1, 1, STRIPE_HEIGHT, STRIPE_WIDTH)
            if tuple(clean_stripe.shape) != expected:
                raise ValueError(f"Image shape mismatch for {image_path}: {tuple(clean_stripe.shape)} != {expected}")

            clean_folded = to_folded(clean_stripe)

            pixel_start, pixel_end = compute_mask_pixel_range(
                text=full_text,
                char_start=target["char_start"],
                word_length=target["length"],
                padding_px=WORD_PADDING_PX,
            )
            metadata["mask_pixel_start"] = pixel_start
            metadata["mask_pixel_end"] = pixel_end

            mask_folded = build_word_mask_folded(
                device=DEVICE,
                text=full_text,
                char_start=target["char_start"],
                word_length=target["length"],
                padding_px=WORD_PADDING_PX,
            )

            infer_seed = stable_u32_seed(f"infer|{image_rel}|{target['char_start']}|{target['word']}", SEED)
            infer_gen = torch.Generator(device=DEVICE).manual_seed(infer_seed)

            inpainted_folded = run_repaint(
                model=model,
                scheduler=scheduler,
                clean_folded=clean_folded,
                mask_folded=mask_folded,
                start_timestep=start_timestep,
                resampling_number=resampling_number,
                generator=infer_gen,
                show_progress=SHOW_REPAINT_PROGRESS,
                use_amp=use_amp,
            )

            save_folded_as_stripe_png(inpainted_folded, output_path)
            append_jsonl(log_path, metadata)
            logged_outputs.add(metadata["output_path"])
            produced += 1

            pbar.set_postfix({"produced": produced, "errors": errors})
        except Exception as exc:
            errors += 1
            append_error_jsonl(
                error_log_path,
                {
                    "image_path": image_rel,
                    "output_path": str(output_path),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
            )
            pbar.set_postfix({"errors": errors})
            continue

    print("Done.")
    print(f"Manifest rows          : {len(rows)}")
    print(f"Generation cap         : {MAX_GENERATED_IMAGES}")
    print(f"Produced new outputs   : {produced}")
    print(f"Skipped existing output: {skipped_existing}")
    print(f"Recovered log entries  : {recovered_log_entries}")
    print(f"Skipped no-word rows   : {skipped_no_word}")
    print(f"Errors                 : {errors}")
    print(f"Error log              : {error_log_path}")
    print(f"Output dir             : {output_dir}")
    print(f"Metadata log           : {log_path}")


if __name__ == "__main__":
    main()