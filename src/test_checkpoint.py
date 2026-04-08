import os
from pathlib import Path

import torch
from PIL import Image
from diffusers import DDPMScheduler, UNet2DModel
from torchvision import transforms

# --- CONFIGURATION ---
# Directory produced by `save_pretrained` in train.py (e.g. ddpm_text_model)
CHECKPOINT_DIR = "ddpm_text_model"

# If set, this image is used. If None, first path from TEST_MANIFEST is used.
INPUT_IMAGE_PATH = None
TEST_MANIFEST = "splits/test.txt"
DATASET_DIR = "stripe_text_dataset"

# Reconstruction settings
NOISE_TIMESTEP = 250  # Higher => harder denoising task
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Output paths
OUTPUT_IMAGE = "review_denoised.png"
OUTPUT_PANEL = "review_panel.png"
# ---------------------


def resolve_input_image() -> Path:
    if INPUT_IMAGE_PATH is not None:
        p = Path(INPUT_IMAGE_PATH)
        if not p.exists():
            raise FileNotFoundError(f"INPUT_IMAGE_PATH not found: {p}")
        return p

    manifest_path = Path(TEST_MANIFEST)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"TEST_MANIFEST not found: {manifest_path}. "
            "Set INPUT_IMAGE_PATH or create split files first."
        )

    lines = [line.strip() for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"No entries found in manifest: {manifest_path}")

    img_path = Path(DATASET_DIR) / lines[0]
    if not img_path.exists():
        raise FileNotFoundError(f"Image from manifest does not exist: {img_path}")
    return img_path


def to_stripe_image(tensor_128: torch.Tensor) -> Image.Image:
    # tensor_128: shape (1, 1, 128, 128) in [-1, 1]
    stripe = tensor_128.view(1, 1, 16, 1024)
    stripe = (stripe / 2 + 0.5).clamp(0, 1)
    pil = transforms.ToPILImage()(stripe[0].cpu())
    return pil.convert("L")


def to_pil_gray(tensor: torch.Tensor) -> Image.Image:
    # tensor: shape (1, 1, 16, 1024) in [-1, 1]
    vis = (tensor / 2 + 0.5).clamp(0, 1)
    pil = transforms.ToPILImage()(vis[0].cpu())
    return pil.convert("L")


def build_review_panel(original: Image.Image, noisy: Image.Image, denoised: Image.Image) -> Image.Image:
    w, h = original.size
    # Stack vertically: Original (top), Noisy (middle), Denoised (bottom).
    panel = Image.new("L", (w, h * 3), color=255)
    panel.paste(original, (0, 0))
    panel.paste(noisy, (0, h))
    panel.paste(denoised, (0, h * 2))
    return panel


def main() -> None:
    checkpoint_dir = Path(CHECKPOINT_DIR)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"CHECKPOINT_DIR not found: {checkpoint_dir}")

    input_image_path = resolve_input_image()

    print(f"Loading checkpoint from: {checkpoint_dir}")
    model = UNet2DModel.from_pretrained(str(checkpoint_dir)).to(DEVICE)
    noise_scheduler = DDPMScheduler.from_pretrained(str(checkpoint_dir))
    model.eval()

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    img = Image.open(input_image_path).convert("L")
    clean = transform(img).unsqueeze(0).to(DEVICE)  # (1, 1, 16, 1024)

    expected_shape = (1, 1, 16, 1024)
    if tuple(clean.shape) != expected_shape:
        raise ValueError(f"Input image shape mismatch: got {tuple(clean.shape)}, expected {expected_shape}")

    clean_folded = clean.view(1, 1, 128, 128)

    # Corrupt with noise at a chosen timestep.
    t = torch.tensor([NOISE_TIMESTEP], device=DEVICE, dtype=torch.long)
    noise = torch.randn_like(clean_folded)
    noisy = noise_scheduler.add_noise(clean_folded, noise, t)

    # Iterative denoising from t -> 0.
    sample = noisy
    with torch.no_grad():
        for step_t in range(NOISE_TIMESTEP, -1, -1):
            step_tensor = torch.tensor([step_t], device=DEVICE, dtype=torch.long)
            noise_pred = model(sample, step_tensor, return_dict=False)[0]
            sample = noise_scheduler.step(noise_pred, step_t, sample).prev_sample

    denoised_pil = to_stripe_image(sample)
    denoised_pil.save(OUTPUT_IMAGE)

    original_pil = to_pil_gray(clean)
    noisy_pil = to_stripe_image(noisy)
    panel = build_review_panel(original_pil, noisy_pil, denoised_pil)
    panel.save(OUTPUT_PANEL)

    print(f"Input image   : {input_image_path}")
    print(f"Device        : {DEVICE}")
    print(f"Noise timestep: {NOISE_TIMESTEP}")
    print(f"Saved output  : {OUTPUT_IMAGE}")
    print(f"Saved panel   : {OUTPUT_PANEL} (original | noisy | denoised)")


if __name__ == "__main__":
    main()
