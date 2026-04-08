from pathlib import Path

import torch
from PIL import Image
from diffusers import DDPMScheduler, UNet2DModel
from tqdm.auto import tqdm
from torchvision import transforms

# --- CONFIGURATION ---
CHECKPOINT_DIR = "checkpoints/epoch_0001"

# If set, this image is used. If None, first path from TEST_MANIFEST is used.
INPUT_IMAGE_PATH = None
TEST_MANIFEST = "splits/test.txt"
DATASET_DIR = "stripe_text_dataset"

# RePaint parameters
START_TIMESTEP = 250      # Initial corruption level (<= scheduler train timesteps - 1)
RESAMPLING_NUMBER = 3     # Number of denoise/jump cycles per timestep
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

# Mask definition in folded 128x128 coordinates
# 1.0 = known/keep, 0.0 = unknown/inpaint
MASK_Y_START = 60
MASK_Y_END = 70
MASK_X_START = 0
MASK_X_END = 128

# Output paths
OUTPUT_INPAINTED = "repaint_inpainted.png"
OUTPUT_PANEL = "repaint_panel.png"
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


def to_pil_gray(tensor_stripe: torch.Tensor) -> Image.Image:
    # tensor_stripe: shape (1, 1, 16, 1024) in [-1, 1]
    vis = (tensor_stripe / 2 + 0.5).clamp(0, 1)
    pil = transforms.ToPILImage()(vis[0].cpu())
    return pil.convert("L")


def build_mask(device: str) -> torch.Tensor:
    mask = torch.ones((1, 1, 128, 128), device=device)
    mask[:, :, MASK_Y_START:MASK_Y_END, MASK_X_START:MASK_X_END] = 0.0
    return mask


def build_review_panel(
    original: Image.Image,
    masked_input: Image.Image,
    inpainted: Image.Image,
    mask_vis: Image.Image,
) -> Image.Image:
    # Vertical stack for easier reading of long stripes.
    w, h = original.size
    panel = Image.new("L", (w, h * 4), color=255)
    panel.paste(original, (0, 0))
    panel.paste(masked_input, (0, h))
    panel.paste(inpainted, (0, h * 2))
    panel.paste(mask_vis, (0, h * 3))
    return panel


def main() -> None:
    checkpoint_dir = Path(CHECKPOINT_DIR)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"CHECKPOINT_DIR not found: {checkpoint_dir}")

    input_image_path = resolve_input_image()

    print(f"Loading checkpoint from: {checkpoint_dir}")
    model = UNet2DModel.from_pretrained(str(checkpoint_dir)).to(DEVICE)
    scheduler = DDPMScheduler.from_pretrained(str(checkpoint_dir))
    model.eval()

    torch.manual_seed(SEED)
    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(SEED)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    img = Image.open(input_image_path).convert("L")
    clean_stripe = transform(img).unsqueeze(0).to(DEVICE)  # (1, 1, 16, 1024)

    expected_shape = (1, 1, 16, 1024)
    if tuple(clean_stripe.shape) != expected_shape:
        raise ValueError(f"Input image shape mismatch: got {tuple(clean_stripe.shape)}, expected {expected_shape}")

    clean_folded = clean_stripe.view(1, 1, 128, 128)
    mask = build_mask(DEVICE)

    # Start from a noised version at START_TIMESTEP.
    if START_TIMESTEP < 1 or START_TIMESTEP >= scheduler.config.num_train_timesteps:
        raise ValueError(
            f"START_TIMESTEP must be in [1, {scheduler.config.num_train_timesteps - 1}], got {START_TIMESTEP}"
        )

    start_t = torch.tensor([START_TIMESTEP], device=DEVICE, dtype=torch.long)
    x_t = scheduler.add_noise(clean_folded, torch.randn_like(clean_folded), start_t)

    timesteps = list(range(START_TIMESTEP, 0, -1))
    pbar = tqdm(timesteps, desc="RePaint")

    with torch.no_grad():
        for step_t in pbar:
            for rep in range(RESAMPLING_NUMBER):
                t = torch.tensor([step_t], device=DEVICE, dtype=torch.long)

                # Unknown region proposal from the model.
                noise_pred = model(x_t, t, return_dict=False)[0]
                x_t_minus_1_unknown = scheduler.step(noise_pred, step_t, x_t).prev_sample

                # Known region forced to match ground truth at t-1.
                t_prev = torch.tensor([step_t - 1], device=DEVICE, dtype=torch.long)
                x_t_minus_1_known = scheduler.add_noise(clean_folded, torch.randn_like(clean_folded), t_prev)

                # RePaint composition.
                x_t_minus_1 = (mask * x_t_minus_1_known) + ((1.0 - mask) * x_t_minus_1_unknown)

                # Jump back to t except on the final resample iteration.
                if rep < RESAMPLING_NUMBER - 1:
                    x_t = scheduler.add_noise(x_t_minus_1, torch.randn_like(x_t_minus_1), t)
                else:
                    x_t = x_t_minus_1

    inpainted_pil = to_stripe_image(x_t)
    inpainted_pil.save(OUTPUT_INPAINTED)

    # For visual review, masked input keeps known regions and whites out unknown regions.
    white_tensor = torch.ones_like(clean_folded)
    masked_folded = (mask * clean_folded) + ((1.0 - mask) * white_tensor)

    original_pil = to_pil_gray(clean_stripe)
    masked_input_pil = to_stripe_image(masked_folded)
    mask_vis_pil = to_stripe_image(mask * 2.0 - 1.0)  # map {0,1} -> {-1,1}

    panel = build_review_panel(original_pil, masked_input_pil, inpainted_pil, mask_vis_pil)
    panel.save(OUTPUT_PANEL)

    print(f"Input image      : {input_image_path}")
    print(f"Device           : {DEVICE}")
    print(f"Start timestep   : {START_TIMESTEP}")
    print(f"Resampling number: {RESAMPLING_NUMBER}")
    print(f"Saved inpainted  : {OUTPUT_INPAINTED}")
    print(f"Saved panel      : {OUTPUT_PANEL} (original | masked input | inpainted | mask)")


if __name__ == "__main__":
    main()
