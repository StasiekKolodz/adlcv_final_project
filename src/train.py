import os
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from diffusers import UNet2DModel, DDPMScheduler
from accelerate import Accelerator
from tqdm.auto import tqdm

# --- CONFIGURATION ---
DATASET_DIR = "stripe_text_dataset"
SPLITS_DIR = "splits"
TRAIN_MANIFEST = os.path.join(SPLITS_DIR, "train.txt")
VAL_MANIFEST = os.path.join(SPLITS_DIR, "val.txt")
OUTPUT_DIR = "ddpm_text_model"
CHECKPOINT_DIR = "checkpoints"
BATCH_SIZE = 8            # Lowered from 32 to fit inside 8GB of VRAM
NUM_EPOCHS = 50           # Start with 50 to see initial convergence
LEARNING_RATE = 1e-4
VALIDATE_EVERY = 1        # Run validation every N epochs
SAVE_EVERY = 1            # Save checkpoint every N epochs
RESUME_FROM = "latest"    # "latest", specific checkpoint path, or None
# ---------------------

class StripeDataset(Dataset):
    """
    Loads your 16x1024 text stripes, converts them to grayscale, 
    and 'folds' them into 128x128 squares for the standard U-Net.
    """
    def __init__(self, image_dir, manifest_path=None):
        self.image_dir = image_dir

        if manifest_path is not None:
            if not os.path.exists(manifest_path):
                raise FileNotFoundError(f"Split manifest not found: {manifest_path}")
            with open(manifest_path, "r", encoding="utf-8") as f:
                rel_paths = [line.strip() for line in f if line.strip()]
            self.image_paths = [os.path.join(image_dir, p) for p in rel_paths]
        else:
            self.image_paths = [
                os.path.join(image_dir, f)
                for f in os.listdir(image_dir)
                if f.endswith('.png')
            ]

        if len(self.image_paths) == 0:
            source = manifest_path if manifest_path is not None else image_dir
            raise RuntimeError(f"No images found for dataset source: {source}")
        
        # We only need ToTensor (scales pixels to 0.0 - 1.0) 
        # and Normalize (scales pixels to -1.0 - 1.0, which diffusion models prefer)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        # Open as Grayscale ('L') since text doesn't need RGB colors
        img = Image.open(img_path).convert("L") 
        
        # Tensor shape becomes: (1, 16, 1024)
        tensor = self.transform(img)

        expected_shape = (1, 16, 1024)
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"Invalid image shape for '{img_path}': got {tuple(tensor.shape)}, "
                f"expected {expected_shape}."
            )
        
        # THE RESHAPE HACK: Fold the 16x1024 stripe into a 128x128 square
        folded_tensor = tensor.view(1, 128, 128)
        
        return folded_tensor


def evaluate(model, dataloader, noise_scheduler, accelerator):
    model.eval()
    val_loss_sum = 0.0
    val_steps = 0

    with torch.no_grad():
        # Use a fixed RNG stream so validation is comparable across epochs.
        generator = torch.Generator(device=accelerator.device).manual_seed(42)
        for clean_images in dataloader:
            noise = torch.randn(clean_images.shape, generator=generator, device=clean_images.device)
            bs = clean_images.shape[0]
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (bs,),
                generator=generator,
                device=clean_images.device,
            ).long()

            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
            noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
            loss = F.mse_loss(noise_pred, noise)

            val_loss_sum += loss.item()
            val_steps += 1

    totals = accelerator.reduce(
        torch.tensor([val_loss_sum, float(val_steps)], device=accelerator.device),
        reduction="sum",
    )
    global_val_steps = int(totals[1].item())
    if global_val_steps == 0:
        return float("nan")
    return totals[0].item() / global_val_steps


def _parse_checkpoint_epoch(dirname: str):
    prefix = "epoch_"
    if not dirname.startswith(prefix):
        return None
    suffix = dirname[len(prefix) :]
    if not suffix.isdigit():
        return None
    return int(suffix)


def resolve_resume_checkpoint(resume_from, checkpoint_root):
    if resume_from is None:
        return None

    if resume_from != "latest":
        return resume_from

    if not os.path.isdir(checkpoint_root):
        return None

    candidates = []
    for name in os.listdir(checkpoint_root):
        epoch_num = _parse_checkpoint_epoch(name)
        if epoch_num is None:
            continue
        full_path = os.path.join(checkpoint_root, name)
        if os.path.isdir(full_path):
            candidates.append((epoch_num, full_path))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def save_training_checkpoint(
    accelerator,
    checkpoint_root,
    epoch,
    global_step,
    best_val_loss,
):
    ckpt_dir = os.path.join(checkpoint_root, f"epoch_{epoch + 1:04d}")
    accelerator.wait_for_everyone()
    accelerator.save_state(ckpt_dir)

    if accelerator.is_main_process:
        trainer_state = {
            "epoch": epoch,
            "global_step": global_step,
            "best_val_loss": best_val_loss,
        }
        with open(os.path.join(ckpt_dir, "trainer_state.json"), "w", encoding="utf-8") as f:
            json.dump(trainer_state, f, indent=2)

    accelerator.wait_for_everyone()


def load_training_checkpoint(accelerator, ckpt_dir):
    accelerator.load_state(ckpt_dir)

    trainer_state_path = os.path.join(ckpt_dir, "trainer_state.json")
    if os.path.exists(trainer_state_path):
        with open(trainer_state_path, "r", encoding="utf-8") as f:
            trainer_state = json.load(f)
        start_epoch = int(trainer_state.get("epoch", -1)) + 1
        global_step = int(trainer_state.get("global_step", 0))
        best_val_loss = float(trainer_state.get("best_val_loss", float("inf")))
    else:
        start_epoch = 0
        global_step = 0
        best_val_loss = float("inf")

    return start_epoch, global_step, best_val_loss

def main():
    # 1. Initialize Accelerator (Handles GPU setup automatically)
    accelerator = Accelerator()
    print(f"Training on device: {accelerator.device}")

    # 2. Load Dataset and DataLoader
    print("Loading dataset...")
    train_dataset = StripeDataset(DATASET_DIR, manifest_path=TRAIN_MANIFEST)
    val_dataset = StripeDataset(DATASET_DIR, manifest_path=VAL_MANIFEST)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    # 3. Define the U-Net Model
    # This is a standard image diffusion architecture, completely off-the-shelf
    model = UNet2DModel(
        sample_size=128,          # Our folded target resolution
        in_channels=1,            # 1 channel for grayscale
        out_channels=1,
        layers_per_block=2,
        block_out_channels=(64, 128, 256, 512), 
        down_block_types=(
            "DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D",
        ),
    )

    # 4. Define the Noise Scheduler (DDPM framework)
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    # 5. Define Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # 6. Prepare everything with Accelerator
    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        model,
        optimizer,
        train_dataloader,
        val_dataloader,
    )

    if accelerator.is_main_process:
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    accelerator.wait_for_everyone()

    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")

    checkpoint_to_resume = resolve_resume_checkpoint(RESUME_FROM, CHECKPOINT_DIR)
    if checkpoint_to_resume is not None:
        if not os.path.isdir(checkpoint_to_resume):
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_to_resume}")
        start_epoch, global_step, best_val_loss = load_training_checkpoint(accelerator, checkpoint_to_resume)
        if accelerator.is_main_process:
            print(
                f"Resumed from {checkpoint_to_resume} at epoch={start_epoch}, "
                f"global_step={global_step}, best_val_loss={best_val_loss:.4f}"
            )
    elif accelerator.is_main_process and RESUME_FROM is not None:
        print("No checkpoint found to resume. Starting from scratch.")

    # 7. Training Loop
    print("Starting training loop...")
    
    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train()
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch+1}")
        
        epoch_loss = 0.0

        for step, clean_images in enumerate(train_dataloader):
            # Sample random noise to add to the images
            noise = torch.randn(clean_images.shape, device=clean_images.device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image in the batch
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device
            ).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (This is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                # Predict the noise residual using the U-Net
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]

                # Calculate MSE Loss (How close was the U-Net's guess to the actual noise we added?)
                loss = F.mse_loss(noise_pred, noise)

                # Backpropagation
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss.item()
            progress_bar.update(1)
            progress_bar.set_postfix({"loss": loss.item()})
            global_step += 1

        # Calculate average loss for the epoch
        if len(train_dataloader) == 0:
            raise RuntimeError("Train dataloader is empty. Check split files and BATCH_SIZE settings.")

        avg_loss = epoch_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} completed. Train Loss: {avg_loss:.4f}")

        val_loss = None
        if (epoch + 1) % VALIDATE_EVERY == 0:
            val_loss = evaluate(model, val_dataloader, noise_scheduler, accelerator)
            if accelerator.is_main_process:
                print(f"Epoch {epoch+1} validation loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if accelerator.is_main_process:
                    print(f"New best validation loss: {best_val_loss:.4f}")

        if (epoch + 1) % SAVE_EVERY == 0:
            save_training_checkpoint(
                accelerator=accelerator,
                checkpoint_root=CHECKPOINT_DIR,
                epoch=epoch,
                global_step=global_step,
                best_val_loss=best_val_loss,
            )
            if accelerator.is_main_process:
                print(f"Saved checkpoint for epoch {epoch+1} to '{CHECKPOINT_DIR}'.")

    # 8. Save the final model
    print(f"Saving model to {OUTPUT_DIR}...")
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(OUTPUT_DIR)
        noise_scheduler.save_pretrained(OUTPUT_DIR)
        print("Training complete!")

if __name__ == "__main__":
    main()