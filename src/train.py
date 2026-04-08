import os
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
OUTPUT_DIR = "ddpm_text_model"
BATCH_SIZE = 32           # Adjust based on your RunPod GPU VRAM (32-64 is usually safe for 128x128)
NUM_EPOCHS = 50           # Start with 50 to see initial convergence
LEARNING_RATE = 1e-4
# ---------------------

class StripeDataset(Dataset):
    """
    Loads your 16x1024 text stripes, converts them to grayscale, 
    and 'folds' them into 128x128 squares for the standard U-Net.
    """
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')]
        
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
        
        # THE RESHAPE HACK: Fold the 16x1024 stripe into a 128x128 square
        folded_tensor = tensor.view(1, 128, 128)
        
        return folded_tensor

def main():
    # 1. Initialize Accelerator (Handles GPU setup automatically)
    accelerator = Accelerator()
    print(f"Training on device: {accelerator.device}")

    # 2. Load Dataset and DataLoader
    print("Loading dataset...")
    dataset = StripeDataset(DATASET_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

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
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    # 7. Training Loop
    global_step = 0
    print("Starting training loop...")
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        progress_bar = tqdm(total=len(dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch+1}")
        
        epoch_loss = 0.0

        for step, clean_images in enumerate(dataloader):
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
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")

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