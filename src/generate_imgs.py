import os
from datasets import load_dataset
from tqdm import tqdm
from PIL import Image

# Import PIXEL renderer components
from pixel_renderer import PixelRendererProcessor
from font_download import FontConfig
from font_download.example_fonts.noto_sans import FONTS_NOTO_SANS

# --- CONFIGURATION ---
NUM_IMAGES_TO_GENERATE = 5  
MAX_CHARS = 55                  # Safe limit for a 1024-pixel wide strip
TARGET_HEIGHT = 16              # 1 patch high
TARGET_WIDTH = 1024             # 64 patches wide (64 * 16)
OUTPUT_DIR = "stripe_text_dataset"
# ---------------------

def truncate_text(text, max_chars):
    """
    Truncates text to a maximum character count, breaking at a space 
    so words aren't visually sliced in half.
    """
    # Remove newlines to ensure the text renders on a single horizontal line
    text = text.replace('\n', ' ').strip()
    
    if len(text) <= max_chars:
        return text
    
    chunk = text[:max_chars]
    last_space = chunk.rfind(' ')
    
    if last_space > 0:
        return chunk[:last_space]
    return chunk

def pad_to_stripe(img, target_width, target_height, background_color=(255, 255, 255)):
    """
    Pads the rendered image to the exact 16x1024 stripe dimensions.
    Assumes a white background (255,255,255).
    """
    # Create a blank stripe image
    new_img = Image.new(img.mode, (target_width, target_height), background_color)
    
    # Paste the rendered text image starting at the far left (0, 0)
    new_img.paste(img, (0, 0))
    return new_img

def main():
    print(f"Ensuring output directory '{OUTPUT_DIR}' exists...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Initializing PIXEL Renderer...")
    font_config = FontConfig(sources=FONTS_NOTO_SANS)
    pixel_processor = PixelRendererProcessor(font=font_config)

    print("Loading FineWeb-1B dataset (Streaming Mode)...")
    dataset = load_dataset("VisionTheta/fineweb-1B", split="train", streaming=True)
    dataset_iter = iter(dataset)

    print(f"Generating {NUM_IMAGES_TO_GENERATE} stripe images of size {TARGET_WIDTH}x{TARGET_HEIGHT}...")
    
    generated_count = 0
    pbar = tqdm(total=NUM_IMAGES_TO_GENERATE)

    while generated_count < NUM_IMAGES_TO_GENERATE:
        try:
            row = next(dataset_iter)
            raw_text = row.get('text', '')

            # 1. Chunk and clean the text (force single line)
            short_text = truncate_text(raw_text, MAX_CHARS)
            
            if not short_text:
                continue

            # 2. Render the text to an image
            rendered_image = pixel_processor.render_text_image(short_text)

            # 3. Ensure the image fits inside our target dimensions
            # If the text is somehow taller than 16px or wider than 1024px, crop it
            if rendered_image.width > TARGET_WIDTH or rendered_image.height > TARGET_HEIGHT:
                rendered_image = rendered_image.crop((0, 0, min(rendered_image.width, TARGET_WIDTH), min(rendered_image.height, TARGET_HEIGHT)))

            # 4. Pad the right side to reach exactly 1024 pixels
            final_image = pad_to_stripe(rendered_image, TARGET_WIDTH, TARGET_HEIGHT)

            # 5. Save to disk
            save_path = os.path.join(OUTPUT_DIR, f"stripe_{generated_count:06d}.png")
            final_image.save(save_path)

            generated_count += 1
            pbar.update(1)

        except StopIteration:
            print("\nReached the absolute end of the dataset!")
            break
        except Exception as e:
            continue

    pbar.close()
    print(f"\nSuccessfully generated {generated_count} images in '{OUTPUT_DIR}'.")
    os._exit(0)
if __name__ == "__main__":
    main()