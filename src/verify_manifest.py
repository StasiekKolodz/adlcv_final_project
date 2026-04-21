import json
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# --- CONFIGURATION ---
MANIFEST_PATH = Path("splits/test_manifest.jsonl")
DATASET_DIR = Path("stripe_text_dataset")
OUTPUT_DIR = Path("verification_outputs")

# We will use the same font you used for generation to keep it consistent
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
# ---------------------

def load_manifest(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
        
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def verify_random_image():
    # 1. Load the manifest and pick a random row
    print("Loading reconstructed manifest...")
    manifest = load_manifest(MANIFEST_PATH)
    if not manifest:
        raise ValueError("Manifest is empty!")
        
    target_row = random.choice(manifest)
    image_filename = target_row["image_path"]
    ground_truth_text = target_row["text"]
    
    print(f"Selected Image: {image_filename}")
    print(f"Manifest Text : '{ground_truth_text}'")
    
    # 2. Load the actual image
    image_path = DATASET_DIR / image_filename
    if not image_path.exists():
        raise FileNotFoundError(f"Dataset image missing: {image_path}")
        
    original_img = Image.open(image_path).convert("RGB")
    
    # 3. Create a verification canvas (Adding space below the stripe for text)
    canvas_width = original_img.width
    canvas_height = original_img.height + 44 
    canvas = Image.new("RGB", (canvas_width, canvas_height), color="white")
    
    # Paste the original stripe at the very top
    canvas.paste(original_img, (0, 0))
    
    # 4. Draw the ground truth text below it
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype(FONT_PATH, 16)
    except OSError:
        print(f"Warning: Could not load {FONT_PATH}, falling back to default font.")
        font = ImageFont.load_default()
        
    # Draw a separator line
    draw.line([(0, original_img.height + 2), (canvas_width, original_img.height + 2)], fill="red", width=1)
    
    # Draw the EXACT text at EXACTLY x=0 with NO prefix
    text_y_position = original_img.height + 10
    draw.text((0, text_y_position), ground_truth_text, fill="black", font=font)
    
    # 5. Save the output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_filename = OUTPUT_DIR / f"verify_{image_filename}"
    canvas.save(output_filename)
    
    print(f"\nSuccess! Verification image saved to: {output_filename}")
    print("Open this file to visually confirm the text matches the image pixels exactly.")

if __name__ == "__main__":
    verify_random_image()