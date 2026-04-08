import os
from collections import Counter
from datasets import load_from_disk
from tqdm import tqdm
from PIL import Image
import concurrent.futures

# --- CONFIGURATION ---
NUM_IMAGES_TO_GENERATE = 10  
MAX_CHARS = 55                  # Adjust this after testing 1 image with your specific font size
TARGET_HEIGHT = 16              # 1 patch high
TARGET_WIDTH = 1024             # 64 patches wide (64 * 16)
OUTPUT_DIR = "stripe_text_dataset"
LOCAL_DATASET_PATH = "./local_fineweb" # Path to where you saved the dataset locally
MONO_FONT_PATH = "./NotoSansMono-Regular.ttf"  # Local monospaced TTF for fixed glyph advance
NUM_WORKERS = 8                 # Match this to your SLURM --cpus-per-task
CHUNKSIZE = 64                  # Larger chunks reduce multiprocessing overhead
PNG_COMPRESS_LEVEL = 1          # Lower compression is faster to write
MAX_ERROR_EXAMPLES = 10         # Keep a few example failures for debugging
SENTENCE_ENDINGS = (".", "!", "?", "...", "…", "؟", "।")
# ---------------------

# Global variable for the worker processes to hold their own renderer instance
worker_pixel_processor = None

def init_worker():
    """
    Initializes the Pango/Cairo renderer INSIDE each independent CPU worker.
    This prevents the C-extensions from crashing during multiprocessing.
    """
    global worker_pixel_processor
    from pixel_renderer import PixelRendererProcessor
    from font_download import FontConfig
    from font_download.fonts import FontSource
    
    mono_font_abs = os.path.abspath(MONO_FONT_PATH)
    if not os.path.exists(mono_font_abs):
        raise FileNotFoundError(
            f"Monospaced font file not found: {mono_font_abs}. "
            "Set MONO_FONT_PATH to a valid .ttf file."
        )

    # Use local monospaced TTF through a file:// source so FontConfig can load it.
    mono_source = FontSource(url=f"file://{mono_font_abs}")
    font_config = FontConfig(sources=[mono_source])
    worker_pixel_processor = PixelRendererProcessor(font=font_config)

def truncate_text(text, max_chars):
    """
    Cleans and truncates text while avoiding sentence-final boundaries when possible.
    This keeps line ends from consistently matching sentence ends.
    """
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text

    # Prefer cutting on a word boundary.
    cut_idx = text.rfind(" ", 0, max_chars + 1)
    if cut_idx <= 0:
        cut_idx = max_chars

    candidate = text[:cut_idx].rstrip()

    # If there is remaining text, avoid ending this line at sentence punctuation.
    if cut_idx < len(text):
        while candidate.endswith(SENTENCE_ENDINGS):
            prev_space = candidate.rfind(" ")
            if prev_space <= 0:
                break
            candidate = candidate[:prev_space].rstrip()

    return candidate if candidate else text[:max_chars].rstrip()

def pad_to_stripe(img, target_width, target_height, background_color=(255, 255, 255)):
    """
    Pads the rendered text image to exactly 16x1024.
    The blank space on the right acts as visual padding/EOS for the model.
    """
    new_img = Image.new(img.mode, (target_width, target_height), background_color)
    new_img.paste(img, (0, 0))
    return new_img

def process_single_item(item):
    """The generation function executed by each CPU core in parallel."""
    idx, raw_text = item
    try:
        if worker_pixel_processor is None:
            raise RuntimeError("Worker renderer is not initialized")

        short_text = truncate_text(raw_text, MAX_CHARS)
        if not short_text:
            return {
                "ok": False,
                "idx": idx,
                "error_type": "EmptyText",
                "message": "Text is empty after cleaning/truncation",
            }

        # Render using this specific worker's initialized processor
        rendered_image = worker_pixel_processor.render_text_image(short_text)

        # Crop just in case a weird unicode character pushes the height past 16px
        if rendered_image.width > TARGET_WIDTH or rendered_image.height > TARGET_HEIGHT:
            rendered_image = rendered_image.crop((0, 0, min(rendered_image.width, TARGET_WIDTH), min(rendered_image.height, TARGET_HEIGHT)))

        # Pad to perfect stripe dimensions
        final_image = pad_to_stripe(rendered_image, TARGET_WIDTH, TARGET_HEIGHT)
        
        # Save to disk
        save_path = os.path.join(OUTPUT_DIR, f"stripe_{idx:06d}.png")
        final_image.save(save_path, optimize=False, compress_level=PNG_COMPRESS_LEVEL)
        return {"ok": True, "idx": idx}
    except Exception as exc:
        return {
            "ok": False,
            "idx": idx,
            "error_type": type(exc).__name__,
            "message": str(exc)[:200],
        }

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # STEP 1: Load from local disk (Zero network latency!)
    print(f"Loading dataset from local disk at {LOCAL_DATASET_PATH}...")
    dataset = load_from_disk(LOCAL_DATASET_PATH)

    dataset_len = len(dataset)
    num_items = min(NUM_IMAGES_TO_GENERATE, dataset_len)
    print(f"Preparing to render {num_items} rows (requested {NUM_IMAGES_TO_GENERATE}, available {dataset_len})...")

    if num_items == 0:
        print("No rows available to render. Exiting.")
        os._exit(0)

    # Stream items directly to workers to reduce memory pressure and startup latency.
    text_items = ((i, dataset[i]["text"]) for i in range(num_items))

    # STEP 2: Render in parallel
    print(f"\nRendering {num_items} images using {NUM_WORKERS} CPU cores...")
    success_count = 0
    failure_count = 0
    error_counts = Counter()
    error_examples = []
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS, initializer=init_worker) as executor:
        # Stream results instead of collecting all booleans in memory.
        progress = tqdm(
            executor.map(process_single_item, text_items, chunksize=CHUNKSIZE),
            total=num_items,
            desc="Rendering",
        )
        for i, result in enumerate(progress, start=1):
            if result["ok"]:
                success_count += 1
            else:
                failure_count += 1
                error_type = result.get("error_type", "UnknownError")
                error_counts[error_type] += 1
                if len(error_examples) < MAX_ERROR_EXAMPLES:
                    error_examples.append(result)

            if i % 200 == 0 or i == num_items:
                progress.set_postfix({"ok": success_count, "fail": failure_count})

    print(f"\nSuccessfully generated {success_count}/{num_items} images in '{OUTPUT_DIR}'.")
    if failure_count > 0:
        print(f"Failed items: {failure_count}")
        print("Failure types:")
        for error_type, count in error_counts.most_common():
            print(f"  - {error_type}: {count}")

        print("Sample failures:")
        for sample in error_examples:
            print(
                f"  - idx={sample.get('idx')} type={sample.get('error_type')} "
                f"message={sample.get('message')}"
            )
    
    # STEP 3: Safe exit
    # Force an immediate, clean exit to bypass the Pango/Cairo C-extension teardown crashes
    os._exit(0) 

if __name__ == "__main__":
    main()