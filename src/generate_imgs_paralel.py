import os
import textwrap
from collections import Counter
from datasets import load_from_disk
from tqdm import tqdm
from PIL import Image
import concurrent.futures

# --- CONFIGURATION ---
NUM_IMAGES_TO_GENERATE = 10  # Number of dataset rows to process
TARGET_HEIGHT = 16              # 1 patch high
TARGET_WIDTH = 1024             # 64 patches wide (64 * 16)
MAX_CHARS = TARGET_WIDTH // TARGET_HEIGHT  # Character-aligned target: 1 char per 16px patch -> 64
MONO_FONT_SIZE = 16             # Render size matching the patch height
OUTPUT_DIR = "stripe_text_dataset"
LOCAL_DATASET_PATH = "./local_fineweb" # Path to where you saved the dataset locally
MONO_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"  # Linux system monospaced font
NUM_WORKERS = 8                 # Match this to your SLURM --cpus-per-task
CHUNKSIZE = 64                  # Larger chunks reduce multiprocessing overhead
PNG_COMPRESS_LEVEL = 1          # Lower compression is faster to write
MAX_ERROR_EXAMPLES = 10         # Keep a few example failures for debugging
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

        clean_text = " ".join(raw_text.split())
        if not clean_text:
            return {
                "ok": False,
                "idx": idx,
                "error_type": "EmptyText",
                "message": "Text is empty after cleaning",
            }

        text_chunks = textwrap.wrap(
            clean_text,
            width=MAX_CHARS,
            break_long_words=True,
            break_on_hyphens=False,
        )

        if not text_chunks:
            return {
                "ok": False,
                "idx": idx,
                "error_type": "NoChunks",
                "message": "No chunks produced after wrapping",
            }

        saved_count = 0

        for chunk_idx, chunk_text in enumerate(text_chunks):
            # Render using this specific worker's initialized processor
            rendered_image = worker_pixel_processor.render_text_image(
                chunk_text,
                block_size=TARGET_HEIGHT,
                font_size=MONO_FONT_SIZE,
            )

            # Crop just in case a weird unicode character pushes the height past 16px
            if rendered_image.width > TARGET_WIDTH or rendered_image.height > TARGET_HEIGHT:
                rendered_image = rendered_image.crop(
                    (0, 0, min(rendered_image.width, TARGET_WIDTH), min(rendered_image.height, TARGET_HEIGHT))
                )

            # Pad to perfect stripe dimensions
            final_image = pad_to_stripe(rendered_image, TARGET_WIDTH, TARGET_HEIGHT)

            # Save to disk. One input row can produce multiple aligned stripe images.
            save_path = os.path.join(OUTPUT_DIR, f"stripe_{idx:06d}_{chunk_idx:02d}.png")
            final_image.save(save_path, optimize=False, compress_level=PNG_COMPRESS_LEVEL)
            saved_count += 1

        return {"ok": True, "idx": idx, "chunks_saved": saved_count}
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
    total_chunks_saved = 0
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
                total_chunks_saved += result.get("chunks_saved", 0)
            else:
                failure_count += 1
                error_type = result.get("error_type", "UnknownError")
                error_counts[error_type] += 1
                if len(error_examples) < MAX_ERROR_EXAMPLES:
                    error_examples.append(result)

            if i % 200 == 0 or i == num_items:
                progress.set_postfix({"rows_ok": success_count, "rows_fail": failure_count, "chunks": total_chunks_saved})

    print(f"\nProcessed rows: {success_count}/{num_items} (failed: {failure_count})")
    print(f"Generated stripe images: {total_chunks_saved} in '{OUTPUT_DIR}'.")
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