import os
import textwrap
import multiprocessing as mp
import time
from collections import Counter
from datasets import load_from_disk
from tqdm import tqdm
from PIL import Image
import concurrent.futures

# --- CONFIGURATION ---
CPU_COUNT = os.cpu_count() or 1
NUM_IMAGES_TO_GENERATE = 15000  # Number of dataset rows to process

TARGET_HEIGHT = 16              # 1 patch high
TARGET_WIDTH = 1024             # 64 patches wide (64 * 16)
MAX_CHARS = 104  # Natural fill: 1024px / ~9.6px per char = ~106. We use 104 for a safe margin.
MONO_FONT_SIZE = 16             # Render size matching the patch height
OUTPUT_DIR = "stripe_text_dataset"
LOCAL_DATASET_PATH = "./local_fineweb" # Path to where you saved the dataset locally
MONO_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"  # Linux system monospaced font
NUM_WORKERS = min(32, max(8, CPU_COUNT // 4))  # Stable default for high-core servers
CHUNKSIZE = 256                 # Larger chunks reduce multiprocessing overhead
MAX_IN_FLIGHT = max(128, NUM_WORKERS * 8)  # Keep workers fed to avoid idle CPU time
PNG_COMPRESS_LEVEL = 1          # Lower compression is faster to write
MAX_ERROR_EXAMPLES = 10         # Keep a few example failures for debugging
PRELOAD_TEXTS = True            # Better throughput: avoids slow per-row parent indexing
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
        start_time = time.perf_counter()
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

        words = clean_text.split()
        if not words:
            return {
                "ok": False,
                "idx": idx,
                "error_type": "NoChunks",
                "message": "No chunks produced after cleaning",
            }

        # Pixel-aware chunking: add words until the candidate line no longer fits.
        # This guarantees saved lines contain only fully visible words.
        text_chunks = []
        current_chunk = ""
        skipped_overlong_words = 0

        for word in words:
            candidate = word if not current_chunk else f"{current_chunk} {word}"
            candidate_img = worker_pixel_processor.render_text_image(
                candidate,
                block_size=TARGET_HEIGHT,
                font_size=MONO_FONT_SIZE,
            )

            if candidate_img.width <= TARGET_WIDTH and candidate_img.height <= TARGET_HEIGHT:
                current_chunk = candidate
                continue

            if current_chunk:
                text_chunks.append(current_chunk)
                single_word_img = worker_pixel_processor.render_text_image(
                    word,
                    block_size=TARGET_HEIGHT,
                    font_size=MONO_FONT_SIZE,
                )
                if single_word_img.width <= TARGET_WIDTH and single_word_img.height <= TARGET_HEIGHT:
                    current_chunk = word
                else:
                    skipped_overlong_words += 1
                    current_chunk = ""
            else:
                skipped_overlong_words += 1

        if current_chunk:
            text_chunks.append(current_chunk)

        if not text_chunks:
            return {
                "ok": False,
                "idx": idx,
                "error_type": "NoFittingChunks",
                "message": "No whole-word chunk fits target dimensions",
            }

        saved_count = 0
        skipped_overlong_chunks = skipped_overlong_words
        render_save_time = 0.0

        for chunk_idx, chunk_text in enumerate(text_chunks):
            chunk_start = time.perf_counter()
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
            render_save_time += time.perf_counter() - chunk_start

        if saved_count == 0 and skipped_overlong_chunks > 0:
            return {
                "ok": False,
                "idx": idx,
                "error_type": "OverlongWordsOnly",
                "message": "All chunks were over MAX_CHARS and skipped",
            }

        return {
            "ok": True,
            "idx": idx,
            "chunks_saved": saved_count,
            "chunks_skipped_overlong": skipped_overlong_chunks,
            "row_time_s": time.perf_counter() - start_time,
            "render_save_time_s": render_save_time,
        }
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

    if PRELOAD_TEXTS:
        # Batched column read is much faster than per-row indexing for large runs.
        print("Preloading text column into memory...")
        texts = dataset[:num_items]["text"]
        text_items = ((i, text) for i, text in enumerate(texts))
        print("Text preload complete.")
    else:
        text_items = ((i, dataset[i]["text"]) for i in range(num_items))

    # STEP 2: Render in parallel
    print(f"\nRendering {num_items} images using {NUM_WORKERS} CPU cores...")
    print(f"CPU detected: {os.cpu_count()} | workers={NUM_WORKERS} | chunksize={CHUNKSIZE} | preload_texts={PRELOAD_TEXTS}")
    success_count = 0
    total_chunks_saved = 0
    failure_count = 0
    error_counts = Counter()
    error_examples = []
    total_row_time_s = 0.0
    total_render_save_time_s = 0.0
    first_result_time_s = None
    loop_start = time.perf_counter()
    
    ctx = mp.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=NUM_WORKERS,
        initializer=init_worker,
        mp_context=ctx,
    ) as executor:
        # Use out-of-order completion so long early samples don't stall progress at 0%.
        progress = tqdm(total=num_items, desc="Rendering")
        items_iter = iter(text_items)
        futures = {}

        def submit_next() -> bool:
            try:
                item = next(items_iter)
            except StopIteration:
                return False
            future = executor.submit(process_single_item, item)
            futures[future] = item[0]
            return True

        initial_submit = min(MAX_IN_FLIGHT, num_items)
        for _ in range(initial_submit):
            if not submit_next():
                break

        processed = 0
        while futures:
            done, _ = concurrent.futures.wait(
                futures,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )

            for future in done:
                idx = futures.pop(future)
                try:
                    result = future.result()
                except Exception as exc:
                    result = {
                        "ok": False,
                        "idx": idx,
                        "error_type": type(exc).__name__,
                        "message": str(exc)[:200],
                    }

                processed += 1
                progress.update(1)

                if first_result_time_s is None:
                    first_result_time_s = time.perf_counter() - loop_start

                while len(futures) < MAX_IN_FLIGHT and submit_next():
                    pass

                if result["ok"]:
                    success_count += 1
                    total_chunks_saved += result.get("chunks_saved", 0)
                    total_row_time_s += result.get("row_time_s", 0.0)
                    total_render_save_time_s += result.get("render_save_time_s", 0.0)
                else:
                    failure_count += 1
                    error_type = result.get("error_type", "UnknownError")
                    error_counts[error_type] += 1
                    if len(error_examples) < MAX_ERROR_EXAMPLES:
                        error_examples.append(result)

            if processed % 200 == 0 or processed == num_items:
                progress.set_postfix({"rows_ok": success_count, "rows_fail": failure_count, "chunks": total_chunks_saved})

        progress.close()

    print(f"\nProcessed rows: {success_count}/{num_items} (failed: {failure_count})")
    print(f"Generated stripe images: {total_chunks_saved} in '{OUTPUT_DIR}'.")
    if first_result_time_s is not None:
        print(f"Time to first result: {first_result_time_s:.2f}s")
    if success_count > 0:
        print(f"Avg row worker time: {total_row_time_s / success_count:.4f}s")
        print(f"Avg render/save time per successful row: {total_render_save_time_s / success_count:.4f}s")
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