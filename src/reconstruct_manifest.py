import os
import re
import json
from pathlib import Path
from collections import defaultdict
from datasets import load_from_disk
from tqdm.auto import tqdm

# --- CONFIGURATION ---
TEST_SPLIT_FILE = Path("splits/test.txt")
OUTPUT_MANIFEST = Path("splits/test_manifest.jsonl")
ERROR_LOG_PATH = Path("splits/reconstruct_manifest_errors.jsonl")
LOCAL_DATASET_PATH = "./local_fineweb"
TARGET_HEIGHT = 16
TARGET_WIDTH = 1024
MONO_FONT_SIZE = 16
MONO_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"

# Regex to extract idx and chunk_idx from 'stripe_000123_01.png'
FILENAME_PATTERN = re.compile(r"stripe_(\d+)_(\d+)\.png")
# ---------------------

def init_renderer():
    """Boots up the exact same renderer used in generation."""
    from pixel_renderer import PixelRendererProcessor
    from font_download import FontConfig
    from font_download.fonts import FontSource
    
    mono_font_abs = os.path.abspath(MONO_FONT_PATH)
    mono_source = FontSource(url=f"file://{mono_font_abs}")
    font_config = FontConfig(sources=[mono_source])
    return PixelRendererProcessor(font=font_config)

def get_text_chunks(raw_text, renderer):
    """Replicates the exact chunking logic from generate_imgs_paralel.py"""
    clean_text = " ".join(raw_text.split())
    words = clean_text.split()
    if not words:
        return []

    text_chunks = []
    current_chunk = ""

    for word in words:
        candidate = word if not current_chunk else f"{current_chunk} {word}"
        candidate_img = renderer.render_text_image(candidate, block_size=TARGET_HEIGHT, font_size=MONO_FONT_SIZE)

        if candidate_img.width <= TARGET_WIDTH and candidate_img.height <= TARGET_HEIGHT:
            current_chunk = candidate
            continue

        if current_chunk:
            text_chunks.append(current_chunk)
            single_word_img = renderer.render_text_image(word, block_size=TARGET_HEIGHT, font_size=MONO_FONT_SIZE)
            if single_word_img.width <= TARGET_WIDTH and single_word_img.height <= TARGET_HEIGHT:
                current_chunk = word
            else:
                current_chunk = ""
        else:
            pass # Skipped overlong word

    if current_chunk:
        text_chunks.append(current_chunk)

    return text_chunks

def main():
    if not TEST_SPLIT_FILE.exists():
        raise FileNotFoundError(f"Cannot find {TEST_SPLIT_FILE}")

    # 1. Parse the required indices from the test split filenames
    print("Parsing required indices from test split...")
    tasks = defaultdict(list)
    malformed_filenames = []
    with open(TEST_SPLIT_FILE, "r") as f:
        for line in f:
            filename = line.strip()
            if not filename:
                continue
            match = FILENAME_PATTERN.search(filename)
            if match:
                idx = int(match.group(1))
                chunk_idx = int(match.group(2))
                tasks[idx].append((chunk_idx, filename))
            else:
                malformed_filenames.append(filename)

    if malformed_filenames:
        print(
            f"Warning: {len(malformed_filenames)} lines in {TEST_SPLIT_FILE} did not match expected pattern "
            f"'{FILENAME_PATTERN.pattern}'."
        )
        preview = malformed_filenames[:10]
        for bad_name in preview:
            print(f"  - malformed split entry: {bad_name}")
        if len(malformed_filenames) > len(preview):
            print(f"  ... and {len(malformed_filenames) - len(preview)} more malformed entries.")
    
    print(f"Found {sum(len(v) for v in tasks.values())} images spanning {len(tasks)} dataset rows.")

    # 2. Load dataset and renderer
    print(f"Loading dataset from {LOCAL_DATASET_PATH}...")
    dataset = load_from_disk(LOCAL_DATASET_PATH)
    print("Booting pixel renderer...")
    renderer = init_renderer()

    # 3. Process and write manifest
    OUTPUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    success_count = 0
    error_count = 0
    error_rows = []

    with open(OUTPUT_MANIFEST, "w", encoding="utf-8") as out:
        pbar = tqdm(tasks.items(), desc="Reconstructing Manifest")
        
        for idx, chunk_requests in pbar:
            try:
                # Fetch original text from HuggingFace dataset
                raw_text = dataset[idx]["text"]
                
                # Re-run the exact pixel-aware mathematical chunking
                chunks = get_text_chunks(raw_text, renderer)
                
                # Fulfill all requested chunk images for this row
                for chunk_idx, filename in chunk_requests:
                    if chunk_idx < len(chunks):
                        ground_truth_text = chunks[chunk_idx]
                        manifest_row = {"image_path": filename, "text": ground_truth_text}
                        out.write(json.dumps(manifest_row) + "\n")
                        success_count += 1
                    else:
                        error_count += 1
                        error_rows.append(
                            {
                                "idx": idx,
                                "filename": filename,
                                "error_type": "MissingChunk",
                                "message": (
                                    f"chunk_idx={chunk_idx} out of range for reconstructed chunks "
                                    f"(available={len(chunks)})"
                                ),
                            }
                        )
            except Exception as e:
                error_count += len(chunk_requests)
                error_rows.append(
                    {
                        "idx": idx,
                        "filenames": [fname for _, fname in chunk_requests],
                        "error_type": type(e).__name__,
                        "message": str(e),
                    }
                )
                
            pbar.set_postfix({"Saved": success_count, "Errors": error_count})

    print(f"\nDone! Successfully reconstructed {success_count} exact ground-truth lines.")
    if error_count > 0:
        print(f"Failed to match {error_count} chunks (likely due to dataset misalignment).")
        ERROR_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(ERROR_LOG_PATH, "w", encoding="utf-8") as f:
            for row in error_rows:
                f.write(json.dumps(row, ensure_ascii=True) + "\n")
        print(f"Detailed reconstruction errors saved to: {ERROR_LOG_PATH}")

    if malformed_filenames:
        print(
            f"Malformed split entries (not processed): {len(malformed_filenames)}. "
            f"See warnings above to fix {TEST_SPLIT_FILE}."
        )

if __name__ == "__main__":
    main()