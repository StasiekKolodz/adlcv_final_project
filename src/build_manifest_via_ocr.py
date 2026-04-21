import json
from pathlib import Path
from tqdm.auto import tqdm

try:
    import easyocr
except ImportError as exc:
    raise ImportError("easyocr is required. Install it with: pip install easyocr") from exc

# --- CONFIGURATION ---
DATASET_DIR = Path("stripe_text_dataset")
TEST_SPLIT_FILE = Path("splits/test.txt")
OUTPUT_MANIFEST = Path("splits/test_manifest.jsonl")

# Restrict to your exact dataset characters to prevent wild hallucinations
ALLOWED_CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,-!'()?"
TARGET_COUNT = 1000  # The exact number of images we want for Task 4
# ---------------------


def load_existing_image_paths(manifest_path: Path) -> set[str]:
    if not manifest_path.exists():
        return set()

    seen = set()
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # Skip malformed historical lines and continue.
                continue
            image_path = row.get("image_path")
            if isinstance(image_path, str) and image_path:
                seen.add(image_path)
    return seen

def build_auto_manifest():
    if not TEST_SPLIT_FILE.exists():
        raise FileNotFoundError(f"Cannot find {TEST_SPLIT_FILE}")
        
    with open(TEST_SPLIT_FILE, "r") as f:
        filenames = [line.strip() for line in f if line.strip()]
        
    print(f"Found {len(filenames)} images in test split. Booting EasyOCR...")
    reader = easyocr.Reader(['en'], gpu=True)
    
    existing_paths = load_existing_image_paths(OUTPUT_MANIFEST)
    successful_rows = len(existing_paths)
    ocr_failures = 0

    if successful_rows >= TARGET_COUNT:
        print(
            f"Manifest already has {successful_rows} rows (>= TARGET_COUNT={TARGET_COUNT}). "
            "Nothing to do."
        )
        return

    OUTPUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    
    with open(OUTPUT_MANIFEST, "a", encoding="utf-8") as out:
        pbar = tqdm(filenames, desc="Reverse-engineering Ground Truth")
        
        for filename in pbar:
            # STOP CONDITION: We hit our 1,000 image target
            if successful_rows >= TARGET_COUNT:
                break

            # Resume-safe: skip rows already written in prior runs.
            if filename in existing_paths:
                continue
                
            img_path = DATASET_DIR / filename
            if not img_path.exists():
                continue
                
            # Run OCR
            try:
                result = reader.readtext(str(img_path), allowlist=ALLOWED_CHARS, detail=0, paragraph=False)
                extracted_text = " ".join(result).strip()
            except Exception:
                ocr_failures += 1
                pbar.set_postfix({"Verified": successful_rows, "OCR_failures": ocr_failures})
                continue
            
            # Geometry Safety Check: Skip empty strings or weirdly short reads
            # This ensures we don't accidentally chop a word in half later due to bad OCR spacing
            if len(extracted_text) < 15:
                continue
                
            # Save to manifest
            manifest_row = {"image_path": filename, "text": extracted_text}
            out.write(json.dumps(manifest_row) + "\n")
            successful_rows += 1
            existing_paths.add(filename)
            
            pbar.set_postfix({"Verified": successful_rows, "OCR_failures": ocr_failures})
            
    if successful_rows >= TARGET_COUNT:
        print(f"\nSuccess! Built manifest to target size ({successful_rows} rows, target={TARGET_COUNT}).")
    else:
        print(
            f"\nCompleted, but target not reached: {successful_rows}/{TARGET_COUNT} rows. "
            "Increase split size or relax filtering if needed."
        )
    print(f"OCR failures skipped: {ocr_failures}")
    print("You can now safely run generate_test_set.py")

if __name__ == "__main__":
    build_auto_manifest()