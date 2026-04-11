import random
import re
import os
from pathlib import Path

# --- CONFIGURATION ---
DATASET_DIR = Path("../stripe_text_dataset")
OUTPUT_DIR = Path("../splits")
TRAIN_RATIO = 0.9
VAL_RATIO = 0.05
TEST_RATIO = 0.05
SEED = 42
# ---------------------

# Expected generated filename format: stripe_<row_id>_<chunk_id>.png
ROW_ID_PATTERN = re.compile(r"^stripe_(\d+)_\d+\.png$")


def validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-8:
        raise ValueError(
            f"Split ratios must sum to 1.0, got {total:.8f} "
            f"(train={train_ratio}, val={val_ratio}, test={test_ratio})"
        )


def parse_row_id(filename: str) -> str | None:
    match = ROW_ID_PATTERN.match(filename)
    if match is None:
        return None
    return match.group(1)


def iter_png_entries(dataset_dir: Path):
    # Use scandir for faster large-directory traversal with less overhead.
    with os.scandir(dataset_dir) as entries:
        for entry in entries:
            if not entry.is_file():
                continue
            if not entry.name.lower().endswith(".png"):
                continue
            yield entry.name


def main() -> None:
    validate_ratios(TRAIN_RATIO, VAL_RATIO, TEST_RATIO)

    dataset_dir = DATASET_DIR.resolve()
    output_dir = OUTPUT_DIR.resolve()

    if not dataset_dir.exists() or not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    # Pass 1: collect unique row IDs only (low memory footprint).
    row_ids_set: set[str] = set()
    total_pngs = 0
    skipped = 0

    for filename in iter_png_entries(dataset_dir):
        total_pngs += 1
        row_id = parse_row_id(filename)
        if row_id is None:
            skipped += 1
            continue
        row_ids_set.add(row_id)

    if total_pngs == 0:
        raise RuntimeError(f"No PNG files found in: {dataset_dir}")

    if not row_ids_set:
        raise RuntimeError(
            "No files matched expected pattern 'stripe_<row_id>_<chunk_id>.png'. "
            f"Checked {total_pngs} png files in {dataset_dir}"
        )

    # Sort before shuffling so split assignment is deterministic for a fixed seed.
    row_ids = sorted(row_ids_set)
    rng = random.Random(SEED)
    rng.shuffle(row_ids)

    n_rows = len(row_ids)
    n_train = int(n_rows * TRAIN_RATIO)
    n_val = int(n_rows * VAL_RATIO)
    # Remainder goes to test to ensure full coverage.
    n_test = n_rows - n_train - n_val

    train_rows = set(row_ids[:n_train])
    val_rows = set(row_ids[n_train : n_train + n_val])
    test_rows = set(row_ids[n_train + n_val :])

    assert len(train_rows | val_rows | test_rows) == n_rows
    assert len(train_rows & val_rows) == 0
    assert len(train_rows & test_rows) == 0
    assert len(val_rows & test_rows) == 0

    output_dir.mkdir(parents=True, exist_ok=True)
    train_manifest = output_dir / "train.txt"
    val_manifest = output_dir / "val.txt"
    test_manifest = output_dir / "test.txt"

    train_files = 0
    val_files = 0
    test_files = 0

    # Pass 2: stream files directly into manifests (no giant file lists in RAM).
    with (
        open(train_manifest, "w", encoding="utf-8") as train_f,
        open(val_manifest, "w", encoding="utf-8") as val_f,
        open(test_manifest, "w", encoding="utf-8") as test_f,
    ):
        for filename in iter_png_entries(dataset_dir):
            row_id = parse_row_id(filename)
            if row_id is None:
                continue

            rel_path = filename  # Files are expected to be directly under DATASET_DIR.
            if row_id in train_rows:
                train_f.write(rel_path + "\n")
                train_files += 1
            elif row_id in val_rows:
                val_f.write(rel_path + "\n")
                val_files += 1
            else:
                test_f.write(rel_path + "\n")
                test_files += 1

    print("Split creation complete")
    print(f"Dataset dir : {dataset_dir}")
    print(f"Output dir  : {output_dir}")
    print(f"Seed        : {SEED}")
    print(f"Rows total  : {n_rows} (train={len(train_rows)}, val={len(val_rows)}, test={len(test_rows)})")
    print(
        "Files total : "
        f"{train_files + val_files + test_files} "
        f"(train={train_files}, val={val_files}, test={test_files})"
    )
    if skipped > 0:
        print(f"Skipped {skipped} PNG files that did not match expected naming pattern.")


if __name__ == "__main__":
    main()
