import random
import re
from pathlib import Path

# --- CONFIGURATION ---
DATASET_DIR = Path("stripe_text_dataset")
OUTPUT_DIR = Path("splits")
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


def write_manifest(path: Path, rel_paths: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # One relative path per line for easy loading in Dataset.
    path.write_text("\n".join(rel_paths) + ("\n" if rel_paths else ""), encoding="utf-8")


def main() -> None:
    validate_ratios(TRAIN_RATIO, VAL_RATIO, TEST_RATIO)

    dataset_dir = DATASET_DIR.resolve()
    output_dir = OUTPUT_DIR.resolve()

    if not dataset_dir.exists() or not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    all_pngs = sorted(p for p in dataset_dir.iterdir() if p.is_file() and p.suffix.lower() == ".png")
    if not all_pngs:
        raise RuntimeError(f"No PNG files found in: {dataset_dir}")

    row_to_files: dict[str, list[str]] = {}
    skipped = 0

    for png_path in all_pngs:
        row_id = parse_row_id(png_path.name)
        if row_id is None:
            skipped += 1
            continue

        rel_path = str(png_path.relative_to(dataset_dir))
        row_to_files.setdefault(row_id, []).append(rel_path)

    if not row_to_files:
        raise RuntimeError(
            "No files matched expected pattern 'stripe_<row_id>_<chunk_id>.png'. "
            f"Checked {len(all_pngs)} png files in {dataset_dir}"
        )

    row_ids = list(row_to_files.keys())
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

    train_files: list[str] = []
    val_files: list[str] = []
    test_files: list[str] = []

    for row_id, files in row_to_files.items():
        files_sorted = sorted(files)
        if row_id in train_rows:
            train_files.extend(files_sorted)
        elif row_id in val_rows:
            val_files.extend(files_sorted)
        else:
            test_files.extend(files_sorted)

    write_manifest(output_dir / "train.txt", sorted(train_files))
    write_manifest(output_dir / "val.txt", sorted(val_files))
    write_manifest(output_dir / "test.txt", sorted(test_files))

    print("Split creation complete")
    print(f"Dataset dir : {dataset_dir}")
    print(f"Output dir  : {output_dir}")
    print(f"Seed        : {SEED}")
    print(f"Rows total  : {n_rows} (train={len(train_rows)}, val={len(val_rows)}, test={len(test_rows)})")
    print(
        "Files total : "
        f"{len(train_files) + len(val_files) + len(test_files)} "
        f"(train={len(train_files)}, val={len(val_files)}, test={len(test_files)})"
    )
    if skipped > 0:
        print(f"Skipped {skipped} PNG files that did not match expected naming pattern.")


if __name__ == "__main__":
    main()
