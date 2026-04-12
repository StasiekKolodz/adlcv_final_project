import argparse
import csv
import json
import re
import string
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable

import numpy as np
from tqdm.auto import tqdm

try:
    import easyocr
except ImportError as exc:
    raise ImportError(
        "easyocr is required for evaluate_ocr.py. Install it with: pip install easyocr"
    ) from exc

try:
    from jiwer import cer as jiwer_cer
except ImportError:
    jiwer_cer = None


# --- CONFIGURATION ---
METADATA_LOG_PATH = "test_outputs/metadata_log.jsonl"
EVAL_RESULTS_CSV = "test_outputs/evaluation_results.csv"
DATASET_DIR = "stripe_text_dataset"

ALLOWED_CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,-!'()?"
OCR_FAILURE_CODE = "OCR_FAILURE"

SEED = 42
OCR_GPU = True
# ---------------------


WORD_RE = re.compile(r"[A-Za-z]+")
WS_RE = re.compile(r"\s+")


@dataclass
class EvalRow:
    image_id: str
    target_word: str
    word_length: int
    ocr_full_string: str
    extracted_prediction: str
    is_whitespace_deletion: bool
    word_cer: float
    word_wer: int
    global_sentence_cer: float


def set_seeds(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)


def normalize_sentence(text: str) -> str:
    text = text.lower().strip()
    return WS_RE.sub(" ", text)


def normalize_word(text: str) -> str:
    text = text.lower().strip()
    text = text.strip(string.punctuation + " ")
    text = WS_RE.sub(" ", text)
    return text


def levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            ins = curr[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            curr.append(min(ins, delete, sub))
        prev = curr
    return prev[-1]


def compute_cer(reference: str, hypothesis: str) -> float:
    if jiwer_cer is not None:
        return float(jiwer_cer(reference, hypothesis))

    if len(reference) == 0:
        return 0.0 if len(hypothesis) == 0 else 1.0
    return levenshtein_distance(reference, hypothesis) / max(1, len(reference))


def binary_word_wer(reference_word: str, predicted_word: str) -> int:
    return 0 if reference_word == predicted_word else 1


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Metadata log not found: {path}")

    rows = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {i} in {path}: {exc}") from exc
            rows.append(item)

    if not rows:
        raise RuntimeError(f"No rows found in metadata log: {path}")

    return rows


def ocr_read_line(reader: easyocr.Reader, image_path: Path) -> str:
    result = reader.readtext(
        str(image_path),
        allowlist=ALLOWED_CHARS,
        detail=0,
        paragraph=False,
    )
    return " ".join(result).strip()


def token_spans(text: str) -> list[re.Match[str]]:
    return [m for m in WORD_RE.finditer(text)]


def find_target_index(original_text: str, target_word: str, char_start_idx: int) -> int | None:
    words = token_spans(original_text)
    if not words:
        return None

    for i, m in enumerate(words):
        if m.start() == char_start_idx:
            return i

    for i, m in enumerate(words):
        if m.start() <= char_start_idx < m.end():
            return i

    target_l = target_word.lower()
    candidates = [(i, abs(m.start() - char_start_idx)) for i, m in enumerate(words) if m.group().lower() == target_l]
    if candidates:
        candidates.sort(key=lambda x: x[1])
        return candidates[0][0]

    return min(range(len(words)), key=lambda i: abs(words[i].start() - char_start_idx))


def best_anchor_index(words: list[re.Match[str]], anchor: str, prefer_from: int = 0) -> int | None:
    if not words or not anchor:
        return None

    anchor_l = anchor.lower()
    exact = [i for i, w in enumerate(words) if w.group().lower() == anchor_l]
    if exact:
        return min(exact, key=lambda i: abs(i - prefer_from))

    best_i = None
    best_score = -1.0
    for i, w in enumerate(words):
        score = SequenceMatcher(None, anchor_l, w.group().lower()).ratio()
        if score > best_score:
            best_score = score
            best_i = i

    if best_i is not None and best_score >= 0.62:
        return best_i
    return None


def extract_by_difflib(original_text: str, ocr_text: str, target_start: int, target_len: int) -> str:
    target_end = target_start + max(1, target_len)
    matcher = SequenceMatcher(None, original_text, ocr_text)

    best_block = None
    best_overlap = -1
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        overlap = max(0, min(i2, target_end) - max(i1, target_start))
        if overlap > best_overlap:
            best_overlap = overlap
            best_block = (tag, i1, i2, j1, j2)

    if best_block is None:
        return ""

    tag, i1, i2, j1, j2 = best_block
    if tag == "equal":
        left = max(i1, target_start)
        right = min(i2, target_end)
        if right <= left:
            return ""
        mapped_start = j1 + (left - i1)
        mapped_end = mapped_start + (right - left)
        return ocr_text[mapped_start:mapped_end].strip()

    return ocr_text[j1:j2].strip()


def extract_prediction(
    original_text: str,
    ocr_text: str,
    target_word: str,
    char_start_idx: int,
) -> str:
    original_words = token_spans(original_text)
    ocr_words = token_spans(ocr_text)
    if not original_words:
        return ""

    target_idx = find_target_index(original_text, target_word, char_start_idx)
    if target_idx is None:
        return extract_by_difflib(original_text, ocr_text, char_start_idx, len(target_word))

    prev_anchor = original_words[target_idx - 1].group() if target_idx > 0 else None
    next_anchor = original_words[target_idx + 1].group() if target_idx + 1 < len(original_words) else None

    prev_idx = best_anchor_index(ocr_words, prev_anchor, prefer_from=max(0, target_idx - 1)) if prev_anchor else None
    next_idx = best_anchor_index(ocr_words, next_anchor, prefer_from=target_idx + 1) if next_anchor else None

    if prev_idx is not None and next_idx is not None:
        if next_idx <= prev_idx + 1:
            return ""
        chunk = " ".join(w.group() for w in ocr_words[prev_idx + 1 : next_idx]).strip()
        if chunk:
            return chunk
        return ""

    if prev_idx is not None:
        if prev_idx + 1 < len(ocr_words):
            return ocr_words[prev_idx + 1].group().strip()
        return ""

    if next_idx is not None:
        if next_idx - 1 >= 0:
            return ocr_words[next_idx - 1].group().strip()
        return ""

    return extract_by_difflib(original_text, ocr_text, char_start_idx, len(target_word))


def get_source_image_path(dataset_dir: Path, item: dict) -> Path:
    if "source_image_path" in item:
        return Path(item["source_image_path"])

    image_id = str(item.get("image_id", "")).strip()
    if not image_id:
        raise ValueError("Metadata row missing image_id and source_image_path")
    return dataset_dir / f"{image_id}.png"


def append_csv_row(csv_path: Path, row: EvalRow, write_header: bool) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_id",
                "target_word",
                "word_length",
                "ocr_full_string",
                "extracted_prediction",
                "is_whitespace_deletion",
                "word_cer",
                "word_wer",
                "global_sentence_cer",
            ],
        )
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "image_id": row.image_id,
                "target_word": row.target_word,
                "word_length": row.word_length,
                "ocr_full_string": row.ocr_full_string,
                "extracted_prediction": row.extracted_prediction,
                "is_whitespace_deletion": row.is_whitespace_deletion,
                "word_cer": f"{row.word_cer:.6f}",
                "word_wer": row.word_wer,
                "global_sentence_cer": f"{row.global_sentence_cer:.6f}",
            }
        )


def existing_image_ids(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()

    seen = set()
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = (row.get("image_id") or "").strip()
            if image_id:
                seen.add(image_id)
    return seen


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else float("nan")


def run_baseline_calibration(reader: easyocr.Reader, metadata_rows: list[dict], dataset_dir: Path) -> None:
    cers = []
    failures = 0

    pbar = tqdm(metadata_rows, desc="Baseline OCR calibration", total=len(metadata_rows))
    for item in pbar:
        original_text = str(item.get("original_text", ""))
        try:
            src_img = get_source_image_path(dataset_dir, item)
            ocr_text = ocr_read_line(reader, src_img)
            cers.append(compute_cer(normalize_sentence(original_text), normalize_sentence(ocr_text)))
        except Exception:
            failures += 1
            continue

        pbar.set_postfix({"mean_cer": f"{mean(cers):.4f}", "failures": failures})

    print("Baseline calibration complete.")
    print(f"Samples             : {len(metadata_rows)}")
    print(f"Successful OCR      : {len(cers)}")
    print(f"Failures            : {failures}")
    print(f"Mean sentence CER   : {mean(cers):.6f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate generated inpainted stripes with EasyOCR")
    parser.add_argument("--metadata", default=METADATA_LOG_PATH, help="Path to metadata_log.jsonl from Script A")
    parser.add_argument("--output", default=EVAL_RESULTS_CSV, help="Output CSV path")
    parser.add_argument("--dataset-dir", default=DATASET_DIR, help="Directory with original source stripes")
    parser.add_argument("--cpu", action="store_true", help="Force CPU OCR (disables GPU)")
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Run baseline OCR on original images before evaluating generated outputs",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute rows even if image_id already exists in output CSV",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seeds(SEED)

    metadata_path = Path(args.metadata)
    output_csv = Path(args.output)
    dataset_dir = Path(args.dataset_dir)

    metadata_rows = load_jsonl(metadata_path)

    gpu_enabled = OCR_GPU and (not args.cpu)
    reader = easyocr.Reader(["en"], gpu=gpu_enabled)

    if args.calibrate:
        run_baseline_calibration(reader, metadata_rows, dataset_dir)

    done_ids = set() if args.overwrite else existing_image_ids(output_csv)
    write_header = (not output_csv.exists()) or args.overwrite
    if args.overwrite and output_csv.exists():
        output_csv.unlink()

    produced = 0
    skipped_existing = 0
    failures = 0

    pbar = tqdm(metadata_rows, desc="OCR evaluation", total=len(metadata_rows))
    for item in pbar:
        image_id = str(item.get("image_id", "")).strip()
        original_text = str(item.get("original_text", ""))
        target_word = str(item.get("masked_word", ""))
        word_length = int(item.get("masked_word_length", len(target_word)))
        char_start_idx = int(item.get("char_start_idx", -1))
        output_path = Path(str(item.get("output_path", "")))

        if not image_id:
            failures += 1
            pbar.set_postfix({"failures": failures})
            continue

        if not args.overwrite and image_id in done_ids:
            skipped_existing += 1
            pbar.set_postfix({"skipped": skipped_existing, "produced": produced, "failures": failures})
            continue

        try:
            if not output_path.exists():
                raise FileNotFoundError(f"Generated output image not found: {output_path}")

            ocr_full = ocr_read_line(reader, output_path)
            extracted = extract_prediction(original_text, ocr_full, target_word, char_start_idx)

            target_norm = normalize_word(target_word)
            extracted_norm = normalize_word(extracted)
            sentence_ref = normalize_sentence(original_text)
            sentence_hyp = normalize_sentence(ocr_full)

            whitespace_deletion = extracted_norm == ""

            row = EvalRow(
                image_id=image_id,
                target_word=target_word,
                word_length=word_length,
                ocr_full_string=ocr_full,
                extracted_prediction=extracted,
                is_whitespace_deletion=whitespace_deletion,
                word_cer=compute_cer(target_norm, extracted_norm),
                word_wer=binary_word_wer(target_norm, extracted_norm),
                global_sentence_cer=compute_cer(sentence_ref, sentence_hyp),
            )
        except Exception:
            failures += 1
            row = EvalRow(
                image_id=image_id,
                target_word=target_word,
                word_length=word_length,
                ocr_full_string=OCR_FAILURE_CODE,
                extracted_prediction=OCR_FAILURE_CODE,
                is_whitespace_deletion=False,
                word_cer=1.0,
                word_wer=1,
                global_sentence_cer=1.0,
            )

        append_csv_row(output_csv, row, write_header=write_header)
        write_header = False
        done_ids.add(image_id)

        produced += 1
        pbar.set_postfix({"produced": produced, "skipped": skipped_existing, "failures": failures})

    print("Done.")
    print(f"Metadata rows    : {len(metadata_rows)}")
    print(f"Rows written     : {produced}")
    print(f"Skipped existing : {skipped_existing}")
    print(f"OCR failures     : {failures}")
    print(f"Output CSV       : {output_csv}")


if __name__ == "__main__":
    main()
