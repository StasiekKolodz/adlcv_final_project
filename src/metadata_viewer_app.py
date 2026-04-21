import json
from pathlib import Path
from typing import Any
from urllib.parse import quote

from PIL import Image, ImageFont
from nicegui import app, ui

# --- CONFIGURATION ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
METADATA_LOG_PATH = PROJECT_ROOT / "test_outputs" / "metadata_log.jsonl"
DATASET_DIR = PROJECT_ROOT / "stripe_text_dataset"
COVERED_CACHE_DIR = PROJECT_ROOT / "test_outputs" / "covered_preview"

DEFAULT_WORD_PADDING_PX = 2
STATIC_FILES_ROUTE = "/viewer-files"

# Must match the font the dataset was rendered with (see generate_imgs_paralel.py).
# Pango's "sans" family is aliased to this TTF *inside the renderer workers* via
# fontconfig — but that alias doesn't exist elsewhere, so we measure via PIL on
# the TTF directly to get identical advance widths.
MONO_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
MONO_FONT_SIZE_PX = 16
RENDERER_LEFT_PAD_PX = 5
# ---------------------

_pango_layout = None
_pango_surface = None

try:
    import cairo as _cairo
    import gi as _gi
    _gi.require_version("Pango", "1.0")
    _gi.require_version("PangoCairo", "1.0")
    _gi.require_foreign("cairo")
    from gi.repository import Pango as _Pango, PangoCairo as _PangoCairo
    _PANGO_AVAILABLE = True
except Exception:
    _PANGO_AVAILABLE = False

try:
    _mono_font: ImageFont.FreeTypeFont | None = ImageFont.truetype(MONO_FONT_PATH, MONO_FONT_SIZE_PX)
except OSError:
    _mono_font = None


def _prefix_width_px(text: str) -> int | None:
    if not text:
        return 0
    if _PANGO_AVAILABLE:
        global _pango_layout, _pango_surface
        if _pango_layout is None:
            _pango_surface = _cairo.ImageSurface(_cairo.FORMAT_RGB24, 1, 1)
            ctx = _cairo.Context(_pango_surface)
            _pango_layout = _PangoCairo.create_layout(ctx)
            font_desc = _Pango.font_description_from_string(f"DejaVu Sans Mono {MONO_FONT_SIZE_PX}px")
            _pango_layout.set_font_description(font_desc)
        _pango_layout.set_text(text, -1)
        width, _ = _pango_layout.get_pixel_size()
        return width
    if _mono_font is None:
        return None
    return int(round(_mono_font.getlength(text)))

app.add_static_files(STATIC_FILES_ROUTE, str(PROJECT_ROOT))


def load_metadata_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Metadata log not found: {path}")

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            item["_line_number"] = line_number
            rows.append(item)

    if not rows:
        raise RuntimeError(f"No valid JSON rows found in {path}")
    return rows


def resolve_reconstruction_path(entry: dict[str, Any]) -> Path:
    output_path = Path(str(entry.get("output_path", "")))
    if output_path.is_absolute():
        return output_path
    return PROJECT_ROOT / output_path


def resolve_uncovered_path(entry: dict[str, Any]) -> Path:
    source = entry.get("source_image_path") or entry.get("image_path")
    if isinstance(source, str) and source:
        source_path = Path(source)
        if source_path.is_absolute():
            return source_path
        return PROJECT_ROOT / source_path

    image_id = str(entry.get("image_id", "")).strip()
    if not image_id:
        return DATASET_DIR / "__missing__.png"
    return DATASET_DIR / f"{image_id}.png"


def image_width(path: Path, fallback: int = 1024) -> int:
    try:
        with Image.open(path) as img:
            return img.width
    except Exception:
        return fallback


def compute_mask_pixel_range(entry: dict[str, Any], width: int) -> tuple[int, int]:
    if "mask_pixel_start" in entry and "mask_pixel_end" in entry:
        s = max(0, min(int(entry["mask_pixel_start"]), width))
        e = max(0, min(int(entry["mask_pixel_end"]), width))
        return s, e

    text = str(entry.get("original_text", ""))
    char_start = int(entry.get("char_start_idx", 0))
    word_len = int(entry.get("masked_word_length", 0))
    pad = int(entry.get("word_padding_px", DEFAULT_WORD_PADDING_PX))

    prefix_w = _prefix_width_px(text[:char_start])
    end_w = _prefix_width_px(text[:char_start + word_len])
    if prefix_w is None or end_w is None:
        # Pango not available and no stored range — cannot place the mask accurately.
        return 0, 0

    start = RENDERER_LEFT_PAD_PX + prefix_w - pad
    end = RENDERER_LEFT_PAD_PX + end_w + pad
    start = max(0, min(start, width))
    end = max(0, min(end, width))
    return start, end


def build_covered_preview(entry: dict[str, Any]) -> Path | None:
    uncovered = resolve_uncovered_path(entry)
    if not uncovered.exists():
        return None

    width = image_width(uncovered)
    start, end = compute_mask_pixel_range(entry, width)

    reconstruction = resolve_reconstruction_path(entry)
    cache_name = f"{reconstruction.stem}__s{start}_e{end}__covered.png"
    cache_path = COVERED_CACHE_DIR / cache_name
    if cache_path.exists():
        return cache_path

    COVERED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with Image.open(uncovered) as img:
        masked = img.convert("L")
        _, height = masked.size
        if end > start:
            white_patch = Image.new("L", (end - start, height), color=255)
            masked.paste(white_patch, (start, 0))
        masked.save(cache_path, format="PNG", optimize=True)

    return cache_path


def file_to_url(path: Path | None) -> str:
    if path is None or not path.exists():
        return ""
    try:
        rel = path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return ""
    return f"{STATIC_FILES_ROUTE}/{quote(rel)}"


def metadata_items(entry: dict[str, Any]) -> list[tuple[str, str]]:
    preferred_order = [
        "image_id",
        "image_path",
        "source_image_path",
        "output_path",
        "original_text",
        "masked_word",
        "masked_word_length",
        "char_start_idx",
        "mask_pixel_start",
        "mask_pixel_end",
        "start_timestep",
        "resampling_number",
    ]

    items: list[tuple[str, str]] = []
    for key in preferred_order:
        if key in entry:
            items.append((key, str(entry[key])))

    remaining = [k for k in entry.keys() if k not in preferred_order and not k.startswith("_")]
    for key in sorted(remaining):
        items.append((key, str(entry[key])))

    return items


def highlight_text(text: str, word: str, char_start: int) -> str:
    """Return HTML with the masked word highlighted."""
    if not text:
        return ""
    if not word:
        return text
    # Prefer positional match when char_start is valid.
    if 0 <= char_start <= len(text) and text[char_start:char_start + len(word)].lower() == word.lower():
        before = text[:char_start]
        match = text[char_start:char_start + len(word)]
        after = text[char_start + len(word):]
    else:
        lower = text.lower()
        idx = lower.find(word.lower())
        if idx < 0:
            return text
        before, match, after = text[:idx], text[idx:idx + len(word)], text[idx + len(word):]

    def esc(s: str) -> str:
        return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))

    return (
        f"{esc(before)}"
        f'<span style="background:#fde68a;color:#111;padding:1px 4px;border-radius:3px;font-weight:600;">{esc(match)}</span>'
        f"{esc(after)}"
    )


def main() -> None:
    rows = load_metadata_rows(METADATA_LOG_PATH)
    state = {"idx": 0}

    ui.add_head_html(
        "<style>"
        ".stripe-img img { image-rendering: pixelated; width: 100%; height: auto; display: block; }"
        ".stripe-box { background: #0b1220; padding: 4px; border-radius: 6px; }"
        "</style>"
    )

    with ui.row().classes("w-full items-center justify-between"):
        ui.label("Test Outputs Metadata Viewer").classes("text-2xl font-bold")
        count_label = ui.label("").classes("text-sm text-gray-500")

    options = {
        str(i): f"[{i:03d}] {rows[i].get('image_id') or Path(str(rows[i].get('output_path', ''))).stem or f'row_{i}'}"
                f"  —  {rows[i].get('masked_word', '')}"
        for i in range(len(rows))
    }

    with ui.row().classes("w-full items-center gap-2 flex-wrap"):
        prev_btn = ui.button(icon="chevron_left").props("flat")
        image_select = ui.select(options=options, label="Select image", with_input=True).classes("w-[28rem]")
        next_btn = ui.button(icon="chevron_right").props("flat")
        ui.space()
        open_btn = ui.button("Open reconstruction", icon="open_in_new").props("outline")
        reload_btn = ui.button("Reload log", icon="refresh").props("outline")

    ui.separator()

    text_label = ui.html("").classes("text-base leading-relaxed w-full")

    def stripe_panel(title: str):
        with ui.column().classes("w-full gap-1"):
            ui.label(title).classes("text-sm font-semibold text-gray-700")
            with ui.element("div").classes("stripe-box w-full"):
                img = ui.image().classes("stripe-img w-full")
            status = ui.label("").classes("text-xs text-gray-500")
            return img, status

    with ui.column().classes("w-full gap-3"):
        uncovered_img, uncovered_status = stripe_panel("Uncovered (original)")
        covered_img, covered_status = stripe_panel("Covered (masked input)")
        recon_img, recon_status = stripe_panel("Reconstruction (generated)")

    ui.separator()
    ui.label("Metadata").classes("text-lg font-semibold")
    metadata_grid = ui.grid(columns=2).classes("w-full gap-x-6 gap-y-1")

    def set_image(img_widget: ui.image, status: ui.label, path: Path | None) -> None:
        url = file_to_url(path)
        if url:
            img_widget.set_source(url)
            img_widget.set_visibility(True)
            status.text = str(path.relative_to(PROJECT_ROOT)) if path and path.is_relative_to(PROJECT_ROOT) else str(path)
        else:
            img_widget.set_source("")
            img_widget.set_visibility(False)
            status.text = f"⚠ missing: {path}" if path else "⚠ no path"

    def update_view() -> None:
        if not rows:
            count_label.text = "No entries"
            return

        state["idx"] = max(0, min(state["idx"], len(rows) - 1))
        entry = rows[state["idx"]]

        uncovered = resolve_uncovered_path(entry)
        try:
            covered = build_covered_preview(entry)
        except Exception as exc:  # noqa: BLE001
            covered = None
            ui.notify(f"Failed to build covered preview: {exc}", color="negative")
        recon = resolve_reconstruction_path(entry)

        set_image(uncovered_img, uncovered_status, uncovered)
        set_image(covered_img, covered_status, covered)
        set_image(recon_img, recon_status, recon)

        image_select.value = str(state["idx"])
        count_label.text = f"Entry {state['idx'] + 1} / {len(rows)}"

        text_label.content = highlight_text(
            str(entry.get("original_text", "")),
            str(entry.get("masked_word", "")),
            int(entry.get("char_start_idx", -1)) if str(entry.get("char_start_idx", "")).lstrip("-").isdigit() else -1,
        )

        metadata_grid.clear()
        with metadata_grid:
            for key, value in metadata_items(entry):
                ui.label(key).classes("text-sm text-gray-500")
                ui.label(value).classes("text-sm font-medium break-all")

    def on_select_change() -> None:
        try:
            requested = int(image_select.value)
        except (TypeError, ValueError):
            requested = 0
        state["idx"] = max(0, min(requested, len(rows) - 1))
        update_view()

    def step(delta: int) -> None:
        state["idx"] = (state["idx"] + delta) % len(rows)
        update_view()

    def open_reconstruction() -> None:
        if not rows:
            ui.notify("No entry selected", color="warning")
            return
        entry = rows[state["idx"]]
        recon = resolve_reconstruction_path(entry)
        if not recon.exists():
            ui.notify(f"Reconstruction file not found: {recon}", color="negative")
            return
        ui.run_javascript(f"window.open({json.dumps(file_to_url(recon))}, '_blank')")

    def reload_log() -> None:
        nonlocal rows
        try:
            rows = load_metadata_rows(METADATA_LOG_PATH)
        except Exception as exc:  # noqa: BLE001
            ui.notify(f"Reload failed: {exc}", color="negative")
            return
        new_options = {
            str(i): f"[{i:03d}] {rows[i].get('image_id') or Path(str(rows[i].get('output_path', ''))).stem or f'row_{i}'}"
                    f"  —  {rows[i].get('masked_word', '')}"
            for i in range(len(rows))
        }
        image_select.options = new_options
        image_select.update()
        state["idx"] = min(state["idx"], len(rows) - 1)
        update_view()
        ui.notify(f"Reloaded {len(rows)} entries", color="positive")

    image_select.on_value_change(lambda _: on_select_change())
    prev_btn.on("click", lambda _: step(-1))
    next_btn.on("click", lambda _: step(1))
    open_btn.on("click", lambda _: open_reconstruction())
    reload_btn.on("click", lambda _: reload_log())

    # Keyboard navigation: left / right arrows.
    ui.keyboard(on_key=lambda e: step(-1) if (e.action.keydown and e.key.arrow_left) else
                                 step(1) if (e.action.keydown and e.key.arrow_right) else None)

    image_select.value = "0"
    update_view()


if __name__ in {"__main__", "__mp_main__"}:
    main()
    ui.run(title="Metadata Viewer", reload=False)
