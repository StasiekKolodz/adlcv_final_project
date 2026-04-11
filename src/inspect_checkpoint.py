import json
from pathlib import Path

from diffusers import DDPMScheduler, UNet2DModel

# --- CONFIGURATION ---
# Examples:
#   "checkpoints/latest"
#   "checkpoints/best"
#   "ddpm_text_model"
#   "ddpm_text_model/best_model"
CHECKPOINT_DIR = "checkpoints/latest"
# ---------------------


def bytes_to_human(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{num_bytes} B"


def print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def inspect_trainer_state(checkpoint_dir: Path) -> None:
    trainer_state_path = checkpoint_dir / "trainer_state.json"
    if not trainer_state_path.exists():
        print("trainer_state.json: not found")
        return

    try:
        data = json.loads(trainer_state_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"trainer_state.json: failed to parse ({exc})")
        return

    print("trainer_state.json:")
    print(f"  epoch         : {data.get('epoch')}")
    print(f"  global_step   : {data.get('global_step')}")
    print(f"  best_val_loss : {data.get('best_val_loss')}")


def inspect_diffusers_artifacts(checkpoint_dir: Path) -> None:
    try:
        model = UNet2DModel.from_pretrained(str(checkpoint_dir))
        scheduler = DDPMScheduler.from_pretrained(str(checkpoint_dir))
    except Exception as exc:
        print(f"Diffusers load: failed ({exc})")
        print("This directory may be an Accelerator state checkpoint only.")
        return

    print("Diffusers load: success")

    cfg = model.config
    print("UNet2DModel config:")
    print(f"  sample_size         : {cfg.sample_size}")
    print(f"  in_channels         : {cfg.in_channels}")
    print(f"  out_channels        : {cfg.out_channels}")
    print(f"  layers_per_block    : {cfg.layers_per_block}")
    print(f"  block_out_channels  : {cfg.block_out_channels}")
    print(f"  down_block_types    : {cfg.down_block_types}")
    print(f"  up_block_types      : {cfg.up_block_types}")

    param_count = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("UNet2DModel params:")
    print(f"  total               : {param_count:,}")
    print(f"  trainable           : {trainable_count:,}")

    scfg = scheduler.config
    print("DDPMScheduler config:")
    print(f"  num_train_timesteps : {scfg.num_train_timesteps}")
    print(f"  prediction_type     : {scfg.prediction_type}")
    print(f"  beta_schedule       : {scfg.beta_schedule}")


def inspect_files(checkpoint_dir: Path) -> None:
    all_files = [p for p in checkpoint_dir.rglob("*") if p.is_file()]
    total_size = sum(p.stat().st_size for p in all_files)

    print(f"Files: {len(all_files)}")
    print(f"Total size: {bytes_to_human(total_size)}")

    largest = sorted(all_files, key=lambda p: p.stat().st_size, reverse=True)[:10]
    if largest:
        print("Top files by size:")
        for p in largest:
            rel = p.relative_to(checkpoint_dir)
            print(f"  {rel}  ({bytes_to_human(p.stat().st_size)})")


def main() -> None:
    checkpoint_dir = Path(CHECKPOINT_DIR)
    print_header("Checkpoint Inspector")
    print(f"Path: {checkpoint_dir}")

    if not checkpoint_dir.exists() or not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    inspect_files(checkpoint_dir)

    print_header("Trainer State")
    inspect_trainer_state(checkpoint_dir)

    print_header("Diffusers Artifacts")
    inspect_diffusers_artifacts(checkpoint_dir)


if __name__ == "__main__":
    main()
