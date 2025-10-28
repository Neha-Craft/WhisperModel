# Model Path Formats

The `--model-path` parameter accepts:

## File Path
- **`.pt` format only** (required for AlignAtt policy decoder)

## Directory Path (recommended)
Must contain:
- **`.pt` file** (required for decoder)

May optionally contain:
- **`.bin` file** - faster-whisper model for encoder (requires faster-whisper)
- **`weights.npz`** or **`weights.safetensors`** - for encoder (requires whisper-mlx)