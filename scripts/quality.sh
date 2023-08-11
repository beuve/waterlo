ORIGINALS=/workspaces/watermark/images/ssim-0002-512/originals
WATERMARKED=/workspaces/watermark/images/ssim-0002-512-c/watermarks

echo "DATASET:      $DATASET"

CUDA_LAUNCH_BLOCKING=1 python3 src/quality_check.py --originals $ORIGINALS --watermarked $WATERMARKED