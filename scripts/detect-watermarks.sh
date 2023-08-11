DATASET= # Path to tested data.
WEIGHTS= # Path to trained weights.
OUTPUTS= # Path to save the detection maps.

COMPRESSION_QUALITY=10 # Not taken into account if > 100

echo "DATASET:      $DATASET"

CUDA_LAUNCH_BLOCKING=1 python3 src/detect_watermark.py --dataset $DATASET --weights $WEIGHTS --output $OUTPUTS --compression $COMPRESSION_QUALITY 