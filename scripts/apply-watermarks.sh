DATASET= # Path to pristine data.
WEIGHTS= # Path to trained weights.
OUTPUTS= # Path to save the watermarked images.
DATASET=/workspaces/watermark/images/jnd-02-512/originals_full
WEIGHTS=/workspaces/watermark/weights/mse-0005-512
OUTPUTS=/workspaces/watermark/images/mse-0005-512/watermarks
ALPHA=0.005

echo "DATASET:      $DATASET"

python3 src/apply_watermark.py --dataset $DATASET --weights $WEIGHTS --output $OUTPUTS --alpha $ALPHA
