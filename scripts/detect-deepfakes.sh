DATASET=/workspaces/watermark/images/mse-0005-512/faceshifter
WEIGHTS=/workspaces/watermark/weights/mse-0005-512
OUTPUTS=/workspaces/watermark/images/mse-0005-512/detections/faceshifter

LABEL="d" # w | p | d

echo "DATASET:      $DATASET"

CUDA_LAUNCH_BLOCKING=1 python3 src/detect_deepfakes.py --dataset $DATASET --weights $WEIGHTS --label $LABEL