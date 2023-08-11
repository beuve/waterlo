DATASET= # Path to pristine data.
OUTPUT=  # Path to save the trained weights (and additionally, some intermediary images)

LOSS=ssim
LAMBDA=1
ALPHA=0.005
COMPRESSION= #--compression


echo "DATASET:      $DATASET"

CUDA_LAUNCH_BLOCKING=1 python3 src/main.py --dataset $DATASET --output $OUTPUT --alpha $ALPHA --lambd $LAMBDA --loss $LOSS $COMPRESSION