LOGIN=$(whoami)
ADDR=127.0.0.1
PORT=55333
DEVICES=0
NUM_GPUS=$(python -c "import sys; print(len(sys.argv[1].split(',')))" $DEVICES)

RES_DIR=neonet_logs
CONFIG=configs/neonext.yml
MODEL=neonext_t
IMG_SIZE=384
TAG=test
# 1,1,1,0' for 224x224 and '1,1,1,1' for 384x384
SHIFTS='1,1,1,1'

CUDA_VISIBLE_DEVICES=$DEVICES \
torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --nproc_per_node=$NUM_GPUS \
  --rdzv_endpoint=$ADDR:$PORT \
  --rdzv_backend=static \
  validate.py \
    --dataset_name="imagenet_pytorch" \
    --local_train_data_dir="/mnt/disk/datasets/ImageNet/train" \
    --local_eval_data_dir="/mnt/disk/datasets/ImageNet/val" \
    --train_num_workers=12 --eval_num_workers=12 \
    --train_image_size=$IMG_SIZE --eval_image_size=$IMG_SIZE \
    --config=$CONFIG --outputs_dir=$RES_DIR/$MODEL/$TAG \
    --per_batch_size=64 \
    --model=$MODEL \
    --linear_bias=1 \
    --kernel_spec='4+7' \
    --shifts=$SHIFTS \
    --layer_scale_init_value=1e-6 \
    --ema=1 \
    --pretrain="checkpoints/neonext-t-384/epoch_285.pt"

