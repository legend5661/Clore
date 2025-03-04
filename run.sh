# CUDA_VISIBLE_DEVICES=1,2
GPUS='0'
WORKERS=6
BATCH_SIZE=40
EXP_NAME='focalclick_wtt'
MODE='train-val'
RESUM_DIR='008_focalclick_wtt'
START_EPOCH=98

python train.py models/focalclick/hrnet18_w18_all.py \
  --gpus $GPUS \
  --workers $WORKERS \
  --batch-size $BATCH_SIZE \
  --exp-name $EXP_NAME \
  --mode $MODE \
  --resume-exp $RESUM_DIR \
  --start-epoch $START_EPOCH
