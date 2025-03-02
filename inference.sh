# CUDA_VISIBLE_DEVICES=0
MODEL_DIR='experiments/focalclick/hrnet18v2_w18_all/008_focalclick_wtt'
CHECKPOINT_NAME='val_iou=0.83_118'
DATASETS='bloodcell,glas,monuseg,nucls,digestpath,bns,camelyon17,camelyon16'
GPUS='0'
EVAL_MODE='cvpr'
INFER_SIZE='1024'
python scripts/evaluate_model.py FocalClick\
  --model_dir $MODEL_DIR \
  --checkpoint $CHECKPOINT_NAME\
  --infer-size $INFER_SIZE\
  --datasets $DATASETS\
  --gpus $GPUS\
  --eval-mode $EVAL_MODE \
  --all \
  # --vis
  # --n-clicks=20\
  # --target-iou=0.90\
  # --thresh=0.5\
  