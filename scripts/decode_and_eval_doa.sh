#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

SLAM_DIR=/apdcephfs_cq10/share_1603164/user/schmittzhu/code/SLAM-LLM
code_dir=examples/seld_spatialsoundqa

# ---- Required paths ----
LLM_PATH=/apdcephfs_cq10/share_1603164/user/zhannliu/models/Llama-2-7b-hf
ENCODER_CKPT=/apdcephfs_cq10/share_1603164/user/schmittzhu/code/Spatial-AST/outputs/finetune_binaural/checkpoint-best.pth
MODEL_CKPT=$SLAM_DIR/outputs/bat-binaural-llama2-qformer-lora/bat-binaural_epoch_5_step_500/model.pt

# ---- Dataset selection (task1 vs task2) ----
# Task1 (bat)
#QA_ROOT=$SLAM_DIR/$code_dir/dataset/bat_qa_salmonn
#STAGE=bat

# Task2 (bat2)
QA_ROOT=$SLAM_DIR/$code_dir/dataset/bat2_qa_salmonn
STAGE=bat2

# ---- Output ----
OUTPUT_DIR=$SLAM_DIR/outputs/bat-decode-eval
SPLIT=test
DECODE_LOG=$OUTPUT_DIR/decode_${SPLIT}_beam1

# ---- Decode controls ----
NUM_BEAMS=1
MAX_NEW_TOKENS=64
VAL_BATCH=1
NUM_WORKERS=0

export PYTHONPATH="$SLAM_DIR/$code_dir:$SLAM_DIR/src:$PYTHONPATH"

# 1) Decode
python -u $SLAM_DIR/$code_dir/inference_seld_batch.py \
  --config-path "$SLAM_DIR/$code_dir/conf" \
  hydra.run.dir=$OUTPUT_DIR \
  ++model_config.llm_name=llama-2-7b \
  ++model_config.llm_path=$LLM_PATH \
  ++model_config.llm_dim=4096 \
  ++model_config.encoder_name=SpatialAST \
  ++model_config.encoder_projector=q-former \
  ++model_config.qformer_layers=8 \
  ++model_config.encoder_ckpt=$ENCODER_CKPT \
  ++dataset_config.file=$SLAM_DIR/$code_dir/dataset/spatial_audio_dataset.py:get_spatial_audio_dataset_binaural \
  ++dataset_config.qa_data_root=$QA_ROOT \
  ++dataset_config.stage=$STAGE \
  ++dataset_config.fix_length_audio=10 \
  ++dataset_config.inference_mode=true \
  ++train_config.val_batch_size=$VAL_BATCH \
  ++train_config.num_workers_dataloader=$NUM_WORKERS \
  ++train_config.freeze_llm=true \
  ++train_config.freeze_encoder=true \
  ++train_config.use_peft=true \
  ++decode_log=$DECODE_LOG \
  ++ckpt_path=$MODEL_CKPT

# 2) Evaluate DOA from pred/gt logs
python $SLAM_DIR/$code_dir/scripts/eval_doa_from_decodes.py \
  --pred ${DECODE_LOG}_pred \
  --gt ${DECODE_LOG}_gt \
  --out-metrics $OUTPUT_DIR/${SPLIT}_metrics.json \
  --out-details $OUTPUT_DIR/${SPLIT}_details.json \
  --filter-doa
