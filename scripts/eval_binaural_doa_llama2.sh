#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false

SLAM_DIR=/apdcephfs_cq10/share_1603164/user/schmittzhu/code/SLAM-LLM
code_dir=examples/seld_spatialsoundqa

# ---- Required paths ----
LLM_PATH=/apdcephfs_cq10/share_1603164/user/zhannliu/models/Llama-2-7b-hf
ENCODER_CKPT=/apdcephfs_cq10/share_1603164/user/schmittzhu/code/Spatial-AST/outputs/finetune_binaural/checkpoint-best.pth
MODEL_CKPT=$SLAM_DIR/outputs/bat-binaural-llama2-qformer-lora/bat-binaural_epoch_5_step_500/model.pt

# ---- Dataset selection (task1 vs task2) ----
# Task1 (bat)
QA_ROOT=$SLAM_DIR/$code_dir/dataset/bat_qa_salmonn
STAGE=bat

# # Task2 (bat2)
# QA_ROOT=$SLAM_DIR/$code_dir/dataset/bat2_qa_salmonn
# STAGE=bat2

# ---- Output ----
OUTPUT_DIR=$SLAM_DIR/outputs/bat-eval

# ---- Eval controls ----
SPLITS=DoA_val,DoA_test
MAX_SAMPLES=0       # 0 = full split
BATCH_SIZE=1
NUM_WORKERS=0
PEFT_CKPT=""        # optional: path to PEFT adapter dir if you have one

export PYTHONPATH="$SLAM_DIR/$code_dir:$SLAM_DIR/src:$PYTHONPATH"

echo "Evaluating Binaural DoA with LLaMA-2 model..."
echo "Task stage: $STAGE"
echo "QA root: $QA_ROOT"
echo "Model checkpoint: $MODEL_CKPT"
echo "Encoder checkpoint: $ENCODER_CKPT"
echo "Output directory: $OUTPUT_DIR"

python $SLAM_DIR/$code_dir/scripts/eval_doa_from_model.py \
  --model-ckpt $MODEL_CKPT \
  --llm-path $LLM_PATH \
  --encoder-ckpt $ENCODER_CKPT \
  --dataset-file $SLAM_DIR/$code_dir/dataset/spatial_audio_dataset.py:get_spatial_audio_dataset_binaural \
  --qa-root $QA_ROOT \
  --stage $STAGE \
  --splits $SPLITS \
  --output-dir $OUTPUT_DIR \
  --batch-size $BATCH_SIZE \
  --num-workers $NUM_WORKERS \
  --max-samples $MAX_SAMPLES
