#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false

SLAM_DIR=/apdcephfs_cq10/share_1603164/user/schmittzhu/code/SLAM-LLM
code_dir=examples/seld_spatialsoundqa

MODEL_CKPT=$SLAM_DIR/outputs/bat2-binaural-llama2-qformer-lora/bat2-binaural_epoch_25/model.pt
# PEFT_CKPT=/path/to/your/lora_adapter
LLM_PATH=/apdcephfs_cq10/share_1603164/user/zhannliu/models/Llama-2-7b-hf
ENCODER_CKPT=/apdcephfs_cq10/share_1603164/user/schmittzhu/code/Spatial-AST/outputs/finetune_binaural/checkpoint-best.pth

QA_ROOT=$SLAM_DIR/examples/seld_spatialsoundqa/dataset/bat2_qa_salmonn
STAGE=bat2
OUTPUT_DIR=$SLAM_DIR/outputs/bat-binaural-llama2-qformer-lora/doa_eval_task2

DATASET_FILE=$SLAM_DIR/examples/seld_spatialsoundqa/dataset/spatial_audio_dataset.py:get_spatial_audio_dataset_binaural

PYTHONPATH="$SLAM_DIR/$code_dir:$SLAM_DIR/src:$PYTHONPATH" \
torchrun --nnodes 1 --nproc_per_node 4 --master_port 39503 \
  $SLAM_DIR/$code_dir/scripts/eval_doa_from_model.py \
  --ddp \
  --model-ckpt $MODEL_CKPT \
  --llm-path $LLM_PATH \
  --encoder-ckpt $ENCODER_CKPT \
  --dataset-file $DATASET_FILE \
  --use-peft \
  --qa-root $QA_ROOT \
  --stage $STAGE \
  --splits DoA_test \
  --output-dir $OUTPUT_DIR \
  --batch-size 1
