#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

SLAM_DIR=/apdcephfs_cq10/share_1603164/user/schmittzhu/code/SLAM-LLM
code_dir=examples/seld_spatialsoundqa

export PYTHONPATH="$SLAM_DIR/$code_dir:$SLAM_DIR/src:$PYTHONPATH"

python $SLAM_DIR/$code_dir/scripts/smoke_test_save_decode.py \
  --config-path "$SLAM_DIR/$code_dir/conf" \
  ++model_config.llm_name=llama-2-7b \
  ++model_config.llm_path=/apdcephfs_cq10/share_1603164/user/zhannliu/models/Llama-2-7b-hf \
  ++model_config.llm_dim=4096 \
  ++model_config.encoder_name=SpatialAST \
  ++model_config.encoder_projector=q-former \
  ++model_config.qformer_layers=8 \
  ++model_config.encoder_ckpt=/apdcephfs_cq10/share_1603164/user/schmittzhu/code/Spatial-AST/outputs/finetune_binaural/checkpoint-best.pth \
  ++dataset_config.file=$SLAM_DIR/$code_dir/dataset/spatial_audio_dataset.py:get_spatial_audio_dataset_binaural \
  ++dataset_config.stage=bat \
  ++dataset_config.qa_data_root=$SLAM_DIR/$code_dir/dataset/bat_qa_salmonn \
  ++dataset_config.fix_length_audio=10 \
  ++train_config.output_dir=$SLAM_DIR/outputs/bat-binaural-llama2-qformer-lora \
  ++train_config.decode_output_dir=$SLAM_DIR/outputs/bat-binaural-llama2-qformer-lora/decodes \
  ++train_config.decode_splits=[val] \
  ++train_config.decode_max_samples=5 \
  ++train_config.decode_num_beams=1 \
  ++train_config.decode_max_new_tokens=64
