#!/bin/bash
export CUDA_VISIBLE_DEVICES=4,5,6,7
export TOKENIZERS_PARALLELISM=false

SLAM_DIR=/apdcephfs_cq10/share_1603164/user/schmittzhu/code/SLAM-LLM
cd $SLAM_DIR
code_dir=examples/seld_spatialsoundqa

audio_encoder_path=/apdcephfs_cq10/share_1603164/user/schmittzhu/code/Spatial-AST/finetuned.pth
llm_path=/apdcephfs_cq10/share_1603164/user/zhannliu/models/Llama-2-7b-hf

stage=bat
output_dir=./outputs/bat_binaural_llama2
ckpt_path=examples/seld_spatialsoundqa/model.pt

split=test
decode_log=$output_dir/decode_${split}_beam4

python -u $code_dir/inference_seld_batch.py \
--config-path "conf" \
hydra.run.dir=$output_dir \
++model_config.llm_name=llama-2-7b \
++model_config.llm_path=$llm_path \
++model_config.llm_dim=4096 \
++model_config.encoder_name=SpatialAST \
++model_config.encoder_projector=q-former \
++model_config.qformer_layers=8 \
++model_config.encoder_ckpt=$audio_encoder_path \
++dataset_config.file=examples/seld_spatialsoundqa/dataset/spatial_audio_dataset.py:get_spatial_audio_dataset_binaural \
++dataset_config.qa_data_root=examples/seld_spatialsoundqa/dataset/bat_qa \
++dataset_config.stage=$stage \
++dataset_config.inference_mode=true \
++dataset_config.fix_length_audio=10 \
++train_config.val_batch_size=1 \
++train_config.num_workers_dataloader=1 \
++train_config.freeze_encoder=true \
++train_config.freeze_llm=true \
++train_config.output_dir=$output_dir \
++train_config.use_peft=false \
++log_config.log_file=$output_dir/test.log \
++decode_log=$decode_log \
++ckpt_path=$ckpt_path
