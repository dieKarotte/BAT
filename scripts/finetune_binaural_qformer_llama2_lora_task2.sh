#!/bin/bash

export CUDA_VISIBLE_DEVICES=4,5,6,7
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1

SLAM_DIR=/apdcephfs_cq10/share_1603164/user/schmittzhu/code/SLAM-LLM
cd $SLAM_DIR
code_dir=examples/seld_spatialsoundqa

# provided fine-tuned audio encoder from Spatial-AST
# audio_encoder_path=/apdcephfs_cq10/share_1603164/user/schmittzhu/code/Spatial-AST/finetuned.pth
# self fintuned audio encoder
audio_encoder_path=/apdcephfs_cq10/share_1603164/user/schmittzhu/code/Spatial-AST/outputs/finetune_binaural/checkpoint-best.pth
llm_path=/apdcephfs_cq10/share_1603164/user/zhannliu/models/Llama-2-7b-hf

stage=bat2
output_dir=./outputs/bat2-binaural-llama2-qformer-lora

hydra_args="
hydra.run.dir=$output_dir \
++model_config.llm_name=llama-2-7b \
++model_config.llm_path=$llm_path \
++model_config.llm_dim=4096 \
++model_config.encoder_name=SpatialAST \
++model_config.encoder_projector=q-former \
++model_config.qformer_layers=8 \
++model_config.encoder_ckpt=$audio_encoder_path \
++dataset_config.file=$code_dir/dataset/spatial_audio_dataset.py:get_spatial_audio_dataset_binaural \
++dataset_config.stage=bat2 \
++dataset_config.qa_data_root=$code_dir/dataset/bat2_qa_salmonn \
++dataset_config.fix_length_audio=10 \
++train_config.model_name=bat2-binaural \
++train_config.num_epochs=25 \
++train_config.freeze_encoder=true \
++train_config.freeze_llm=true \
++train_config.batching_strategy=custom \
++train_config.warmup_steps=2000 \
++train_config.total_steps=20000 \
++train_config.lr=1e-4 \
++train_config.validation_interval=2000 \
++train_config.batch_size_training=8 \
++train_config.val_batch_size=1 \
++train_config.num_workers_dataloader=4 \
++train_config.output_dir=$output_dir \
++train_config.use_peft=true \
++peft_config.peft_method=lora \
++train_config.enable_ddp=true \
++train_config.enable_fsdp=false \
++train_config.save_every_epoch=true \
++train_config.decode_every_epoch=true \
++train_config.decode_splits=[val,test,DoA_val,DoA_test] \
++train_config.decode_max_samples=100 \
++train_config.decode_output_dir=$output_dir/decodes \
++train_config.decode_num_beams=1 \
++train_config.decode_max_new_tokens=64 \
++train_config.decode_print_samples=3 \
++metric=acc \
++log_config.log_file=$output_dir/log.txt \
"

torchrun --nnodes 1 --nproc_per_node 4 --master_port 39504 \
    $code_dir/finetune_seld.py \
    --config-path "conf" \
    $hydra_args
