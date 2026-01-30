# <img src="assets/bat.png" alt="SELD_SpatialSoundQA" width="25" height="25"> SELD_SpatialSoundQA

# This repo is part of the [SLAM-LLM](https://github.com/X-LANCE/SLAM-LLM) and change dataloader according to our binaural data, the LICENSE follows as the origin repo.

This repo hosts the code and models of "[BAT: Learning to Reason about Spatial Sounds with Large Language Models](https://arxiv.org/abs/2402.01591)" [ICML 2024 [bib](https://github.com/X-LANCE/SLAM-LLM/tree/main/examples/seld_spatialsoundqa#citation)]. 

Checkout our [demo page](https://zhishengzheng.com/BAT/) and enjoy a QA game with spatial audio.

## Performance evaluation on **SpatialSoundQA**
We use [Spatial-AST](https://huggingface.co/datasets/zhisheng01/SpatialAudio/blob/main/SpatialAST/finetuned.pth) as audio encoder, [llama-2-7b](https://huggingface.co/meta-llama/Llama-2-7b) as LLM backbone. We finetune the model by adding Q-Former and LoRA. To calculate MAP, you can refer to [calculate_map.py](https://github.com/X-LANCE/SLAM-LLM/blob/main/examples/seld_spatialsoundqa/scripts/calculate_map.py)
<img src="assets/performance.png" alt="xxx">


## Checkpoints
Encoder | Projector | LLM | 
|---|---|---|
[Spatial-AST](https://huggingface.co/datasets/zhisheng01/SpatialAudio/blob/main/SpatialAST/finetuned.pth) | [Q-former](https://huggingface.co/datasets/zhisheng01/SpatialAudio/blob/main/BAT/model.pt)(~73.56M) | [llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b) |

## Demo (Spatial Audio Inference)
### Environment setup
```
cd SLAM-LLM/examples/seld_spatialsoundqa/
pip install -r requirements.txt
cd SLAM-LLM/
pip install -e .
```

Then try [`inference.ipynb`](https://github.com/X-LANCE/SLAM-LLM/blob/main/examples/seld_spatialsoundqa/inference.ipynb).


## Data preparation
You need to prepare the data jsonl in this format. Below is an example.  
You can download the SpatialSoundQA dataset from [SpatialAudio](https://huggingface.co/datasets/zhisheng01/SpatialAudio).
```json
{
  "audio_id": "eval/audio/YI-HlrcP6Qg4",
  "reverb_id": "q9vSo1VnCiC/0.npy", 
  "audio_id2": null, 
  "reverb_id2": null, 
  "question_id": 0, 
  "question_type": "CLASSIFICATION", 
  "question": "Enumerate the sound occurrences in the audio clip.", 
  "answer": "accelerating, revving, vroom; car; vehicle"
}

...

{
  "audio_id": "eval/audio/YZX2fVPmUidA", 
  "reverb_id": "q9vSo1VnCiC/32.npy", 
  "audio_id2": "eval/audio/YjNjUU01quLs", 
  "reverb_id2": "q9vSo1VnCiC/31.npy", 
  "question_id": 58, 
  "question_type": "MIXUP_NONBINARY_DISTANCE", 
  "question": "How far away is the sound of the banjo from the sound of the whack, thwack?", 
  "answer": "2m"
}
```

### Binaural BAT QA (SALMONN-style prompts)
We added scripts to build QA pairs from BAT-style json files. The output layout is:
`dataset/bat_qa_salmonn/{stage}/{train,val,test}.json` with `binaural_path` and QA pairs.

```bash
python scripts/convert_bat_to_qa_salmonn.py \
  --input_root /path/to/bat \
  --output_root dataset/bat_qa_salmonn \
  --stage bat \
  --prompt_file /path/to/SALMONN-3D/prompts/train.json \
  --label_map_csv /path/to/class_labels_indices.csv \
  --distance_mode append
```

### BAT2 dual-source QA (gender-conditioned DoA)
This script generates two questions per sample: event detection and gender-specific DoA.
It also normalizes azimuth/elevation into the range used by our trainer.

```bash
python scripts/convert_bat2_to_qa_dualsource.py \
  --input_root /path/to/bat2 \
  --output_root dataset/bat2_qa_salmonn \
  --stage bat2 \
  --prompt_file /path/to/SALMONN-3D/prompts/dual_source.json \
  --label_map_csv /path/to/class_labels_indices.csv \
  --binaural_root /path/to/Spatial-AST
```

### DoA-only splits (for faster decoding/eval)
Filter DoA questions into dedicated splits used by `decode_splits`.

```bash
python scripts/extract_doa_splits.py \
  --input_root dataset/bat_qa_salmonn \
  --stage bat \
  --splits val,test
```

## Train a new model
```bash
cd examples/seld_spatialsoundqa/
bash scripts/finetune_spatial-ast_qformer_llama_2_7b.sh
```

### Binaural Q-Former + LoRA (DDP)
Use the binaural trainer to finetune Q-Former + LoRA. The script runs with DDP and
logs per-epoch decodes (including DoA splits if present).

```bash
cd examples/seld_spatialsoundqa/
bash scripts/finetune_binaural_qformer_llama2_lora.sh
```

For BAT2 (dual-source) use the task2 variant:

```bash
cd examples/seld_spatialsoundqa/
bash scripts/finetune_binaural_qformer_llama2_lora_task2.sh
```

## Decoding with checkpoints
```bash
cd examples/seld_spatialsoundqa/
bash scripts/decode_spatial-ast_qformer_llama_2_7b.sh
```

### Binaural decode + DoA evaluation
Decode predictions and compute DoA error metrics (azimuth-only and azimuth+elevation).

```bash
cd examples/seld_spatialsoundqa/
bash scripts/decode_and_eval_doa.sh
```

### DoA evaluation directly from a model checkpoint
Runs inference and writes per-sample predictions plus summary metrics.

```bash
python scripts/eval_doa_from_model.py \
  --model-ckpt /path/to/model.pt \
  --llm-path /path/to/Llama-2-7b-hf \
  --encoder-ckpt /path/to/Spatial-AST/finetuned.pth \
  --dataset-file examples/seld_spatialsoundqa/dataset/spatial_audio_dataset.py:get_spatial_audio_dataset_binaural \
  --qa-root examples/seld_spatialsoundqa/dataset/bat_qa_salmonn \
  --stage bat \
  --splits test \
  --output-dir /path/to/doa_eval \
  --use-peft
```

DDP evaluation (4 GPUs):

```bash
cd examples/seld_spatialsoundqa/
bash scripts/eval_binaural_doa_llama2_ddp.sh
```


## TODO
- [x] Decode with checkpoints
- [x] Upload SpatialSoundQA dataset
- [x] Upload pretrained checkpoints
- [x] Update model performance

## Citation
```
@article{zheng2024bat,
  author    = {Zheng, Zhisheng and Peng, Puyuan and Ma, Ziyang and Chen, Xie and Choi, Eunsol and Harwath, David},
  title     = {BAT: Learning to Reason about Spatial Sounds with Large Language Models},
  journal   = {arXiv preprint arXiv:2402.01591},
  year      = {2024},
}
```

## Adaption
