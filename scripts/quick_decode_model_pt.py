import argparse
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import torch

from seld_config import ModelConfig, TrainConfig, DataConfig, LogConfig, FSDPConfig, PeftConfig

logger = logging.getLogger(__name__)


import importlib
from pathlib import Path as _Path


def load_module_from_py_file(py_file: str) -> object:
    module_name = _Path(py_file).name
    loader = importlib.machinery.SourceFileLoader(module_name, py_file)
    spec = importlib.util.spec_from_loader(module_name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def get_custom_model_factory(model_config):
    costom_model_path = model_config.file
    if costom_model_path is None:
        raise ValueError('model_config.file is required')

    if ':' in costom_model_path:
        module_path, func_name = costom_model_path.split(':')
    else:
        module_path, func_name = costom_model_path, 'model_factory'

    module_path = _Path(module_path)
    if not module_path.is_file():
        raise FileNotFoundError(module_path.as_posix())
    module = load_module_from_py_file(module_path.as_posix())
    return getattr(module, func_name)


def get_preprocessed_dataset(tokenizer, dataset_config, split: str):
    if ':' in dataset_config.file:
        module_path, func_name = dataset_config.file.split(':')
    else:
        module_path, func_name = dataset_config.file, 'get_custom_dataset'
    module = load_module_from_py_file(module_path)
    return getattr(module, func_name)(dataset_config, tokenizer, split)


def decode(model, tokenizer, dataloader, max_samples, device):
    model.eval()
    results = []
    seen = 0
    with torch.no_grad():
        for batch in dataloader:
            for key in batch.keys():
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
            outputs = model.generate(
                **batch,
                num_beams=1,
                max_new_tokens=64,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
            )
            output_text = model.tokenizer.batch_decode(outputs, add_special_tokens=False, skip_special_tokens=True)
            for key, text, target in zip(batch["keys"], output_text, batch["targets"]):
                clean_text = text.replace("\n", " ")
                results.append({"key": key, "prediction": clean_text, "target": target})
                print(f"[decode] {key}: {clean_text}")
                seen += 1
                if max_samples and seen >= max_samples:
                    return results
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm_path", required=True)
    parser.add_argument("--encoder_ckpt", required=True)
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--dataset_file", required=True)
    parser.add_argument("--qa_data_root", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--split", default="DoA_val")
    parser.add_argument("--max_samples", type=int, default=5)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    train_config = TrainConfig()
    model_config = ModelConfig()
    dataset_config = DataConfig()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    example_dir = os.path.dirname(script_dir)
    model_config.file = os.path.join(example_dir, 'model', 'slam_model_seld.py') + ':model_factory'

    model_config.llm_name = "llama-2-7b"
    model_config.llm_path = args.llm_path
    model_config.llm_dim = 4096
    model_config.encoder_name = "SpatialAST"
    model_config.encoder_projector = "q-former"
    model_config.qformer_layers = 8
    model_config.encoder_ckpt = args.encoder_ckpt

    dataset_config.file = args.dataset_file
    dataset_config.qa_data_root = args.qa_data_root
    dataset_config.stage = args.stage
    dataset_config.fix_length_audio = 10
    dataset_config.inference_mode = True

    model_factory = get_custom_model_factory(model_config)
    model, tokenizer = model_factory(
        train_config,
        model_config,
        ckpt_path=args.ckpt_path,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dataset = get_preprocessed_dataset(tokenizer, dataset_config, split=args.split)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=dataset.collator,
    )

    decode(model, tokenizer, dataloader, args.max_samples, device)


if __name__ == "__main__":
    main()
