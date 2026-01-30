import argparse
import json
import logging
import math
import os
import re
import sys
from dataclasses import asdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from omegaconf import OmegaConf
import torch.distributed as dist
from tqdm import tqdm

script_dir = os.path.dirname(os.path.abspath(__file__))
example_dir = os.path.dirname(script_dir)
repo_root = os.path.dirname(example_dir)
sys.path.insert(0, example_dir)
sys.path.insert(0, os.path.join(repo_root, "src"))

from seld_config import ModelConfig, TrainConfig, DataConfig, LogConfig, FSDPConfig, PeftConfig


logger = logging.getLogger(__name__)


def load_module_from_py_file(py_file: str) -> object:
    import importlib.machinery
    import importlib.util

    module_name = Path(py_file).name
    loader = importlib.machinery.SourceFileLoader(module_name, py_file)
    spec = importlib.util.spec_from_loader(module_name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def get_custom_model_factory(model_config):
    model_path = model_config.file
    if ":" in model_path:
        module_path, func_name = model_path.split(":")
    else:
        module_path, func_name = model_path, "model_factory"
    module = load_module_from_py_file(module_path)
    return getattr(module, func_name)


def get_custom_dataset(dataset_config, tokenizer, split: str):
    if ":" in dataset_config.file:
        module_path, func_name = dataset_config.file.split(":")
    else:
        module_path, func_name = dataset_config.file, "get_custom_dataset"
    module = load_module_from_py_file(module_path)
    return getattr(module, func_name)(dataset_config, tokenizer, split)


def wrap_azimuth_diff(pred: float, gt: float) -> float:
    diff = pred - gt
    while diff > 180:
        diff -= 360
    while diff < -180:
        diff += 360
    return diff


def angular_error_norm(az1, el1, az2, el2) -> Optional[float]:
    if None in (az1, el1, az2, el2):
        return None
    try:
        az1_r = math.radians(az1)
        az2_r = math.radians(az2)
        el1_r = math.radians(el1)
        el2_r = math.radians(el2)

        x1 = math.cos(el1_r) * math.cos(az1_r)
        y1 = math.cos(el1_r) * math.sin(az1_r)
        z1 = math.sin(el1_r)

        x2 = math.cos(el2_r) * math.cos(az2_r)
        y2 = math.cos(el2_r) * math.sin(az2_r)
        z2 = math.sin(el2_r)

        dot = x1 * x2 + y1 * y2 + z1 * z2
        dot = max(min(dot, 1.0), -1.0)
        return math.degrees(math.acos(dot))
    except Exception:
        return None


def summarize(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": float("nan"), "median": float("nan"), "std": float("nan"), "count": 0}
    arr = torch.tensor(values, dtype=torch.float32)
    return {
        "mean": float(arr.mean().item()),
        "median": float(arr.median().item()),
        "std": float(arr.std(unbiased=False).item()),
        "count": int(arr.numel()),
    }


AZ_RE = re.compile(r"azimuth[^\d+-]*([-+]?\d+(?:\.\d+)?)", re.IGNORECASE)
EL_RE = re.compile(r"elevation[^\d+-]*([-+]?\d+(?:\.\d+)?)", re.IGNORECASE)
NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


def parse_az_el(text: str) -> Tuple[Optional[float], Optional[float]]:
    if not text:
        return None, None
    az_m = AZ_RE.search(text)
    el_m = EL_RE.search(text)
    az = float(az_m.group(1)) if az_m else None
    el = float(el_m.group(1)) if el_m else None
    if az is not None and el is not None:
        return az, el

    nums = NUM_RE.findall(text)
    if len(nums) >= 2:
        try:
            return float(nums[0]), float(nums[1])
        except Exception:
            return None, None
    return az, el


def filter_doa_samples(samples: List[Dict]) -> List[Dict]:
    return [s for s in samples if "doa" in str(s.get("question_type", "")).lower()]


def write_json(path: str, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


@dataclass
class EvalConfig:
    model_ckpt: str
    llm_path: str
    encoder_ckpt: str
    dataset_file: str
    qa_root: str
    stage: str
    splits: List[str]
    output_dir: str
    peft_ckpt: Optional[str] = None
    batch_size: int = 1
    num_workers: int = 0
    max_samples: int = 0
    filter_doa: bool = True


def run_split(
    model,
    tokenizer,
    dataset_config,
    split: str,
    output_dir: str,
    max_samples: int,
    filter_doa: bool,
    batch_size: int,
    rank: int,
    world_size: int,
    ddp: bool,
):
    qa_path = Path(dataset_config.qa_data_root) / dataset_config.stage / f"{split}.json"
    raw = json.load(open(qa_path))
    samples = raw["data"] if isinstance(raw, dict) else raw
    if filter_doa:
        samples = filter_doa_samples(samples)

    tmp_root = Path(output_dir) / "_tmp_qa" / dataset_config.stage
    tmp_path = tmp_root / f"{split}.json"
    if not ddp or rank == 0:
        tmp_root.mkdir(parents=True, exist_ok=True)
        json.dump({"data": samples}, open(tmp_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    if ddp:
        dist.barrier()

    dataset_config.qa_data_root = str(tmp_root.parent)
    dataset = get_custom_dataset(dataset_config, tokenizer, split)

    device = next(model.parameters()).device

    results = []
    az_errors = []
    azel_errors = []

    total = len(samples)
    limit = min(total, max_samples) if max_samples else total

    indices = list(range(rank, limit, world_size))
    total_batches = (len(indices) + batch_size - 1) // batch_size if indices else 0
    pbar = tqdm(total=total_batches, desc=f"Decoding {split} (rank {rank})", unit="batch", disable=(ddp and rank != 0))
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start:start + batch_size]
        batch_samples = [dataset[i] for i in batch_indices]
        collated = dataset.collator(batch_samples)
        for key in collated.keys():
            if isinstance(collated[key], torch.Tensor):
                collated[key] = collated[key].to(device)

        with torch.no_grad():
            outputs = model.generate(
                **collated,
                num_beams=1,
                max_new_tokens=64,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
            )
        decoded = model.tokenizer.batch_decode(outputs, add_special_tokens=False, skip_special_tokens=True)

        for i, text in enumerate(decoded):
            raw_item = samples[batch_indices[i]]
            pred_az, pred_el = parse_az_el(text)
            gt_az, gt_el = parse_az_el(raw_item.get("answer", ""))

            if pred_az is not None and gt_az is not None:
                az_err = abs(wrap_azimuth_diff(pred_az, gt_az))
                az_errors.append(az_err)
                ang_err = angular_error_norm(pred_az, pred_el, gt_az, gt_el) if pred_el is not None and gt_el is not None else None
                if ang_err is not None:
                    azel_errors.append(ang_err)

            results.append(
                {
                    "id": raw_item.get("id"),
                    "binaural_path": raw_item.get("binaural_path"),
                    "gender": raw_item.get("gender"),
                    "question": raw_item.get("question"),
                    "answer": raw_item.get("answer"),
                    "prediction": text.replace("\n", " "),
                    "pred_azimuth": pred_az,
                    "pred_elevation": pred_el,
                    "gt_azimuth": gt_az,
                    "gt_elevation": gt_el,
                }
            )
        pbar.update(1)
    pbar.close()

    if ddp:
        pred_lists = [None for _ in range(world_size)] if rank == 0 else None
        az_lists = [None for _ in range(world_size)] if rank == 0 else None
        azel_lists = [None for _ in range(world_size)] if rank == 0 else None
        dist.gather_object(results, pred_lists, dst=0)
        dist.gather_object(az_errors, az_lists, dst=0)
        dist.gather_object(azel_errors, azel_lists, dst=0)
        if rank == 0:
            all_results = [item for sub in pred_lists for item in (sub or [])]
            all_az = [v for sub in az_lists for v in (sub or [])]
            all_azel = [v for sub in azel_lists for v in (sub or [])]
            metrics = {
                "total": int(limit),
                "azimuth_only_error": summarize(all_az),
                "azimuth_elevation_error": summarize(all_azel),
            }
            out_json = os.path.join(output_dir, f"{split}_predictions.json")
            out_metrics = os.path.join(output_dir, f"{split}_metrics.json")
            write_json(out_json, all_results)
            write_json(out_metrics, metrics)
            logger.info(f"[{split}] wrote {out_json} and {out_metrics}")
        dist.barrier()
    else:
        metrics = {
            "total": int(limit),
            "azimuth_only_error": summarize(az_errors),
            "azimuth_elevation_error": summarize(azel_errors),
        }
        out_json = os.path.join(output_dir, f"{split}_predictions.json")
        out_metrics = os.path.join(output_dir, f"{split}_metrics.json")
        write_json(out_json, results)
        write_json(out_metrics, metrics)
        logger.info(f"[{split}] wrote {out_json} and {out_metrics}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-ckpt", required=True, help="Q-former/model.pt checkpoint")
    parser.add_argument("--llm-path", required=True)
    parser.add_argument("--encoder-ckpt", required=True)
    parser.add_argument("--dataset-file", required=True)
    parser.add_argument("--qa-root", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--splits", default="test", help="Comma-separated splits, e.g. test or val,test")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--peft-ckpt", default=None)
    parser.add_argument("--use-peft", action="store_true", help="Initialize LoRA modules before loading model.ckpt")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--no-filter-doa", action="store_true")
    parser.add_argument("--ddp", action="store_true", help="Enable DDP with torchrun")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    train_config = TrainConfig()
    model_config = ModelConfig()
    dataset_config = DataConfig()
    class _DatasetConfigProxy:
        def __init__(self, cfg):
            self._cfg = cfg
        def __getattr__(self, key):
            return getattr(self._cfg, key)
        def __getitem__(self, key):
            return getattr(self._cfg, key)
        def __setitem__(self, key, value):
            setattr(self._cfg, key, value)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    example_dir = os.path.dirname(script_dir)
    model_config.file = os.path.join(example_dir, "model", "slam_model_seld.py") + ":model_factory"

    model_config.llm_name = "llama-2-7b"
    model_config.llm_path = args.llm_path
    model_config.llm_dim = 4096
    model_config.encoder_name = "SpatialAST"
    model_config.encoder_projector = "q-former"
    model_config.qformer_layers = 8
    model_config.encoder_ckpt = args.encoder_ckpt
    if not hasattr(model_config, "get"):
        setattr(model_config, "get", lambda k, d=None: getattr(model_config, k, d))

    dataset_config.file = args.dataset_file
    dataset_config.qa_data_root = args.qa_root
    dataset_config.stage = args.stage
    dataset_config.fix_length_audio = 10
    dataset_config.inference_mode = True
    if not hasattr(dataset_config, "get"):
        setattr(dataset_config, "get", lambda k, d=None: getattr(dataset_config, k, d))
    train_config.val_batch_size = args.batch_size
    train_config.num_workers_dataloader = args.num_workers
    if not hasattr(train_config, "get"):
        setattr(train_config, "get", lambda k, d=None: getattr(train_config, k, d))

    dataset_config = _DatasetConfigProxy(dataset_config)

    ddp = args.ddp or int(os.environ.get("WORLD_SIZE", "1")) > 1
    rank = 0
    world_size = 1
    local_rank = 0
    if ddp:
        dist.init_process_group(backend="nccl")
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)

    train_config.use_peft = args.use_peft or bool(args.peft_ckpt)
    if train_config.use_peft:
        if not isinstance(train_config.peft_config, (dict,)) and not hasattr(train_config.peft_config, "get"):
            train_config.peft_config = OmegaConf.create(asdict(train_config.peft_config))

    model_factory = get_custom_model_factory(model_config)
    model, tokenizer = model_factory(
        train_config,
        model_config,
        ckpt_path=args.model_ckpt,
        peft_ckpt=args.peft_ckpt,
        metric="acc",
    )

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    model.to(device)

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    for split in splits:
        run_split(
            model,
            tokenizer,
            dataset_config,
            split,
            args.output_dir,
            args.max_samples,
            filter_doa=not args.no_filter_doa,
            batch_size=args.batch_size,
            rank=rank,
            world_size=world_size,
            ddp=ddp,
        )

    if ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
