import os
import json
import logging
from dataclasses import dataclass, field
from typing import Optional

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from seld_config import ModelConfig, TrainConfig, DataConfig, LogConfig, FSDPConfig, PeftConfig
from slam_llm.utils.model_utils import get_custom_model_factory
from slam_llm.utils.dataset_utils import get_preprocessed_dataset
from slam_llm.utils.checkpoint_handler import save_model_checkpoint_peft

logger = logging.getLogger(__name__)


@dataclass
class RunConfig:
    dataset_config: DataConfig = field(default_factory=DataConfig)
    model_config: ModelConfig = field(default_factory=ModelConfig)
    train_config: TrainConfig = field(default_factory=TrainConfig)
    log_config: LogConfig = field(default_factory=LogConfig)
    fsdp_config: FSDPConfig = field(default_factory=FSDPConfig)
    peft_config: PeftConfig = field(default_factory=PeftConfig)
    debug: bool = field(default=False)
    metric: str = field(default="acc")
    ckpt_path: Optional[str] = field(default=None)
    peft_ckpt: Optional[str] = field(default=None)


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _decode_and_save(model, tokenizer, dataloader, train_config, out_path, device):
    base_model = _unwrap_model(model)
    base_model.eval()
    results = []
    seen = 0

    max_samples = train_config.decode_max_samples
    num_beams = train_config.decode_num_beams
    max_new_tokens = train_config.decode_max_new_tokens
    do_sample = train_config.decode_do_sample
    temperature = train_config.decode_temperature
    top_p = train_config.decode_top_p
    print_limit = train_config.decode_print_samples

    with torch.no_grad():
        for batch in dataloader:
            for key in batch.keys():
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
            model_outputs = base_model.generate(
                **batch,
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
            )
            output_text = base_model.tokenizer.batch_decode(
                model_outputs, add_special_tokens=False, skip_special_tokens=True
            )
            for key, text, target in zip(batch["keys"], output_text, batch["targets"]):
                clean_text = text.replace("\n", " ")
                results.append({"key": key, "prediction": clean_text, "target": target})
                if print_limit and seen < print_limit:
                    logger.info(f"[smoke-decode] {key}: {clean_text}")
                seen += 1
                if max_samples and seen >= max_samples:
                    break
            if max_samples and seen >= max_samples:
                break

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


@hydra.main(config_name=None, version_base=None)
def main_hydra(cfg: DictConfig):
    run_config = RunConfig()
    cfg = OmegaConf.merge(run_config, cfg)

    log_level = getattr(logging, cfg.get("log_level", "INFO").upper())
    logging.basicConfig(level=log_level)

    train_config = cfg.train_config
    model_config = cfg.model_config
    dataset_config = cfg.dataset_config

    # DDP setup (optional)
    rank = 0
    local_rank = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if train_config.enable_ddp:
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")

    model_factory = get_custom_model_factory(model_config, logger)
    model, tokenizer = model_factory(
        train_config,
        model_config,
        ckpt_path=cfg.get("ckpt_path", None),
        peft_ckpt=cfg.get("peft_ckpt", None),
        metric=cfg.get("metric", "acc"),
    )
    model.to(device)

    if train_config.enable_ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # save checkpoint (peft) to verify save path works
    if (not train_config.enable_ddp) or rank == 0:
        save_dir = train_config.output_dir
        os.makedirs(save_dir, exist_ok=True)
        save_model_checkpoint_peft(model, None, rank, train_config, checkpoint_name="smoke_test")

    # decode a few samples
    if (not train_config.enable_ddp) or rank == 0:
        dataset_config.inference_mode = True
        for split in train_config.decode_splits:
            dataset_decode = get_preprocessed_dataset(tokenizer, dataset_config, split=split)
            dataloader = torch.utils.data.DataLoader(
                dataset_decode,
                batch_size=train_config.val_batch_size,
                shuffle=False,
                num_workers=train_config.num_workers_dataloader,
                collate_fn=dataset_decode.collator,
            )
            out_path = os.path.join(
                train_config.decode_output_dir or os.path.join(train_config.output_dir, "decodes"),
                f"smoke_{split}.json",
            )
            logger.info(f"[smoke] decoding {split} -> {out_path}")
            _decode_and_save(model, tokenizer, dataloader, train_config, out_path, device)

    if train_config.enable_ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main_hydra()
