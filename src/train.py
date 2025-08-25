"""train.py — single entry‑point to run any gaze‑prediction experiment.

Key design goals
-----------------
1.  **No model‑specific logic in here.**  All heavy lifting is delegated to
    *builder* functions that you register with simple decorators.
2.  **Reuse the proven _train loop** from `src/training.py` unchanged.
3.  **DDP‑aware but optional.**  Run on one GPU or many with the same CLI.

Add new components by dropping a file such as `models/dinogaze_spade.py` that
contains

```python
from train import register_model

@register_model("dinogaze_spade")
def build(cfg):
    …                                    # return a fully‑assembled nn.Module
```

Exactly the same for datasets via `@register_data("SALICON")`.

Run examples
------------
```bash
# SALICON pre‑train
python train.py \
    --config configs/salicon_dinogaze.yaml \
    stage.name=salicon_pretrain                                \
    stage.model_key=dinogaze_spade                             \
    stage.dataset_key=SALICON

# MIT scan‑path fine‑tune, override LR on the fly
python train.py --config configs/mit_scanpath_dg3.yaml stage.lr=1e-5
```
"""
from __future__ import annotations

import argparse
import dataclasses as _dc
import importlib
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.distributed as dist
from src.registry import MODEL_REGISTRY, DATA_REGISTRY
from torch.nn.parallel import DistributedDataParallel as DDP
import yaml

PROJECT_ROOT = Path(__file__).parent.parent.resolve()

torch.set_float32_matmul_precision('high')

# -----------------------------------------------------------------------------
#  0.  Small helper: reproducibility  -----------------------------------------
# -----------------------------------------------------------------------------

def _fix_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False  # better throughput, ok for saliency
    torch.backends.cudnn.benchmark = True

# -----------------------------------------------------------------------------
#  1.  Config schema  ----------------------------------------------------------
# -----------------------------------------------------------------------------

@_dc.dataclass
class StageCfg:
    kind: str
    name: str                              # folder name; can default to kind
    model_key: str                         # registry key
    dataset_key: str                       # registry key

    # hyper‑parameters
    lr: float = 5e-4
    milestones: Tuple[int, ...] = (20, 40)
    batch_size: int = 8
    grad_acc_steps: int = 1
    min_lr: float = 1e-7
    val_every: int = 1
    early_stopping_patience: int = 3       # New parameter with a default of 3
    resume_ckpt: str | None = None
    extra: Dict[str, Any] = _dc.field(default_factory=dict)


@_dc.dataclass
class RunCfg:
    stage: StageCfg
    compile: bool = False
    seed: int = 123
    num_workers: int | str = 'auto'

    # paths
    paths: Dict[str, Path] = _dc.field(default_factory=lambda: {
        "dataset_dir": Path("./data/pysaliency_datasets"),
        "train_dir": Path("./experiments"),
        "lmdb_dir": Path("./data/lmdb_caches"),
    })


# -----------------------------------------------------------------------------
#  3.  DDP helpers  ------------------------------------------------------------
# -----------------------------------------------------------------------------

class _DDPCtx:
    """Simple container for DD‑related flags."""
    def __init__(self):
        self.rank = int(os.environ.get("RANK", 0))
        self.world = int(os.environ.get("WORLD_SIZE", 1))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.enabled = self.world > 1
        if self.enabled:
            torch.cuda.set_device(self.local_rank)
            dist.init_process_group("nccl", init_method="env://")
        self.device = torch.device(f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu")
        self.is_master = self.rank == 0

    def barrier(self):
        """Wrapper for torch.distributed.barrier(). Does nothing if DDP is not enabled."""
        if self.enabled:
            dist.barrier()

    def cleanup(self):
        if self.enabled:
            dist.destroy_process_group()

# -----------------------------------------------------------------------------
#  4.  Optim / sched factories  ------------------------------------------------
# -----------------------------------------------------------------------------

def make_optim_and_sched(model: torch.nn.Module, cfg: StageCfg):
    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.Adam(params, lr=cfg.lr)
    sched = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=list(cfg.milestones))
    return optim, sched


def _auto_import_modules():
    """
    Dynamically imports Python modules from specified subdirectories
    to ensure their builders are registered.
    """
    logger = logging.getLogger("AutoImport")
    current_dir = Path(__file__).parent
    subdirs_to_scan = ["models", "datasets"]
    
    for subdir in subdirs_to_scan:
        scan_path = current_dir / subdir
        if not scan_path.is_dir():
            logger.warning(f"Auto-import directory not found, skipping: {scan_path}")
            continue
            
        for file_path in scan_path.glob("*.py"):
            if file_path.name.startswith("_"):
                continue
            
            module_name = f"src.{subdir}.{file_path.stem}"
            
            try:
                importlib.import_module(module_name)
                logger.info(f"Successfully auto-imported and registered: {module_name}")
            except Exception:
                # Use exc_info=True to see the full traceback of the import error
                logger.error(f"Failed to auto-import module {module_name}", exc_info=True)
                
# -----------------------------------------------------------------------------
#  5.  Generic train‑stage wrapper  -------------------------------------------
# -----------------------------------------------------------------------------

from src.training import _train as train_loop  # ← reuse your proven routine


def train_stage(run_cfg: RunCfg) -> None:
    """
    Manages the setup and execution of a single training stage, incorporating
    the correct "Load-then-Freeze" logic for fine-tuning.
    """
    ddp = _DDPCtx()
    log_level = logging.INFO if ddp.is_master else logging.ERROR
    logging.basicConfig(
        level=log_level,
        format=f"[%(asctime)s][RANK {ddp.rank}][%(name)s][%(levelname)s] - %(message)s",
        force=True
    )
    logger = logging.getLogger(__name__)
    if ddp.is_master:
        logger.info("DDP: world=%s rank=%s local_rank=%s", ddp.world, ddp.rank, ddp.local_rank)
        logger.info("Logging configured: Master rank will log INFO, other ranks will log WARNING and above.")

    fold_number = run_cfg.stage.extra.get('fold', 0)
    _fix_seed(run_cfg.seed + ddp.rank + fold_number)

    # --- 1. Build Model via Registry ---
    logger.info("Building model on CPU...")
    if run_cfg.stage.model_key not in MODEL_REGISTRY:
        raise KeyError(f"Model '{run_cfg.stage.model_key}' not registered")
    model = MODEL_REGISTRY[run_cfg.stage.model_key](run_cfg)

    # --- 2. Handle Fine-Tuning: "Load-then-Freeze" ---
    # This block handles loading weights from a previous stage *before* freezing anything.
    is_finetuning_from_ckpt = run_cfg.stage.resume_ckpt and Path(run_cfg.stage.resume_ckpt).exists()
    
    if is_finetuning_from_ckpt:
        logger.info(f"Fine-tuning mode detected. Loading weights from: {run_cfg.stage.resume_ckpt}")
        
        # --- LOAD ---
        # Load the checkpoint onto the CPU model.
        ckpt = torch.load(run_cfg.stage.resume_ckpt, map_location='cpu', weights_only=False)
        # Be flexible: checkpoint could be the state_dict itself or a dict containing it.
        model_state_dict = ckpt.get('model', ckpt) 
        
        # Load weights with strict=False to ignore missing keys (e.g., new scanpath head)
        # and unexpected keys (e.g., old optimizer params not part of the model).
        missing_keys, unexpected_keys = model.load_state_dict(model_state_dict, strict=False)
        if ddp.is_master:
            logger.info(f"Loaded weights. Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")

        # --- FREEZE ---
        # Apply freezing logic *after* the correct weights have been loaded.
        if run_cfg.stage.extra.get('freeze_saliency_network', False):
            if hasattr(model, 'saliency_network'):
                for param in model.saliency_network.parameters():
                    param.requires_grad = False
                logger.info("Saliency network has been frozen AFTER loading weights.")
            else:
                logger.warning("Config requested freezing, but `saliency_network` not found on model.")

    # --- 3. Move to GPU, Wrap with DDP ---
    model.to(ddp.device)
    if ddp.enabled:
        # If parts of the model are frozen, DDP needs `find_unused_parameters=True`
        # to avoid errors when it doesn't see gradients for the frozen layers.
        find_unused = run_cfg.stage.extra.get('freeze_saliency_network', True)
        model = DDP(model, device_ids=[ddp.device.index], find_unused_parameters=find_unused)
        if ddp.is_master:
            logger.info(f"Wrapped model with DDP (find_unused_parameters={find_unused})")

    # --- 4. Build Dataset & Optimizer (needed before potential warmup) ---
    if ddp.is_master: logger.info("Building dataset...")
    if run_cfg.stage.dataset_key not in DATA_REGISTRY:
        raise KeyError(f"Dataset '{run_cfg.stage.dataset_key}' not registered")
    train_dl, val_dl, baseline_ll = DATA_REGISTRY[run_cfg.stage.dataset_key](run_cfg, ddp, logger)

    if ddp.is_master: logger.info("Creating optimizer and scheduler...")
    optim, sched = make_optim_and_sched(model, run_cfg.stage)

    # --- 5. Compile and Warmup ---
    if run_cfg.compile:
        if ddp.is_master: logger.info("Compiling model with torch.compile...")
        model = torch.compile(model, mode="reduce-overhead")

        # --- WARMUP STEP to compile the training graph before any inference ---
        if ddp.is_master:
            logger.info("Running a single-batch training warmup to compile the backward pass graph...")
        try:
            warmup_batch = next(iter(train_dl))
            
            # Prepare batch for model call
            model_call_args = []
            model_call_kwargs = {k: v.to(ddp.device, non_blocking=True) for k, v in warmup_batch.items() if isinstance(v, torch.Tensor)}
            if 'image' in model_call_kwargs: model_call_args.append(model_call_kwargs.pop('image'))
            if 'centerbias' in model_call_kwargs: model_call_args.append(model_call_kwargs.pop('centerbias'))

            # Perform a single forward/backward pass
            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                from src.metrics import log_likelihood # Assuming this is your loss metric
                log_density = model(*model_call_args, **model_call_kwargs)
                target_mask = model_call_kwargs.get('fixation_mask')
                if isinstance(target_mask, torch.sparse.Tensor): target_mask = target_mask.to_dense()
                loss = -log_likelihood(log_density, target_mask, weights=model_call_kwargs.get('weight'))

            dummy_scaler = torch.cuda.amp.GradScaler(enabled=(ddp.device.type == 'cuda'))
            dummy_scaler.scale(loss).backward()
            optim.zero_grad(set_to_none=True) # Reset grads immediately, we don't want to step
            if ddp.is_master: logger.info("Warmup successful. Compiled graph is ready.")
        except Exception:
            logger.error("The model warmup step failed. This indicates a potential issue.", exc_info=True)
            raise # Re-raise the exception to stop the run

    # --- 6. Call the Generic Training Loop ---
    train_sampler_for_loop = (
        train_dl.batch_sampler if hasattr(train_dl, "batch_sampler") and train_dl.batch_sampler is not None
        else train_dl.sampler if hasattr(train_dl, "sampler") else None
    )
    
    # We pass `startwith=run_cfg.stage.resume_ckpt` if it's NOT a fine-tuning run.
    # If it IS a fine-tuning run, we pass `None` because we have already handled
    # the initial load. The train_loop will then manage its own resumption for
    # the current stage if it gets interrupted.
    startwith_for_loop = None if is_finetuning_from_ckpt else run_cfg.stage.resume_ckpt
    
    train_loop(
        this_directory=run_cfg.paths["train_dir"] / run_cfg.stage.kind / run_cfg.stage.name,
        model=model,
        train_loader=train_dl,
        train_baseline_log_likelihood=baseline_ll["train"],
        val_loader=val_dl,
        val_baseline_log_likelihood=baseline_ll["val"],
        optimizer=optim,
        lr_scheduler=sched,
        gradient_accumulation_steps=run_cfg.stage.grad_acc_steps,
        minimum_learning_rate=run_cfg.stage.min_lr,
        validation_epochs=run_cfg.stage.val_every,
        early_stopping_patience=run_cfg.stage.early_stopping_patience,
        startwith=startwith_for_loop,
        device=ddp.device,
        is_distributed=ddp.enabled,
        is_master=ddp.is_master,
        logger=logging.getLogger("trainer"),
        train_sampler=train_sampler_for_loop,
        is_finetuning_run=is_finetuning_from_ckpt
    )

    ddp.cleanup()

# -----------------------------------------------------------------------------
#  6.  Minimal CLI / YAML loader  ---------------------------------------------
# -----------------------------------------------------------------------------

def _parse_cli() -> Tuple[argparse.Namespace, Dict[str, str]]:
    parser = argparse.ArgumentParser(description="Unified trainer for gaze‑prediction models")
    parser.add_argument("--config", type=str, required=True, help="YAML file with default settings")
    parser.add_argument("overrides", nargs="*", help="Any key=value pair to override cfg (dot notation)")
    args = parser.parse_args()

    kv = {}
    for ov in args.overrides:
        if "=" not in ov:
            raise ValueError(f"Override '{ov}' must be key=value")
        k, v = ov.split("=", 1)
        kv[k] = v
    return args, kv


def _deep_set(obj: Any, dotted_key: str, value: str):
    """Set dataclass or dict fields with dotted path."""
    parts = dotted_key.split(".")
    cur = obj
    for p in parts[:-1]:
        cur = getattr(cur, p) if _dc.is_dataclass(cur) else cur[p]
    last = parts[-1]
    v_cast: Any = value
    # naive cast: int, float, bool else str
    for _typ in (int, float):
        try:
            v_cast = _typ(value)
            break
        except ValueError:
            pass
    if value.lower() in ("true", "false"):
        v_cast = value.lower() == "true"
    if _dc.is_dataclass(cur):
        setattr(cur, last, v_cast)
    else:
        cur[last] = v_cast


def _load_cfg(path: str) -> RunCfg:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    # --- Stage ---------------------------------------------------------
    stage_raw = raw.get("stage", {})
    stage_cfg = StageCfg(
            kind        = stage_raw["kind"],
            name        = stage_raw.get("name"),
            model_key   = stage_raw["model_key"],
            dataset_key = stage_raw["dataset_key"],
            lr = stage_raw.get("lr", 5e-4),
            milestones = tuple(stage_raw.get("milestones", ())),
            batch_size = stage_raw.get("batch_size", 8),
            grad_acc_steps = stage_raw.get("grad_acc_steps", 1),
            min_lr = stage_raw.get("min_lr", 1e-7),
            val_every = stage_raw.get("val_every", 1),
            early_stopping_patience = stage_raw.get("early_stopping_patience", 3),
            resume_ckpt = stage_raw.get("resume_ckpt"),
            extra       = stage_raw.get("extra", {}),

    )
    paths = {k: Path(v) for k, v in raw.get("paths", {}).items()}

    cfg = RunCfg(stage=stage_cfg,
                compile=raw.get("compile", False),
                seed=raw.get("seed", 123),
                paths=paths,
                num_workers=raw.get("num_workers", 'auto'))
    return cfg

# -----------------------------------------------------------------------------
#  7.  Main  -------------------------------------------------------------------
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    
    _auto_import_modules()
    
    cli_ns, overrides = _parse_cli()
    cfg = _load_cfg(cli_ns.config)

    # apply CLI overrides
    for k, v in overrides.items():
        _deep_set(cfg, k, v)
        
    # Resolve 'auto' for num_workers into an integer
    if isinstance(cfg.num_workers, str) and cfg.num_workers.lower() == 'auto':
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        try:
            # Get the number of available CPUs for this process
            cpu_count = len(os.sched_getaffinity(0))
        except AttributeError:
            cpu_count = os.cpu_count() or 1
        
        # Calculate workers per process, with a reasonable cap (e.g., 8)
        workers_per_process = min(8, cpu_count // world_size if world_size > 0 else cpu_count)
        cfg.num_workers = workers_per_process
    else:
        # Ensure it's an integer if not 'auto'
        cfg.num_workers = int(cfg.num_workers)

    # ensure working dirs
    if cfg.paths["train_dir"]:
        cfg.paths["train_dir"].mkdir(parents=True, exist_ok=True)

    train_stage(cfg)
    