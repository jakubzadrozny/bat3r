# RUN:
#   accelerate launch --gpu_ids=0,1,2,3 --num_processes=4 scripts/train_distrib.py
#
"""
Copyright 2025 University of Oxford
Author: Ben Kaye
Licence: BSD-3-Clause

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import copy
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import hydra
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler
import torch.utils.data as data
import wandb
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed, DataLoaderConfiguration
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from bat3r.pointmaps import ConvUnet, PointmapModule, SequentialUnet
from bat3r.third_party.prepare_dataloder import _prepare_data_loader

WANDB_ENABLED = False
WANDB_RUN_NAME = None


def configure_logger():
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        ))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return get_logger(__name__)

logger = configure_logger()

@dataclass
class TrainingLoopConfig:
    steps: int = 100000
    save_every: int = 5000
    val_every: int = 20
    log_every: int = 10
    save_path: str = "weights"
    gradient_clip_value: float | None = 1.0
    gradient_accumulation_steps: int = 1
    start_delta: float = 0.1
    end_delta: float = 100.0
    alpha: float = 1.0


def training_loop(
    accelerator: Accelerator,
    module: PointmapModule,
    train_loader: data.DataLoader,
    val_loader: data.DataLoader | None = None,
    train_cfg: DictConfig | TrainingLoopConfig | None = None,
) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    """
    Standard NN training loop

    Args:
        module: PointmapModule
        train_loader: DataLoader
        val_loader: DataLoader
        train_cfg: TrainingLoopConfig

    Returns:
        training_losses: list[tuple[int, float]] (step_index, loss)
        validation_losses: list[tuple[int, float]] (step_index, loss)
    """
    train_cfg = train_cfg or TrainingLoopConfig()

    model: nn.Module = module.model
    optimizer: optim.Optimizer = module.optimizer

    if optimizer is None:
        raise ValueError("Optimizer required for training")

    scheduler: optim.lr_scheduler.LRScheduler | None = module.scheduler

    num_iters = 0
    epoch = 0
    model.train()

    grad_clip: bool = (
        train_cfg.gradient_clip_value is not None and train_cfg.gradient_clip_value > 0
    )

    pbar = tqdm(
        total=train_cfg.steps,
        desc="Training..",
        disable=True,
    )

    val_loss = float("inf")
    val_loss_smoothed = float("inf")
    loss_moving_average = torch.zeros(train_cfg.log_every)
    train_losses = []
    train_grad_norms = []
    val_losses = []
    val_canon_losses = []
    val_posed_losses = []
    val_occupancy_losses = []
    val_confidences = []

    no_save = train_cfg.save_path is None
    save_path = None
    if not no_save and accelerator.is_main_process:
        save_path = Path(train_cfg.save_path)
        if WANDB_ENABLED and WANDB_RUN_NAME is not None:
            save_path = save_path / WANDB_RUN_NAME
        save_path.mkdir(parents=True, exist_ok=True)

    def step(batch: tuple) -> tuple[float, float, float, float, bool, float]:
        if not model.training:
            model.train()

        progress = num_iters / train_cfg.steps
        warped_progress = progress ** train_cfg.alpha
        current_delta = train_cfg.start_delta * ((train_cfg.end_delta / train_cfg.start_delta) ** warped_progress)

        with accelerator.accumulate(model):
            # pixelwise_loss_sum, pixelwise_pixels, occupancy_loss_sum, occupancy_pixels = module.training_step(batch, reduction='none')
            (pixelwise_loss_sum, pixelwise_pixels, occupancy_loss_sum, occupancy_pixels), output, batch = module.training_step(
                batch, reduction='none', return_preds=True, delta=current_delta,
            )
            
            with torch.no_grad():
                eval_losses_train, _, _, _ = module.compute_eval_losses(output, batch.model_targets, mask=batch.mask)
                per_sample_total_loss = eval_losses_train["total_loss"]
                sample_ids = batch.data_id
                
                high_loss_indices = (per_sample_total_loss > 0.4).nonzero(as_tuple=True)[0]
                if len(high_loss_indices) > 0:
                    high_loss_ids = [sample_ids[i] for i in high_loss_indices.tolist()]
                    high_loss_vals = [per_sample_total_loss[i].item() for i in high_loss_indices.tolist()]
                    logger.warning(f"High training loss detected: {list(zip(high_loss_ids, high_loss_vals))}")

            if pixelwise_loss_sum.isnan() or occupancy_loss_sum.isnan():
                raise ValueError("Loss is NaN")
            
            # Scale loss to match single-GPU reduction (global average over pixels)
            pixelwise_pixels = pixelwise_pixels.detach()
            occupancy_pixels = occupancy_pixels.detach()

            pixelwise_pixels_global = accelerator.reduce(pixelwise_pixels, reduction="sum")
            occupancy_pixels_global = accelerator.reduce(occupancy_pixels, reduction="sum")

            if pixelwise_pixels_global == 0: pixelwise_pixels_global = 1e-6
            if occupancy_pixels_global == 0: occupancy_pixels_global = 1e-6
            
            scale_pixelwise = accelerator.num_processes / pixelwise_pixels_global
            scale_occupancy = accelerator.num_processes / occupancy_pixels_global
            
            loss = (pixelwise_loss_sum * scale_pixelwise) + (occupancy_loss_sum * scale_occupancy)

            # NOTE: When using `accelerator.accumulate(model)`, Accelerate automatically 
            # scales the loss (i.e. divides by gradient_accumulation_steps) under the hood 
            # during `backward()`. The magnitude of gradients is kept consistent.
            accelerator.backward(loss)

            loss_val = accelerator.reduce(loss.detach(), reduction="mean").item()

            total_norm = 0.0
            clipped_percentage = 0.0
            grad_quantile_99 = 0.0
            
            synced = accelerator.sync_gradients

            if synced and grad_clip:
                with torch.no_grad():
                    grads = [p.grad.abs().flatten() for p in model.parameters() if p.grad is not None]
                    if grads:
                        all_grads = torch.cat(grads)
                        clipped_percentage = (all_grads > train_cfg.gradient_clip_value).float().mean().item() * 100
                    
                    if all_grads.numel() > 1_000_000:
                        indices = torch.randint(0, all_grads.numel(), (1_000_000,), device=all_grads.device)
                        sample_grads = all_grads[indices]
                    else:
                        sample_grads = all_grads
                    grad_quantile_99 = torch.quantile(sample_grads.float(), 0.99).item()
                accelerator.clip_grad_value_(model.parameters(), train_cfg.gradient_clip_value) # <- BEFORE, causes training instabilities
                # Compute the total norm consistently with clip_grad_norm_ by setting max_norm=inf
                total_norm = accelerator.clip_grad_norm_(model.parameters(), float('inf'))
                if isinstance(total_norm, torch.Tensor):
                    total_norm = total_norm.item()

            # Note: optimizer and scheduler step must be outside the `if synced:` guard 
            # because accelerate.accumulate wraps them and handles skipping automatically
            optimizer.step()
            optimizer.zero_grad()

            if scheduler:
                scheduler.step()

        return loss_val, total_norm, clipped_percentage, grad_quantile_99, synced, current_delta

    canon_loss_global = float("inf")
    posed_loss_global = float("inf")
    occupancy_loss_global = float("inf")
    confidence_global = 0.0
    grad_norm_moving_max = 0.0
    grad_norm_max_iter = 0
    loss_moving_max = float("-inf")
    loss_max_iter = 0
    clipped_pct_moving_max = 0.0
    grad_q99_moving_max = 0.0
    
    accumulated_loss = 0.0
    accumulation_counter = 0
    accumulated_delta = 0.0

    while num_iters < train_cfg.steps:
        logger.info(f"begin epoch {epoch}")

        val_loader.sampler.set_epoch(epoch)
        val_iter = iter(val_loader) if val_loader is not None else None

        for batch in train_loader:
            loss, grad_norm, clipped_pct, grad_q99, synced, current_delta = step(batch)
            accumulated_loss += loss
            accumulation_counter += 1
            accumulated_delta += current_delta

            if not synced:
                continue

            num_iters += 1
            pbar.update(1)

            avg_loss = accumulated_loss / accumulation_counter
            avg_delta = accumulated_delta / accumulation_counter
            accumulated_loss = 0.0
            accumulation_counter = 0
            accumulated_delta = 0.0

            if grad_norm > grad_norm_moving_max:
                grad_norm_moving_max = grad_norm
                grad_norm_max_iter = num_iters

            if avg_loss > loss_moving_max:
                loss_moving_max = avg_loss
                loss_max_iter = num_iters

            if clipped_pct > clipped_pct_moving_max:
                clipped_pct_moving_max = clipped_pct

            if grad_q99 > grad_q99_moving_max:
                grad_q99_moving_max = grad_q99

            if WANDB_ENABLED and accelerator.is_main_process:
                wandb.log({"train/loss": avg_loss})

            loss_moving_average[num_iters % train_cfg.log_every] = avg_loss

            # DEBUG:
            # if len(loss_history) > 0:
            #     avg_loss = sum(loss_history) / len(loss_history)
            #     if (loss - avg_loss) > (max(loss_history) - min(loss_history)):
            #         batch = PointmapBatch(*batch)
            #         logger.warning(f"Loss spike detected at step {num_iters}: {loss:.4f} (avg last 50: {avg_loss:.4f})")
            #         logger.warning(f"data ids: {batch.data_id}", main_process_only=False)
            # loss_history.append(loss)
            # if len(loss_history) > 50: loss_history.pop(0)
            # END DEBUG

            if val_loader is not None and not num_iters % train_cfg.val_every:
                val_batch = next(val_iter)

                model.eval()
                with torch.no_grad():
                    (val_pixelwise_loss, val_pixelwise_pixels, val_occupancy_loss, val_occupancy_pixels), output, batch = module.validation_step(val_batch, reduction='none')
                    
                    eval_losses, _, _, _ = module.compute_eval_losses(output, batch.model_targets, mask=batch.mask)

                    val_pixelwise_loss_global = accelerator.reduce(val_pixelwise_loss, reduction="sum")
                    val_occupancy_loss_global = accelerator.reduce(val_occupancy_loss, reduction="sum")
                    val_pixelwise_pixels_global = accelerator.reduce(val_pixelwise_pixels, reduction="sum")
                    val_occupancy_pixels_global = accelerator.reduce(val_occupancy_pixels, reduction="sum")

                    if val_pixelwise_pixels_global == 0: val_pixelwise_pixels_global = 1e-6
                    if val_occupancy_pixels_global == 0: val_occupancy_pixels_global = 1e-6

                    # Exponential Moving Average for validation loss (recent weighted more heavily
                    if val_loss_smoothed == float("inf"):
                        val_loss_smoothed = val_loss
                    else:
                        val_loss_smoothed = 0.7 * val_loss_smoothed + 0.3 * val_loss
                    _val_loss = (val_pixelwise_loss_global / val_pixelwise_pixels_global) + (val_occupancy_loss_global / val_occupancy_pixels_global)
                    val_loss = _val_loss.item()
                    # logger.info(f"Train step {num_iters}, val loss={val_loss:.4f}")

                    # Breakdown components
                    canon_loss_local = eval_losses["canon_loss"].mean()
                    posed_loss_local = eval_losses["posed_loss"].mean()
                    occupancy_loss_local = eval_losses["occupancy_loss"].mean()
                    confidence_local = eval_losses["confidence"].mean()

                    canon_loss_global = accelerator.reduce(canon_loss_local, reduction="mean").item()
                    posed_loss_global = accelerator.reduce(posed_loss_local, reduction="mean").item()
                    occupancy_loss_global = accelerator.reduce(occupancy_loss_local, reduction="mean").item()
                    confidence_global = accelerator.reduce(confidence_local, reduction="mean").item()

                    val_canon_losses.append((num_iters, canon_loss_global))
                    val_posed_losses.append((num_iters, posed_loss_global))
                    val_occupancy_losses.append((num_iters, occupancy_loss_global))
                    val_confidences.append((num_iters, confidence_global))

                    if WANDB_ENABLED and accelerator.is_main_process:
                        wandb.log({"val/loss": val_loss})

                    val_losses.append((num_iters, val_loss))
            
            if (not num_iters % train_cfg.log_every):
                loss_ = loss_moving_average.mean().item()
                train_losses.append((num_iters, loss_))
                train_grad_norms.append((num_iters, grad_norm_moving_max))
                pbar.set_description(
                    f"Training loss: {loss_:.4f}, Val loss: {val_loss_smoothed:.4f}"
                )
                # logger.info(f"Iter: {num_iters}, Epoch: {epoch}, Train loss: {loss_:.4f}, Val loss: {val_loss:.4f} (C: {canon_loss_global:.4f}, P: {posed_loss_global:.4f}, O: {occupancy_loss_global:.4f}, Conf: {confidence_global:.4f}, Grad: {grad_norm_moving_max:.4f} @ {grad_norm_max_iter}, Max Loss: {loss_moving_max:.4f} @ {loss_max_iter}, Clip: {clipped_pct_moving_max:.4f}%, Q99: {grad_q99_moving_max:.4f})")
                # logger.info(f"It: {num_iters}, Ep: {epoch} | Trn: {loss_:.4f} Val: {val_loss:.4f} (C: {canon_loss_global:.4f} P: {posed_loss_global:.4f} O: {occupancy_loss_global:.4f} Cf: {confidence_global:.4f}) | MaxL: {loss_moving_max:.4f} @ {loss_max_iter} Grad: {grad_norm_moving_max:.2f} @ {grad_norm_max_iter} Clp: {clipped_pct_moving_max:.4f}% Q99: {grad_q99_moving_max:.4f} Dlt: {avg_delta:.2f}")
                logger.info(f"It: {num_iters}, Ep: {epoch} | Trn: {loss_:.4f} Val: {val_loss:.4f} (C: {canon_loss_global:.4f} P: {posed_loss_global:.4f} O: {occupancy_loss_global:.4f} Cf: {confidence_global:.4f}) | MaxL: {loss_moving_max:.4f} @ {loss_max_iter} Grad: {grad_norm_moving_max:.2f} @ {grad_norm_max_iter} Dlt: {avg_delta:.2f}")
                grad_norm_moving_max = 0.0
                grad_norm_max_iter = 0
                loss_moving_max = float("-inf")
                loss_max_iter = 0
                clipped_pct_moving_max = 0.0
                grad_q99_moving_max = 0.0

            if not no_save and not num_iters % train_cfg.save_every:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    torch.save(
                        dict(
                            model_state=accelerator.unwrap_model(model).state_dict(),
                            optim_state=optimizer.state_dict(),
                            scheduler_state=scheduler.state_dict() if scheduler else None,
                            train_losses=train_losses,
                            train_grad_norms=train_grad_norms,
                            val_losses=val_losses,
                            val_canon_losses=val_canon_losses,
                            val_posed_losses=val_posed_losses,
                            val_occupancy_losses=val_occupancy_losses,
                            val_confidences=val_confidences,
                        ),
                        save_path / f"weights_{num_iters}.pth",
                    )

            if num_iters >= train_cfg.steps:
                break

        epoch += 1

    return train_losses, val_losses


def get_module(config: DictConfig, training: bool) -> tuple[SequentialUnet | ConvUnet, optim.Optimizer, LRScheduler]:
    """
    Load the model, optimizer and scheduler.
    """
    model: SequentialUnet | ConvUnet = hydra.utils.instantiate(config.model)
    
    optim_state, scheduler_state = None, None
    if config.train_config.get("use_weights", None) is not None:
        weights_path = Path(config.train_config.use_weights)
        if weights_path.exists():
            # map to CPU
            state_dicts = torch.load(weights_path, map_location="cpu")
            if "model_state" in state_dicts:
                model_weight_dict = state_dicts["model_state"]
            else:
                model_weight_dict = state_dicts

            if "optim_state" in state_dicts:
                optim_state = state_dicts["optim_state"]
            if "scheduler_state" in state_dicts:
                scheduler_state = state_dicts["scheduler_state"]

            model.load_state_dict(model_weight_dict)
        else:
            raise FileNotFoundError(f"Weights file not found: {weights_path}")

    if training:
        optimizer: optim.Optimizer = hydra.utils.instantiate(
            config.optimizer, params=model.parameters(), _convert_="all"
        )
        scheduler: LRScheduler = (
            hydra.utils.instantiate(config.scheduler, optimizer=optimizer, _convert_="all")
            if config.get("scheduler", None) is not None
            else None
        )

        if optim_state is not None:
            optimizer.load_state_dict(optim_state)
        if scheduler_state is not None:
            scheduler.load_state_dict(scheduler_state)

    model.train(training)

    return model, optimizer, scheduler


def get_source_dir() -> str:
    import bat3r

    return Path(bat3r.__file__).parent.absolute().as_posix()


def sample_indices(ind, N):
    pi = np.random.permutation(len(ind))
    return ind[pi[:N]]


def get_filtered_dataset(dataset, cfg, base_train_ids, dataset_name="dataset"):
    df_path = Path(dataset.root) / "train.csv"
    train_ids = base_train_ids.copy()
    
    if df_path.exists():
        df = pd.read_csv(str(df_path), dtype={"file_id": "string"})
        conditions = [
            df['cd_canon'] < cfg.cd_canon_thresh,
            df['cd_fitted_to_rec'] < cfg.cd_fitted_thresh,
            df['confidence'] > cfg.conf_thresh,
            df['fitting_total_loss'] < cfg.fitting_loss_thresh,
            df['joints_dist'] < cfg.joints_dist_thresh,
            df['cd_fitted_to_can'] < cfg.fitted_to_can_thresh,
        ]
        is_valid = np.logical_and.reduce(conditions)
        valid_ids = set(df.loc[is_valid, "file_id"].tolist())
        train_ids = train_ids & valid_ids
        logger.info(f"All {dataset_name} samples: {len(df)}, of which valid: {len(valid_ids)}, of which train: {len(train_ids)}")
    
    train_ind = [idx for idx, file_id in enumerate(dataset.ids) if file_id.split('_')[0] in train_ids]
    filtered_dataset = data.Subset(dataset, train_ind)
    logger.info(f"Length of {dataset_name} train set: {len(filtered_dataset)}")
    return filtered_dataset


@hydra.main(config_path="../configs", config_name="main", version_base="1.3")
def main(cfg: DictConfig):
    """initialise and train a network"""
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)

    seed = cfg.get("seed", None)
    dataloader_config = DataLoaderConfiguration(
        even_batches=True,
        split_batches=True,
        use_seedable_sampler=True,
        data_seed=seed,
        non_blocking=True,
    )
    accelerator = Accelerator(
        dataloader_config=dataloader_config,
        gradient_accumulation_steps=cfg.train_config.get("gradient_accumulation_steps", 1),
    )

    global WANDB_RUN_NAME
    global WANDB_ENABLED
    wandb_cfg = cfg.get("wandb", None)
    WANDB_ENABLED = wandb_cfg is not None and wandb_cfg.enabled
    if WANDB_ENABLED and accelerator.is_main_process:
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        wandb.init(
            project=cfg.wandb.project,
            mode="online",
            tags=[str(x) for x in cfg.wandb.tags],
            config=cfg_dict | dict(local_repository=Path(__file__).parent.parent),
            settings=wandb.Settings(code_dir="src"),
            dir=cfg.wandb.dir,
        )

        WANDB_RUN_NAME = wandb.run.name

    logger.info(f"Source directory: {get_source_dir()}")
    logger.info(f"Log dir: {cfg.train_config.save_path}")
    logger.info(f"Data root: {cfg.dataset_root}")

    if seed is not None:
        set_seed(seed)

    model, optimizer, scheduler = get_module(cfg, training=True)
    module: PointmapModule = hydra.utils.instantiate(cfg.module, device=accelerator.device, model=model)

    train_ids_path = Path(cfg.dataset_root) / "train_ids.json"
    test_ids_file = Path(cfg.val_dataset_root) / "test_ids.json"
    with train_ids_path.open() as f:
        train_ids = set(json.load(f))
    with test_ids_file.open() as f:
        test_ids = set(json.load(f))

    train_ids_orig = train_ids.copy()

    dataset = hydra.utils.instantiate(cfg.dataset)
    train_dataset = get_filtered_dataset(dataset, cfg, train_ids_orig, dataset_name="primary dataset")

    secondary_dataset_root = cfg.get("secondary_dataset_root", None)
    if secondary_dataset_root is not None:
        sec_dataset_cfg = copy.deepcopy(cfg.dataset)
        sec_dataset_cfg.root = secondary_dataset_root
        secondary_dataset = hydra.utils.instantiate(sec_dataset_cfg)
        secondary_dataset = get_filtered_dataset(secondary_dataset, cfg, train_ids_orig, dataset_name="secondary dataset")
        train_dataset = data.ConcatDataset([train_dataset, secondary_dataset])

    val_dataset = hydra.utils.instantiate(cfg.val_dataset)
    test_ind = [idx for idx in range(len(val_dataset)) if val_dataset.ids[idx] in test_ids]
    val_dataset = data.Subset(val_dataset, test_ind)

    train_loader = hydra.utils.instantiate(cfg.dataloader, dataset=train_dataset, pin_memory=True)
    # CONSIDER:
    # train_loader = hydra.utils.instantiate(cfg.dataloader, dataset=train_dataset, batch_size=batch_size, drop_last=True)

    val_sampler = data.DistributedSampler(
        val_dataset,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        shuffle=True,
        seed=seed,
        drop_last=False,
    )
    if cfg.train_config.batch_size % accelerator.num_processes != 0:
        raise ValueError(f"batch size {cfg.train_config.batch_size} not divisible by num workers {accelerator.num_processes}")
    val_batch_size = cfg.train_config.batch_size // accelerator.num_processes
    logger.info(f"Local batch size: {val_batch_size}")
    val_loader = (
        hydra.utils.instantiate(cfg.val_loader, dataset=val_dataset, batch_size=val_batch_size, sampler=val_sampler, shuffle=False, pin_memory=True)
        if cfg.get("val_loader", None) is not None
        else None
    )

    if accelerator.num_processes > 1:
        model, optimizer, train_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, scheduler)
    else:
        model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
        train_loader = _prepare_data_loader(accelerator, train_loader)

    module.model = model
    module.optimizer = optimizer
    module.scheduler = scheduler

    logger.info(f"Training on {len(train_dataset)} samples, {len(train_loader)} batches")

    try:
        training_loop(
            accelerator,
            module,
            train_loader,
            val_loader,
            train_cfg=cfg.train_config,
        )
    finally:
        accelerator.end_training()


if __name__ == "__main__":
    main()
