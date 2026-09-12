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

import logging
from typing import Literal

import einops
import torch
import torch.nn.functional as F
import torch.optim as opt
from torch import Tensor

from bat3r.dataset import MeshToDualPointmap, PointmapBatch
from bat3r.models import ConvUnet, SequentialUnet

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
    return logger

logger = configure_logger()

ACTIVATED_WARNING_FLAG = [False]
UNACTIVATED_WARNING_FLAG = [False]


def make_dual_clip_hook(name: str, val_clip: float = 0.002, max_norm: float = 0.1):
    def hook(grad: Tensor) -> Tensor:
        val_clipped_grad = torch.clamp(grad, min=-val_clip, max=val_clip)
        clip_pct = (grad.abs() > val_clip).float().mean().item() * 100
        if clip_pct > 0.05:
            logger.warning(f"{name} Value Clipped {clip_pct:.3f}% of pixels.")

        g_norm = torch.linalg.vector_norm(val_clipped_grad)
        if g_norm > 2*max_norm:
            logger.warning(f"{name} Norm Scaled! (was {g_norm:.4f}, reduced to {max_norm})")
        if g_norm > max_norm:
            # Scale down the already direction-corrected gradient
            final_grad = val_clipped_grad * (max_norm / (g_norm + 1e-6))
            return final_grad

        return val_clipped_grad
    return hook


def extract_from_amodal_pointmap(
    pointmap: Tensor, num_layers: int, activate_occupancy: bool, is_targets: bool = False,
    exp_confidence: bool = False,
) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
    """Extracts canonical positions, posed positions, occupancy, and confidence from a multi-layer pointmap.


    Args:
        pointmap (Tensor): Input tensor of shape (B, N*C, H, W) or (N*C, H, W) where:
            - C is channels per layer (7 or 8 depending on confidence)
            - N is number of layers
        num_layers (int): Number of layers in the pointmap

    Returns:
        tuple[Tensor]: Tuple containing:

        < B H W N C >
    """

    if pointmap.requires_grad and activate_occupancy:
        if not ACTIVATED_WARNING_FLAG[0]:
            logger.warning(
                "It appears you are training and activating occupancy.\n\
                this may lead to unintended effects!"
            )
            ACTIVATED_WARNING_FLAG[0] = True
    elif not pointmap.requires_grad and not activate_occupancy and not is_targets:
        if not UNACTIVATED_WARNING_FLAG[0]:
            logger.warning(
                "It appears you are evaluating and not activating occupancy\n\
                Does not apply if using with model targets.\n\
                this may lead to unintended effects!"
            )
            UNACTIVATED_WARNING_FLAG[0] = True

    is_batched = pointmap.dim() == 4
    if not is_batched:
        pointmap = pointmap[None]

    assert pointmap.dim() == 4, "pointmap should be (B, C, H, W) or (C, H, W)"

    pointmap = einops.rearrange(pointmap, "... (n c) h w -> ... h w n c", n=num_layers)

    has_confidence = (num_layers > 1 and pointmap.shape[-1] == 8) or (
        num_layers == 1 and pointmap.shape[-1] == 7
    )

    occupancy, confidence = None, None
    if num_layers > 1:
        if has_confidence:
            canon, posed, occupancy, confidence = pointmap.split([3, 3, 1, 1], dim=-1)
        else:
            canon, posed, occupancy = pointmap.split([3, 3, 1], dim=-1)
    else:
        if has_confidence:
            canon, posed, confidence = pointmap.split([3, 3, 1], dim=-1)
        else:
            canon, posed = pointmap.split([3, 3], dim=-1)

    if confidence is not None:
        if confidence.requires_grad:
            # This intercepts the gradient flowing back from the loss into the confidence channel.
            # It hard-caps the gradient values between -1.0 and 1.0.
            confidence.register_hook(make_dual_clip_hook("confidence", val_clip=0.0002, max_norm=0.002))
        if exp_confidence:
            print('using old exp activation for confidence')
            confidence = 1 + torch.exp(confidence)
            # confidence = 1e-4 + torch.exp(confidence)
        else:
            confidence = 1e-3 + F.softplus(confidence) # -> use this by default

    if activate_occupancy and occupancy is not None:
        occupancy = F.sigmoid(occupancy)

    if not is_batched:
        canon = canon[0]
        posed = posed[0]
        occupancy = occupancy[0] if occupancy is not None else None
        confidence = confidence[0] if confidence is not None else None

    return canon, posed, occupancy, confidence


def evaluate_layers(parameter: Tensor, occupancy: Tensor):
    # parameter: (B H W N C)

    is_batched = parameter.dim() == 5

    if not is_batched:
        parameter = parameter[None]
        occupancy = occupancy[None]

    occupancy = occupancy.squeeze(dim=4)
    assert occupancy.dim() == 4, "occupancy should be (B, H, W, N, (1))"
    assert parameter.dim() == 5, "parameter should be (B, H, W, N, C)"

    assert parameter.shape[:3] == occupancy.shape[:3], (
        "parameter and occupancy must have the same shape"
    )
    result = [
        [
            parameter[b, :, :, layer][occupancy[b, :, :, layer] > 0.5] # -> HERE OCC THRESH
            for layer in range(parameter.shape[-2])
        ]
        for b in range(parameter.shape[0])
    ]
    if not is_batched:
        return result[0]
    return result


def extract_predictions(
    amodal_pointmap: Tensor,
    num_layers: int,
    activate_occupancy: bool,
    return_type: Literal["pointmap", "layered_pointcloud", "pointcloud"],
    mask: Tensor | None = None,
    exp_confidence: bool = False,
) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
    """
    Separate the amodal pointmap into canonical, posed, occupancy, and confidence.

    Args:
        amodal_pointmap: (B, N*C, H, W)
        num_layers: int
        activate_occupancy: bool
        return_type: str
        mask: (B, H, W) | None

    return_type==pointmap:
        tensors (B, H, W, N, Ci)
    return_type==layered_pointcloud:
        B lists of N lists of tensors (Lk, Ci)
    return_type==pointcloud:
        B lists of tensors (Li, Ci)

    Returns:
        canon: C=3
        posed: C=3
        occupancy: (B, H, W, N, 1)
        confidence: C=1
    """
    as_pointmap, combine_layers = False, None
    match return_type:
        case "pointmap":
            as_pointmap = True
        case "layered_pointcloud":
            combine_layers = False
        case "pointcloud":
            combine_layers = True
        case _:
            raise ValueError(f"Invalid return type: {return_type}")

    canon, posed, occupancy, confidence = extract_from_amodal_pointmap(
        amodal_pointmap, num_layers, activate_occupancy, exp_confidence=exp_confidence
    )

    if mask is not None:
        if occupancy is None:
            occupancy = (
                mask[..., None, None]
                .clone(deep=True)
                .expand(-1, -1, -1, num_layers, -1)
            )
        else:
            occupancy = occupancy * mask[..., None, None]

    assert occupancy.min() >= 0 and occupancy.max() <= 1, (
        "occupancy should be between 0 and 1"
    )

    if as_pointmap:
        return canon, posed, occupancy, confidence

    if occupancy is None and mask is None:
        raise ValueError("mask is required if occupancy is not predicted")

    canon, posed, confidence = (
        evaluate_layers(t, occupancy) if t is not None else None
        for t in [canon, posed, confidence]
    )

    if combine_layers:
        canon, posed, confidence = (
            [torch.cat(t_el) for t_el in t] if t is not None else None
            for t in [canon, posed, confidence]
        )

    return canon, posed, occupancy, confidence


def layer_occupancy_loss(
    prediction: Tensor,
    target_occupancy: Tensor,
    soft_label: float | None,
    reduction: str = 'full',
) -> Tensor | tuple[Tensor, Tensor]:
    """
    Computes the occupancy loss for a single layer as the expectation over the segmentation mask of the BCE loss

    Args:
        prediction: (B, H, W, N)
        target_occupancy: (B, H, W, N)
        soft_label: float

        returns expected value (0 dim)
    """

    if soft_label is None:
        soft_label = 0.0

    mask = target_occupancy[:, :, :, [0]]

    clipped_labels = target_occupancy.clamp(soft_label, 1 - soft_label)

    loss = mask * F.binary_cross_entropy_with_logits(
        prediction,
        clipped_labels,
        reduction="none",
    )
    
    num_pixels = mask.sum() * target_occupancy.shape[-1]

    if reduction == 'per-layer':
        return loss.sum((1, 2)) / mask.sum((1, 2))
    elif reduction == 'per-sample':
        return loss.sum((-1, -2, -3)) / mask.sum((-1, -2, -3)) / target_occupancy.shape[-1]
    elif reduction == 'full':
        return loss.sum() / num_pixels
    elif reduction == 'none':
        return loss.sum(), num_pixels
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


class PointmapModule:
    """
    Encapsulate training of a DualPM model
    """

    confidence_alpha: float
    soft_label: float | None
    device: str = "cuda"
    renderer: MeshToDualPointmap | None
    num_layers: int
    model_confidence: bool
    optimizer: opt.Optimizer | None
    scheduler: torch.optim.lr_scheduler._LRScheduler | None

    def __init__(
        self,
        model: SequentialUnet | ConvUnet,
        optimizer: opt.Optimizer | None = None,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
        render_on_module: bool = True,
        confidence_alpha: float = 1.0,
        device: str = "cuda",
        soft_label: float | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.confidence_alpha = confidence_alpha
        self.device = device
        self.soft_label = soft_label

        # Determine if model outputs confidence values based on output channels
        self.num_layers = self.model.num_layers
        self.model_confidence = (
            True
            if self.num_layers == 1
            and self.model.channels_per_layer == 7
            or self.model.channels_per_layer == 8
            else False
        )

        if render_on_module:
            res = self.model.resolution
            self.renderer = MeshToDualPointmap(
                im_size=(res, res), num_layers=self.num_layers, device=self.device, return_on_cpu=False
            )
            # except Exception as e:
            #     logger.warning(f"Failed to initialize renderer: {e}")
            #     self.renderer = None
        else:
            self.renderer = None

    def render_batch(self, batch: PointmapBatch) -> PointmapBatch:
        """returns a new pointmap batch including the model targets if not already present"""
        if isinstance(batch, tuple) or isinstance(batch, list):
            batch = PointmapBatch(*batch)

        if self.renderer is None:
            if batch.model_targets is None or not batch.model_targets.any():
                raise ValueError(
                    "model targets are not provided and renderer is not enabled"
                )
            return batch

        # no way to render
        if batch.render_args is None:
            return batch

        # already there
        if batch.model_targets is not None and batch.model_targets.any():
            return batch

        render_args = {
            k: v.to(self.device)
            if isinstance(v, Tensor)
            else [i.to(self.device) for i in v]
            for k, v in batch.render_args.items()
            if k in ["pose_verts", "canonical_verts", "faces", "model_view", "focal_length"]
        }
        model_targets = einops.rearrange(
            self.renderer(**render_args), "b h w n c-> b (n c) h w"
        )

        # generate a new pointmap batch with the new model targets
        batch = PointmapBatch(*batch)
        batch.model_targets = model_targets
        return batch

    def process_batch(self, batch: tuple | PointmapBatch) -> PointmapBatch:
        """Prepare a batch for model forward pass"""
        batch = PointmapBatch(*batch)
        batch.to(self.device)
        batch = self.render_batch(batch)
        return batch

    def step(self, batch: tuple | PointmapBatch) -> tuple[Tensor, PointmapBatch]:
        """
        Render model targets if needed, and forward pass.

        Args:
            batch: tuple or PointmapBatch

        Returns:
            tuple[Tensor, PointmapBatch]: network output and pointmap batch
        """
        batch = self.process_batch(batch)
        network_output = self.model(batch.input_image)
        return network_output, batch

    def predict(
        self,
        input_image: Tensor,
        mask: Tensor | None = None,
        confidence_threshold: float | None = 2.0,
        output: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        predict a pointcloud from batched input image and mask

        Args:
            input_image: (B, H, W, Cin)
            mask: (B, H, W)
            return_type: "pointmap", "layered_pointcloud", "pointcloud"

        Returns:
            canon: (B, N, 3)
            posed: (B, N, 3)
            occupancy: (B, H, W, N, 1)
            confidence: (B, N, 1)
        """
        if output is None:
            with torch.inference_mode():
                output = self.model(input_image.to(self.device))

        canon, posed, occupancy, confidence = extract_predictions(
            output,
            self.num_layers,
            activate_occupancy=True,
            mask=mask.to(self.device) if mask is not None else None,
            return_type="pointcloud",
            exp_confidence=getattr(self, 'exp_confidence', False)
        )

        # combine the sequences with padding

        lens_ = [c.shape[0] for c in canon]
        max_len = max(lens_)
        canon, posed, confidence = (
            torch.stack([F.pad(c, (0, 0, 0, max_len - len(c))) for c in tensor])
            for tensor in (canon, posed, confidence)
        )

        # make sequence mask
        seq_mask = torch.stack(
            [
                F.pad(
                    torch.ones(L, dtype=torch.bool, device=canon.device),
                    (0, max_len - L),
                )
                for L in lens_
            ]
        )

        if confidence_threshold is not None:
            seq_mask &= confidence[..., 0] >= confidence_threshold

        return canon, posed, seq_mask

    def training_step(
        self, batch: tuple | PointmapBatch,
        return_preds: bool = False,
        use_confidence: bool = True,
        reduction: str = 'full',
        delta: float | None = None,
    ) -> Tensor | tuple[Tensor, Tensor] | tuple[Tensor | tuple[Tensor, Tensor], Tensor, PointmapBatch]:
        """inference step and compute loss"""
        output, batch = self.step(batch)
        loss_ret = self.compute_loss(
            output,
            batch.model_targets,
            mask=batch.mask,
            use_confidence=use_confidence,
            reduction=reduction,
            delta=delta,
        )

        if return_preds:
            return loss_ret, output, batch
        return loss_ret

    def validation_step(
        self, batch: tuple, reduction: str = 'full'
    ) -> tuple[Tensor | tuple[Tensor, Tensor], Tensor, PointmapBatch]:
        with torch.no_grad():
            return self.training_step(batch, return_preds=True, use_confidence=False, reduction=reduction)

    def compute_loss(
        self,
        output: Tensor,
        target: Tensor,
        mask: Tensor | None = None,
        use_confidence: bool = True,
        reduction: str = 'full',
        delta: float | None = None,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """
        Compute the DUSt3r loss confidence weighted loss and binary cross entropy occupancy loss
        """

        canon, posed, occupancy, confidence = extract_from_amodal_pointmap(
            output, self.num_layers, activate_occupancy=False,
            exp_confidence=getattr(self, 'exp_confidence', False)
        )
        canon_target, posed_target, occupancy_target, _ = extract_from_amodal_pointmap(
            target, self.num_layers, activate_occupancy=False, is_targets=True,
            exp_confidence=getattr(self, 'exp_confidence', False)
        )

        if canon.requires_grad:
            canon.register_hook(make_dual_clip_hook("canon"))
        if posed.requires_grad:
            posed.register_hook(make_dual_clip_hook("posed"))

        # Compute pixelwise l2 loss

        # B H W N C -> B H W N
        if delta is None:
            canon_loss = F.mse_loss(canon, canon_target, reduction="none").sum(-1)
            posed_loss = F.mse_loss(posed, posed_target, reduction="none").sum(-1)
        else:
            canon_loss = 2*F.huber_loss(canon, canon_target, reduction="none", delta=delta).sum(-1)
            posed_loss = 2*F.huber_loss(posed, posed_target, reduction="none", delta=delta).sum(-1)

        # weight each layer by the confidence
        if use_confidence and (confidence is not None):
            confidence = confidence.squeeze(4)
            log_confidence = torch.log(confidence)
            canon_loss = (
                confidence * canon_loss - self.confidence_alpha * log_confidence
            )
            posed_loss = (
                confidence * posed_loss - self.confidence_alpha * log_confidence
            )
            # c_for_geom = confidence.detach()
            # log_confidence = torch.log(confidence)
            # canon_loss_clipped = torch.clamp(canon_loss.detach(), max=10.0)
            # canon_loss = (
            #     c_for_geom * canon_loss + confidence * canon_loss_clipped - self.confidence_alpha * log_confidence
            # )
            # posed_loss_clipped = torch.clamp(posed_loss.detach(), max=10.0)
            # posed_loss = (
            #     c_for_geom * posed_loss + confidence * posed_loss_clipped - self.confidence_alpha * log_confidence
            # )

        if occupancy_target is None:
            # OCC Target: (B, H, W, N)
            occupancy_target = (
                mask[:, None].float().expand(-1, -1, -1, -1, self.num_layers)
            )

        # expectation over occupancy Ground truth of dust3r loss
        occupancy_target = occupancy_target.squeeze(4)
        total_valid_pixels = occupancy_target.sum()

        if reduction == 'per-sample':
            pixelwise_loss = (
                0.5 * (canon_loss + posed_loss) * occupancy_target
            ).sum((-1, -2, -3)) / occupancy_target.sum((-1, -2, -3))
        elif reduction == 'full':
            pixelwise_loss = (
                0.5 * (canon_loss + posed_loss) * occupancy_target
            ).sum() / total_valid_pixels
        elif reduction == 'none':
            pixelwise_loss = (
                0.5 * (canon_loss + posed_loss) * occupancy_target
            ).sum()
        else:
            raise ValueError(f"reduction {reduction} unknown")

        # Occupancy loss. expectation over segemnation mask of standard BCE loss
        if occupancy is not None:
            occ_ret = layer_occupancy_loss(
                occupancy.squeeze(4), occupancy_target, soft_label=self.soft_label, reduction=reduction
            )

        if reduction == 'none':
            occupancy_loss_sum, occupancy_pixels = occ_ret
            return pixelwise_loss, total_valid_pixels, occupancy_loss_sum, occupancy_pixels
        else:
            return pixelwise_loss + occ_ret
    

    def compute_eval_losses(
        self, output: Tensor, target: Tensor, mask: Tensor | None = None
    ) -> Tensor:
        canon, posed, occupancy, confidence = extract_from_amodal_pointmap(
            output, self.num_layers, activate_occupancy=False,
            exp_confidence=getattr(self, 'exp_confidence', False)
        )
        canon_target, posed_target, occupancy_target, _ = extract_from_amodal_pointmap(
            target, self.num_layers, activate_occupancy=False, is_targets=True,
            exp_confidence=getattr(self, 'exp_confidence', False)
        )

        # Compute pixelwise l2 loss

        # B H W N C -> B H W N
        canon_loss = F.mse_loss(canon, canon_target, reduction="none").sum(-1)
        posed_loss = F.mse_loss(posed, posed_target, reduction="none").sum(-1)

        if occupancy_target is None:
            # OCC Target: (B, H, W, N)
            occupancy_target = (
                mask[:, None].float().expand(-1, -1, -1, -1, self.num_layers)
            )

        # expectation over occupancy Ground truth of dust3r loss
        occupancy_target = occupancy_target.squeeze(4)
        confidence = confidence.squeeze(4)
        per_layer_canon_loss = (canon_loss * occupancy_target).sum((1, 2)) / occupancy_target.sum((1, 2))
        per_sample_canon_loss = (canon_loss * occupancy_target).sum((1, 2, 3)) / occupancy_target.sum((1, 2, 3))
        per_layer_posed_loss = (posed_loss * occupancy_target).sum((1, 2)) / occupancy_target.sum((1, 2))
        per_sample_posed_loss = (posed_loss * occupancy_target).sum((1, 2, 3)) / occupancy_target.sum((1, 2, 3))
        per_layer_confidence = (confidence * occupancy_target).sum((1, 2)) / occupancy_target.sum((1, 2))
        per_sample_confidence = (confidence * occupancy_target).sum((1, 2, 3)) / occupancy_target.sum((1, 2, 3))

        per_layer_occupancy_loss = layer_occupancy_loss(
            occupancy.squeeze(4), occupancy_target, soft_label=self.soft_label, reduction='per-layer'
        )
        per_sample_occupancy_loss = torch.mean(per_layer_occupancy_loss, dim=-1)

        per_sample_total_loss = 0.5 * (per_sample_canon_loss + per_sample_posed_loss)

        eval_losses = {
            "total_loss": per_sample_total_loss,
            "canon_loss": per_sample_canon_loss,
            "posed_loss": per_sample_posed_loss,
            "occupancy_loss": per_sample_occupancy_loss,
            "confidence": per_sample_confidence,
            "per_layer_canon_loss": per_layer_canon_loss,
            "per_layer_posed_loss": per_layer_posed_loss,
            "per_layer_occupancy_loss": per_layer_occupancy_loss,
            "per_layer_confidence": per_layer_confidence,
        }

        occupancy = F.sigmoid(occupancy)
        return eval_losses, canon, posed, occupancy
