from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

def build_warmup_cosine_scheduler(
    optimizer,
    warmup_steps: int,
    total_steps: int,
    warmup_start_factor: float = 0.1,
    eta_min: float = 1e-5,
):
    """Factory to safely distribute the optimizer to all nested schedulers."""
    warmup = LinearLR(
        optimizer, 
        start_factor=warmup_start_factor, 
        total_iters=warmup_steps
    )
    cosine = CosineAnnealingLR(
        optimizer, 
        T_max=(total_steps - warmup_steps), 
        eta_min=eta_min
    )
    return SequentialLR(
        optimizer, 
        schedulers=[warmup, cosine], 
        milestones=[warmup_steps]
    )
