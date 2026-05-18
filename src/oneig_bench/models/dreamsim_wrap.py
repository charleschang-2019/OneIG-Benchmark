"""DreamSim loader (lazy)."""
from __future__ import annotations


def load_dreamsim(device: str = "cuda"):
    import torchvision

    torchvision.disable_beta_transforms_warning()
    from dreamsim import dreamsim

    return dreamsim(pretrained=True, device=device)
