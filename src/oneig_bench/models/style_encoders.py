"""CSD + Style Encoder embedding wrappers (no module-level model loads)."""
from __future__ import annotations

import torch
import torchvision.transforms.functional as F
from PIL import Image
from torchvision import transforms
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection


class CSDStyleEmbedding:
    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = torch.device(device)
        self.model = self._load_model(model_path).to(self.device)
        self.preprocess = transforms.Compose(
            [
                transforms.Resize(size=224, interpolation=F.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

    def _load_model(self, model_path: str):
        # Imported lazily — CSD_config pulls in `clip` which is a heavy dep.
        from scripts.utils.CSD_config import CSD_CLIP, convert_state_dict

        model = CSD_CLIP("vit_large", "default")
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        state_dict = convert_state_dict(checkpoint["model_state_dict"])
        model.load_state_dict(state_dict, strict=False)
        return model

    def get_style_embedding(self, image_path: str):
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, _, style_output = self.model(image_tensor)
        return style_output


class SEStyleEmbedding:
    def __init__(
        self,
        pretrained_path: str = "xingpng/OneIG-StyleEncoder",
        device: str = "cuda",
        dtype=torch.bfloat16,
    ):
        self.device = torch.device(device)
        self.dtype = dtype
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(pretrained_path)
        self.image_encoder.to(self.device, dtype=self.dtype)
        self.image_encoder.eval()
        self.processor = CLIPImageProcessor()

    @staticmethod
    def _l2_normalize(x):
        return torch.nn.functional.normalize(x, p=2, dim=-1)

    def get_style_embedding(self, image_path: str):
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").pixel_values.to(
            self.device, dtype=self.dtype
        )
        with torch.no_grad():
            outputs = self.image_encoder(inputs)
            image_embeds = outputs.image_embeds
            return self._l2_normalize(image_embeds)
