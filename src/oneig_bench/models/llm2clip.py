"""LLM2CLIP text/image similarity wrapper."""
from __future__ import annotations

import torch
from PIL import Image
from transformers import AutoConfig, AutoModel, AutoTokenizer, CLIPImageProcessor


class LLM2CLIP:
    def __init__(
        self,
        processor_model: str = "openai/clip-vit-large-patch14-336",
        model_name: str = "microsoft/LLM2CLIP-Openai-L-14-336",
        llm_model_name: str = "microsoft/LLM2CLIP-Llama-3-8B-Instruct-CC-Finetuned",
        device: str = "cuda",
    ):
        self.processor = CLIPImageProcessor.from_pretrained(processor_model)
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(device).eval()

        self.llm_model_name = llm_model_name
        self.config = AutoConfig.from_pretrained(self.llm_model_name, trust_remote_code=True)
        self.llm_model = AutoModel.from_pretrained(
            self.llm_model_name,
            torch_dtype=torch.bfloat16,
            config=self.config,
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name)

        # Workaround required by the upstream LLM2VEC implementation.
        self.llm_model.config._name_or_path = "meta-llama/Meta-Llama-3-8B-Instruct"

        from scripts.utils.llm2clip.llm2vec import LLM2Vec

        self.l2v = LLM2Vec(
            self.llm_model, self.tokenizer, pooling_mode="mean", max_length=512, doc_max_length=512
        )
        self.device = device

    def text_img_similarity_score(self, image_path_list, text_prompt):
        try:
            captions = [text_prompt]
            images = [Image.open(p) for p in image_path_list]
            input_pixels = self.processor(images=images, return_tensors="pt").pixel_values.to(
                self.device
            )
            text_features = self.l2v.encode(captions, convert_to_tensor=True).to(self.device)

            with torch.no_grad(), torch.amp.autocast(self.device):
                image_features = self.model.get_image_features(input_pixels)
                text_features = self.model.get_text_features(text_features)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                text_probs = (image_features @ text_features.T).cpu().tolist()
            return [tp[0] for tp in text_probs]
        except Exception as e:  # noqa: BLE001  (mirrors original behavior)
            print(f"Error: {e}")
            return None
