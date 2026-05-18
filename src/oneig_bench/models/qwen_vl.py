"""Qwen2.5-VL batch inferencer (semantic Q/A + OCR)."""
from __future__ import annotations

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor


class Qwen2_5VLBatchInferencer:
    def __init__(
        self,
        model_path: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: str = "cuda",
        dtype=torch.bfloat16,
        use_flash_attention: bool = True,
    ):
        attn_impl = "flash_attention_2" if use_flash_attention else "eager"
        from transformers import Qwen2_5_VLForConditionalGeneration

        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=dtype,
            attn_implementation=attn_impl,
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.device = torch.device(device)
        self.TEXT_PROMPT = (
            "Recognize the text in the image, only reply with the text content, "
            "but avoid repeating previously mentioned content. "
            "If no text is recognized, please reply with 'No text recognized'."
        )

    def batch_inference(self, messages, max_new_tokens: int = 128):
        texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages
        ]
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            return self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

    def infer_semantic(self, images_path: list[str], question: str) -> list[str]:
        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": f"{question}. Please answer 'Yes' or 'No' only."},
                    ],
                }
            ]
            for image_path in images_path
        ]
        return self.batch_inference(messages)

    def infer_ocr(self, images_path: list[str], max_new_tokens: int = 128) -> list[str]:
        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": self.TEXT_PROMPT},
                    ],
                }
            ]
            for image_path in images_path
        ]
        return self.batch_inference(messages, max_new_tokens=max_new_tokens)
