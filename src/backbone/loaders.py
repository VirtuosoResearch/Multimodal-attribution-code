from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer


_TEXT_BACKBONES = {
    "openllama_3b": "openlm-research/open_llama_3b",
    "openllama-3b": "openlm-research/open_llama_3b",
    "open_llama_3b": "openlm-research/open_llama_3b",
}

_VISION_BACKBONES = {
    "dinov2_vits14": "facebook/dinov2-small",
    "dinov2-vit-s-14": "facebook/dinov2-small",
    "dinov2-small": "facebook/dinov2-small",
}


def _canonicalize(name: str, table: dict[str, str], kind: str) -> tuple[str, str]:
    key = name.strip().lower()
    if key not in table:
        raise ValueError(f"Unsupported {kind} backbone: {name}")
    return key, table[key]


def _default_dtype(device: str) -> torch.dtype:
    return torch.float16 if device.startswith("cuda") else torch.float32


@dataclass
class LoadedTextBackbone:
    name: str
    model_id: str
    tokenizer: AutoTokenizer
    model: AutoModel
    device: str

    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 8,
        max_length: int = 128,
        normalize: bool = False,
    ) -> torch.Tensor:
        outputs: list[torch.Tensor] = []
        self.model.eval()
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                batch = list(texts[start:start + batch_size])
                encoded = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                )
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                result = self.model(**encoded)
                hidden = result.last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                pooled = pooled.float()
                if normalize:
                    pooled = F.normalize(pooled, dim=1)
                outputs.append(pooled.cpu())
        return torch.cat(outputs, dim=0)


@dataclass
class LoadedVisionBackbone:
    name: str
    model_id: str
    processor: AutoImageProcessor
    model: AutoModel
    device: str

    def encode(
        self,
        images: Sequence[Image.Image],
        *,
        batch_size: int = 16,
        normalize: bool = False,
    ) -> torch.Tensor:
        outputs: list[torch.Tensor] = []
        self.model.eval()
        with torch.inference_mode():
            for start in range(0, len(images), batch_size):
                batch = [image.convert("RGB") for image in images[start:start + batch_size]]
                encoded = self.processor(images=batch, return_tensors="pt")
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                result = self.model(**encoded)
                pooled = getattr(result, "pooler_output", None)
                if pooled is None:
                    pooled = result.last_hidden_state[:, 0]
                pooled = pooled.float()
                if normalize:
                    pooled = F.normalize(pooled, dim=1)
                outputs.append(pooled.cpu())
        return torch.cat(outputs, dim=0)


def load_text_backbone(
    name: str,
    *,
    device: str = "cuda:0",
    torch_dtype: torch.dtype | None = None,
) -> LoadedTextBackbone:
    canonical_name, model_id = _canonicalize(name, _TEXT_BACKBONES, "text")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    try:
        model = AutoModel.from_pretrained(
            model_id,
            dtype=torch_dtype or _default_dtype(device),
            low_cpu_mem_usage=True,
        )
    except ValueError as exc:
        if "torch to at least v2.6" in str(exc):
            raise RuntimeError(
                f"{model_id} only exposes PyTorch .bin weights here. "
                "Current transformers requires torch>=2.6 to load them safely; "
                "upgrade torch in the target env or switch to a safetensors variant."
            ) from exc
        raise
    model.to(device)
    return LoadedTextBackbone(
        name=canonical_name,
        model_id=model_id,
        tokenizer=tokenizer,
        model=model,
        device=device,
    )


def load_vision_backbone(
    name: str,
    *,
    device: str = "cuda:0",
    torch_dtype: torch.dtype | None = None,
) -> LoadedVisionBackbone:
    canonical_name, model_id = _canonicalize(name, _VISION_BACKBONES, "vision")
    processor = AutoImageProcessor.from_pretrained(model_id, use_fast=False)
    model = AutoModel.from_pretrained(
        model_id,
        dtype=torch_dtype or _default_dtype(device),
        low_cpu_mem_usage=True,
    )
    model.to(device)
    return LoadedVisionBackbone(
        name=canonical_name,
        model_id=model_id,
        processor=processor,
        model=model,
        device=device,
    )
