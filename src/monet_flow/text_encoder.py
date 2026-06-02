from __future__ import annotations

from collections.abc import Sequence

import torch


class FrozenCLIPTextEncoder:
    """Small frozen text encoder used to cache prompt embeddings for training."""

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        try:
            from transformers import AutoTokenizer, CLIPTextModelWithProjection
        except ImportError as exc:
            raise ImportError("Install transformers to encode text prompts.") from exc

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = CLIPTextModelWithProjection.from_pretrained(model_name, torch_dtype=dtype)
        self.model.to(self.device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.inference_mode()
    def encode(self, texts: Sequence[str], batch_size: int = 64) -> torch.Tensor:
        outputs = []
        for start in range(0, len(texts), batch_size):
            chunk = list(texts[start : start + batch_size])
            tokens = self.tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).to(self.device)
            encoded = self.model(**tokens).text_embeds
            encoded = torch.nn.functional.normalize(encoded.float(), dim=-1)
            outputs.append(encoded.cpu())
        return torch.cat(outputs, dim=0)
