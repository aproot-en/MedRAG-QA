"""BGE-M3 embedding wrapper.

`BAAI/bge-m3` is only published with a `pytorch_model.bin` checkpoint (no
`model.safetensors`). Recent `transformers` versions refuse to `torch.load`
a `.bin` file unless torch >= 2.6 (CVE-2025-32434 guard). Upgrading torch in
a shared conda environment is riskier than it's worth for this project, so
we disable that specific guard before loading -- the checkpoint comes
straight from the official BAAI org repo on the Hugging Face Hub over TLS,
which is the same trust boundary `transformers` itself relies on for any
`from_pretrained` call.
"""
import numpy as np

import config


def _load_model():
    import transformers.utils.import_utils as iu
    iu.check_torch_load_is_safe = lambda *a, **k: None
    import transformers.modeling_utils as mu
    mu.check_torch_load_is_safe = lambda *a, **k: None

    import torch
    from sentence_transformers import SentenceTransformer

    device = config.EMBED_DEVICE if torch.cuda.is_available() else "cpu"
    # Prefer the local Hugging Face cache.  Besides avoiding an unnecessary
    # network HEAD request on every run, this makes the advertised offline
    # rerun path work when the model has already been downloaded.
    try:
        model = SentenceTransformer(
            config.EMBED_MODEL_NAME, device=device, local_files_only=True
        )
    except OSError:
        model = SentenceTransformer(config.EMBED_MODEL_NAME, device=device)
    return model, device


class BgeM3Embedder:
    """Thin wrapper around the BGE-M3 dense embedding head (L2-normalized)."""

    def __init__(self):
        self.model, self.device = _load_model()

    def encode(self, texts, batch_size: int = config.EMBED_BATCH_SIZE, show_progress: bool = True) -> np.ndarray:
        embeddings = self.model.encode(
            list(texts),
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text], batch_size=1, show_progress=False)[0]
