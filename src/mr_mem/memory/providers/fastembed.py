"""Local pretrained ONNX embedding, with explicitly verified model artifacts."""

import hashlib
from importlib.metadata import version
from pathlib import Path

from mr_mem.memory.embedding import EmbeddingIdentity, validate_vector
from mr_mem.memory.providers.errors import ProviderPackageMissing, ProviderStorageUnavailable
from mr_mem.memory.retrieval import RetrievalProviderUnavailable

FASTEMBED_VERSION = "0.8.0"


def model_revision(model_dir: Path) -> str:
    """Fingerprint model/tokenizer artifacts, excluding downloader bookkeeping."""
    files = sorted(
        p
        for p in model_dir.rglob("*")
        if p.is_file() and not any(part.startswith(".") for part in p.relative_to(model_dir).parts)
    )
    if not any(p.suffix == ".onnx" for p in files):
        raise ProviderStorageUnavailable("local ONNX model artifacts are absent")
    digest = hashlib.sha256()
    for path in files:
        name = path.relative_to(model_dir).as_posix().encode()
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        with path.open("rb") as stream:
            file_digest = hashlib.file_digest(stream, "sha256").digest()
        digest.update(file_digest)
    return f"fastembed-{FASTEMBED_VERSION}:sha256:{digest.hexdigest()}"


class FastEmbedEmbedding:
    def __init__(self, identity: EmbeddingIdentity, *, model_dir: Path):
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            raise ProviderPackageMissing(
                "install mr-mem[vector] for FastEmbed"
            ) from exc
        path = Path(model_dir)
        if (
            version("fastembed") != FASTEMBED_VERSION
            or identity.provider != "fastembed"
            or identity.revision != model_revision(path)
        ):
            raise ProviderStorageUnavailable("embedding implementation/model revision mismatch")
        supported = {m["model"]: m for m in TextEmbedding.list_supported_models()}
        if (
            identity.model_id not in supported
            or supported[identity.model_id]["dim"] != identity.dimension
        ):
            raise ProviderStorageUnavailable("embedding model/dimension configuration mismatch")
        try:
            self._model = TextEmbedding(
                model_name=identity.model_id,
                specific_model_path=str(path),
                cache_dir=str(path),
                local_files_only=True,
                threads=1,
                providers=["CPUExecutionProvider"],
                cuda=False,
            )
        except Exception as exc:
            raise ProviderStorageUnavailable(
                "cannot load configured local embedding model"
            ) from exc
        self._identity = identity

    @property
    def identity(self) -> EmbeddingIdentity:
        return self._identity

    def embed(self, text: str) -> tuple[float, ...]:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("embedding requires nonempty canonical/query text")
        try:
            rows = iter(self._model.embed([text]))
            vector = tuple(float(v) for v in next(rows))
            if next(rows, None) is not None:
                raise ValueError("embedding returned multiple vectors for one input")
            return validate_vector(vector, self.identity)
        except Exception as exc:
            raise RetrievalProviderUnavailable("local embedding execution failed") from exc
