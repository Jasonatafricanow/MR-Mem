from mr_mem.memory.retrieval import RetrievalProviderUnavailable


class ProviderPackageMissing(RetrievalProviderUnavailable):
    """An explicitly requested optional package is absent."""


class ProviderStorageUnavailable(RetrievalProviderUnavailable):
    """Configured index/model storage is missing or incompatible."""
