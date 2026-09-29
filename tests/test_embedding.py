import pytest

from mr_mem.memory.embedding import EmbeddingIdentity, projection_target, validate_vector


def test_embedding_identity_and_projection_target_are_derived_state():
    identity = EmbeddingIdentity("fastembed", "model", 3, "revision")
    assert projection_target(identity) == projection_target(identity)
    assert validate_vector((1, 2.0, 3), identity) == (1.0, 2.0, 3.0)


def test_invalid_embedding_vectors_fail_closed():
    identity = EmbeddingIdentity("fastembed", "model", 3, "revision")
    with pytest.raises(ValueError):
        validate_vector((1.0, 2.0), identity)
    with pytest.raises(ValueError):
        validate_vector((0.0, 0.0, 0.0), identity)
