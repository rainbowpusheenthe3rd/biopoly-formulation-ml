"""L4 — learned polymer representation: embeddings and their cosine geometry."""

from __future__ import annotations

import numpy as np
import pytest

from biopoly.representation import (
    cosine_similarity_matrix,
    formulation_embeddings,
    learn_embeddings,
)

pytestmark = pytest.mark.layer(4)  # features, splitting & representation


def test_embeddings_cover_polymers(small_df):
    emb = learn_embeddings(small_df)
    assert emb.shape[0] >= 5  # most/all polymers appear as a dominant polymer
    assert np.isfinite(emb.to_numpy()).all()


def test_cosine_matrix_is_valid(small_df):
    cos = cosine_similarity_matrix(learn_embeddings(small_df))
    x = cos.to_numpy()
    assert np.allclose(np.diag(x), 1.0, atol=1e-6)  # self-similarity is 1
    assert np.allclose(x, x.T, atol=1e-6)  # symmetric
    assert x.min() >= -1.0001 and x.max() <= 1.0001  # bounded to [-1, 1]


def test_formulation_embedding_shape(small_df):
    emb = learn_embeddings(small_df)
    fe = formulation_embeddings(small_df, emb)
    assert len(fe) == len(small_df)
    assert fe.shape[1] == emb.shape[1]
    assert np.isfinite(fe.to_numpy()).all()
