"""Learned polymer embeddings vs the descriptor baseline.

The forward model already sees each polymer as a fraction column plus a categorical
``primary_polymer``. This module asks whether a *learned* dense representation of each
polymer — one that captures how the polymer actually behaves — carries structure the
raw descriptors miss.

The embedding is supervised and simple: each polymer's standardised **mean property
signature** across the dataset (a target-encoding), which is a genuinely learned-from-
data representation. We then (1) inspect its cosine-similarity geometry — do chemically
similar polymers sit close? — and (2) honestly test whether feeding a formulation's
blended embedding to the forward model beats descriptors alone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from biopoly import TARGETS
from biopoly.data.chemistry import POLYMERS

_SIGNATURE_COLS = [*TARGETS, "process_temp_c"]


def polymer_signatures(df: pd.DataFrame) -> pd.DataFrame:
    """Per-polymer standardised mean signature over the targets + processing temp.

    For each polymer, average the measured properties over rows where it is the
    dominant polymer, then standardise each column across polymers. Rows: polymers;
    columns: the signature features.
    """
    rows: dict[str, list[float]] = {}
    for p in POLYMERS:
        mask = df["primary_polymer"] == p
        if not mask.any():
            continue
        rows[p] = [float(df.loc[mask, c].mean()) for c in _SIGNATURE_COLS]
    sig = pd.DataFrame.from_dict(rows, orient="index", columns=_SIGNATURE_COLS)
    sig = sig.fillna(sig.mean())
    z = StandardScaler().fit_transform(sig)
    return pd.DataFrame(z, index=sig.index, columns=_SIGNATURE_COLS)


def learn_embeddings(df: pd.DataFrame) -> pd.DataFrame:
    """The learned polymer representation: the standardised property signature."""
    return polymer_signatures(df)


def project_2d(embeddings: pd.DataFrame, *, seed: int = 0) -> pd.DataFrame:
    """PCA-project embeddings to 2-D for visualisation."""
    k = min(2, embeddings.shape[1], max(1, embeddings.shape[0] - 1))
    xy = PCA(n_components=k, random_state=seed).fit_transform(embeddings)
    return pd.DataFrame(xy, index=embeddings.index, columns=[f"pc{i + 1}" for i in range(k)])


def cosine_similarity_matrix(embeddings: pd.DataFrame) -> pd.DataFrame:
    """Pairwise cosine similarity between polymer embeddings (rows = polymers)."""
    x = embeddings.to_numpy(dtype=float)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    xn = x / np.clip(norm, 1e-9, None)
    sim = xn @ xn.T
    return pd.DataFrame(sim, index=embeddings.index, columns=embeddings.index)


def formulation_embeddings(df: pd.DataFrame, embeddings: pd.DataFrame) -> pd.DataFrame:
    """Blend each row's polymers into a single fraction-weighted embedding vector."""
    frac = np.vstack([df[f"frac_{p}"].to_numpy() for p in POLYMERS]).T  # (n, n_polymers)
    emb = np.vstack([embeddings.loc[p].to_numpy() for p in POLYMERS])  # (n_polymers, dim)
    blended = frac @ emb  # (n, dim)
    cols = [f"pemb_{c}" for c in embeddings.columns]
    return pd.DataFrame(blended, index=df.index, columns=cols)
