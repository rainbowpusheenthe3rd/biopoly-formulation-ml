"""Active learning: choose the next experiment by expected information gain.

Data is the binding constraint in a scarce domain like bioplastics, so each real
measurement should be the most *informative* one available. This module frames that
decision information-theoretically and — the elegant part — reuses the inverse-design
search machinery: inverse design searches formulation space to *minimise*
distance-to-target; active learning searches the *same* space to *maximise*
information gain. Same loop, different objective.

Epistemic vs aleatoric — the honest bit. The forward model's quantile band is
dominated by *aleatoric* noise (irreducible measurement scatter it has learned);
sampling where that is large just re-measures noise. Active learning must instead
target *epistemic* uncertainty — the model's ignorance, which data can reduce. We
estimate it by **query-by-committee**: a bootstrap ensemble whose *disagreement* on a
candidate is a Monte-Carlo estimate of the BALD mutual information between the
would-be measurement and the model. Higher disagreement -> more to learn.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from biopoly import TARGETS
from biopoly.data.generate import _sample_formulation
from biopoly.data.schema import FEATURE_COLS
from biopoly.features import make_x, split
from biopoly.inverse.common import describe
from biopoly.models.forward import ForwardModel
from biopoly.models.metrics import TOLERANCE

_COMMITTEE_PARAMS = {"n_estimators": 120, "learning_rate": 0.08}


@dataclass
class Committee:
    """A bootstrap ensemble of forward models; disagreement ~ epistemic uncertainty."""

    members: list[ForwardModel]

    @staticmethod
    def fit(
        pool: pd.DataFrame, *, k: int = 5, params: dict | None = None, seed: int = 0
    ) -> Committee:
        """Fit ``k`` forward models, each on a bootstrap resample of ``pool``."""
        rng = np.random.default_rng(seed)
        n = len(pool)
        members = [
            ForwardModel(params or _COMMITTEE_PARAMS).fit(pool.iloc[rng.integers(0, n, size=n)])
            for _ in range(k)
        ]
        return Committee(members)

    def _member_values(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        """Point predictions per target, stacked as ``(k_members, n_rows)``."""
        stacks: dict[str, list[np.ndarray]] = {t: [] for t in TARGETS}
        for m in self.members:
            pred = m.predict(df)
            for t in TARGETS:
                stacks[t].append(pred[t]["value"])
        return {t: np.vstack(v) for t, v in stacks.items()}

    def disagreement(self, df: pd.DataFrame) -> np.ndarray:
        """Per-row acquisition: committee std summed over targets in tolerance units.

        This is the epistemic-uncertainty (BALD) proxy; its argmax is the most
        informative next experiment. Tolerance-normalising makes a 3 MPa tensile
        spread and a 2% water spread comparable before they are combined.
        """
        vals = self._member_values(df)
        score = np.zeros(len(df))
        for t in TARGETS:
            score += vals[t].std(axis=0) / TOLERANCE[t]
        return score / len(TARGETS)

    def predict_mean(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        """Ensemble point prediction per target (the committee mean)."""
        vals = self._member_values(df)
        return {t: vals[t].mean(axis=0) for t in TARGETS}


def _candidate_frame(forms) -> pd.DataFrame:
    """Feature frame for a list of Formulations, at nominal feedstock quality."""
    rows = []
    for f in forms:
        row = f.as_row()
        row["primary_polymer"] = max(f.polymer_frac, key=f.polymer_frac.get)
        row["tensile_protocol"] = "ISO527"
        row["feedstock_quality"] = 1.0
        rows.append(row)
    return make_x(pd.DataFrame(rows)[FEATURE_COLS])


def propose_experiment(
    committee: Committee, *, n_candidates: int = 2000, top_k: int = 1, seed: int = 0
) -> list[dict]:
    """Generate the most-informative new formulation(s) to measure next.

    The inverse-design pattern with the objective flipped: sample candidate
    formulations, score each by committee disagreement (information gain), and keep
    the highest — "what should we measure next?" rather than "what hits the target?".
    """
    rng = np.random.default_rng(seed)
    forms = [_sample_formulation(rng) for _ in range(n_candidates)]
    scores = committee.disagreement(_candidate_frame(forms))
    order = np.argsort(scores)[::-1][:top_k]
    return [
        {"information_gain": round(float(scores[i]), 4), "formulation": describe(forms[i])}
        for i in order
    ]


def _mean_r2(df: pd.DataFrame, pred: dict[str, np.ndarray]) -> float:
    """Mean R^2 across targets (scale-free), on the non-missing rows of each target."""
    scores = []
    for t in TARGETS:
        y = df[t].to_numpy(dtype=float)
        mask = ~np.isnan(y)
        if mask.sum() < 2:
            continue
        yt, yp = y[mask], pred[t][mask]
        ss_tot = float(np.sum((yt - yt.mean()) ** 2))
        if ss_tot <= 0:
            continue
        scores.append(1.0 - float(np.sum((yt - yp) ** 2)) / ss_tot)
    return float(np.mean(scores)) if scores else 0.0


def active_learning_curve(
    df: pd.DataFrame,
    *,
    seed_size: int = 60,
    batch: int = 30,
    rounds: int = 6,
    k: int = 4,
    test_size: float = 0.2,
    seed: int = 0,
) -> dict[str, list[float]]:
    """Pool-based active learning: committee-disagreement selection vs random.

    Both strategies start from the same random seed set, then repeatedly reveal a
    batch of labels — the active one by max committee disagreement, the control at
    random — refit, and score mean R^2 on a fixed held-out test set. Returns
    ``{"labels": [...], "active": [r2...], "random": [r2...]}``; the active curve
    reaching higher R^2 per labelled example is the whole point.
    """
    train_pool, test = split(df, test_size=test_size, mode="random", seed=seed)
    train_pool = train_pool.reset_index(drop=True)
    test = test.reset_index(drop=True)
    curves: dict[str, list[float]] = {}

    for strategy in ("active", "random"):
        rng = np.random.default_rng(seed)
        labeled = set(rng.choice(len(train_pool), size=seed_size, replace=False).tolist())
        labels_x: list[float] = []
        r2s: list[float] = []
        for r in range(rounds + 1):
            committee = Committee.fit(train_pool.iloc[sorted(labeled)], k=k, seed=seed + r)
            r2s.append(_mean_r2(test, committee.predict_mean(test)))
            labels_x.append(float(len(labeled)))
            if r == rounds:
                break
            unlabeled = [i for i in range(len(train_pool)) if i not in labeled]
            if strategy == "active":
                acq = committee.disagreement(train_pool.iloc[unlabeled])
                chosen = [unlabeled[j] for j in np.argsort(acq)[::-1][:batch]]
            else:
                take = min(batch, len(unlabeled))
                chosen = rng.choice(unlabeled, size=take, replace=False).tolist()
            labeled.update(chosen)
        curves[strategy] = r2s
        curves["labels"] = labels_x
    return curves


def active_learning_shift_curve(
    df: pd.DataFrame,
    *,
    seed_size: int = 40,
    batch: int = 25,
    rounds: int = 5,
    k: int = 4,
    test_size: int = 200,
    params: dict | None = None,
    seed: int = 0,
) -> dict[str, list[float]]:
    """Benchmark active learning under the mid-2025 supplier shift.

    A distribution-shift setup: the labelled seed is drawn from **pre-shift** (S1)
    data only; the unlabelled pool adds **post-shift** (S2) samples; the test set is
    held-out post-shift data. The hypothesis is that a committee trained on the old
    regime disagrees most on the new one, so disagreement-driven selection would
    request the informative post-shift samples faster than random.

    In practice, on this synthetic problem, that edge does **not** materialise — see
    the honest benchmark in ``docs/RESULTS.md``: committee disagreement lands at
    parity with (or slightly behind) random. Uncertainty sampling's advantage needs
    genuine label sparsity or a sharper epistemic signal than a bootstrap committee
    gives here. The method is kept for the domains where it pays.

    Returns ``{"labels": [...], "active": [r2...], "random": [r2...]}`` scored on the
    post-shift test set.
    """
    s1 = df[df.supplier_batch == "S1"].reset_index(drop=True)
    s2 = df[df.supplier_batch == "S2"].reset_index(drop=True)
    rng0 = np.random.default_rng(seed)
    n_test = min(test_size, len(s2) // 2)
    test_pos = rng0.choice(len(s2), size=n_test, replace=False)
    test = s2.iloc[test_pos].reset_index(drop=True)
    pool_s2 = s2.drop(index=test_pos).reset_index(drop=True)
    curves: dict[str, list[float]] = {}

    for strategy in ("active", "random"):
        rng = np.random.default_rng(seed)
        seed_pos = rng.choice(len(s1), size=seed_size, replace=False)
        labeled = s1.iloc[seed_pos].reset_index(drop=True)
        unlabeled = pd.concat([s1.drop(index=seed_pos), pool_s2], ignore_index=True)
        labels_x: list[float] = []
        r2s: list[float] = []
        for r in range(rounds + 1):
            committee = Committee.fit(labeled, k=k, params=params, seed=seed + r)
            r2s.append(_mean_r2(test, committee.predict_mean(test)))
            labels_x.append(float(len(labeled)))
            if r == rounds:
                break
            if strategy == "active":
                pick = np.argsort(committee.disagreement(unlabeled))[::-1][:batch]
            else:
                pick = rng.choice(len(unlabeled), size=min(batch, len(unlabeled)), replace=False)
            labeled = pd.concat([labeled, unlabeled.iloc[pick]], ignore_index=True)
            unlabeled = unlabeled.drop(index=unlabeled.index[pick]).reset_index(drop=True)
        curves[strategy] = r2s
        curves["labels"] = labels_x
    return curves
