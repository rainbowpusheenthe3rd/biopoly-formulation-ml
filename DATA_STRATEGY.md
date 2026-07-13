# Data strategy

Bioplastics data is scarce and expensive — that is the binding constraint this project is built around
(and the one it exists to reason about honestly). This note sets out how the project bootstraps data in
that regime, and where the honest limits are. It is a **synthetic-data demo**; the point is the *method*.

## The ladder

1. **Physics-informed synthetic generation** *(have)* — a transparent ground truth
   ([`chemistry.py`](src/biopoly/data/chemistry.py)) plus messiness that makes it a real ML problem:
   measurement noise, a protocol covariate, structured (not-at-random) missingness, a seasonal feedstock
   signal, a realized-crystallinity latent, and a mid-2025 supplier-purity shift. See
   [`DATA_CARD.md`](DATA_CARD.md).
2. **Literature-derived priors** *(have)* — every anchor in the generator is a plausible literature
   mid-range, so the synthetic data has realistic structure rather than arbitrary numbers.
3. **Active learning** *(have — machinery)* — when real experiments *can* be run, spend them where they
   are most informative ([`active_learning.py`](src/biopoly/active_learning.py)). Benchmarked honestly:
   it does not beat random on the synthetic problem, but the acquisition loop is in place.
4. **A real-data seed** *(started — this)* — begin folding in real literature values, smallest first.
5. **Real formulations with metadata** *(next)* — real recipes *with processing conditions and measured
   outputs* are what actually augment training; sourcing those is the real work ahead.

## The seed so far

[`data/real_seed.csv`](data/real_seed.csv) holds **indicative literature mid-ranges for the neat
polymers** (PLA, PHA, PBAT, PBS, TPS, PCL), and only the commonly and robustly reported properties —
tensile strength, water absorption, optical clarity. Melt-flow index and 60-day biodegradation are
strongly condition-dependent and are deliberately **left out** until primary citations are attached.

**Provenance — read honestly:** these are *indicative reference values* from commercial datasheets
(e.g. NatureWorks Ingeo for PLA, BASF Ecoflex for PBAT) and biopolymer review literature, rounded to
mid-range. They are a *starting point*, not primary-cited measurements; attach and verify specific
citations before treating any value as authoritative.

## First use: anchoring, not training

Neat-polymer literature points carry no processing metadata, so they cannot yet train the model. Their
honest first use is **anchoring** — checking the synthetic ground truth against reality
([`real_seed.py`](src/biopoly/data/real_seed.py) `synthetic_vs_real()`):

- The synthetic neat-polymer outputs sit within a **~12% median gap** of the literature seed — the
  generator is anchored in reality, not invented.
- The seed also **flags where it is not**: PBAT tensile strength (synthetic ~12 MPa vs literature ~20)
  and TPS water absorption (~25% vs ~40) are the clearest calibration targets. This is real data doing
  real work — surfacing where the synthetic priors should be tightened.

## Honest limitations

- Indicative mid-ranges, not primary-cited measurements; partial (3 of 5 properties).
- Neat polymers only — no blends, no processing conditions, so not training-ready yet.
- Six rows: enough to anchor and to start the habit of folding real data in, nothing more.

## Growing it

Attach primary citations to each value; add real **formulations** (recipe + processing + measured
properties) as they can be sourced; then either blend a small real set into training or hold it out as a
real-world validation set for the synthetic-trained model. The scarce-data machinery — priors, active
learning, calibrated uncertainty, drift/retrain — is already in place to make the most of each real point.
