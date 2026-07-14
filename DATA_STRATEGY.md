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
5. **Real formulations** *(have — schema-complete)* — [`data/real_formulations.csv`](data/real_formulations.csv):
   real PLA/PBAT and PLA/PBS melt-blends with reported tensile strength, now enriched with processing
   metadata (composition, melt temperature, protocol) so each is a **schema-complete training row**
   ([`real_seed.py`](src/biopoly/data/real_seed.py) `real_formulations_training_frame()`). Only tensile
   is reported, so the other four targets stay missing and the forward model's per-target NaN masking
   uses each row for tensile alone. Their honest use is a **real-world validation set** (sim-to-real
   transfer, see below), with training augmentation available but off by default. The remaining work is
   the *full property set* on these blends (MFI, biodegradation, water, clarity with primary citations).

## The seed so far

[`data/real_seed.csv`](data/real_seed.csv) holds neat-polymer reference values (PLA, PHA, PBAT, PBS,
TPS, PCL) for the commonly and robustly reported properties. Melt-flow index and 60-day biodegradation
are strongly condition-dependent and are deliberately **left out** until they can be sourced properly.

**Provenance — read honestly, per column:**
- **Tensile strength — sourced.** Each row carries a literature range (`tensile_range_mpa`) and a
  per-row `source`; values are mid-range. PLA is anchored on a NatureWorks Ingeo datasheet (7000D
  ~65 MPa) and PBAT notes the datasheet caveat that oriented **film** grades (BASF ecoflex ~35 MPa)
  read far higher than **bulk** PBAT (~10–20). See the References below.
- **Water absorption & optical clarity — still indicative.** Not yet tied to primary citations;
  treat as placeholders pending sourcing.

## First use: anchoring, not training

Neat-polymer literature points carry no processing metadata, so they cannot yet train the model. Their
honest first use is **anchoring** — checking the synthetic ground truth against reality
([`real_seed.py`](src/biopoly/data/real_seed.py) `synthetic_vs_real()`):

- The synthetic neat-polymer outputs sit within a **~12% median gap** of the literature seed — the
  generator is anchored in reality, not invented.
- The seed also **flags where it is not**: the clearest *sourced* target is **PCL tensile** (synthetic
  ~16 MPa vs literature ~20–30). (Sourcing PBAT properly also *corrected an earlier guess* — bulk PBAT
  is ~10–20 MPa, so the synthetic anchor of ~12 is fine; the 35 MPa datasheet figure is oriented film.)
  This is real data doing real work — both validating the priors and correcting them.

## Second use: sim-to-real validation, and an honest augmentation check

The real *formulations* go one rung further than the neat seed. Enriched with processing metadata they
line up with the feature schema, so the synthetic-trained forward model can predict their tensile and
be scored against the literature value ([`real_seed.py`](src/biopoly/data/real_seed.py)
`evaluate_sim_to_real()`): a genuine out-of-distribution transfer check, reported with the fraction of
blends whose conformally-calibrated p10–p90 band contains reality (see `docs/RESULTS.md`).

`augmentation_experiment()` then asks the honest question — *do these points help as training data?* —
by leave-one-out (synthetic-only vs synthetic + the other real blends). Five points against thousands
of synthetic rows barely move the tensile error, which is the expected, honest answer: at this scale
real data is worth more as **validation and anchoring** (and, next, targeted fine-tuning) than as raw
augmentation. `biopoly-train --augment-real` folds them in for anyone who wants to; the champion
pipeline stays purely synthetic and reproducible by default.

## Honest limitations

- Tensile is literature-sourced (datasheet + review ranges); water absorption and optical clarity are
  still indicative; MFI and 60-day biodegradation are omitted. So: partial, and secondary-sourced.
- The neat seed is neat polymers only; the real formulations report tensile only, and their melt
  residence time is a representative extrusion value rather than a per-study figure.
- A handful of rows: enough to anchor, validate sim-to-real transfer, and start the habit of folding
  real data in — not enough to move a synthetic-trained model by augmentation alone.

## Growing it

Attach primary citations to each value; add real **formulations** (recipe + processing + measured
properties) as they can be sourced; then either blend a small real set into training or hold it out as a
real-world validation set for the synthetic-trained model. The scarce-data machinery — priors, active
learning, calibrated uncertainty, drift/retrain — is already in place to make the most of each real point.

## References

Sources consulted for the tensile seed (secondary sources; attach primary citations before
authoritative use):

- NatureWorks Ingeo — Injection-molding technical data sheets (e.g. 3100HP, 7000D ~65 MPa):
  <https://www.natureworksllc.com/Technical-Resources>
- BASF ecoflex® F Blend C1200 — product data sheet (~35 MPa MD film, ISO 527):
  <https://plastics-rubber.basf.com/global/en/performance_polymers/products/ecoflex>
- "Biodegradable Plastics Compared: PHA, PHB, PBAT, PLA, PBS, PCL, TPS" — property overview
  (tensile ranges used here): <https://specialty-polymer.com/biodegradable-compared-pha-phb-pbat-pla-pbs-pcl-and-tps/>
- "Critical Review on Polylactic Acid: Properties, Structure, Processing, Biocomposites, and
  Nanocomposites", *Materials* (PMC9228835): <https://pmc.ncbi.nlm.nih.gov/articles/PMC9228835/>
- "Recent advances in biodegradable polymer blends and their biocomposites: a comprehensive review",
  *Green Chemistry* (RSC, 2025), DOI 10.1039/D5GC01294E: <https://pubs.rsc.org/en/content/articlehtml/2025/gc/d5gc01294e>

Blend datapoints (`real_formulations.csv`):

- "Influence of PBS, PBAT and TPS content on tensile and processing properties of PLA-based blends",
  *J. Materials Science* (2023), DOI 10.1007/s10853-022-08081-z:
  <https://link.springer.com/article/10.1007/s10853-022-08081-z>
- "Polylactide (PLA) and Its Blends with Poly(butylene succinate) (PBS): A Brief Review", *Polymers*
  (PMC6680981): <https://pmc.ncbi.nlm.nih.gov/articles/PMC6680981/>
- "Understanding the Role of PBAT Content ... on 3D-Printed PLA/PBAT Objects", *Polymers* (2026),
  DOI 10.3390/polym18030339: <https://doi.org/10.3390/polym18030339>
