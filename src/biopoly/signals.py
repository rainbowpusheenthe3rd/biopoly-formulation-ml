"""Synthetic instrument signals + a DSP feature-extraction pipeline.

Polymer *characterisation* data is rarely tabular — it is **signal**: a DSC
thermogram (heat flow vs temperature, with melting endotherms), an FTIR spectrum,
a TGA curve. The information a scientist actually reads off those instruments —
where the melt peaks sit, how sharp they are (crystallinity), their areas — has to
be *extracted* with signal processing before it can feed a model. Hand-waving the
measurement away and modelling only the nominal recipe throws that away.

This module makes the point end-to-end on a **synthetic DSC-like signal**:

1. :func:`synth_dsc` — generate a heat-flow thermogram from a formulation: each
   polymer contributes a melting endotherm (a Gaussian) at its characteristic
   melt temperature, with area ∝ mass fraction and sharpness ∝ crystallinity;
   a nucleating agent raises crystallinity, a plasticiser depresses both Tm and
   crystallinity. A sloping baseline + noise stand in for a real instrument.
2. :func:`process_signal` — the standard DSP clean-up with ``scipy.signal``:
   baseline correction then Savitzky-Golay smoothing.
3. :func:`extract_features` — ``find_peaks`` + ``peak_widths`` → a fixed feature
   vector (peak count, dominant melt temperature, total area, mean width, height).

The extracted features *recover composition and crystallinity from the curve* —
which is exactly what makes them informative. (Synthetic — illustrative of the
approach, not of any real material.)
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import minimum_filter1d
from scipy.signal import find_peaks, peak_widths, savgol_filter

from biopoly.data.chemistry import POLYMERS

# Characteristic DSC melting temperatures (°C), plausible literature mid-ranges.
MELT_TEMP: dict[str, float] = {
    "PLA": 155.0,
    "PHA": 170.0,
    "PBAT": 118.0,
    "PBS": 114.0,
    "TPS": 95.0,  # thermoplastic starch: broad/amorphous, low crystallinity
    "PCL": 60.0,
}
# Baseline crystallinity per polymer (peak-sharpness proxy; 0 amorphous .. 1 crystalline).
CRYSTALLINITY: dict[str, float] = {
    "PLA": 0.90,
    "PHA": 0.80,
    "PBAT": 0.50,
    "PBS": 0.80,
    "TPS": 0.30,
    "PCL": 0.70,
}


def temperature_axis(lo: float = 30.0, hi: float = 200.0, n: int = 400) -> np.ndarray:
    """The DSC temperature sweep (°C) the thermogram is sampled on."""
    return np.linspace(lo, hi, n)


def synth_dsc(
    polymer_frac: dict[str, float],
    *,
    nucleating: float = 0.0,
    plasticizer: float = 0.0,
    crystallinity_scale: float = 1.0,
    rng: np.random.Generator | None = None,
    x: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthesise a DSC-thermogram-like heat-flow signal for a formulation.

    Args:
        polymer_frac: Polymer -> mass fraction.
        nucleating: Nucleating-agent fraction (raises crystallinity → sharper/taller).
        plasticizer: Plasticiser fraction (depresses Tm and crystallinity).
        crystallinity_scale: Realized-crystallinity multiplier (~1.0) — the batch
            latent (see :func:`biopoly.data.chemistry.forward_true`) the thermogram
            reveals: higher means sharper, taller endotherms.
        rng: Optional generator for measurement noise; omit for a clean signal.
        x: Optional temperature axis; defaults to :func:`temperature_axis`.

    Returns:
        ``(x_temperature_c, y_heatflow)`` arrays of equal length.
    """
    x = temperature_axis() if x is None else x
    y = np.zeros_like(x, dtype=float)
    for p in POLYMERS:
        f = float(polymer_frac.get(p, 0.0))
        if f <= 0.0:
            continue
        cryst = (
            CRYSTALLINITY[p] * (1.0 + 1.5 * nucleating - 1.2 * plasticizer) * crystallinity_scale
        )
        cryst = float(np.clip(cryst, 0.05, 1.4))
        tm = MELT_TEMP[p] - 40.0 * plasticizer  # plasticiser depresses the melt
        width = 6.0 / (0.3 + cryst)  # more crystalline → sharper endotherm
        y += (f * cryst) * np.exp(-0.5 * ((x - tm) / width) ** 2)
    y += 0.02 * (x - x.min()) / (x.max() - x.min())  # sloping instrument baseline
    if rng is not None:
        y += rng.normal(0.0, 0.01, size=x.shape)
    return x, y


def _odd_window(window: int, n: int) -> int:
    """Clamp a Savitzky-Golay window to an odd value <= n."""
    w = min(window, n if n % 2 == 1 else n - 1)
    return w if w % 2 == 1 else w - 1


def correct_baseline(y: np.ndarray, window: int = 101) -> np.ndarray:
    """Subtract a slowly-varying baseline (smoothed rolling minimum)."""
    w = _odd_window(window, len(y))
    base = minimum_filter1d(y, size=max(3, w))
    base = savgol_filter(base, window_length=_odd_window(w, len(y)), polyorder=2)
    return y - base


def smooth(y: np.ndarray, window: int = 11, poly: int = 3) -> np.ndarray:
    """Savitzky-Golay smoothing (denoise while preserving peak shape)."""
    w = _odd_window(window, len(y))
    if w <= poly:
        return y.copy()
    return savgol_filter(y, window_length=w, polyorder=poly)


def process_signal(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Baseline-correct then smooth a raw thermogram (returns the processed signal)."""
    return smooth(correct_baseline(y))


def extract_features(
    x: np.ndarray, y_processed: np.ndarray, *, prominence: float = 0.02
) -> dict[str, float]:
    """Extract interpretable DSC features from a processed thermogram.

    Returns a fixed-key dict: number of detected melt peaks, the dominant (tallest)
    peak's temperature, total endotherm area, mean peak width (a crystallinity
    proxy — narrower is more crystalline), and the max peak height.
    """
    peaks, _ = find_peaks(y_processed, prominence=prominence)
    if len(peaks) == 0:
        return {
            "n_peaks": 0,
            "dominant_temp_c": float("nan"),
            "total_area": 0.0,
            "mean_width_c": 0.0,
            "max_height": 0.0,
        }
    widths, _, _, _ = peak_widths(y_processed, peaks, rel_height=0.5)
    dx = float(x[1] - x[0])
    heights = y_processed[peaks]
    dominant = peaks[int(np.argmax(heights))]
    areas = heights * widths * dx
    return {
        "n_peaks": int(len(peaks)),
        "dominant_temp_c": float(x[dominant]),
        "total_area": float(areas.sum()),
        "mean_width_c": float(np.mean(widths) * dx),
        "max_height": float(heights.max()),
    }


def dsc_from_row(
    row,
    *,
    crystallinity_scale: float = 1.0,
    rng: np.random.Generator | None = None,
    x: np.ndarray | None = None,
):
    """Convenience: synthesise a thermogram from a dataset row (frac_*/add_* columns)."""
    pfrac = {p: float(row.get(f"frac_{p}", 0.0)) for p in POLYMERS}
    return synth_dsc(
        pfrac,
        nucleating=float(row.get("add_nucleating", 0.0)),
        plasticizer=float(row.get("add_plasticizer", 0.0)),
        crystallinity_scale=crystallinity_scale,
        rng=rng,
        x=x,
    )
