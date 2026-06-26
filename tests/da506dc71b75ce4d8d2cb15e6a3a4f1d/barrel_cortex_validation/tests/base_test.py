"""
base_test.py

Shared base class for all barrel cortex validation tests.

Centralises the stimulus construction parameters that every test
inherits - matching the global contrast/dpi
constants used in the visual cortex.
"""

import os
import sciunit
import numpy as np
import matplotlib.pyplot as plt
from sciunit.scores import ZScore, NAScore


# Global stimulus constants  (analogous to low_contrast / high_contrast / dpi)
DT          = 0.1    # ms   simulation timestep
STIM_ONSET  = 10.0   # ms   pulse delivered at this time
DURATION    = 50.0   # ms   default single-pulse trial length
AMPLITUDE   = 5.0    # a.u  default deflection amplitude
PLOTS_DIR   = 'plots'
DPI         = 200


class BaseBarrelTest(sciunit.Test):
    """
    Abstract base class for barrel cortex SciUnit tests.

    Provides
    
    _build_stimulus: constructs the (time_vector, amplitudes) pair
    _zscore_or_na: safe Z-score that returns NAScore on missing spikes
    _gaussian_pdf: helper for plotting the in-vivo normal distribution
    _save_figure: standardised figure save to PLOTS_DIR
    """

    score_type = ZScore

   
    # Stimulus construction
  
    def _build_stimulus(self, duration_ms=DURATION, stim_onset_ms=STIM_ONSET,
                        amplitude=AMPLITUDE, n_whiskers=1):
        """
        Build a single-pulse time-series stimulus.

        Returns
        -------
        t    : np.ndarray, shape (T,)  — time axis in ms
        amps : np.ndarray, shape (n_whiskers, T) — amplitude matrix
        """
        t    = np.arange(0.0, duration_ms, DT)
        amps = np.zeros((n_whiskers, len(t)))
        idx  = int(stim_onset_ms / DT)
        amps[:, idx] = amplitude
        return t, amps



    def _zscore_or_na(self, prediction, observation, missing_msg='No spike detected'):
        """
        Return a ZScore if prediction is a finite number, NAScore otherwise.
        Prevents float('inf') from silently propagating into compute_score.
        """
        if prediction is None or not np.isfinite(prediction):
            return NAScore(missing_msg)
        mean = observation['mean']
        std  = observation['std']
        return ZScore((prediction - mean) / std)

    # ------------------------------------------------------------------
    # Plotting helpers
    # ------------------------------------------------------------------

    def _gaussian_pdf(self, mean, std, n_points=200):
        """Return (x, y) arrays for a Gaussian PDF centred on mean±3*std."""
        x = np.linspace(mean - 3 * std, mean + 3 * std, n_points)
        y = (1.0 / (std * np.sqrt(2 * np.pi))) * np.exp(
            -0.5 * ((x - mean) / std) ** 2
        )
        return x, y

    def _save_figure(self, fig, filename):
        """Save figure to PLOTS_DIR/<filename> at standard DPI."""
        os.makedirs(PLOTS_DIR, exist_ok=True)
        path = os.path.join(PLOTS_DIR, filename)
        fig.savefig(path, dpi=DPI)
        plt.close(fig)
