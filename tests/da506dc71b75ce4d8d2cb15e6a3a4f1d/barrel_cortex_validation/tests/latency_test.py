"""
latency_test.py

TEST 1: Absolute first-spike latency in the principal barrel (C2).

Validates that the model's population onset latency following a single
whisker deflection falls within the in-vivo distribution reported by
Brecht & Sakmann 2002.  Scoring is via Z-score; a |Z| < 2 (?Is that accurate enough?) is a pass.

Inherits stimulus construction and plotting utilities from BaseBarrelTest,
mirroring the pattern in the visual cortex where analysis steps
are composed from shared helpers.
"""

import numpy as np
import matplotlib.pyplot as plt
import sciunit

from tests.base_test import BaseBarrelTest, STIM_ONSET
from tests.capabilities import StimulusInput, PopulationOutput


class LatencyTest(BaseBarrelTest):
    """
    Validates population response latency parameters using Z-scores. (Self note: Choose more statisticallly accurate scoring methods when improving tests)

    Observation keys expected
    
    mean  : float — in-vivo mean first-spike latency in ms
    std   : float — in-vivo standard deviation in ms
    """

    required_capabilities = (StimulusInput, PopulationOutput)

    
    def generate_prediction(self, model):
        """
        Deliver a single-pulse stimulus to barrel C2 and return the
        first-spike latency relative to stimulus onset.

        Returns None if no spike is detected (handled gracefully by
        _zscore_or_na in compute_score).
        """
        t, amps = self._build_stimulus()
        model.apply_waveform(
            time_vector=t,
            amplitudes=amps,
            barrel_mapping={'C2': 0},
        )
        spikes = model.get_barrel_spikes('C2')

        if len(spikes) == 0:
            return None
        return float(spikes[0]) - STIM_ONSET

    
    def compute_score(self, observation, prediction):
        score = self._zscore_or_na(prediction, observation)
        self._plot(prediction, observation, model_name=self.name)
        return score

    
    def _plot(self, prediction, observation, model_name):
        mean, std = observation['mean'], observation['std']
        x, y = self._gaussian_pdf(mean, std)

        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(x, y, color='black', lw=1.5, label='In Vivo Baseline')

        if prediction is not None and np.isfinite(prediction):
            ax.axvline(
                prediction, color='crimson', linestyle='--', lw=2,
                label=f'Model ({prediction:.1f} ms)',
            )
        else:
            ax.set_title(f'Latency Validation: {model_name} — NO SPIKE')

        ax.set_title(f'Latency Validation: {model_name}')
        ax.set_xlabel('Latency (ms)')
        ax.set_ylabel('Density')
        ax.legend(frameon=False)
        fig.tight_layout()
        self._save_figure(fig, f'{model_name}_distribution.png')
