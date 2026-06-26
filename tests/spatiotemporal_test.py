"""
spatiotemporal_test.py:

TEST 3: Lateral propagation delay from principal (C2) to adjacent (C3)
barrel column.

Validates that the inter-columnar spatiotemporal spread falls within
the in-vivo distribution reported by Estebanez et al. 2018.

If either column produces no spike the prediction returns None and
compute_score issues an NAScore """

import numpy as np
import matplotlib.pyplot as plt

from tests.base_test import BaseBarrelTest, STIM_ONSET
from tests.capabilities import StimulusInput, PopulationOutput


class SpatiotemporalTest(BaseBarrelTest):
    """
    Validates lateral propagation delays across adjacent barrel columns.

    Observation keys expected:

    mean  : float — in-vivo mean C2→C3 propagation delay in ms
    std   : float — in-vivo standard deviation in ms
    """

    required_capabilities = (StimulusInput, PopulationOutput)

    
    def generate_prediction(self, model):
        """
        Stimulate C2, measure first-spike latency in both C2 and C3,
        and return the propagation delay (latency_C3 - latency_C2).

        Returns None if either column fails to fire.
        """
        t, amps = self._build_stimulus()
        model.apply_waveform(
            time_vector=t,
            amplitudes=amps,
            barrel_mapping={'C2': 0},
        )

        spikes_c2 = model.get_barrel_spikes('C2')
        spikes_c3 = model.get_barrel_spikes('C3')

        if len(spikes_c2) == 0 or len(spikes_c3) == 0:
            return None

        latency_c2 = float(spikes_c2[0]) - STIM_ONSET
        latency_c3 = float(spikes_c3[0]) - STIM_ONSET
        return latency_c3 - latency_c2

    
    def compute_score(self, observation, prediction):
        if prediction is None:
            from sciunit.scores import NAScore
            return NAScore('No spike in C2 or C3 — propagation delay undefined')

        score = self._zscore_or_na(prediction, observation)
        self._plot(prediction, observation, model_name=self.name)
        return score

    
    def _plot(self, prediction, observation, model_name): #(Self note: Not exactly necessary plot, right?)
        mean, std = observation['mean'], observation['std']
        x, y = self._gaussian_pdf(mean, std)

        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(x, y, color='black', lw=1.5, label='In Vivo Lateral Spread')
        ax.axvline(
            prediction, color='darkviolet', linestyle='--', lw=2,
            label=f'Model Delay ({prediction:.1f} ms)',
        )
        ax.set_title(f'Spatiotemporal Propagation: {model_name}')
        ax.set_xlabel('Lateral Delay C2 \u2192 C3 (ms)')
        ax.set_ylabel('Density')
        ax.legend(frameon=False)
        fig.tight_layout()
        self._save_figure(fig, f'{model_name}_spread.png')
