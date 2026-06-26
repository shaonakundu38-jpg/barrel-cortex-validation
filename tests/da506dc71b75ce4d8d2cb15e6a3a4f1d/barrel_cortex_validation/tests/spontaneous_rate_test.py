"""
spontaneous_rate_test.py
TEST 6: Spontaneous firing rate of L4 excitatory cells at rest.

Models frequently over-fire at rest due to excessive recurrent
excitation.  This test delivers no stimulus for 1 000 ms and
verifies that the mean L4 Exc firing rate stays below the in-vivo
baseline (~0.3 Hz, Brecht et al. 2003).

Mirrors the InternalStimulus / TrialAveragedFiringRate analysis step
used in the visual cortex pipeline for spontaneous activity checks.
"""

import numpy as np
import matplotlib.pyplot as plt
import sciunit
from sciunit.scores import ZScore, NAScore

from tests.base_test import BaseBarrelTest
from tests.capabilities import StimulusInput, PopulationOutput, SpontaneousOutput


class SpontaneousRateTest(BaseBarrelTest):
    """
    Validates spontaneous L4 Exc firing rate against in-vivo baseline.

    Observation keys expected:
    mean  : float — in-vivo mean spontaneous rate in Hz
    std   : float — in-vivo standard deviation in Hz
    """

    required_capabilities = (SpontaneousOutput,)

    WINDOW_MS = 1000.0  # ms

    

    def generate_prediction(self, model):
        """
        Query the model's spontaneous rate directly via SpontaneousOutput.
        The model is responsible for simulating a no-stimulus period of
        WINDOW_MS and returning the mean L4 Exc firing rate in Hz.
        """
        rate = model.get_spontaneous_rate()
        return float(rate)

    

    def compute_score(self, observation, prediction):
        score = self._zscore_or_na(prediction, observation)
        self._plot(prediction, observation, model_name=self.name)
        return score

    

    def _plot(self, prediction, observation, model_name):
        mean, std = observation['mean'], observation['std']
        x, y = self._gaussian_pdf(mean, std)

        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(x, y, color='black', lw=1.5, label='In Vivo Spontaneous')
        ax.axvline(
            prediction, color='darkorange', linestyle='--', lw=2,
            label=f'Model ({prediction:.2f} Hz)',
        )
        ax.set_title(f'Spontaneous Rate: {model_name}')
        ax.set_xlabel('Firing Rate (Hz)')
        ax.set_ylabel('Density')
        ax.legend(frameon=False)
        fig.tight_layout()
        self._save_figure(fig, f'{model_name}_spontaneous_rate.png')
