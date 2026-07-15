"""
adaptation_test.py
------------------
TEST 5: Spike-count adaptation ratio under repetitive whisker stimulation.

Real barrel cortex neurons show strong short-term synaptic depression
under repetitive deflections (Petersen 2002, Neuron).  A train of
5 pulses at 8 Hz is delivered; the adaptation ratio is defined as:

    ratio = spike_count(pulse_5) / spike_count(pulse_1)

In-vivo values cluster around 0.35 (±0.08).  A model that does not
depress will produce a ratio near 1.0 and fail with a high Z-score.

Spike counting window
---------------------
Each pulse activates the barrel for stim_change_time (5 ms in the
Kremer model).  We count spikes in a RESPONSE_WINDOW_MS window
starting at each pulse onset.  This must be shorter than the ISI
(125 ms at 8 Hz) to avoid counting spikes from the next pulse, and
long enough to capture the full L23 response (which typically peaks
within 30-50 ms of L4 activation).  We use 60 ms as a conservative
window that captures the response without overlapping the next pulse.
"""

import numpy as np
import matplotlib.pyplot as plt
from sciunit.scores import NAScore

from tests.base_test import BaseBarrelTest
from tests.capabilities import RepetitiveInput, PopulationOutput


class AdaptationTest(BaseBarrelTest):
    """
    Validates short-term spike-count adaptation under 8 Hz pulse trains.

    Observation keys expected
    -------------------------
    mean  : float — in-vivo mean adaptation ratio (pulse5 / pulse1)
    std   : float — in-vivo standard deviation
    """

    required_capabilities = (RepetitiveInput, PopulationOutput)

    N_PULSES         = 5
    FREQ_HZ          = 8.0
    ISI_MS           = 1000.0 / FREQ_HZ      # 125 ms
    RESPONSE_WINDOW_MS = 60.0                 # count window per pulse
    # Must satisfy: RESPONSE_WINDOW_MS < ISI_MS  (125 ms) — satisfied.

    # ------------------------------------------------------------------

    def generate_prediction(self, model):
        """
        Deliver a 5-pulse 8 Hz train to C2 and extract the per-pulse
        spike count within a 60 ms response window after each pulse onset.
        Returns the adaptation ratio (pulse 5 / pulse 1) and the full
        count list for plotting.
        """
        duration_ms = self.ISI_MS * (self.N_PULSES + 1)

        model.apply_pulse_train(
            barrel_name='C2',
            n_pulses=self.N_PULSES,
            isi_ms=self.ISI_MS,
            amplitude=5.0,
            duration_ms=duration_ms,
        )

        spikes = model.get_barrel_spikes('C2')

        counts = []
        for pulse_idx in range(self.N_PULSES):
            onset  = self.ISI_MS * (pulse_idx + 1)
            offset = onset + self.RESPONSE_WINDOW_MS   # 60 ms window only
            pulse_spikes = [s for s in spikes if onset <= s < offset]
            counts.append(len(pulse_spikes))

        if counts[0] == 0:
            return None

        ratio = counts[-1] / counts[0]
        return {'ratio': ratio, 'counts': counts}

    # ------------------------------------------------------------------

    def compute_score(self, observation, prediction):
        if prediction is None:
            return NAScore('No spike on pulse 1 — adaptation ratio undefined')

        score = self._zscore_or_na(prediction['ratio'], observation)
        self._plot(prediction, model_name=self.name)
        return score

    # ------------------------------------------------------------------

    def _plot(self, prediction, model_name):
        counts = prediction['counts']
        pulses = np.arange(1, len(counts) + 1)

        fig, ax = plt.subplots(figsize=(5, 3))
        ax.bar(pulses, counts, color='steelblue', alpha=0.85)
        ax.set_xlabel('Pulse Number')
        ax.set_ylabel(f'Spike Count (first {int(self.RESPONSE_WINDOW_MS)} ms)')
        ax.set_title(
            f'Adaptation: {model_name}\n'
            f'Ratio (P5/P1) = {prediction["ratio"]:.2f}'
        )
        ax.set_xticks(pulses)
        fig.tight_layout()
        self._save_figure(fig, f'{model_name}_adaptation.png')
