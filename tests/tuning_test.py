"""
tuning_test.py
--------------
TEST 2: Multi-directional selectivity via Vector Summation DSI.

Measures the Direction Selectivity Index (DSI) across 8 evenly spaced
whisker-deflection angles using vector summation, and scores it against
the in-vivo distribution from Andermann & Moore 2006.

The preferred angle is also computed and labelled on the polar plot,
following the convention in the visual cortex pipeline where tuning
curves always show the preferred orientation alongside the selectivity
metric.
"""

import numpy as np
import matplotlib.pyplot as plt

from tests.base_test import BaseBarrelTest, DURATION
from tests.capabilities import SweepInput, AngularOutput


class TuningTest(BaseBarrelTest):
    """
    Measures multi-directional selectivity via Vector Summation DSI.

    Observation keys expected:
    mean  : float — in-vivo mean DSI
    std   : float — in-vivo standard deviation of DSI
    """

    required_capabilities = (SweepInput, AngularOutput)

    N_ANGLES = 8   # evenly sampled around the full circle,(?do we need more? )

   

    def generate_prediction(self, model):
        """
        Sweep 8 angles, get per-angle spike counts, compute DSI
        via vector summation, and return both the scalar DSI and the
        raw spike dict for plotting.
        """
        t      = np.arange(0.0, DURATION * 2, 0.1)   # longer trial for tuning
        angles = np.linspace(0, 2 * np.pi, self.N_ANGLES, endpoint=False)

        model.apply_angular_suite(angles=angles, time_vector=t, amplitude=5.0)
        spike_dict = model.get_binned_directional_matrix()

        vx, vy, total = 0.0, 0.0, 0
        for angle, spikes in spike_dict.items():
            count  = len(spikes)
            vx    += count * np.cos(angle)
            vy    += count * np.sin(angle)
            total += count

        dsi           = np.sqrt(vx**2 + vy**2) / total if total > 0 else 0.0
        preferred_rad = np.arctan2(vy, vx) if total > 0 else 0.0

        return {'dsi': dsi, 'preferred_rad': preferred_rad, 'raw': spike_dict}

   

    def compute_score(self, observation, prediction):
        score = self._zscore_or_na(prediction['dsi'], observation)
        self._plot(prediction, model_name=self.name)
        return score

   

    def _plot(self, prediction, model_name):
        spike_dict    = prediction['raw']
        preferred_rad = prediction['preferred_rad']

        angles = np.array(list(spike_dict.keys()))
        counts = np.array([len(spike_dict[a]) for a in angles])

        angles_closed = np.append(angles, angles[0])
        counts_closed = np.append(counts, counts[0])

        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'},
                               figsize=(4, 4))
        ax.plot(angles_closed, counts_closed,
                color='royalblue', lw=2, marker='o', markersize=5)
        ax.axvline(preferred_rad, color='crimson', lw=1.5,
                   linestyle='--', label=f'Preferred: {np.degrees(preferred_rad):.0f}°')
        ax.set_title(
            f'Direction Tuning: {model_name}\nDSI = {prediction["dsi"]:.2f}',
            va='bottom', fontsize=10,
        )
        ax.legend(loc='lower right', fontsize=8, frameon=False)
        fig.tight_layout()
        self._save_figure(fig, f'{model_name}_polar.png')
