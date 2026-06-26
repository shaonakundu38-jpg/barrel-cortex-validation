"""
laminar_latency_test.py

TEST 4: Laminar first-spike latency ordering (L4 -> L2/3 -> L5).
 
Barrel cortex has a highly stereotyped laminar response sequence:
L4 spiny stellates fire first (~7-8 ms), L2/3 pyramidals follow
(~11 ms), L5 thick-tufted cells last (~14-15 ms) [Petersen 2003].
 
This test checks:
  (a) that each layer's mean latency Z-scores within the in-vivo range, AND
  (b) that the ordering constraint L4 < L2/3 < L5 is satisfied.
 
A combined score is returned: the maximum absolute Z across the three
layers.  An ordering violation is flagged as a separate NAScore.
 
validate_observation is overridden because the observation dict has a
nested per-layer structure that SciUnit's default ZScore validator does
not accept.
"""
 import numpy as np
import matplotlib.pyplot as plt
import sciunit
from sciunit.scores import ZScore, NAScore
 
from tests.base_test import BaseBarrelTest, STIM_ONSET
from tests.capabilities import StimulusInput, PopulationOutput
 
 
class LaminarLatencyTest(BaseBarrelTest):
    """
    Validates laminar first-spike latency ordering (L4, L2/3, L5).
 
    Observation format
    ------------------
    {
        'L4':  {'mean': float, 'std': float},
        'L23': {'mean': float, 'std': float},
        'L5':  {'mean': float, 'std': float},
    }
    """
 
    required_capabilities = (StimulusInput, PopulationOutput)
 
    LAYERS = ['L4', 'L23', 'L5']
   
    def validate_observation(self, observation):
        required_layers = {'L4', 'L23', 'L5'}
        if not required_layers.issubset(observation.keys()):
            raise sciunit.errors.ObservationError(
                f'Observation must contain keys: {required_layers}'
            )
        for layer in required_layers:
            if 'mean' not in observation[layer] or 'std' not in observation[layer]:
                raise sciunit.errors.ObservationError(
                    f'Layer {layer} must have "mean" and "std" keys.'
                )
        return observation
 
   
 
    def generate_prediction(self, model):
        """
        Stimulate C2 and extract mean first-spike latency per layer.
        Returns {layer: latency_ms} or None per layer if no spikes.
        """
        t, amps = self._build_stimulus(duration_ms=80.0)
        model.apply_waveform(
            time_vector=t,
            amplitudes=amps,
            barrel_mapping={'C2': 0},
        )
 
        latencies = {}
        for layer in self.LAYERS:
            sheet = model.get_layer_spikes(layer=layer, cell_type='Exc')
            first_spikes = [
                times[0] for times in sheet.values()
                if hasattr(times, '__len__') and len(times) > 0
            ]
            if first_spikes:
                latencies[layer] = float(np.mean(first_spikes)) - STIM_ONSET
            else:
                latencies[layer] = None
 
        return latencies
 
   
 
    def compute_score(self, observation, prediction):
        vals = [prediction.get(l) for l in self.LAYERS]
 
        if any(v is None for v in vals):
            missing = [l for l, v in zip(self.LAYERS, vals) if v is None]
            return NAScore(f'No spikes in layer(s): {missing}')
 
        l4_lat, l23_lat, l5_lat = vals
        if not (l4_lat < l23_lat < l5_lat):
            return NAScore(
                f'Laminar ordering violated: '
                f'L4={l4_lat:.1f} L23={l23_lat:.1f} L5={l5_lat:.1f} ms'
            )
 
        z_vals = []
        for layer in self.LAYERS:
            obs  = observation[layer]
            pred = prediction[layer]
            z_vals.append(abs((pred - obs['mean']) / obs['std']))
 
        worst_z = max(z_vals)
        self._plot(prediction, observation, model_name=self.name)
        return ZScore(worst_z)
 
   
 
    def _plot(self, prediction, observation, model_name):
        layers    = self.LAYERS
        pred_vals = [prediction[l] for l in layers]
        obs_means = [observation[l]['mean'] for l in layers]
        obs_stds  = [observation[l]['std']  for l in layers]
 
        x     = np.arange(len(layers))
        width = 0.35
 
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(x - width / 2, obs_means, width, yerr=obs_stds,
               color='steelblue', alpha=0.7, label='In Vivo', capsize=5)
        ax.bar(x + width / 2, pred_vals, width,
               color='coral', alpha=0.9, label='Model')
        ax.set_xticks(x)
        ax.set_xticklabels(['L4', 'L2/3', 'L5'])
        ax.set_ylabel('First-Spike Latency (ms)')
        ax.set_title(f'Laminar Latency Profile: {model_name}')
        ax.legend(frameon=False)
        fig.tight_layout()
        self._save_figure(fig, f'{model_name}_laminar_latency.png')
 
