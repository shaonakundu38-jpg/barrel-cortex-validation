"""
dummy_barrel_model.py
---------------------
Reference dummy model implementing all barrel cortex capabilities.

This model produces biologically plausible responses with Gaussian
trial-to-trial jitter so that the validation tests exercise realistic
score ranges rather than perfectly deterministic outputs.

A second class — BadLatencyModel — deliberately produces incorrect
latencies, allowing you to verify that each test correctly rejects a
bad model (i.e. produces |Z| > 2).

Architecture mirrors the visual cortex pipeline's sheet convention:
    Barrel_Exc_L4   — principal L4 excitatory population  (C2 barrel)
    Barrel_Exc_L23  — supragranular L2/3 excitatory population
    Barrel_Exc_L5   — infragranular L5 excitatory population
"""

import numpy as np
import sciunit

from tests.capabilities import (
    StimulusInput, PopulationOutput,
    SweepInput, AngularOutput,
    RepetitiveInput, SpontaneousOutput,
)
# Noise seed for reproducibility across test runs

RNG = np.random.default_rng(seed=42)


class UnifiedBarrelModel(
    
    sciunit.Model,
    StimulusInput, PopulationOutput,
    SweepInput, AngularOutput,
    RepetitiveInput, SpontaneousOutput,
):
    """
    Biologically plausible dummy barrel cortex model.

    Spike times include Gaussian jitter to simulate trial-to-trial
    variability (sigma = 0.3 ms for L4, 0.5 ms for L2/3 and L5),
    matching the noise levels reported in Brecht & Sakmann 2002.

    Laminar latency values are set to the in-vivo means so that a
    fresh validation run produces Z-scores near 0, confirming the
    test pipeline is wired correctly before plugging in a real model.
    """

    
    # Laminar latency parameters  (relative to stimulus onset, ms)
    
    _LAYER_PARAMS = {
        'L4':  {'latency': 7.5,  'jitter': 0.3, 'n_cells': 2},
        'L23': {'latency': 11.0, 'jitter': 0.5, 'n_cells': 2},
        'L5':  {'latency': 14.5, 'jitter': 0.5, 'n_cells': 2},
    }
    _STIM_ONSET   = 10.0   # ms

    def __init__(self, name='UnifiedBarrelModel'):
        super().__init__(name=name)
        self._barrel_spikes     = {}   # {barrel_name: [spike_times]}
        self._layer_spikes      = {}   # {layer: {cell_idx: np.ndarray}}
        self._directional_mat   = {}   # {angle_rad: [spike_times]}
        self._train_spikes      = []   # flat list for pulse-train trials

    
    # StimulusInput
    
    def apply_waveform(self, time_vector, amplitudes, barrel_mapping):
        """
        Inject a single-pulse waveform and populate barrel + layer spikes.
        C3 (adjacent column) fires with an additional 5.5 ms lateral delay.
        """
        max_amp   = float(np.max(amplitudes))
        n_spikes  = max(1, int(max_amp * 3)) if max_amp > 0 else 0
        onset     = self._STIM_ONSET

        self._barrel_spikes = {}
        self._layer_spikes  = {}

        if 'C2' in barrel_mapping:
            # Principal barrel — fast L4 onset
            self._barrel_spikes['C2'] = self._jittered_train(
                onset + 7.5, n_spikes, isi=1.5, jitter=0.3
            )
            # Adjacent barrel — lateral propagation adds ~5.5 ms
            self._barrel_spikes['C3'] = self._jittered_train(
                onset + 7.5 + 5.5, max(1, n_spikes // 3), isi=2.0, jitter=0.4
            )

        # Laminar population readout
        for layer, params in self._LAYER_PARAMS.items():
            self._layer_spikes[layer] = {
                cell_idx: np.array(self._jittered_train(
                    onset + params['latency'], n_spikes,
                    isi=1.5, jitter=params['jitter'],
                ))
                for cell_idx in range(params['n_cells'])
            }

    
    # PopulationOutput
    
    def get_barrel_spikes(self, barrel_name):
        return self._barrel_spikes.get(barrel_name, [])

    def get_layer_spikes(self, layer, cell_type='Exc'):
        return self._layer_spikes.get(layer, {})

    
    # SweepInput
    def apply_angular_suite(self, angles, time_vector, amplitude):
        """
        Directional tuning: cosine preference peaking at 0 rad,
        with Poisson-like count noise.
        """
        self._directional_mat = {}
        for angle in angles:
            tuning    = (np.cos(angle) + 1.0) / 2.0
            n_spikes  = max(1, int(tuning * 12 + RNG.normal(0, 1.0)))
            self._directional_mat[angle] = [
                15.0 + i * 2.0 + float(RNG.normal(0, 0.5))
                for i in range(n_spikes)
            ]

    
    # AngularOutput
    def get_binned_directional_matrix(self):
        return self._directional_mat

    
    # RepetitiveInput
    def apply_pulse_train(self, barrel_name, n_pulses, isi_ms,
                          amplitude, duration_ms):
        """
        Multi-pulse train with exponential spike-count depression.
        Depression constant tau = 2 pulses, matching Petersen 2002.
        """
        self._train_spikes = []
        tau_depression = 2.0   # pulses

        for pulse_idx in range(n_pulses):
            onset         = isi_ms * (pulse_idx + 1)
            # Exponential adaptation: spike count decays across pulses
            depression    = np.exp(-pulse_idx / tau_depression)
            n_spikes      = max(0, int(amplitude * 3 * depression
                                       + RNG.normal(0, 0.5)))
            pulse_spikes  = [
                onset + 7.5 + i * 1.5 + float(RNG.normal(0, 0.3))
                for i in range(n_spikes)
            ]
            self._train_spikes.extend(pulse_spikes)

        self._barrel_spikes['C2'] = sorted(self._train_spikes)

    
    # SpontaneousOutput
    def get_spontaneous_rate(self):
        """Return a plausible low spontaneous rate (~0.3 Hz ± noise)."""
        return max(0.0, float(RNG.normal(0.3, 0.05)))

    
    # Internal helper
    @staticmethod
    def _jittered_train(onset_ms, n_spikes, isi=1.5, jitter=0.3):
        """Return a list of n_spikes spike times starting at onset_ms."""
        return [
            onset_ms + i * isi + float(RNG.normal(0, jitter))
            for i in range(n_spikes)
        ]



# Deliberately bad model for pipeline verification
class BadLatencyModel(UnifiedBarrelModel):
    """
    A model with pathologically late L4 responses (~20 ms instead of ~7.5 ms).

    Use this to confirm that LatencyTest and LaminarLatencyTest both
    produce |Z| > 2 — i.e. that the tests would actually catch a bad model.
    """

    def __init__(self, name='BadLatencyModel'):
        super().__init__(name=name)

    def apply_waveform(self, time_vector, amplitudes, barrel_mapping):
        super().apply_waveform(time_vector, amplitudes, barrel_mapping)
        onset = self._STIM_ONSET

        # Shift all L4 spikes 12.5 ms later than ground truth
        if 'C2' in self._barrel_spikes:
            self._barrel_spikes['C2'] = [s + 12.5
                                         for s in self._barrel_spikes['C2']]
        if 'L4' in self._layer_spikes:
            self._layer_spikes['L4'] = {
                idx: times + 12.5
                for idx, times in self._layer_spikes['L4'].items()
            }
