"""
capabilities.py
SciUnit capability interfaces for rodent barrel cortex models.

Each capability defines inputs that a model must satisfy
to be testable by the corresponding validation test. (? Check if the inputs are actually stuff found in papers?) This resembles
the sheet-based capability separation used in the visual cortex
pipeline (V1_Exc_L4, V1_Inh_L4, etc.)  here adapted to the
barrel column architecture (principal barrel, adjacent barrels,
laminar sheets).

Capabilities: 
StimulusInput        single-pulse tactile waveform injection
PopulationOutput     spike readout per barrel column and per layer
SweepInput           multi-angle directional deflection protocol
AngularOutput        directional spike-count matrix
RepetitiveInput      multi-pulse train for adaptation protocols
SpontaneousOutput    baseline (no-stimulus) spike rate readout
"""

import sciunit

# Input capabilities

class StimulusInput(sciunit.Capability):
    """
    Enforces continuous time-series tactile stimulation inputs.

    apply_waveform must inject an amplitude waveform (whiskers x time)
    into the named barrel columns specified by barrel_mapping.

    Parameters
    time_vector   : 1-D array, shape (T,), time axis in ms
    amplitudes    : 2-D array, shape (n_whiskers, T), deflection amplitudes
    barrel_mapping: dict mapping barrel name (str) to whisker row index (int)
                    e.g. {'C2': 0}
    """
    def apply_waveform(self, time_vector, amplitudes, barrel_mapping):
        raise NotImplementedError


class SweepInput(sciunit.Capability):
    """
    Enforces experimental protocols using structured directional sweeping.

    apply_angular_suite delivers a single deflection at each supplied
    angle (in radians) and must store per-angle spike trains for
    subsequent retrieval via AngularOutput.

    Parameters
    angles      : array-like of deflection angles in radians
    time_vector : 1-D array, time axis in ms
    amplitude   : float, deflection amplitude (common across all angles)
    """
    def apply_angular_suite(self, angles, time_vector, amplitude):
        raise NotImplementedError


class RepetitiveInput(sciunit.Capability):
    """
    Parameters
    barrel_name : str, target barrel column (e.g. 'C2')
    n_pulses    : int, number of pulses in the train
    isi_ms      : float, inter-pulse interval in ms
    amplitude   : float, deflection amplitude per pulse
    duration_ms : float, total simulation duration in ms
    """
    def apply_pulse_train(self, barrel_name, n_pulses, isi_ms,
                          amplitude, duration_ms):
        raise NotImplementedError

# Output capabilities

class PopulationOutput(sciunit.Capability):
    """
    Enforces high-density readout of spiking data across layer sheets.

    get_layer_spikes returns a dict mapping neuron index to spike-time array
    for the requested layer and cell type.

    get_barrel_spikes returns the list of spike times for the first cell
    recorded in the named barrel column (used for latency extraction).
    """
    def get_layer_spikes(self, layer, cell_type='Exc'):
        """
        Parameters:
        layer     : str, one of 'L4', 'L23', 'L5'
        cell_type : str, 'Exc' or 'Inh'

        Returns:
        dict : {neuron_index (int): spike_times (np.ndarray, ms)}
        """
        raise NotImplementedError

    def get_barrel_spikes(self, barrel_name):
        """
        Parameters

        barrel_name : str, e.g. 'C2' or 'C3'

        Returns: 
        list or np.ndarray of spike times in ms 
        """
        raise NotImplementedError


class AngularOutput(sciunit.Capability):
    """
    Enforces directional metrics matching population tuning tables.

    get_binned_directional_matrix returns the per-angle spike lists
    recorded during the most recent apply_angular_suite call.

    Returns:
    dict : {angle_rad (float): spike_times (list)}
    """
    def get_binned_directional_matrix(self):
        raise NotImplementedError


class SpontaneousOutput(sciunit.Capability):
    """
    Enforces readout of spontaneous (no-stimulus) firing activity.

    get_spontaneous_rate returns the mean firing rate across all L4
    excitatory neurons over the recorded window.

    Returns:
    float : mean firing rate in Hz
    """
    def get_spontaneous_rate(self):
        raise NotImplementedError
