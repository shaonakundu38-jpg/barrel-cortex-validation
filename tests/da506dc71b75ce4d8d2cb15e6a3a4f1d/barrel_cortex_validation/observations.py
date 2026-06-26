"""
observations.py: 

Ground-truth empirical observations for barrel cortex validation tests.

Each entry follows the convention used in the visual cortex analysis:
    {
        'mean'   : float,   # population mean from literature
        'std'    : float,   # population standard deviation
        'units'  : str,     # physical units of the measured quantity
        'source' : str,     # citation (Author Year, Journal)
    }
"""

OBSERVATIONS = {
    # TEST 1: Absolute first-spike latency in the principal barrel (C2)
    # following a single-deflection tactile stimulus.
    # Source: Brecht & Sakmann 2002 (L4 spiny stellates, ~7–8 ms onset).
   
    'latency': {
        'mean'   : 7.5,
        'std'    : 0.8,
        'units'  : 'ms',
        'source' : 'Brecht & Sakmann 2002, J Physiol',
    },

   
    # TEST 2: Direction Selectivity Index (DSI) computed via vector
    # summation across 8 evenly-spaced whisker-deflection angles.
    # Source: Andermann & Moore 2006, Nat Neurosci.
   
    'direction_selectivity': {
        'mean'   : 0.38,
        'std'    : 0.05,
        'units'  : 'dimensionless (0–1)',
        'source' : 'Andermann & Moore 2006, Nat Neurosci',
    },

   
    # TEST 3: Lateral propagation delay from principal (C2) to adjacent
    # (C3) barrel column — the inter-columnar spatiotemporal spread.
    # Source: Estebanez et al. 2018, J Neurosci.
   
    'spatiotemporal_spread': {
        'mean'   : 5.5,
        'std'    : 0.6,
        'units'  : 'ms',
        'source' : 'Estebanez et al. 2018, J Neurosci',
    },

   
    # TEST 4: Laminar first-spike latency ordering.
    # L4 leads, L2/3 follows, L5 is last.  Values are relative to
    # stimulus onset; gaps are as important as absolutes.
    # Source: Petersen et al. 2003, Neuron.
   
    'laminar_latency': {
        'L4'  : {'mean': 7.5,  'std': 0.8,  'units': 'ms'},
        'L23' : {'mean': 11.0, 'std': 1.2,  'units': 'ms'},
        'L5'  : {'mean': 14.5, 'std': 1.5,  'units': 'ms'},
        'source' : 'Petersen et al. 2003, Neuron',
    },

   
    # TEST 5: Spike-count adaptation ratio (pulse 5 / pulse 1) under an
    # 8 Hz repetitive whisker-deflection train.
    # Source: Petersen 2002, Neuron (short-term depression in L4).
   
    'adaptation_ratio': {
        'mean'   : 0.35,
        'std'    : 0.08,
        'units'  : 'dimensionless (spike count ratio)',
        'source' : 'Petersen 2002, Neuron',
    },

   
    # TEST 6: Spontaneous firing rate of L4 excitatory cells at rest
    # (no whisker stimulus, no direct stimulation).
    # Source: Brecht et al. 2003, J Physiol.
   
    'spontaneous_rate': {
        'mean'   : 0.3,
        'std'    : 0.15,
        'units'  : 'Hz',
        'source' : 'Brecht et al. 2003, J Physiol',
    },
}
