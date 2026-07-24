# Barrel Cortex Validation Library

A SciUnit-based validation test library for computational models of the rodent barrel cortex.
The library provides six quantitative validation tests grounded in peer-reviewed in-vivo electrophysiology literature, a Brian 2 wrapper for the Kremer et al. 2011 barrel cortex model, and a modular architecture that allows any future barrel cortex model to be validated with the same test suite.

---

## Table of Contents

- [Background](#background)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Validation Tests](#validation-tests)
- [The Kremer et al. 2011 Model](#the-kremer-et-al-2011-model)
- [The Model Wrapper](#the-model-wrapper)
- [Observations and Literature Sources](#observations-and-literature-sources)
- [Running Validation](#running-validation)
- [Interpreting Results](#interpreting-results)
- [Known Limitations](#known-limitations)
- [Adding a New Model](#adding-a-new-model)
- [License](#license)

---

## Background

The rodent whisker-to-barrel cortex pathway is one of the best-characterised sensory systems in neuroscience. Each facial whisker maps to a discrete cortical column (a "barrel") in primary somatosensory cortex (S1). Within each barrel column, thalamocortical input arrives in Layer 4 (L4), propagates to Layer 2/3 (L2/3), and subsequently to Layer 5 (L5). Neurons across all layers respond selectively to the angular direction of whisker deflection.

This library uses the [SciUnit](https://github.com/scidash/sciunit) framework to formalise validation of barrel cortex models. SciUnit separates three concerns:

- **Capabilities** — the API a model must implement to be testable
- **Tests** — the scientific protocol and scoring criterion
- **Observations** — the empirical in-vivo targets drawn from the literature

This design means the tests are model-agnostic. Any model that implements the required capabilities can be validated without modifying the test code.

## Project Structure

```
barrel_cortex_validation/
│
├── run_validation.py          # Top-level pipeline — run this
├── observations.py            # Empirical targets with literature citations
│
├── tests/
│   ├── __init__.py
│   ├── capabilities.py        # SciUnit capability definitions
│   ├── base_test.py           # Shared base class for all tests
│   ├── latency_test.py        # Test 1: Population response latency
│   ├── tuning_test.py         # Test 2: Directional selectivity (DI)
│   ├── spatiotemporal_test.py # Test 3: Lateral propagation delay
│   ├── laminar_latency_test.py# Test 4: Laminar ordering
│   ├── adaptation_test.py     # Test 5: Spike-count adaptation
│   └── spontaneous_rate_test.py # Test 6: Spontaneous firing rate
│
├── models/
│   ├── dummy_barrel_model.py  # Reference dummy model for pipeline testing
│   └── kremer_model_wrapper.py# Brian 2 wrapper for Kremer et al. 2011
│
└── plots/                     # Auto-generated figures (created at runtime)
```

---

## Requirements

### Python version

Python 3.9 or higher is recommended. The library has been tested on Python 3.14.

### Core dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `sciunit` | ≥ 0.2.5 | Validation framework |
| `brian2` | ≥ 2.5.0 | Neural simulation (for Kremer model) |
| `numpy` | ≥ 1.23 | Numerical computation |
| `matplotlib` | ≥ 3.5 | Plot generation |

### Installing dependencies

It is strongly recommended to use a virtual environment:

```bash
python -m venv val_env
source val_env/bin/activate        # On Linux/macOS
val_env\Scripts\activate           # On Windows
```

Then install all dependencies:

```bash
pip install sciunit brian2 numpy matplotlib
```

Or using the requirements file if provided:

```bash
pip install -r requirements.txt
```

### requirements.txt

```
sciunit>=0.2.5
brian2>=2.5.0
numpy>=1.23
matplotlib>=3.5
```

### Brian 2 note

This library uses **Brian 2**, not Brian 1. The two are not compatible. If you see `from brian import *` anywhere, that is Brian 1. This library uses `from brian2 import *`.

Brian 2 is set to use the `numpy` codegen target throughout (`prefs.codegen.target = 'numpy'`). This avoids the need for a C compiler and is reliable across all platforms, at the cost of some simulation speed.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/barrel_cortex_validation.git
cd barrel_cortex_validation
```

2. Create and activate a virtual environment:

```bash
python -m venv val_env
source val_env/bin/activate
```

3. Install dependencies:

```bash
pip install sciunit brian2 numpy matplotlib
```

4. Verify the installation by running the dummy model pipeline:

```bash
PYTHONPATH=. python -B run_validation.py
```

You should see the validation suite run against the `UnifiedBarrelModel` dummy model and produce scores for all six tests.


## Quick Start

### Running validation on the dummy model

The dummy model (`models/dummy_barrel_model.py`) produces biologically plausible spike responses with Gaussian jitter. It is useful for verifying that the test pipeline is working correctly before connecting a real simulator.

```bash
PYTHONPATH=. python -B run_validation.py
```

### Running validation on the Kremer et al. 2011 model

Edit the `__main__` block at the bottom of `run_validation.py`:

```python
if __name__ == '__main__':
    from models.kremer_model_wrapper import KremerBarrelModel

    kremer = KremerBarrelModel(barrelarraysize=3)  # use 3 to avoid memory errors
    kremer.build()       # constructs Brian 2 network (~1-2 minutes)
    kremer.warmup(duration_s=5)   # STDP training run
    run_barrel_validation(kremer)
```

Then run:

```bash
PYTHONPATH=. python -B run_validation.py
```

The `-B` flag prevents Python from writing `.pyc` bytecode files. The `PYTHONPATH=.` ensures all local imports resolve correctly from the project root.


## Validation Tests

All tests inherit from `BaseBarrelTest` which provides shared utilities: stimulus construction, Z-score safety checks, Gaussian PDF plotting, and standardised figure saving.

### Test 1 — Population Latency

**File:** `tests/latency_test.py`

**What it measures:** The first-spike latency of L2/3 excitatory neurons in the principal barrel (C2) following a single-pulse tactile stimulus, measured relative to stimulus onset.

**Protocol:** A single-pulse waveform is injected into barrel C2. First-spike time is extracted and stimulus onset is subtracted to give latency in ms. This is repeated across 10 trials and the mean latency is used, reducing the influence of Poisson noise in stochastic models.

**Scoring:** Z-score against the in-vivo distribution. (Note: Requires better scoring methods, Use KSS scoring)

**Pass criterion:** |Z| < 2.0

**Literature source:** Brecht & Sakmann 2002, J Physiol 543:49–70

---

### Test 2 — Directional Selectivity

**File:** `tests/tuning_test.py`

**What it measures:** The Direction Index (DI) of the C2 barrel population response across 8 evenly-spaced whisker deflection angles.

**Protocol:** 8 deflection angles (0°, 45°, 90°, ..., 315°) are presented sequentially. Spike counts per angle are recorded. DI is computed as:

```
DI = spike count at preferred direction / mean spike count across all directions
```

A DI of 1.0 means no directional preference (all angles evoke the same response). A DI > 1 indicates a directional preference, with higher values reflecting sharper tuning.

**Scoring:** Z-score against the in-vivo population distribution.

**Pass criterion:** |Z| < 2.0

**Literature source:** Wilent & Contreras 2005, J Neurosci 25(11):2983–2991

---

### Test 3 — Spatiotemporal Spread

**File:** `tests/spatiotemporal_test.py`

**What it measures:** The lateral propagation delay from the principal barrel (C2) to the adjacent barrel (C3) following C2 stimulation.

**Protocol:** A single pulse is delivered to C2. First-spike latencies are extracted from both C2 and C3. The propagation delay is:

```
delay = latency(C3) - latency(C2)
```

If either barrel produces no spike, an NAScore is returned.

**Scoring:** Z-score against the in-vivo inter-columnar delay.

**Pass criterion:** |Z| < 2.0

**Literature source:** Estebanez et al. 2018, J Neurosci

> **Note:** The observation value (mean 5.5 ms, std 0.6 ms) is a literature-informed estimate and should be verified against the primary source before drawing scientific conclusions from this test.

---

### Test 4 — Laminar Latency

**File:** `tests/laminar_latency_test.py`

**What it measures:** First-spike latency per layer (L4, L2/3, L5) and whether the canonical ordering L4 < L2/3 < L5 is preserved.

**Protocol:** A single pulse is delivered to C2. Mean first-spike latency is computed across all excitatory neurons in each layer. The test checks both the absolute latency values against in-vivo targets and the ordering constraint.

**Scoring:** If the ordering constraint is violated, an NAScore is returned with a description of the violation. Otherwise, the maximum absolute Z-score across the three layers is returned.

**Pass criterion:** Ordering holds AND max |Z| < 2.0

**Literature source:** Petersen et al. 2003, Neuron

> **Note for the Kremer model:** L4 in this model is a PoissonGroup (no integrate-and-fire dynamics, so no meaningful first-spike latency) and L5 is not implemented. This test will always return NAScore for those layers when run against the Kremer model. This is correct behaviour — the test is honestly reporting a model limitation.

---

### Test 5 — Spike-Count Adaptation

**File:** `tests/adaptation_test.py`

**What it measures:** The degree of short-term synaptic depression under repetitive whisker stimulation, quantified as the adaptation ratio.

**Protocol:** A train of 5 pulses is delivered at 8 Hz (inter-pulse interval = 125 ms). Spike counts are extracted in a 60 ms response window following each pulse onset. The adaptation ratio is:

```
ratio = spike count(pulse 5) / spike count(pulse 1)
```

The 60 ms window is chosen to be long enough to capture the full L2/3 response while being short enough to avoid overlap with the next pulse.

**Scoring:** Z-score against the in-vivo adaptation ratio.

**Pass criterion:** |Z| < 2.0

**Literature source:** Petersen 2002, Neuron

---

### Test 6 — Spontaneous Firing Rate

**File:** `tests/spontaneous_rate_test.py`

**What it measures:** Mean firing rate of L2/3 excitatory neurons in the C2 barrel during a no-stimulus period.

**Protocol:** No stimulus is applied. A 1000 ms silent settling period is run first to allow any residual post-stimulus activity to dissipate. The mean firing rate is then measured over a 2000 ms window.

**Scoring:** Z-score against the in-vivo spontaneous rate.

**Pass criterion:** |Z| < 2.0

**Literature source:** Brecht et al. 2003, J Physiol 553:243–265

> **Note:** This test measures L2/3 neurons (not L4) because L4 in the Kremer model is a Poisson source. The observation target (mean 1.0 Hz, std 0.5 Hz) reflects L2/3 spontaneous rates in anesthetised rat preparations.

---

## The Kremer et al. 2011 Model

The Kremer et al. 2011 model simulates the emergence of direction selectivity in barrel cortex through Spike-Timing-Dependent Plasticity (STDP).

**STDP** (Spike-Timing-Dependent Plasticity) is a biological learning rule where the strength of a synapse changes depending on the relative timing of pre- and post-synaptic spikes. If the pre-synaptic neuron fires shortly before the post-synaptic neuron, the synapse is strengthened (potentiation). If the order is reversed, the synapse is weakened (depression).

### Model architecture

| Component | Description |
|-----------|-------------|
| Layer 4 | PoissonGroup — direction-tuned Poisson firing rates driven by a moving bar stimulus |
| Layer 2/3 Exc | Integrate-and-fire neurons with adaptive threshold, arranged on a 2D grid |
| Layer 2/3 Inh | Integrate-and-fire inhibitory neurons |
| Feedforward | L4 → L2/3 Exc with STDP (on_pre / on_post rules in Brian 2 Synapses) |
| Recurrent Exc | Distance-dependent Gaussian connectivity within L2/3 (E→E and E→I) |
| Recurrent Inh | Distance-dependent Gaussian I→E connections |


### Brian 1 to Brian 2 port

The original Kremer 2011 code was written for Brian 1 (`from brian import *`). This library uses a Brian 2 port with the following key changes:

- `PoissonGroup.rate` → `PoissonGroup.rates` (units required)
- `Connection` + `ExponentialSTDP` → `Synapses` with on_pre / on_post STDP rules
- `SpikeMonitor.spiketimes` → `SpikeMonitor.spike_trains()`
- `layer4.subgroup(N)` → `layer4[start:end]`
- All numerical parameters require explicit Brian 2 units

---

## The Model Wrapper

The wrapper (`models/kremer_model_wrapper.py`) translates between the SciUnit capability API and the Brian 2 simulator without modifying the model's equations, synapses, or STDP rule.

### Design principle

The wrapper is a thin translation layer. It only controls `PoissonGroup.rates` to inject stimuli and reads `SpikeMonitor.spike_trains()` to extract results. Everything else — equations, connectivity, STDP weights — is left untouched.

### Barrel coordinate mapping

The Kremer model uses a grid indexed by `(row, col)`. Named barrels are mapped as:

```python
'C2' -> (centre, centre)       # principal barrel
'C3' -> (centre, centre + 1)   # adjacent barrel
```

### Warmup

The Kremer model develops direction selectivity through STDP and therefore requires a training period before validation. The `warmup()` method replicates the original moving-bar stimulation protocol:

```python
kremer.build()
kremer.warmup(duration_s=5)  # STDP training
run_barrel_validation(kremer)
```

After warmup, a 500 ms cooldown period is run with no stimulus before the trained snapshot is stored. Every validation trial then restores this trained-and-settled state.

---

## Observations and Literature Sources

All empirical targets are stored in `observations.py` with their units and citations. The `_obs()` helper in `run_validation.py` strips non-SciUnit fields (`source`, `units`) before passing to tests, while preserving them for printed output.

| Test | Mean | Std | Units | Source |
|------|------|-----|-------|--------|
| Latency | 7.5 | 0.8 | ms | Brecht & Sakmann 2002 |
| Direction Index | 2.18 | 0.50 | dimensionless | Wilent & Contreras 2005 |
| Spread delay | 5.5 | 0.6 | ms | Estebanez et al. 2018 ⚠️ |
| L4 latency | 7.5 | 0.8 | ms | Petersen et al. 2003 |
| L2/3 latency | 11.0 | 1.2 | ms | Petersen et al. 2003 |
| L5 latency | 14.5 | 1.5 | ms | Petersen et al. 2003 |
| Adaptation ratio | 0.35 | 0.08 | dimensionless | Petersen 2002 ⚠️ |
| Spontaneous rate | 1.0 | 0.5 | Hz | Brecht et al. 2003 ⚠️ |

⚠️ Values marked with this symbol are literature-informed estimates that have not been directly verified against the exact figure or table in the cited paper. They should be confirmed before using this library for publication-level validation.

---

## Running Validation

### Pass/fail thresholds

| Verdict | Criterion |
|---------|-----------|
| PASS | \|Z\| < 2.0 |
| BORDERLINE | 2.0 ≤ \|Z\| < 3.0 |
| FAIL | \|Z\| ≥ 3.0 |
| N/A | Model does not implement the required capability or produces no spikes |

### Output

Running `run_validation.py` prints a per-test summary and saves one figure per test to the `plots/` directory. 

### Reproducibility

The Kremer model uses stochastic Poisson neurons. To get reproducible results across runs, add a Brian 2 seed at the top of `run_validation.py`:

```python
from brian2 import seed
seed(42)
```

Remove this for production runs to verify robustness across random seeds.

### Kremer model results summary

| Test | Result | Interpretation |
| Population Latency | Variable (PASS/BORDERLINE) | L2/3 onset timing is approximately correct; variability from Poisson L4 |
| Directional Selectivity | BORDERLINE (DI ≈ 1.0) | Expected — the model is titled "Late emergence"; 5s warmup is insufficient for full tuning |
| Spatiotemporal Spread | FAIL | Model spread is faster than in-vivo; observation value also needs verification |
| Laminar Latency | N/A | Correct — L4 is Poisson, L5 absent in this model |
| Spike Adaptation | In progress | Response window fix applied |
| Spontaneous Rate | In progress | Settling period fix applied |

## Adding a New Model

To validate a new barrel cortex model, create a new wrapper class in `models/` that inherits from `sciunit.Model` and implements the required capabilities.

### Minimum capabilities for all six tests

```python
from tests.capabilities import (
    StimulusInput,    # apply_waveform()
    PopulationOutput, # get_barrel_spikes(), get_layer_spikes()
    SweepInput,       # apply_angular_suite()
    AngularOutput,    # get_binned_directional_matrix()
    RepetitiveInput,  # apply_pulse_train()
    SpontaneousOutput # get_spontaneous_rate()
)
```

### Minimal wrapper skeleton

```python
import sciunit
from tests.capabilities import StimulusInput, PopulationOutput

class MyBarrelModel(sciunit.Model, StimulusInput, PopulationOutput):

    def __init__(self, name='MyModel'):
        super().__init__(name=name)

    def apply_waveform(self, time_vector, amplitudes, barrel_mapping):
        # inject stimulus into your simulator
        pass

    def get_barrel_spikes(self, barrel_name):
        # return list of spike times in ms relative to stimulus onset
        pass

    def get_layer_spikes(self, layer, cell_type='Exc'):
        # return {neuron_idx: np.ndarray of spike times in ms}
        pass
```

Once your wrapper implements all required capabilities, pass it to `run_barrel_validation()`:

```python
from run_validation import run_barrel_validation
my_model = MyBarrelModel()
run_barrel_validation(my_model)
```

## Acknowledgements

This project was developed under the supervision of Dr Andrew Davison, NEUROPSI, CNRS Saclay. The validation framework is modelled after a visual cortex analysis pipeline developed by the lab. The Kremer et al. 2011 model was ported from Brian 1 to Brian 2 as part of this work.

**Key references:**

- Kremer et al. 2011 — Late emergence of the vibrissa direction selectivity map in the rat barrel cortex. PLOS ONE.
- Brecht & Sakmann 2002 — Dynamic representation of whisker deflection by synaptic potentials in spiny stellate and pyramidal cells. J Physiol 543:49–70.
- Wilent & Contreras 2005 — Stimulus-dependent changes in spike threshold enhance feature selectivity in rat barrel cortex neurons. J Neurosci 25(11):2983–2991.
- Petersen et al. 2003 — Interaction of sensory responses with spontaneous depolarization in layer 2/3 barrel cortex. Neuron.
- Petersen 2002 — Short-term dynamics of synaptic transmission within the excitatory neuronal network of rat layer 4 barrel cortex. J Neurophysiol.
- Estebanez et al. 2018 — A focal cortical dysfunction and its rescue identify a common pathway for sensorimotor integration. J Neurosci.
