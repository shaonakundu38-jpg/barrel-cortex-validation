"""
kremer_model_wrapper.py
-----------------------
Thin SciUnit wrapper around the Brian 2 port of the Kremer et al. 2011
barrel cortex model.

FIXES in this version:
- Added weight_scale parameter (default 0.3) to tone down synaptic weights
- Increased default warmup duration to 20 s for better STDP training
- Fixed missing self._spike_mon_l4 assignment
- Added automatic angle conversion (degrees→radians) in apply_angular_suite
- Debug print for L4 rate sums (commented out by default)
"""

import numpy as np
import sciunit

from tests.capabilities import (
    StimulusInput, PopulationOutput,
    SweepInput, AngularOutput,
    RepetitiveInput, SpontaneousOutput,
)

try:
    from brian2 import (
        PoissonGroup, NeuronGroup, Synapses, SpikeMonitor,
        Network, ms, mV, Hz, prefs, start_scope,
    )
    from numpy import zeros, linspace, meshgrid, arange, exp, cos, clip, pi, inf
    prefs.codegen.target = 'numpy'
    BRIAN_AVAILABLE = True
except ImportError:
    BRIAN_AVAILABLE = False

def _make_barrel_grid(barrelarraysize):
    centre = barrelarraysize // 2
    grid = {'C2': (centre, centre)}
    if centre + 1 < barrelarraysize:
        grid['C3'] = (centre, centre + 1)
    return grid


class KremerBarrelModel(
    sciunit.Model,
    StimulusInput, PopulationOutput,
    SweepInput, AngularOutput,
    RepetitiveInput, SpontaneousOutput,
):
    def __init__(self, barrelarraysize=5, weight_scale=0.8,
                 name='KremerEtAl2011_Brian2'):
        if not BRIAN_AVAILABLE:
            raise ImportError('Brian2 is required: pip install brian2')
        super().__init__(name=name)
        self.barrelarraysize = barrelarraysize
        self.weight_scale    = weight_scale   # <-- new: scale all weights
        self._barrel_grid    = _make_barrel_grid(barrelarraysize)
        self._built          = False
        self._warmed_up      = False
        self._barrel_spikes  = {}
        self._layer_spikes   = {}
        self._dir_matrix     = {}
        self._namespace      = {}

        if 'C3' not in self._barrel_grid:
            print(f'WARNING: barrelarraysize={barrelarraysize} is too small for C3. '
                  f'SpatiotemporalTest will return NAScore.')

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self):
        if self._built:
            return

        start_scope()
        size     = self.barrelarraysize
        Nbarrels = size ** 2

        # ---- model parameters ------------------------------------------
        M4, M23exc, M23inh = 22, 25, 12
        N4     = M4 ** 2
        N23exc = M23exc ** 2
        N23inh = M23inh ** 2

        taum, taue, taui = 10 * ms, 2 * ms, 25 * ms
        El               = -70 * mV
        Vt, vt_inc, tauvt = -55 * mV, 2 * mV, 50 * ms
        taup, taud       = 5 * ms, 25 * ms
        Ap, Ad           = 0.05, -0.04
        EPSP, IPSP       = 1 * mV, -1 * mV
        EPSC = EPSP * (taue / taum) ** (taum / (taue - taum))
        IPSC = IPSP * (taui / taum) ** (taum / (taui - taum))

        # ---- scale weights ---------------------------------------------
        ws = self.weight_scale
        EPSC_scaled = EPSC * ws
        IPSC_scaled = IPSC * ws   # inhibitory also scaled to keep balance

        # ---- save namespace --------------------------------------------
        self._namespace = dict(
            taum=taum, taue=taue, taui=taui,
            El=El, Vt=Vt, vt_inc=vt_inc, tauvt=tauvt,
            taup=taup, taud=taud, Ap=Ap, Ad=Ad,
            EPSC=EPSC_scaled, IPSC=IPSC_scaled,
            Nbarrels=Nbarrels, N23exc=N23exc,
        )

        # ---- stimulation constants -------------------------------------
        self._stim_change_time = 5 * ms
        Fmax = 0.5 / self._stim_change_time
        self._Fmax   = Fmax
        self._tuning = lambda theta: clip(cos(theta), 0, inf) * Fmax

        # ---- neuron equations ------------------------------------------
        eqs = '''
            dv/dt  = (ge+gi+El-v)/taum : volt
            dge/dt = -ge/taue          : volt
            dgi/dt = -gi/taui          : volt
            dvt/dt = (Vt-vt)/tauvt     : volt
            x : 1
            y : 1
        '''

        # ---- Layer 4 ---------------------------------------------------
        layer4 = PoissonGroup(N4 * Nbarrels, rates=0 * Hz)

        barrels4      = {}
        barrelindices = {}
        for i in range(size):
            for j in range(size):
                idx = i * size + j
                barrels4[(i, j)]      = layer4[idx * N4:(idx + 1) * N4]
                barrelindices[(i, j)] = slice(idx * N4, (idx + 1) * N4)

        layer4.add_attribute('selectivity')
        layer4.selectivity = zeros(len(layer4))
        for (i, j), inds in barrelindices.items():
            layer4.selectivity[inds] = linspace(0, 2 * pi, N4)

        # ---- Layer 2/3 -------------------------------------------------
        layer23 = NeuronGroup(
            Nbarrels * (N23exc + N23inh), model=eqs,
            threshold='v > vt', reset='v = El; vt += vt_inc',
            refractory=2 * ms, method='exact',
            namespace=self._namespace,
        )
        layer23.v  = El
        layer23.vt = Vt

        layer23exc = layer23[0:Nbarrels * N23exc]
        xc, yc = meshgrid(arange(M23exc) / M23exc, arange(M23exc) / M23exc)
        xc, yc = xc.flatten(), yc.flatten()
        barrels23 = {}
        for i in range(size):
            for j in range(size):
                idx = i * size + j
                barrels23[(i, j)] = layer23exc[idx * N23exc:(idx + 1) * N23exc]
                barrels23[(i, j)].x = xc + i
                barrels23[(i, j)].y = yc + j

        layer23inh = layer23[Nbarrels * N23exc:]
        xi, yi = meshgrid(arange(M23inh) / M23inh, arange(M23inh) / M23inh)
        xi, yi = xi.flatten(), yi.flatten()
        barrels23inh = {}
        for i in range(size):
            for j in range(size):
                idx = i * size + j
                barrels23inh[(i, j)] = layer23inh[idx * N23inh:(idx + 1) * N23inh]
                barrels23inh[(i, j)].x = xi + i
                barrels23inh[(i, j)].y = yi + j

        print('Building synapses, please wait...')

        # ---- Feedforward L4->L23 Exc (STDP) ----------------------------
        feedforward = Synapses(
            layer4, layer23exc,
            model='''w : volt
                     dda/dt = -da/taup : 1 (event-driven)
                     ddb/dt = -db/taud : 1 (event-driven)''',
            on_pre='''ge += w
                      da += Ap
                      w = clip(w + db*volt, 0*volt, EPSC)''',
            on_post='''db += Ad
                       w = clip(w + da*volt, 0*volt, EPSC)''',
            namespace=self._namespace,
        )
        i_idx, j_idx = [], []
        for i in range(size):
            for j in range(size):
                idx = i * size + j
                for s_i in range(idx * N4, (idx + 1) * N4):
                    n = np.random.binomial(N23exc, 0.5)
                    targets = np.random.choice(
                        np.arange(idx * N23exc, (idx + 1) * N23exc),
                        size=n, replace=False,
                    )
                    i_idx.extend([s_i] * n)
                    j_idx.extend(targets)
        feedforward.connect(i=np.array(i_idx), j=np.array(j_idx))
        feedforward.w = EPSC_scaled * 0.5   # scaled
        print(f'Feedforward synapses: {len(feedforward)}')

        # ---- Recurrent Exc ---------------------------------------------
        recurrent_exc = Synapses(
            layer23exc, layer23, model='w : volt', on_pre='ge += w',
            namespace=self._namespace,
        )
        recurrent_exc.connect(
            condition='j < Nbarrels*N23exc',
            p='0.15*exp(-0.5*(((x_pre-x_post)/0.4)**2+((y_pre-y_post)/0.4)**2))',
            namespace=self._namespace,
        )
        recurrent_exc.connect(
            condition='j >= Nbarrels*N23exc',
            p='0.15*exp(-0.5*(((x_pre-x_post)/0.4)**2+((y_pre-y_post)/0.4)**2))',
            namespace=self._namespace,
        )
        recurrent_exc.w['j < Nbarrels*N23exc']  = EPSC_scaled * 0.3
        recurrent_exc.w['j >= Nbarrels*N23exc'] = EPSC_scaled

        # ---- Recurrent Inh ---------------------------------------------
        recurrent_inh = Synapses(
            layer23inh, layer23exc, model='w : volt', on_pre='gi += w',
            namespace=self._namespace,
        )
        recurrent_inh.connect(
            p='exp(-0.5*(((x_pre-x_post)/0.2)**2+((y_pre-y_post)/0.2)**2))'
        )
        recurrent_inh.w = IPSC_scaled

        # ---- Monitors --------------------------------------------------
        spike_mon_exc = SpikeMonitor(layer23exc)
        spike_mon_inh = SpikeMonitor(layer23inh)
        spike_mon_l4  = SpikeMonitor(layer4)

        # ---- Explicit Network ------------------------------------------
        self._net = Network(
            layer4, layer23, feedforward,
            recurrent_exc, recurrent_inh,
            spike_mon_exc, spike_mon_inh, spike_mon_l4,
        )

        # Store references
        self._layer4        = layer4
        self._layer23exc    = layer23exc
        self._layer23inh    = layer23inh
        self._barrels4      = barrels4
        self._barrels23     = barrels23
        self._barrelindices = barrelindices
        self._spike_mon_exc = spike_mon_exc
        self._spike_mon_inh = spike_mon_inh
        self._spike_mon_l4  = spike_mon_l4
        self._N4            = N4
        self._N23exc        = N23exc
        self._N23inh        = N23inh

        self._net.store('clean')
        self._built = True
        print('Network built and snapshot stored.')

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fresh_trial(self):
        name = 'trial_start' if self._warmed_up else 'clean'
        self._net.restore(name)
        self._layer4.rates = 0 * Hz
        self._clear_monitors()

    def _activate_barrel(self, barrel_name, direction_rad=0.0):
        grid_pos    = self._barrel_grid[barrel_name]
        inds        = self._barrelindices[grid_pos]
        selectivity = self._layer4.selectivity[inds]
        rates = self._tuning(selectivity - direction_rad)
        self._barrels4[grid_pos].rates = rates

        # (Optional) uncomment to see that rates change with direction:
        # print(f"Direction {np.degrees(direction_rad):.0f}°: sum rates = {np.sum(rates):.1f} Hz")

    def _silence_barrel(self, barrel_name):
        self._barrels4[self._barrel_grid[barrel_name]].rates = 0 * Hz

    def _run(self, duration_ms):
        self._net.run(duration_ms * ms, namespace=self._namespace)

    def _clear_monitors(self):
        import numpy as np
        for mon in (self._spike_mon_exc, self._spike_mon_inh, self._spike_mon_l4):
            mon._mon_i = np.array([], dtype=int)
            mon._mon_t = np.array([], dtype=float)

    def _extract_barrel_spikes_ms(self, barrel_name, sim_start_ms=0.0):
        if barrel_name not in self._barrel_grid:
            return []
        grid_pos       = self._barrel_grid[barrel_name]
        global_indices = list(self._barrels23[grid_pos].i[:])
        trains         = self._spike_mon_exc.spike_trains()
        all_spikes     = []
        for idx in global_indices:
            if idx in trains:
                t_ms = np.array(trains[idx] / ms)
                t_ms = t_ms[t_ms >= sim_start_ms] - sim_start_ms
                all_spikes.extend(t_ms.tolist())
        return sorted(all_spikes)

    def _extract_layer_spikes_ms(self, layer, cell_type='Exc', sim_start_ms=0.0):
        if layer in ('L4', 'L5'):
            return {}
        monitor = self._spike_mon_exc if cell_type == 'Exc' else self._spike_mon_inh
        trains  = monitor.spike_trains()
        result  = {}
        for idx, times in trains.items():
            t_ms = np.array(times / ms)
            t_ms = t_ms[t_ms >= sim_start_ms] - sim_start_ms
            if len(t_ms) > 0:
                result[int(idx)] = t_ms
        return result

    # ------------------------------------------------------------------
    # StimulusInput
    # ------------------------------------------------------------------

    def apply_waveform(self, time_vector, amplitudes, barrel_mapping):
        self._require_built()
        self._fresh_trial()

        dt_ms    = float(time_vector[1] - time_vector[0])
        total_ms = float(time_vector[-1] - time_vector[0]) + dt_ms
        pulse_col = int(np.argmax(np.any(amplitudes > 0, axis=0)))
        onset_ms  = float(time_vector[pulse_col])
        target    = list(barrel_mapping.keys())[0]

        if onset_ms > 0:
            self._run(onset_ms)

        self._activate_barrel(target, direction_rad=0.0)
        self._run(float(self._stim_change_time / ms))

        self._silence_barrel(target)
        remainder = total_ms - onset_ms - float(self._stim_change_time / ms)
        if remainder > 0:
            self._run(remainder)

        self._barrel_spikes = {
            bname: self._extract_barrel_spikes_ms(bname, sim_start_ms=onset_ms)
            for bname in self._barrel_grid
        }
        self._layer_spikes = {
            'L23': self._extract_layer_spikes_ms('L23', 'Exc', onset_ms),
            'L4' : {},
            'L5' : {},
        }

    # ------------------------------------------------------------------
    # PopulationOutput
    # ------------------------------------------------------------------

    def get_barrel_spikes(self, barrel_name):
        return self._barrel_spikes.get(barrel_name, [])

    def get_layer_spikes(self, layer, cell_type='Exc'):
        return self._layer_spikes.get(layer, {})

    # ------------------------------------------------------------------
    # SweepInput
    # ------------------------------------------------------------------

    def apply_angular_suite(self, angles, time_vector, amplitude):
        self._require_built()
        self._dir_matrix = {}
        trial_ms = float(time_vector[-1] - time_vector[0])

        # auto‑convert degrees to radians if needed
        if np.any(np.array(angles) > 2 * np.pi):
            use_degrees = True
        else:
            use_degrees = False

        for angle in angles:
            if use_degrees:
                direction_rad = np.radians(angle)
            else:
                direction_rad = float(angle)

            self._fresh_trial()
            self._run(10.0)  # baseline
            self._activate_barrel('C2', direction_rad=direction_rad)
            self._run(float(self._stim_change_time / ms))
            self._silence_barrel('C2')
            remainder = max(0.0, trial_ms - 10.0 - float(self._stim_change_time / ms))
            if remainder > 0:
                self._run(remainder)
            self._dir_matrix[float(angle)] = self._extract_barrel_spikes_ms(
                'C2', sim_start_ms=10.0
            )

    # ------------------------------------------------------------------
    # AngularOutput
    # ------------------------------------------------------------------

    def get_binned_directional_matrix(self):
        return self._dir_matrix

    # ------------------------------------------------------------------
    # RepetitiveInput
    # ------------------------------------------------------------------

    def apply_pulse_train(self, barrel_name, n_pulses, isi_ms,
                          amplitude, duration_ms):
        self._require_built()
        self._fresh_trial()

        pulse_dur_ms = float(self._stim_change_time / ms)
        elapsed      = 0.0

        for pulse_idx in range(n_pulses):
            gap_ms = isi_ms - pulse_dur_ms if pulse_idx > 0 else isi_ms
            if gap_ms > 0:
                self._run(gap_ms)
                elapsed += gap_ms
            self._activate_barrel(barrel_name, direction_rad=0.0)
            self._run(pulse_dur_ms)
            elapsed += pulse_dur_ms
            self._silence_barrel(barrel_name)

        remainder = duration_ms - elapsed
        if remainder > 0:
            self._run(remainder)

        self._barrel_spikes[barrel_name] = self._extract_barrel_spikes_ms(
            barrel_name, sim_start_ms=0.0
        )

    # ------------------------------------------------------------------
    # SpontaneousOutput
    # ------------------------------------------------------------------

    def get_spontaneous_rate(self):
        self._require_built()
        self._fresh_trial()
        self._layer4.rates = 0 * Hz

        settle_ms = 1000.0
        self._run(settle_ms)

        self._clear_monitors()
        self._layer4.rates = 0 * Hz

        measure_ms = 2000.0
        self._run(measure_ms)

        grid_pos       = self._barrel_grid['C2']
        global_indices = list(self._barrels23[grid_pos].i[:])
        trains         = self._spike_mon_exc.spike_trains()
        total_spikes   = sum(len(trains[i]) for i in global_indices if i in trains)
        rate_hz        = total_spikes / (len(global_indices) * measure_ms / 1000.0)
        return float(rate_hz)

    # ------------------------------------------------------------------
    # Warmup (STDP training)
    # ------------------------------------------------------------------

    def warmup(self, duration_s=60):   # increased from 5 to 20 seconds
        self._require_built()
        print(f'Running {duration_s}s warmup (STDP training)...')
        self._net.restore('clean')

        import numpy as np
        from brian2 import Hz

        size = self.barrelarraysize
        dt_ms      = float(self._stim_change_time / ms)
        stimspeed  = 1.0 / dt_ms
        stimzonecentre = np.array([size / 2.0, size / 2.0])
        stimradius  = (11.0 * dt_ms * stimspeed + 1.0) * 0.5
        stimradius2 = float(stimradius ** 2)

        direction  = float(np.random.rand() * 2.0 * np.pi)
        stimnorm   = np.array([np.cos(direction), np.sin(direction)], dtype=float)
        stimcentre = stimzonecentre - stimnorm * stimradius

        n_steps = int(duration_s * 1000.0 / dt_ms)

        for step in range(n_steps):
            stimcentre = stimcentre + stimnorm * (stimspeed * dt_ms)

            if float(np.sum((stimcentre - stimzonecentre) ** 2)) > stimradius2:
                direction  = float(np.random.rand() * 2.0 * np.pi)
                stimnorm   = np.array([np.cos(direction), np.sin(direction)], dtype=float)
                stimcentre = stimzonecentre - stimnorm * stimradius

            for (i, j), barrel in self._barrels4.items():
                whiskerpos = np.array([float(i), float(j)]) + 0.5
                isactive   = abs(float(np.dot(whiskerpos - stimcentre, stimnorm))) < 0.5
                if isactive:
                    inds = self._barrelindices[(i, j)]
                    barrel.rates = self._tuning(
                        self._layer4.selectivity[inds] - direction
                    )
                else:
                    barrel.rates = 0 * Hz

            self._net.run(self._stim_change_time, namespace=self._namespace)

            if step % 200 == 0:
                print(f'  Warmup: {int(step / n_steps * 100)}%  ({step}/{n_steps} steps)')

        self._layer4.rates = 0 * Hz
    
        self._net.store('trial_start')
        self._warmed_up = True
        print('Warmup complete. Trained snapshot stored.')

    # ------------------------------------------------------------------

    def _require_built(self):
        if not self._built:
            raise RuntimeError('Call model.build() before running validation tests.')