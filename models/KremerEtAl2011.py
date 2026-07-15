from brian2 import *
import matplotlib.pyplot as plt

prefs.codegen.target = 'numpy'

# PARAMETERS
# Neuron numbers
M4, M23exc, M23inh = 22, 25, 12 # side of each barrel (in neurons)
N4, N23exc, N23inh = M4**2, M23exc**2, M23inh**2 # neurons per barrel
barrelarraysize = 5 # Choose 3 or 4 if memory error
Nbarrels = barrelarraysize**2

# Stimulation
stim_change_time = 5*ms
Fmax = .5 / stim_change_time # maximum firing rate in layer 4 (.5 spike / stimulation)

# Neuron parameters
taum, taue, taui = 10*ms, 2*ms, 25*ms
El = -70*mV
Vt, vt_inc, tauvt = -55*mV, 2*mV, 50*ms # adaptive threshold

# STDP
taup, taud = 5*ms, 25*ms
Ap, Ad = .05, -.04

# EPSPs/IPSPs
v_target = -70*mV # Helper for unit matching in Brian 2
EPSP, IPSP = 1*mV, -1*mV
EPSC = EPSP * (taue/taum)**(taum/(taue-taum))
IPSC = IPSP * (taui/taum)**(taum/(taui-taum))

# Model: IF with adaptive threshold (Brian 2 requires explicit units for every variable)
eqs = '''
dv/dt = (ge + gi + El - v) / taum : volt
dge/dt = -ge / taue                 : volt
dgi/dt = -gi / taui                 : volt
dvt/dt = (Vt - vt) / tauvt           : volt # adaptation
x                                   : 1
y                                   : 1
'''

# Tuning curve
tuning = lambda theta: clip(cos(theta), 0, inf) * Fmax

# Layer 4
layer4 = PoissonGroup(N4 * Nbarrels, rates=0*Hz)

barrels4 = {}
barrelindices = {}
for i in range(barrelarraysize):
    for j in range(barrelarraysize):
        idx = i * barrelarraysize + j
        start_idx = idx * N4
        end_idx = (idx + 1) * N4
        barrels4[(i, j)] = layer4[start_idx:end_idx]
        barrelindices[(i, j)] = slice(start_idx, end_idx)

barrels4active = dict((ij, False) for ij in barrels4)
layer4.add_attribute('selectivity')
layer4.selectivity = zeros(len(layer4))
for (i, j), inds in barrelindices.items():
    layer4.selectivity[inds] = linspace(0, 2*pi, N4)

# Layer 2/3
layer23 = NeuronGroup(Nbarrels * (N23exc + N23inh), model=eqs, 
                      threshold='v > vt', reset='v = El; vt += vt_inc', 
                      refractory=2*ms, method='exact')
layer23.v = El
layer23.vt = Vt

# Layer 2/3 excitatory
layer23exc = layer23[0 : Nbarrels * N23exc]
x_coord, y_coord = meshgrid(arange(M23exc) * 1. / M23exc, arange(M23exc) * 1. / M23exc)
x_coord, y_coord = x_coord.flatten(), y_coord.flatten()

barrels23 = {}
for i in range(barrelarraysize):
    for j in range(barrelarraysize):
        idx = i * barrelarraysize + j
        barrels23[(i, j)] = layer23exc[idx * N23exc : (idx + 1) * N23exc]
        barrels23[(i, j)].x = x_coord + i
        barrels23[(i, j)].y = y_coord + j

# Layer 2/3 inhibitory
layer23inh = layer23[Nbarrels * N23exc : ]
x_coord_inh, y_coord_inh = meshgrid(arange(M23inh) * 1. / M23inh, arange(M23inh) * 1. / M23inh)
x_coord_inh, y_coord_inh = x_coord_inh.flatten(), y_coord_inh.flatten()

barrels23inh = {}
for i in range(barrelarraysize):
    for j in range(barrelarraysize):
        idx = i * barrelarraysize + j
        barrels23inh[(i, j)] = layer23inh[idx * N23inh : (idx + 1) * N23inh]
        barrels23inh[(i, j)].x = x_coord_inh + i
        barrels23inh[(i, j)].y = y_coord_inh + j

print("Building synapses, please wait...")

# Feedforward connections with STDP modeled directly within Synapses
feedforward = Synapses(layer4, layer23exc,
                       model='''w : volt
                                dda/dt = -da/taup : 1 (event-driven)
                                ddb/dt = -db/taud : 1 (event-driven)''',
                       on_pre='''ge += w
                                 da += Ap
                                 w = clip(w + db*volt, 0*volt, EPSC)''',
                       on_post='''db += Ad
                                  w = clip(w + da*volt, 0*volt, EPSC)''')

# Connect structural barrels randomly with 50% sparseness
print("Connecting feedforward synapses...")
i_indices = []
j_indices = []

for i in range(barrelarraysize):
    for j in range(barrelarraysize):
        idx = i * barrelarraysize + j
        start_i = idx * N4
        end_i = (idx + 1) * N4
        start_j = idx * N23exc
        end_j = (idx + 1) * N23exc
        
        # For every source neuron in this Layer 4 barrel, sample 50% of the target neurons
        for s_i in range(start_i, end_i):
            # Flip a coin for each target neuron using a binomial selection
            n_connections = np.random.binomial(N23exc, 0.5)
            targets = np.random.choice(np.arange(start_j, end_j), size=n_connections, replace=False)
            
            i_indices.extend([s_i] * n_connections)
            j_indices.extend(targets)

# Pass matching, flat 1-dimensional arrays directly to Brian 2
feedforward.connect(i=np.array(i_indices), j=np.array(j_indices))
feedforward.w = EPSC * 0.5
print(f"Successfully created {len(feedforward)} feedforward synapses.")

# Excitatory lateral connections (E -> E and E -> I)
recurrent_exc = Synapses(layer23exc, layer23, model='w : volt', on_pre='ge += w')

# Connect E -> E (target indices j are less than Nbarrels*N23exc)
recurrent_exc.connect(condition='j < Nbarrels*N23exc',
                      p='0.15 * exp(-0.5 * (((x_pre - x_post)/0.4)**2 + ((y_pre - y_post)/0.4)**2))')

# Connect E -> I (target indices j are greater than or equal to Nbarrels*N23exc)
recurrent_exc.connect(condition='j >= Nbarrels*N23exc',
                      p='0.15 * exp(-0.5 * (((x_pre - x_post)/0.4)**2 + ((y_pre - y_post)/0.4)**2))')

# Assign weights dynamically based on target type
recurrent_exc.w['j < Nbarrels*N23exc'] = EPSC * 0.3
recurrent_exc.w['j >= Nbarrels*N23exc'] = EPSC

# Inhibitory lateral connections (I -> E)
recurrent_inh = Synapses(layer23inh, layer23exc, model='w : volt', on_pre='gi += w')
recurrent_inh.connect(p='exp(-0.5 * (((x_pre - x_post)/0.2)**2 + ((y_pre - y_post)/0.2)**2))')
recurrent_inh.w = IPSC
# Stimulation details
stimspeed = 1. / stim_change_time
direction = 0.0
stimzonecentre = ones(2) * barrelarraysize / 2.
stimcentre = zeros(2)
stimnorm = zeros(2)
stimradius = (11 * stim_change_time * stimspeed + 1) * .5
stimradius2 = stimradius**2

def new_direction():
    global direction, stimnorm, stimcentre
    direction = rand() * 2 * pi
    stimnorm[0], stimnorm[1] = cos(direction), sin(direction)
    stimcentre = stimzonecentre - stimnorm * stimradius

# Network Operation replacement to update sensory states at every time-step
@network_operation
def stimulation(t):
    global direction, stimcentre
    stimcentre += stimspeed * stimnorm * defaultclock.dt
    if sum((stimcentre - stimzonecentre)**2) > stimradius2:
        new_direction()
    for (i, j), b in barrels4.items():
        whiskerpos = array([i, j], dtype=float) + 0.5
        isactive = abs(dot(whiskerpos - stimcentre, stimnorm)) < .5
        if barrels4active[i, j] != isactive:
            barrels4active[i, j] = isactive
            if isactive:
                b.rates = tuning(layer4.selectivity[barrelindices[i, j]] - direction)
            else:
                b.rates = 0 * Hz

new_direction()

# Add monitors to read network states
spikemon = SpikeMonitor(layer23exc)

print("Starting simulation...")
run(5 * second, report='text')

# Plotting metrics
print("Plotting results...")
plt.figure()
# Extract structural connectivity mappings for alignment
weights_matrix = np.zeros((len(layer4), len(layer23exc)))
weights_matrix[feedforward.i, feedforward.j] = feedforward.w / volt

selectivity = []
for i in range(len(layer23exc)):
    w_vec = weights_matrix[:, i]
    complex_val = mean(w_vec * exp(layer4.selectivity * 1j))
    pref_dir = (arctan2(complex_val.imag, complex_val.real) % (2 * pi)) * 180. / pi
    selectivity.append(pref_dir)
selectivity = array(selectivity)

I = zeros((barrelarraysize * M23exc, barrelarraysize * M23exc))
ix = array(around(layer23exc.x * M23exc) - 1, dtype=int)
iy = array(around(layer23exc.y * M23exc) - 1, dtype=int)
I[iy, ix] = selectivity

plt.figure()
plt.imshow(I, cmap='hsv')
plt.colorbar(label='Preferred Direction (Degrees)')

max_ix = max(ix)
max_iy = max(iy)
for i in range(1, barrelarraysize + 1):
    plt.plot([i * max_ix / barrelarraysize, i * max_ix / barrelarraysize], [0, max_iy], 'k')
    plt.plot([0, max_ix], [i * max_iy / barrelarraysize, i * max_iy / barrelarraysize], 'k')
plt.title("Preferred Orientation Pinwheel Map")

# Save the map as an image file
plt.savefig('barrel_map.png', dpi=300)
print("Saved orientation map to barrel_map.png")

plt.figure()
plt.hist(selectivity, bins=20)
plt.xlabel('Preferred Direction')
plt.ylabel('Count')
plt.title("Distribution of Preferred Directions")

# Save the histogram as a separate image file
plt.savefig('direction_histogram.png', dpi=300)
print("Saved distribution histogram to direction_histogram.png")