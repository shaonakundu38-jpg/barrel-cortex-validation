"""
run_validation.py: 
Validation pipeline for the rodent barrel cortex model.

  1. Resolve which sheets / layers are available in the model.
  2. Run all analysis (tests) against the model.
  3. Produce all figures alongside the scores.
  4. Print a formatted pass/fail summary table.
 
Pass criterion  : |Z-score| < 2.0  (within 2 S.D mean)
Borderline      : 2.0 <= |Z| < 3.0
Fail            : |Z| >= 3.0
"""
 
import sciunit
from sciunit.scores import ZScore, NAScore
 
from observations import OBSERVATIONS
 
from models.dummy_barrel_model import UnifiedBarrelModel, BadLatencyModel
 
from tests.latency_test          import LatencyTest
from tests.tuning_test           import TuningTest
from tests.spatiotemporal_test   import SpatiotemporalTest
from tests.laminar_latency_test  import LaminarLatencyTest
from tests.adaptation_test       import AdaptationTest
from tests.spontaneous_rate_test import SpontaneousRateTest
 
 # Pass/fail thresholds
Z_PASS       = 2.0
Z_BORDERLINE = 3.0

def _obs(key):
    """Return only the mean/std keys SciUnit's ZScore validator accepts."""
    o = OBSERVATIONS[key]
    return {'mean': o['mean'], 'std': o['std']}
 
 
def _laminar_obs():
    """Return the nested laminar observation with only mean/std per layer."""
    o = OBSERVATIONS['laminar_latency']
    return {
        'L4':  {'mean': o['L4']['mean'],  'std': o['L4']['std']},
        'L23': {'mean': o['L23']['mean'], 'std': o['L23']['std']},
        'L5':  {'mean': o['L5']['mean'],  'std': o['L5']['std']},
    }
 
 # Helpers
 
def _verdict(score):
    """
    Return a human-readable verdict string for a SciUnit score.
 
    ZScore.__float__ succeeds for real scores; NAScore raises TypeError,
    which is how we detect a missing-spike result.
    """
    if isinstance(score, NAScore):
        return f'N/A  ({score})'
    try:
        z = abs(float(score))
    except (TypeError, ValueError):
        return 'N/A  (unscored)'
    if z < Z_PASS:
        return f'PASS  (|Z| = {z:.2f})'
    if z < Z_BORDERLINE:
        return f'BORDERLINE  (|Z| = {z:.2f})'
    return f'FAIL  (|Z| = {z:.2f})'
 
 
def _section(title):
    print('\n' + '=' * 65)
    print(f'  {title}')
    print('=' * 65)
 

# Analysis pipeline

def run_barrel_validation(model):
    """
    Run the full validation suite against a single model and return a
    dict of {test_name: score}.
    """
 
    _section(f'BARREL CORTEX VALIDATION  --  {model.name}')
 
    results = {}
 
    # TEST 1: Absolute Response Latency
    test_latency = LatencyTest(
        observation=_obs('latency'),
        name='Population_Latency',
    )
    score_1 = test_latency.judge(model)
    results[test_latency.name] = score_1
    print(f'\n[TEST 1] {test_latency.name}')
    print(f'         Source : {OBSERVATIONS["latency"]["source"]}')
    print(f'         Score  : {score_1}   ->  {_verdict(score_1)}')
 
    
    # TEST 2: Multi-Directional Selectivity
    test_tuning = TuningTest(
        observation=_obs('direction_selectivity'),
        name='Directional_Selectivity',
    )
    score_2 = test_tuning.judge(model)
    results[test_tuning.name] = score_2
    print(f'\n[TEST 2] {test_tuning.name}')
    print(f'         Source : {OBSERVATIONS["direction_selectivity"]["source"]}')
    print(f'         Score  : {score_2}   ->  {_verdict(score_2)}')
 
   
    # TEST 3: Spatiotemporal Lateral Propagation Delay
    test_spatio = SpatiotemporalTest(
        observation=_obs('spatiotemporal_spread'),
        name='Spatiotemporal_Spread',
    )
    score_3 = test_spatio.judge(model)
    results[test_spatio.name] = score_3
    print(f'\n[TEST 3] {test_spatio.name}')
    print(f'         Source : {OBSERVATIONS["spatiotemporal_spread"]["source"]}')
    print(f'         Score  : {score_3}   ->  {_verdict(score_3)}')
 
    
    # TEST 4: Laminar Latency Ordering
    test_laminar = LaminarLatencyTest(
        observation=_laminar_obs(),
        name='Laminar_Latency',
    )
    score_4 = test_laminar.judge(model)
    results[test_laminar.name] = score_4
    print(f'\n[TEST 4] {test_laminar.name}')
    print(f'         Source : {OBSERVATIONS["laminar_latency"]["source"]}')
    print(f'         Score  : {score_4}   ->  {_verdict(score_4)}')
 
    
    # TEST 5: Spike-Count Adaptation
    test_adapt = AdaptationTest(
        observation=_obs('adaptation_ratio'),
        name='Spike_Adaptation',
    )
    score_5 = test_adapt.judge(model)
    results[test_adapt.name] = score_5
    print(f'\n[TEST 5] {test_adapt.name}')
    print(f'         Source : {OBSERVATIONS["adaptation_ratio"]["source"]}')
    print(f'         Score  : {score_5}   ->  {_verdict(score_5)}')
 
    
    # TEST 6: Spontaneous Firing Rate
    test_spont = SpontaneousRateTest(
        observation=_obs('spontaneous_rate'),
        name='Spontaneous_Rate',
    )
    score_6 = test_spont.judge(model)
    results[test_spont.name] = score_6
    print(f'\n[TEST 6] {test_spont.name}')
    print(f'         Source : {OBSERVATIONS["spontaneous_rate"]["source"]}')
    print(f'         Score  : {score_6}   ->  {_verdict(score_6)}')
 
    
    # Summary
    
    _section('SUMMARY')
    passed = sum(
        1 for s in results.values()
        if isinstance(s, ZScore) and abs(s.score) < Z_PASS
    )
    print(f'  Passed : {passed} / {len(results)}  (|Z| < {Z_PASS})')
    print(f'  Plots  : see ./plots/\n')
 
    return results
 
 
if __name__ == '__main__':
    # --- Good model: should produce mostly PASS scores ---
    good_model = UnifiedBarrelModel(name='L4_Cortical_Sheet')
    run_barrel_validation(good_model)
 
    # --- Bad model: should produce FAIL on latency tests ---
    print('\n')
    bad_model = BadLatencyModel(name='BadLatencyModel')
    run_barrel_validation(bad_model)
 
