import pandas as pd
import numpy as np

def calculate_approx_decomposition(head, model, scenario_targets, baseline_targets):
    """
    Basic approximation of change drivers using Coeff * ΔShock logic.
    Labeled as approximation because it doesn't account for recursive feedback 
    through sectoral channels, only first-order impacts on the tax head equation.
    """
    if not hasattr(model, 'params'):
        return pd.DataFrame()
    
    params = model.params
    # scenario_targets are for year 1 (approximation)
    # We compare scenario vs baseline
    
    # 1. GDP Effect
    g_scen = scenario_targets.get('gdp_growth', 0)
    g_base = baseline_targets.get('gdp_growth', 0)
    # Since model is log-log, coeff is elasticity
    g_coeff = params.get('log_gdp_hat', params.get('log_gdp', 0))
    g_impact = g_coeff * (g_scen - g_base)
    
    # 2. Inflation Effect (Semi-elasticity)
    inf_scen = scenario_targets.get('inflation', 0)
    inf_base = baseline_targets.get('inflation', 0)
    inf_coeff = params.get('inflation', 0)
    inf_impact = inf_coeff * (inf_scen - inf_base)
    
    # 3. FX Effect
    fx_scen = scenario_targets.get('exrate_growth', 0)
    fx_base = baseline_targets.get('exrate_growth', 0)
    fx_coeff = params.get('log_exrate', 0)
    fx_impact = fx_coeff * (fx_scen - fx_base)
    
    # 4. Policy Rate Effect
    pr_scen = scenario_targets.get('policy_rate', 0)
    pr_base = baseline_targets.get('policy_rate', 0)
    pr_coeff = params.get('policy rate', 0)
    pr_impact = pr_coeff * (pr_scen - pr_base)
    
    # Standardize result in percentage contribution to the delta
    total = g_impact + inf_impact + fx_impact + pr_impact
    if abs(total) < 1e-6: total = 1.0 # Avoid div by zero
    
    rows = [
        {'Driver': 'GDP Growth', 'Impact': g_impact},
        {'Driver': 'Inflation', 'Impact': inf_impact},
        {'Driver': 'Exchange Rate', 'Impact': fx_impact},
        {'Driver': 'Policy Rate', 'Impact': pr_impact}
    ]
    
    return pd.DataFrame(rows)
