import pandas as pd
import numpy as np
import math

def calculate_log_path(last_val, growth_rates):
    """
    Computes level path (in logs) based on growth rate targets.
    Formula: log_val[t] = log_val[t-1] + ln(1 + growth/100)
    """
    path = []
    current_log = math.log(last_val)
    for g in growth_rates:
        # Avoid math domain error on negative growth close to -100
        delta = math.log(max(1 + g/100.0, 0.001))
        current_log += delta
        path.append(current_log)
    return path

def generate_default_path(horizon, gdp=10.8, inf=6.1, fx=1.0, rate=11.2):
    """Generates a default flat target path for the given horizon."""
    df = pd.DataFrame({
        'Year': range(1, horizon + 1),
        'Nominal GDP Growth (%)': [gdp] * horizon,
        'Inflation (%)': [inf] * horizon,
        'Exchange Rate Depr (%)': [fx] * horizon,
        'Policy Rate (%)': [rate] * horizon
    })
    return df
