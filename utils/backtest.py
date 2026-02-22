import pandas as pd
import numpy as np
import math
from engine.pipeline_stable import ForecastingPipeline
from engine.scenario_stable import ScenarioEngine

def run_expanding_window_backtest(df_raw, n_test=5, mode='strict'):
    """
    Leakage-safe backtesting using expanding window.
    'strict' mode re-fits the model at each step.
    'fast' mode fits once on (N - n_test) and uses fixed coefficients.
    """
    results = []
    actual_years = df_raw['year_end'].iloc[-n_test:].tolist()
    
    # If fast mode, fit once at the start
    pipeline_fast = None
    if mode == 'fast':
        initial_train = df_raw.iloc[:len(df_raw) - n_test]
        pipeline_fast = ForecastingPipeline(initial_train)
        pipeline_fast.run_full_pipeline()

    # We will forecast 1-step ahead for each year in the test set
    for i in range(len(df_raw) - n_test, len(df_raw)):
        train_df = df_raw.iloc[:i]
        actual_row = df_raw.iloc[i]
        target_year = actual_row['year_end']
        
        if mode == 'strict':
            # Fit models on training data every time
            pipeline = ForecastingPipeline(train_df)
            pipeline.run_full_pipeline()
        else:
            pipeline = pipeline_fast
            
        # Scenario Engine for the test point
        # We use ACTUAL macro realization for the target year
        engine = ScenarioEngine(train_df, pipeline.best_models, pipeline.channel_models)
        
        # Calculate realized growth rates for macro targets
        prev_log_gdp = train_df['log_gdp'].iloc[-1]
        g_rate = math.exp(actual_row['log_gdp'] - prev_log_gdp) - 1
        
        prev_exrate = train_df['exrate'].iloc[-1]
        fx_rate = (actual_row['exrate'] / prev_exrate) - 1
        
        scenario_targets = {
            'gdp_growth': g_rate,
            'inflation': actual_row['inflation'],
            'exrate_growth': fx_rate,
            'policy_rate': actual_row['policy rate'],
            'active_dummies': [],
            'all_dummies': pipeline.dummies['all']
        }
        
        # Forecast 1 step
        fc_results, _, _ = engine.run_scenario(1, scenario_targets)
        
        # Collect results (units are PKR Billion)
        summary = {'year': target_year}
        for head in fc_results.keys():
            summary[f'{head}_pred'] = fc_results[head]['scenario'].iloc[0]
            
            if head in actual_row.index:
                summary[f'{head}_actual'] = actual_row[head] / 1000.0
            elif head == 'total':
                # Sum components if total column is missing
                actual_sum = sum([actual_row.get(h, 0) for h in ['dt', 'gst', 'fed', 'customs']])
                summary['total_actual'] = actual_sum / 1000.0
            else:
                summary[f'{head}_actual'] = 0.0
            
        results.append(summary)
        
    return pd.DataFrame(results)

def compute_metrics(actuals, preds):
    """Standard econometric evaluation metrics."""
    actuals = np.array(actuals)
    preds = np.array(preds)
    
    mask = (actuals != 0) & (~np.isnan(actuals))
    actuals = actuals[mask]
    preds = preds[mask]
    
    if len(actuals) == 0: return {}
    
    err = preds - actuals
    mae = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err**2))
    smape = 100 * np.mean(2 * np.abs(err) / (np.abs(actuals) + np.abs(preds)))
    
    # Directional Accuracy
    def get_signs(v):
        return np.sign(np.diff(np.concatenate([[v[0]], v])))
    
    # Simplified sign check for YoY change if history allowed, here just comparing pred vs actual sign relative to train end
    # For now, just basic RMSE%
    rmse_pct = (rmse / np.mean(actuals)) * 100
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'sMAPE%': smape,
        'RMSE%': rmse_pct
    }
