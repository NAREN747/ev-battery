"""
Feature Engineering for EV Battery Health Monitoring
Extracts statistical, physical, and historical features per cycle.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


DATA_DIR = Path(__file__).parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"


def ensure_dirs():
    """Create necessary directories."""
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)


def compute_soh(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute State of Health (SOH) from discharge capacity.
    SOH = current_capacity / initial_capacity * 100%
    """
    df = df.copy()
    
    # Get initial capacity per battery (first cycle)
    initial_capacity = df.groupby('battery_id')['capacity'].first().rename('initial_capacity')
    df = df.merge(initial_capacity, on='battery_id', how='left')
    
    # Compute SOH as percentage
    df['soh'] = (df['capacity'] / df['initial_capacity']) * 100
    
    # Capacity retention
    df['capacity_retention'] = df['capacity'] / df['initial_capacity']
    
    # Capacity fade (cumulative)
    df['capacity_fade'] = 1 - df['capacity_retention']
    
    # Capacity fade per cycle (degradation rate)
    df['cycle_degradation'] = df.groupby('battery_id')['capacity'].diff().fillna(0)
    df['cycle_degradation_pct'] = df.groupby('battery_id')['soh'].diff().fillna(0)
    
    return df


def compute_rul(df: pd.DataFrame, eol_threshold: float = 70.0, min_cycles: int = 20) -> pd.DataFrame:
    """
    Compute Remaining Useful Life (RUL) - cycles until SOH <= 70%.
    Only computes for batteries where EOL is observed or can be reasonably extrapolated.
    Filters out batteries with too few cycles.
    Returns both RUL and a flag indicating whether EOL was observed.
    """
    df = df.copy()
    
    # Filter batteries with sufficient cycles
    cycle_counts = df.groupby('battery_id')['cycle'].count()
    valid_batteries = cycle_counts[cycle_counts >= min_cycles].index.tolist()
    df = df[df['battery_id'].isin(valid_batteries)].copy()
    print(f"  Kept {len(valid_batteries)} batteries with >= {min_cycles} cycles (removed {len(cycle_counts) - len(valid_batteries)} batteries)")
    
    # For each battery, find the cycle where SOH crosses EOL threshold
    eol_cycles = {}
    for bid in df['battery_id'].unique():
        battery_data = df[df['battery_id'] == bid].sort_values('cycle')
        soh_values = battery_data['soh'].values
        cycles = battery_data['cycle'].values
        
        # Find first cycle where SOH <= threshold
        below_threshold = np.where(soh_values <= eol_threshold)[0]
        if len(below_threshold) > 0:
            eol_cycle = cycles[below_threshold[0]]
        else:
            # Extrapolate based on current degradation rate
            last_soh = soh_values[-1]
            last_cycle = cycles[-1]
            avg_degradation = np.mean(np.diff(soh_values)) if len(soh_values) > 1 else -0.1
            if avg_degradation < 0:
                cycles_to_eol = (last_soh - eol_threshold) / abs(avg_degradation)
                eol_cycle = last_cycle + int(cycles_to_eol)
            else:
                eol_cycle = last_cycle + 1000  # Arbitrary large number
        
        eol_cycles[bid] = eol_cycle
    
    # Map EOL cycle to each row
    df['eol_cycle'] = df['battery_id'].map(eol_cycles)
    
    # Compute RUL
    df['rul'] = df['eol_cycle'] - df['cycle']
    df['rul'] = df['rul'].clip(lower=0)  # No negative RUL
    
    # Mark whether EOL was observed in data
    observed_eol = {}
    for bid in df['battery_id'].unique():
        battery_data = df[df['battery_id'] == bid]
        observed_eol[bid] = (battery_data['soh'] <= eol_threshold).any()
    df['eol_observed'] = df['battery_id'].map(observed_eol)
    
    return df


def compute_degradation_severity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify degradation severity into Normal, Accelerated, Critical.
    Uses ONLY current cycle's degradation rate (observable), not full-lifetime statistics.
    """
    df = df.copy()
    
    # Use current cycle's absolute degradation rate (observable at this cycle)
    # Classify based on absolute thresholds, not population quantiles
    current_degradation = df['cycle_degradation_pct'].abs()
    
    # Absolute thresholds for degradation rate per cycle (% SOH per cycle)
    # These are domain-defined, not data-driven quantiles
    NORMAL_THRESH = 0.10      # < 0.1% SOH loss per cycle = Normal
    ACCELERATED_THRESH = 0.30  # 0.1-0.3% = Accelerated
    # > 0.3% = Critical
    
    def classify_severity(rate):
        if rate <= NORMAL_THRESH:
            return 'Normal'
        elif rate <= ACCELERATED_THRESH:
            return 'Accelerated'
        else:
            return 'Critical'
    
    df['degradation_severity'] = current_degradation.apply(classify_severity)
    
    # Also add numeric encoding
    severity_map = {'Normal': 0, 'Accelerated': 1, 'Critical': 2}
    df['degradation_severity_encoded'] = df['degradation_severity'].map(severity_map)
    
    return df, {'normal_thresh': NORMAL_THRESH, 'accelerated_thresh': ACCELERATED_THRESH}


def compute_failure_risk(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify failure risk into Low, Medium, High.
    Uses ONLY current/observable features (no retrospective RUL or full-lifetime severity).
    Uses FIXED thresholds (no dataset-dependent quantiles) to prevent leakage.
    Returns both the risk score (for reference) and the encoded label.
    """
    df = df.copy()
    
    # Risk factors based on CURRENT observable state only:
    # 1. Current SOH (lower = higher risk) - observable at current cycle
    # 2. Current degradation rate (cycle_degradation_pct) - observable trend
    # 3. Temperature stress (high temp = higher risk) - observable
    # 4. Voltage/current anomalies - observable
    
    # Normalize risk factors to 0-1 scale using FIXED thresholds
    soh_risk = 1 - (df['soh'] / 100)  # 0 at 100% SOH, 1 at 0% SOH
    
    # Degradation rate risk - use FIXED threshold (not dataset max)
    # Typical degradation: 0-0.5% SOH per cycle
    deg_rate = df['cycle_degradation_pct'].abs()
    deg_rate_risk = (deg_rate / 0.5).clip(0, 1)  # Fixed 0.5% threshold
    
    # Temperature stress - FIXED thresholds
    temp_risk = (df['temp_mean'] - 20) / 60  # Normalize assuming 20-80°C range
    temp_risk = temp_risk.clip(0, 1)
    
    # Voltage anomaly risk (deviation from nominal) - FIXED threshold
    voltage_risk = (df['voltage_mean'] - 3.7).abs() / 2.0  # Nominal ~3.7V
    voltage_risk = voltage_risk.clip(0, 1)
    
    # Weighted risk score (no RUL, no retrospective severity)
    risk_score = (
        0.40 * soh_risk +
        0.30 * deg_rate_risk +
        0.20 * temp_risk +
        0.10 * voltage_risk
    )
    
    df['failure_risk_score'] = risk_score
    
    # Classify into Low, Medium, High using FIXED thresholds (not dataset quantiles)
    # These are domain-defined thresholds
    LOW_THRESH = 0.30
    HIGH_THRESH = 0.65
    
    def classify_risk(score):
        if score <= LOW_THRESH:
            return 'Low'
        elif score <= HIGH_THRESH:
            return 'Medium'
        else:
            return 'High'
    
    df['failure_risk'] = df['failure_risk_score'].apply(
        lambda s: 'Low' if s <= 0.30 else ('Medium' if s <= 0.65 else 'High')
    )
    
    # Numeric encoding
    risk_map = {'Low': 0, 'Medium': 1, 'High': 2}
    df['failure_risk_encoded'] = df['failure_risk'].map(risk_map)
    
    return df, {'low_thresh': 0.30, 'high_thresh': 0.65}


def engineer_statistical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract statistical features from voltage, current, temperature."""
    df = df.copy()
    
    # Voltage features
    df['voltage_range'] = df['voltage_max'] - df['voltage_min']
    df['voltage_cv'] = df['voltage_std'] / (df['voltage_mean'] + 1e-6) if 'voltage_std' in df.columns else 0
    
    # Current features
    df['current_range'] = df['current_max'] - df['current_min']
    df['current_cv'] = df['current_std'] / (df['current_mean'].abs() + 1e-6) if 'current_std' in df.columns else 0
    
    # Temperature features
    df['temp_range'] = df['temp_max'] - df['temp_min']
    df['temp_rise'] = df['temp_max'] - df['ambient_temp']  # Self-heating indicator
    df['temp_cv'] = df['temp_std'] / (df['temp_mean'] + 1e-6) if 'temp_std' in df.columns else 0
    
    # Power-related features
    df['power_mean'] = df['voltage_mean'] * df['current_mean'].abs()
    df['energy_throughput'] = df['power_mean'] * df['duration'] / 3600  # Wh
    
    # Internal resistance proxy (voltage drop under load)
    if 'voltage_min' in df.columns and 'current_max' in df.columns:
        df['ir_proxy'] = (df['voltage_mean'] - df['voltage_min']) / (df['current_max'].abs() + 1e-6)
    
    return df


def engineer_historical_features(df: pd.DataFrame, windows: List[int] = [3, 5, 10]) -> pd.DataFrame:
    """
    Create historical features: rolling statistics, previous cycle values, cycle-to-cycle deltas.
    Uses ONLY past data (shift by 1) to prevent leakage.
    Excludes target-derived features (SOH, capacity, RUL) to prevent leakage.
    """
    df = df.copy()
    df = df.sort_values(['battery_id', 'cycle']).reset_index(drop=True)
    
    # Features to compute rolling stats for - ONLY raw sensor data
    # Excluded: capacity, soh, rul, capacity_retention, capacity_fade, cycle_degradation_pct, soh_*, capacity_*, rul_*
    feature_cols = [
        'voltage_mean', 'voltage_min', 'voltage_max',
        'current_mean', 'temp_mean', 'temp_max', 'duration',
        'ir_proxy' if 'ir_proxy' in df.columns else 'voltage_mean'
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]
    
    for window in windows:
        for col in feature_cols:
            # Rolling stats on PAST data only: shift by 1, then rolling
            shifted = df.groupby('battery_id')[col].shift(1)
            
            # Rolling mean on past data only
            df[f'{col}_roll_mean_{window}'] = shifted.groupby(df['battery_id']).transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            # Rolling std on past data only
            df[f'{col}_roll_std_{window}'] = shifted.groupby(df['battery_id']).transform(
                lambda x: x.rolling(window, min_periods=1).std().fillna(0)
            )
            # Rolling min/max on past data only
            df[f'{col}_roll_min_{window}'] = shifted.groupby(df['battery_id']).transform(
                lambda x: x.rolling(window, min_periods=1).min()
            )
            df[f'{col}_roll_max_{window}'] = shifted.groupby(df['battery_id']).transform(
                lambda x: x.rolling(window, min_periods=1).max()
            )
    
    # Previous cycle values (lag features) - ONLY for raw sensor data
    lag_cols = ['voltage_mean', 'current_mean', 'temp_mean']
    lag_cols = [c for c in lag_cols if c in df.columns]
    for col in lag_cols:
        df[f'{col}_lag1'] = df.groupby('battery_id')[col].shift(1)
        df[f'{col}_lag2'] = df.groupby('battery_id')[col].shift(2)
    
    # Cycle-to-cycle deltas - ONLY for raw sensor data
    for col in lag_cols:
        df[f'{col}_delta1'] = df.groupby('battery_id')[col].diff()
        df[f'{col}_delta2'] = df.groupby('battery_id')[col].diff(2)
    
    # Fill NaN values from lag/delta operations
    df = df.fillna(0)
    
    return df


def engineer_cycle_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer features related to cycle number and progression.
    Excludes future-dependent features (no max_cycle normalization).
    """
    df = df.copy()
    
    # Cycle squared (for non-linear degradation)
    df['cycle_squared'] = df['cycle'] ** 2
    df['cycle_cubed'] = df['cycle'] ** 3
    
    # Log cycle
    df['cycle_log'] = np.log1p(df['cycle'])
    
    # Cycles per day estimate (if duration available)
    if 'duration' in df.columns:
        df['cycles_per_day_est'] = 86400 / (df['duration'] + 1e-6)
    
    # REMOVED: cycle_normalized = cycle / max_cycle (uses future max_cycle)
    
    return df


def select_features(df: pd.DataFrame, target_cols: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Select final feature set for modeling.
    Returns feature matrix X and list of feature names.
    """
    # Exclude target columns and identifier columns
    exclude_cols = set(target_cols + [
        'battery_id', 'cycle', 'initial_capacity', 'eol_cycle', 'eol_observed',
        'degradation_severity', 'failure_risk',  # String labels
        'degradation_severity_encoded', 'failure_risk_encoded',  # Encoded labels
        # Leaky features that directly determine or are used to compute targets:
        'failure_risk_score',  # Used to compute failure_risk label - MAJOR LEAKAGE
        'soh', 'capacity_retention', 'capacity_fade',  # Direct SOH components
        'rul',  # Direct RUL target
        'capacity',  # Used to compute SOH
        'cycle_degradation', 'cycle_degradation_pct',  # Used to compute severity
        'temp_mean', 'voltage_mean',  # Used to compute failure_risk_score
        'power_mean', 'energy_throughput', 'ir_proxy', 'voltage_cv', 'temp_cv',  # Derived from leaked features
    ])
    
    feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype in ['float64', 'int64', 'float32', 'int32']]
    
    X = df[feature_cols].copy()
    
    # Handle any remaining inf or NaN
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return X, feature_cols


def main():
    """Main feature engineering pipeline."""
    print("=" * 60)
    print("EV Battery - Feature Engineering Pipeline")
    print("=" * 60)
    
    ensure_dirs()
    
    # Load processed cycle summary
    input_path = PROCESSED_DIR / "cycle_summary.csv"
    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        return
    
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} rows, {df['battery_id'].nunique()} batteries")
    
    # Step 1: Compute SOH
    print("\n1. Computing State of Health (SOH)...")
    df = compute_soh(df)
    
    # Step 2: Compute RUL
    print("2. Computing Remaining Useful Life (RUL)...")
    df = compute_rul(df, eol_threshold=70.0)
    
    # Step 3: Degradation severity classification
    print("3. Classifying degradation severity...")
    df, severity_thresholds = compute_degradation_severity(df)
    print(f"   Severity thresholds: Normal<={severity_thresholds['normal_thresh']:.6f}, Accelerated<={severity_thresholds['accelerated_thresh']:.6f}")
    print(f"   Distribution: {df['degradation_severity'].value_counts().to_dict()}")
    
    # Step 4: Failure risk classification
    print("4. Computing failure risk...")
    df, risk_thresholds = compute_failure_risk(df)
    print(f"   Risk thresholds: Low<={risk_thresholds['low_thresh']:.4f}, High<={risk_thresholds['high_thresh']:.4f}")
    print(f"   Distribution: {df['failure_risk'].value_counts().to_dict()}")
    
    # Step 5: Statistical features
    print("5. Engineering statistical features...")
    df = engineer_statistical_features(df)
    
    # Step 6: Historical features
    print("6. Engineering historical features...")
    df = engineer_historical_features(df, windows=[3, 5, 10])
    
    # Step 7: Cycle features
    print("7. Engineering cycle features...")
    df = engineer_cycle_features(df)
    
    # Step 8: Feature selection
    print("8. Selecting final feature set...")
    target_cols = ['soh', 'rul', 'degradation_severity_encoded', 'failure_risk_encoded']
    X, feature_names = select_features(df, target_cols)
    print(f"   Selected {len(feature_names)} features")
    
    # Save feature matrix and targets
    features_path = FEATURES_DIR / "features.csv"
    targets_path = FEATURES_DIR / "targets.csv"
    feature_list_path = FEATURES_DIR / "feature_names.txt"
    
    X.to_csv(features_path, index=False)
    df[target_cols].to_csv(targets_path, index=False)
    with open(feature_list_path, 'w') as f:
        f.write('\n'.join(feature_names))
    
    # Also save the full enriched dataframe
    enriched_path = FEATURES_DIR / "enriched_data.csv"
    df.to_csv(enriched_path, index=False)
    
    print(f"\nSaved features to {features_path}")
    print(f"Saved targets to {targets_path}")
    print(f"Saved enriched data to {enriched_path}")
    print(f"Feature names saved to {feature_list_path}")
    
    # Print feature summary
    print("\nFeature categories:")
    categories = {
        'Basic': [c for c in feature_names if not any(x in c for x in ['_roll_', '_lag', '_delta', '_normalized', '_squared', '_cubed', '_log'])],
        'Rolling': [c for c in feature_names if '_roll_' in c],
        'Lag': [c for c in feature_names if '_lag' in c],
        'Delta': [c for c in feature_names if '_delta' in c],
        'Cycle': [c for c in feature_names if 'cycle_' in c and '_roll_' not in c],
    }
    for cat, feats in categories.items():
        print(f"  {cat}: {len(feats)} features")
    
    print(f"\nTarget distributions:")
    for target in target_cols:
        if target in df.columns:
            print(f"  {target}: {df[target].describe().to_dict()}")
    
    return df, X, feature_names


if __name__ == "__main__":
    df, X, features = main()
    print("\nFeature engineering complete!")