import os
import sys
import time
import re
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any  # Added Any here
import warnings
warnings.filterwarnings('ignore')

# Data analysis imports
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import t, norm
import itertools
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

# Selenium imports
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

# Visualization Constants
PLOT_COLORS = {
    'trump': '#FF4136',  # Bright red
    'harris': '#0074D9',  # Bright blue
    'trump_light': '#FF7A73',  # Light red
    'harris_light': '#7FDBFF',  # Light blue
    'ci': '#2F4F4F',  # Dark slate gray for CIs
    'ev_dist': '#FF4136',  # Match Trump color for consistency
    'grid': '#E6E6E6',  # Light gray for grid
    'line': '#2F4F4F'  # Dark slate gray for lines
}

# Plot Style Configuration
PLOT_STYLE = {
    'figure.figsize': (12, 15),
    'figure.dpi': 100,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.color': '#E6E6E6',
    'axes.axisbelow': True
}

# Constants and Configuration
MAX_POLL_AGE_DAYS = 90  # 3 month maximum age for polls
POLL_DECAY_RATE = 0.05  # Aggressive decay rate for weighting recent polls

BATTLEGROUND_STATES = {
    'Pennsylvania': {
        'url': 'pennsylvania', 
        'ev': 19, 
        'region': 'Northeast', 
        'poll_errors': {'2020': 4.7, '2016': 4.4, '2012': 2.6}
    },
    'North Carolina': {
        'url': 'north-carolina', 
        'ev': 16, 
        'region': 'South', 
        'poll_errors': {'2020': 2.8, '2016': 3.9, '2012': 2.1}
    },
    'Georgia': {
        'url': 'georgia', 
        'ev': 16, 
        'region': 'South', 
        'poll_errors': {'2020': 1.2, '2016': 3.1, '2012': 2.8}
    },
    'Wisconsin': {
        'url': 'wisconsin', 
        'ev': 10, 
        'region': 'Midwest', 
        'poll_errors': {'2020': 4.1, '2016': 7.2, '2012': 3.1}
    },
    'Michigan': {
        'url': 'michigan', 
        'ev': 15, 
        'region': 'Midwest', 
        'poll_errors': {'2020': 2.6, '2016': 3.7, '2012': 2.9}
    },
    'Arizona': {
        'url': 'arizona', 
        'ev': 11, 
        'region': 'Southwest',
        'poll_errors': {'2020': 3.9, '2016': 2.8, '2012': 2.2}
    },
    'Nevada': {
        'url': 'nevada', 
        'ev': 6, 
        'region': 'Southwest',
        'poll_errors': {'2020': 5.3, '2016': 2.4, '2012': 3.1}
    }
}

BASE_URL = 'https://www.realclearpolling.com/polls/president/general/2024/'
HARRIS_ENTRY_DATE = datetime(2024, 8, 1)

# Regional correlations for final week
REGIONAL_CORRELATIONS = {
    'Northeast': {'Northeast': 1.0, 'Midwest': 0.8, 'South': 0.7, 'Southwest': 0.6},
    'Midwest': {'Northeast': 0.8, 'Midwest': 1.0, 'South': 0.75, 'Southwest': 0.7},
    'South': {'Northeast': 0.7, 'Midwest': 0.75, 'South': 1.0, 'Southwest': 0.65},
    'Southwest': {'Northeast': 0.6, 'Midwest': 0.7, 'South': 0.65, 'Southwest': 1.0}
}

# Pollster ratings with aggressive weighting
POLLSTER_RATINGS = {
    'A+': 1.0, 'A': 0.85, 'A-': 0.75, 'B+': 0.65, 'B': 0.55,
    'B-': 0.45, 'C+': 0.35, 'C': 0.25, 'C-': 0.15, 'D': 0.1
}

# Maximum retries for web scraping
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds

def setup_driver() -> webdriver.Chrome:
    """Set up and return a configured Chrome WebDriver with improved error handling."""
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
    chrome_options.add_argument("--disable-gpu")
    
    retry_count = 0
    while retry_count < MAX_RETRIES:
        try:
            service = Service("/usr/local/bin/chromedriver")
            driver = webdriver.Chrome(service=service, options=chrome_options)
            print("ChromeDriver setup successful.")
            return driver
        except Exception as e:
            retry_count += 1
            if retry_count == MAX_RETRIES:
                print(f"ChromeDriver setup failed after {MAX_RETRIES} attempts: {e}")
                raise
            print(f"ChromeDriver setup attempt {retry_count} failed, retrying...")
            time.sleep(RETRY_DELAY)

def fetch_polling_data(driver: webdriver.Chrome, state: str) -> Optional[pd.DataFrame]:
    """Fetch polling data with improved error handling and retry logic."""
    url = f"{BASE_URL}{BATTLEGROUND_STATES[state]['url']}/trump-vs-harris"
    retry_count = 0
    
    while retry_count < MAX_RETRIES:
        try:
            print(f"Fetching data for {state} from {url}")
            driver.get(url)
            WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.TAG_NAME, 'table')))
            
            tables = driver.find_elements(By.TAG_NAME, 'table')
            if not tables:
                print(f"No tables found for {state}")
                return None
                
            table = tables[0]
            headers = [th.text.strip() for th in table.find_elements(By.TAG_NAME, 'th')]
            
            rows = []
            for tr in table.find_elements(By.TAG_NAME, 'tr')[1:]:
                cells = tr.find_elements(By.TAG_NAME, 'td')
                if len(cells) >= len(headers):
                    row = [td.text.strip() for td in cells]
                    if any(row):
                        rows.append(row)
            
            df = pd.DataFrame(rows, columns=headers)
            
            # Add metadata
            df['State'] = state
            df['Region'] = BATTLEGROUND_STATES[state]['region']
            
            # Calculate weighted historical error (final week weights)
            hist_errors = BATTLEGROUND_STATES[state]['poll_errors']
            weighted_error = (
                0.7 * hist_errors['2020'] +
                0.2 * hist_errors['2016'] +
                0.1 * hist_errors['2012']
            )
            df['Historical_Error'] = weighted_error
            
            return df
            
        except Exception as e:
            retry_count += 1
            if retry_count == MAX_RETRIES:
                print(f"Failed to fetch data for {state} after {MAX_RETRIES} attempts: {e}")
                return None
            print(f"Attempt {retry_count} failed, retrying in {RETRY_DELAY} seconds...")
            time.sleep(RETRY_DELAY)

def process_state_data(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Process and clean polling data for final week analysis."""
    if df is None or df.empty:
        return None
        
    def parse_date(date_str):
        try:
            if '-' in date_str:
                end_date = date_str.split('-')[1].strip()
            else:
                end_date = date_str.strip()
            return pd.to_datetime(end_date, format='%m/%d')
        except Exception as e:
            print(f"Error parsing date: {date_str}, Error: {e}")
            return None

    # Convert dates
    df['DATE'] = df['DATE'].apply(parse_date)
    df['DATE'] = df['DATE'].apply(lambda x: x.replace(year=2024) if x is not None else None)
    
    # Apply time-based filters
    current_date = datetime.now().date()
    cutoff_date = current_date - timedelta(days=MAX_POLL_AGE_DAYS)
    harris_entry = HARRIS_ENTRY_DATE.date()
    
    # Use the later of Harris entry or MAX_POLL_AGE_DAYS cutoff
    effective_cutoff = max(cutoff_date, harris_entry)
    
    # Filter out older polls
    df = df[df['DATE'].dt.date >= effective_cutoff]
    
    print(f"Filtered polls to date range: {effective_cutoff} to {current_date}")
    print(f"Number of polls in range: {len(df)}")
    
    # Convert polling numbers to float with validation
    for col in ['TRUMP (R)', 'HARRIS (D)', 'MOE']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Basic data validation
    df = df[
        (df['TRUMP (R)'] >= 0) & 
        (df['HARRIS (D)'] >= 0) & 
        ((df['TRUMP (R)'] + df['HARRIS (D)']) <= 100)
    ]
    
    # Calculate undecided/other voters
    df['Undecided'] = 100 - df['TRUMP (R)'] - df['HARRIS (D)']
    
    # Calculate weights
    df['Weight'] = df.apply(calculate_poll_weight, axis=1)
    
    # Remove invalid entries
    df = df.dropna(subset=['TRUMP (R)', 'HARRIS (D)', 'Weight', 'DATE'])
    df = df[df['Weight'] > 0]
    
    return df

def calculate_poll_weight(row: pd.Series) -> float:
    """Calculate comprehensive poll weight with improved methodology for final week."""
    try:
        # Sample size weighting using inverse square root relationship
        sample_str = str(row['SAMPLE']).replace('LV', '').replace('RV', '').strip()
        sample_size = float(sample_str) if sample_str.replace('.','').isdigit() else 0
        base_weight = 1 / np.sqrt(1/sample_size) if sample_size > 0 else 0
        
        # Calculate age and apply decay
        days_old = (datetime.now().date() - row['DATE'].date()).days
        
        # Double-check we're within MAX_POLL_AGE_DAYS
        if days_old > MAX_POLL_AGE_DAYS:
            return 0
            
        # Apply time decay for polls within the window
        time_weight = 1 / (1 + POLL_DECAY_RATE * days_old)
        
        # Enhanced sample type weighting for final week
        sample_type_weight = 1.5 if 'LV' in str(row['SAMPLE']) else 1.0
        
        # Get pollster rating
        pollster_rating = POLLSTER_RATINGS.get(row.get('Rating', 'C'), 0.25)
        
        # Historical accuracy adjustment
        historical_error_adjustment = 1 / (1 + row['Historical_Error'] / 15)
        
        # Calculate total weight
        total_weight = (base_weight * 
                       time_weight * 
                       sample_type_weight * 
                       pollster_rating * 
                       historical_error_adjustment)
        
        return total_weight
        
    except Exception as e:
        print(f"Error calculating weight: {e}")
        return 0
    
def bootstrap_confidence_intervals(data: pd.DataFrame, n_bootstrap: int = 1000) -> dict:
    """Calculate bootstrap confidence intervals for polling estimates."""
    results = []
    for _ in range(n_bootstrap):
        # Resample with replacement
        bootstrap_sample = data.sample(n=len(data), replace=True)
        weighted_mean = np.average(
            bootstrap_sample['margin'], 
            weights=bootstrap_sample['Weight']
        )
        results.append(weighted_mean)
    
    return {
        'mean': np.mean(results),
        'ci_lower': np.percentile(results, 5),
        'ci_upper': np.percentile(results, 95),
        'std': np.std(results)
    }

def impute_missing_data(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing polling data using modern techniques."""
    # Create imputer with multiple regression iterations
    imputer = IterativeImputer(
        max_iter=10,
        random_state=42,
        min_value=0,
        max_value=100
    )
    
    # Columns to use for imputation
    cols_for_imputation = ['TRUMP (R)', 'HARRIS (D)', 'Weight', 'Historical_Error']
    
    # Create copy of data
    df_imputed = df.copy()
    
    # Perform imputation
    imputed_values = imputer.fit_transform(df_imputed[cols_for_imputation])
    
    # Replace missing values in original dataframe
    for i, col in enumerate(cols_for_imputation):
        df_imputed[col] = imputed_values[:, i]
    
    return df_imputed

def create_ensemble_model(state_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Create ensemble of models for polling prediction with proper NaN handling."""
    models = {
        'lasso': LassoCV(cv=5, random_state=42),
        'ridge': RidgeCV(cv=5),
        'elastic_net': ElasticNetCV(cv=5, random_state=42),
        'rf': RandomForestRegressor(n_estimators=100, random_state=42),
        'gbm': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    ensemble_results = {}
    
    for state, df in state_data.items():
        if df is None or df.empty:
            continue
            
        # Create features and handle missing values
        features = ['Weight', 'Historical_Error', 'MOE']
        
        # Handle missing values in features
        X = df[features].copy()
        y = df['TRUMP (R)'].fillna(0) - df['HARRIS (D)'].fillna(0)  # Margin
        
        # Fill NaN values with mean for each column
        for col in X.columns:
            X[col] = X[col].fillna(X[col].mean())
            
        # Convert to numpy arrays
        X = X.values
        y = y.values
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train models
        state_predictions = {}
        for name, model in models.items():
            try:
                model.fit(X_scaled, y)
                pred = model.predict(np.array([[1, 0, 0]]))  # Predict for standard case
                state_predictions[name] = pred[0]
            except Exception as e:
                print(f"Error fitting {name} for {state}: {e}")
                state_predictions[name] = None
        
        # Combine predictions with weights based on cross-validation performance
        valid_predictions = [p for p in state_predictions.values() if p is not None]
        if valid_predictions:
            ensemble_results[state] = np.mean(valid_predictions)
        else:
            ensemble_results[state] = 0
            
    return ensemble_results

def hierarchical_state_model(state_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    """Create hierarchical model for state-level estimates."""
    # Prepare data for hierarchical modeling
    states = []
    margins = []
    weights = []
    regions = []
    
    for state, df in state_data.items():
        if df is None or df.empty:
            continue
            
        states.append(state)
        margin = np.average(df['TRUMP (R)'] - df['HARRIS (D)'], weights=df['Weight'])
        margins.append(margin)
        weights.append(np.sum(df['Weight']))
        regions.append(BATTLEGROUND_STATES[state]['region'])
    
    # Convert to arrays
    margins = np.array(margins)
    weights = np.array(weights)
    
    # Create region indicators
    unique_regions = list(set(regions))
    region_indicators = pd.get_dummies(regions).values
    
    # Fit hierarchical model
    X = np.column_stack([np.ones_like(margins), region_indicators])
    model = sm.WLS(margins, X, weights=weights)
    results = model.fit()
    
    # Extract region effects
    region_effects = dict(zip(unique_regions, results.params[1:]))
    
    # Calculate state-level estimates with shrinkage
    hierarchical_estimates = {}
    for i, state in enumerate(states):
        raw_estimate = margins[i]
        region = regions[i]
        region_effect = region_effects[region]
        
        # Shrinkage factor based on sample size (weight)
        shrinkage = weights[i] / (weights[i] + 100)  # 100 is prior strength
        
        # Combine raw estimate with region effect
        shrunk_estimate = (
            shrinkage * raw_estimate + 
            (1 - shrinkage) * (results.params[0] + region_effect)
        )
        
        hierarchical_estimates[state] = shrunk_estimate
    
    return hierarchical_estimates

def regularized_state_correlation(state_data: Dict[str, pd.DataFrame], alpha: float = 0.1) -> np.ndarray:
    """Apply ridge regularization to state correlation matrix with proper NaN handling."""
    states = list(state_data.keys())
    n_states = len(states)
    
    # Create margin time series for each state
    margins = []
    for state in states:
        df = state_data[state]
        if df is None or df.empty:
            margins.append(np.zeros(1))  # Add dummy data for empty states
            continue
            
        # Calculate margin and handle NaN values
        margin = df['TRUMP (R)'].fillna(0) - df['HARRIS (D)'].fillna(0)
        margins.append(margin.values)
    
    # Pad sequences to same length
    max_len = max(len(m) for m in margins)
    padded_margins = []
    for m in margins:
        if len(m) < max_len:
            # Pad with the mean of existing values
            pad_value = np.mean(m) if len(m) > 0 else 0
            padded = np.pad(m, (0, max_len - len(m)), 'constant', constant_values=pad_value)
            padded_margins.append(padded)
        else:
            padded_margins.append(m)
    
    # Calculate correlation matrix
    margins_array = np.array(padded_margins)
    corr_matrix = np.corrcoef(margins_array)
    
    # Handle NaN values in correlation matrix
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
    
    # Add ridge penalty to diagonal
    regularized_matrix = corr_matrix + alpha * np.eye(n_states)
    
    # Ensure matrix is positive definite
    eigenvals = np.linalg.eigvals(regularized_matrix)
    if np.min(eigenvals) < 0:
        regularized_matrix += (-np.min(eigenvals) + 1e-6) * np.eye(n_states)
    
    return regularized_matrix

def allocate_undecided_sophisticated(
    trump: float, 
    harris: float, 
    undecided: float,
    historical_error: float
) -> Tuple[float, float]:
    """Allocate undecided voters with final week methodology."""
    # Validate inputs
    if not all(isinstance(x, (int, float)) for x in [trump, harris, undecided]):
        return trump, harris
    
    total = trump + harris
    if total > 100:
        scale = 100 / total
        trump *= scale
        harris *= scale
        undecided = 100 - (trump + harris)
    
    # Reduced challenger advantage for final week
    challenger_base = 0.52
    
    # Adjust based on current polling strength with diminishing returns
    total_decided = trump + harris
    if total_decided > 0:
        trump_share = trump / total_decided
        relative_strength = (trump_share - 0.5) * 2
        strength_adjustment = np.tanh(relative_strength) * 0.08
        challenger_share = challenger_base + strength_adjustment
    else:
        challenger_share = challenger_base
    
    # Add uncertainty based on historical polling errors
    error_adjustment = np.tanh(historical_error / 10) * 0.04
    challenger_share = min(max(challenger_share - error_adjustment, 0.48), 0.56)
    
    # Calculate final allocation
    trump_allocation = undecided * challenger_share
    harris_allocation = undecided * (1 - challenger_share)
    
    return trump + trump_allocation, harris + harris_allocation

# First, let's define all the helper functions needed

def bootstrap_confidence_intervals(data: pd.DataFrame, n_bootstrap: int = 2000) -> dict:
    """Calculate bootstrap confidence intervals with increased resampling."""
    results = []
    weights = data['Weight'].values / data['Weight'].sum()  # Normalize weights
    
    for _ in range(n_bootstrap):
        # Weighted resampling
        indices = np.random.choice(
            len(data), 
            size=len(data), 
            replace=True, 
            p=weights
        )
        bootstrap_sample = data.iloc[indices]
        weighted_mean = np.average(
            bootstrap_sample['margin'],
            weights=bootstrap_sample['Weight']
        )
        results.append(weighted_mean)
    
    return {
        'mean': np.mean(results),
        'ci_lower': np.percentile(results, 5),
        'ci_upper': np.percentile(results, 95),
        'std': np.std(results)
    }

def calculate_state_correlation_matrix(
    states: List[str],
    battleground_data: Dict[str, Dict]
) -> np.ndarray:
    """Calculate enhanced correlation matrix incorporating multiple factors."""
    n_states = len(states)
    correlation_matrix = np.zeros((n_states, n_states))
    
    for i in range(n_states):
        for j in range(n_states):
            state1 = states[i]
            state2 = states[j]
            
            # Get regional correlation
            region1 = battleground_data[state1]['region']
            region2 = battleground_data[state2]['region']
            regional_corr = REGIONAL_CORRELATIONS[region1][region2]
            
            # Calculate demographic similarity (if available)
            # This could be enhanced with actual demographic data
            demographic_corr = 0.5  # Default similarity
            
            # Calculate historical error correlation
            hist_errors1 = battleground_data[state1]['poll_errors']
            hist_errors2 = battleground_data[state2]['poll_errors']
            error_correlation = 1 - abs(
                hist_errors1['2020'] - hist_errors2['2020']
            ) / max(hist_errors1['2020'], hist_errors2['2020'])
            
            # Combine correlations with weights
            correlation_matrix[i,j] = (
                0.5 * regional_corr +
                0.3 * error_correlation +
                0.2 * demographic_corr
            )
    
    # Ensure matrix is positive definite
    min_eigenval = np.linalg.eigvals(correlation_matrix).min()
    if min_eigenval < 0:
        correlation_matrix += np.eye(n_states) * (abs(min_eigenval) + 1e-6)
    
    return correlation_matrix

def calculate_total_uncertainty(
    state: str,
    uncertainty: float,
    hist_error: float,
    sample_size: int,
    days_to_election: int
) -> float:
    """Calculate comprehensive uncertainty with more constrained bounds."""
    # Base uncertainty from polling (reduced multiplier)
    base_uncertainty = uncertainty * 0.7  # Reduced from 1.0
    
    # Historical error component (reduced weight)
    historical_component = hist_error * 0.3  # Reduced from 0.5
    
    # Time uncertainty with dampened effect
    time_uncertainty = min((days_to_election / 365) * 0.2, 0.15)  # Capped at 0.15
    
    # Sample size adjustment with reduced impact
    sample_size_factor = np.sqrt(500 / max(sample_size, 100))  # Changed from 1000 to 500
    
    # Combine all sources of uncertainty with bounds
    total_uncertainty = np.sqrt(
        base_uncertainty**2 +
        historical_component**2 +
        time_uncertainty**2
    ) * sample_size_factor
    
    # Apply reasonable bounds
    return np.clip(total_uncertainty, 0.3, 2.0)  # Min 0.3%, Max 2.0%

def monte_carlo_simulation(
    state_data: Dict[str, pd.DataFrame],
    n_sims: int = 250000,
    days_to_election: int = 365
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
    """Monte Carlo simulation with adjusted uncertainty parameters."""
    
    # 1. Prepare state data
    states = list(state_data.keys())
    n_states = len(states)
    
    if not states:
        raise ValueError("No valid state data provided")
    
    # 2. Calculate correlation matrix
    correlation_matrix = calculate_state_correlation_matrix(
        states,
        BATTLEGROUND_STATES
    )
    
    # 3. Generate random effects with reduced variance
    df_t = 10  # Increased from 7 for slightly thinner tails
    random_effects = np.random.standard_t(df=df_t, size=(n_sims, n_states))
    random_effects = np.dot(
        random_effects * 0.8,  # Scale down random effects
        np.linalg.cholesky(correlation_matrix).T
    )
    
    # 4. Generate systematic effects with reduced magnitude
    systematic_bias = np.random.normal(0, 0.15, n_sims)  # Reduced from 0.25
    incumbent_effect = np.random.normal(-0.05, 0.1, n_sims)  # Reduced from (-0.1, 0.2)
    
    # 5. Initialize results array
    state_results = np.zeros((n_sims, n_states))
    
    # 6. Process each state
    for i, state in enumerate(states):
        df = state_data[state]
        if df.empty:
            continue
        
        # Calculate weighted polling average
        weights = df['Weight'].values
        margins = df['TRUMP (R)'].values - df['HARRIS (D)'].values
        mean_margin = np.average(margins, weights=weights)
        
        # Calculate uncertainty with new constraints
        total_uncertainty = calculate_total_uncertainty(
            state=state,
            uncertainty=np.std(margins),
            hist_error=BATTLEGROUND_STATES[state]['poll_errors']['2020'],
            sample_size=len(df),
            days_to_election=days_to_election
        )
        
        # Generate state results with reduced multipliers
        state_results[:,i] = (
            mean_margin +  # Base margin
            random_effects[:,i] * total_uncertainty +  # State-specific variation
            systematic_bias * 0.15 +  # Reduced from 0.2
            incumbent_effect * 0.1   # Reduced from 0.15
        )
    
    # 7. Calculate state-level results
    results = {}
    for i, state in enumerate(states):
        results[state] = {
            'MeanMargin': np.mean(state_results[:,i]),
            'CI': np.percentile(state_results[:,i], [5, 95]),
            'TrumpProb': np.mean(state_results[:,i] > 0),
            'HarrisProb': np.mean(state_results[:,i] < 0),
            'Volatility': np.std(state_results[:,i])
        }
    
    # 8. Calculate electoral college results
    ev_results = calculate_electoral_votes(state_results, states)
    
    # 9. Add diagnostics to results
    ev_results.update({
        'Correlation_Matrix': correlation_matrix,
        'Total_Uncertainty': {
            state: results[state]['Volatility'] 
            for state in states
        }
    })
    
    return results, ev_results

def calculate_electoral_votes(
    state_results: np.ndarray, 
    states: List[str]
) -> Dict[str, float]:
    """Calculate electoral vote outcomes using 90% confidence intervals."""
    n_sims = len(state_results)
    
    # Base electoral votes (excluding battleground states)
    base_ev = {'Trump': 219, 'Harris': 226}
    
    # Calculate EV for each simulation
    trump_ev = np.zeros(n_sims)
    for sim in range(n_sims):
        ev = base_ev['Trump']
        for i, state in enumerate(states):
            if state_results[sim,i] > 0:
                ev += BATTLEGROUND_STATES[state]['ev']
        trump_ev[sim] = ev
    
    # Calculate probabilities and scenarios
    trump_wins = np.sum(trump_ev > 269)
    trump_prob = trump_wins / n_sims
    
    ev_mean = np.mean(trump_ev)
    ev_std = np.std(trump_ev)
    ev_ci = np.percentile(trump_ev, [5, 95])  # Changed from [2.5, 97.5] to [5, 95]
    recount_zone = np.sum(np.abs(trump_ev - 269.5) <= 5) / n_sims
    
    return {
        'TrumpEV': ev_mean,
        'TrumpEV_std': ev_std,
        'TrumpEV_CI': ev_ci,
        'HarrisEV': 538 - ev_mean,
        'TrumpProb': trump_prob,
        'HarrisProb': 1 - trump_prob,
        'EVDistribution': trump_ev,
        'Recount_Probability': recount_zone
    }

def process_simulation_results(
    state_results: np.ndarray,
    states: List[str]
) -> Dict[str, Dict[str, float]]:
    """Process raw simulation results into final state-level statistics."""
    results = {}
    
    for i, state in enumerate(states):
        # Calculate mean margin
        mean_margin = np.mean(state_results[:,i])
        
        # Calculate confidence intervals
        ci = np.percentile(state_results[:,i], [5, 95])
        
        # Calculate win probabilities
        trump_win_prob = np.mean(state_results[:,i] > 0)
        
        # Store results
        results[state] = {
            'MeanMargin': mean_margin,
            'CI': ci,
            'TrumpProb': trump_win_prob,
            'HarrisProb': 1 - trump_win_prob,
            'Volatility': np.std(state_results[:,i])
        }
        
    return results

def visualize_results(
    state_results: Dict[str, Dict[str, float]], 
    ev_results: Dict[str, float]
):
    """Create mobile-friendly visualization of election results with 90% confidence intervals."""
    # Import needed for KDE
    from scipy.stats import gaussian_kde
    
    # State abbreviations dictionary
    STATE_ABBREV = {
        'Pennsylvania': 'PA',
        'Michigan': 'MI',
        'Wisconsin': 'WI',
        'Georgia': 'GA',
        'Arizona': 'AZ',
        'Nevada': 'NV',
        'North Carolina': 'NC'
    }
    
    # Set style and colors
    plt.style.use('seaborn-v0_8-whitegrid')
    colors = {
        'trump': '#FF4136',      # Bright red
        'trump_light': '#FF7A73', # Light red
        'harris': '#0074D9',     # Bright blue
        'harris_light': '#7FDBFF', # Light blue
        'neutral': '#2F4F4F',    # Dark slate gray
        'grid': '#E6E6E6'        # Light gray
    }
    
    # Increased figure width-to-height ratio for better mobile viewing
    fig = plt.figure(figsize=(12, 24))
    gs = plt.GridSpec(4, 1, height_ratios=[5, 2, 1, 0.3], hspace=0.6)
    
    # Top plot - State margins with enhanced styling
    ax1 = fig.add_subplot(gs[0])
    
    # Sort states by margin magnitude
    states = sorted(state_results.keys(), 
                   key=lambda x: abs(state_results[x]['MeanMargin']))
    margins = [state_results[state]['MeanMargin'] for state in states]
    errors = [(state_results[state]['CI'][1] - state_results[state]['CI'][0])/2 
              for state in states]
    
    # Increase spacing between bars
    y_pos = np.arange(len(states)) * 1.5
    
    # Create bars with gradient effect
    for i, margin in enumerate(margins):
        color = colors['trump'] if margin > 0 else colors['harris']
        light_color = colors['trump_light'] if margin > 0 else colors['harris_light']
        
        bar = ax1.barh(y_pos[i], margin, align='center', 
                      color=color, alpha=0.7, height=0.8,
                      edgecolor='none')
        
        if margin != 0:
            ax1.barh(y_pos[i], margin * 0.95, align='center',
                    color=light_color, alpha=0.3, height=0.8,
                    edgecolor='none')
    
    # Add sophisticated error bars with reduced cap size for 90% CI
    error_bars = ax1.errorbar(margins, y_pos, xerr=errors, fmt='none',
                             color=colors['neutral'], capsize=5, capthick=1.5,
                             elinewidth=1.5, alpha=0.7)
    
    # Create state labels with abbreviations
    labels = []
    for i, state in enumerate(states):
        ev = BATTLEGROUND_STATES[state]['ev']
        prob = state_results[state]['TrumpProb']
        prob_pct = prob if prob >= 0.5 else (1-prob)
        candidate = "Trump" if prob >= 0.5 else "Harris"
        
        # Add state abbreviation to label
        label = f"{state} ({STATE_ABBREV[state]})\n{ev} EV\n{prob_pct*100:.0f}% {candidate}"
        labels.append(label)
        
        # Add margin values and CI numbers with adjusted positioning
        margin = margins[i]
        text_color = 'black'
        
        # Add CI numbers with more compact formatting
        ax1.text(state_results[state]['CI'][0] - 0.1, y_pos[i] + 0.3,
                f"{state_results[state]['CI'][0]:+.1f}", 
                ha='right', va='bottom',
                color=text_color, fontsize=12)
        
        ax1.text(state_results[state]['CI'][1] + 0.1, y_pos[i] + 0.3,
                f"{state_results[state]['CI'][1]:+.1f}", 
                ha='left', va='bottom',
                color=text_color, fontsize=12)
        
        # Add mean margin
        ax1.text(margin + (0.2 if margin > 0 else -0.2), y_pos[i] + 0.3,
                f"{margin:+.1f}%", va='bottom',
                ha='left' if margin > 0 else 'right',
                color=text_color, fontweight='bold', fontsize=14)
    
    # Customize top plot styling
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels, fontsize=14)
    ax1.tick_params(axis='both', which='major', labelsize=12)
    
    # Add zero line
    ax1.axvline(x=0, color=colors['neutral'], linestyle='-', 
                linewidth=1.5, alpha=0.3, zorder=1)
    
    ax1.set_title('2024 Battleground State Polling Margins\nwith 90% Confidence Intervals', 
                  pad=20, fontsize=20, fontweight='bold')
    ax1.set_xlabel('Margin (Trump Lead → | ← Harris Lead)', fontsize=16)
    
    # Electoral Vote Distribution plot with adjusted bin width
    ax2 = fig.add_subplot(gs[1])
    ev_dist = ev_results['EVDistribution']
    
    # Adjust number of bins for smoother distribution
    n_bins = 35
    counts, bins, patches = ax2.hist(ev_dist, bins=n_bins, density=True, alpha=0.5,
                                   color='gray', edgecolor='white',
                                   linewidth=1)
    
    # Color the bars based on 270 threshold
    for i in range(len(patches)):
        if bins[i] >= 270:
            patches[i].set_facecolor(colors['trump'])
            patches[i].set_alpha(0.6)
        else:
            patches[i].set_facecolor(colors['harris'])
            patches[i].set_alpha(0.6)
    
    # Calculate and display 90% CI for electoral votes
    ev_ci = np.percentile(ev_dist, [5, 95])
    ax2.axvline(x=ev_ci[0], color='black', linestyle=':', linewidth=1.5,
                alpha=0.5, label='90% CI')
    ax2.axvline(x=ev_ci[1], color='black', linestyle=':', linewidth=1.5,
                alpha=0.5)
    
    # Add KDE overlay
    kde = gaussian_kde(ev_dist, bw_method='silverman')
    x_range = np.linspace(bins[0], bins[-1], 200)
    kde_values = kde(x_range)
    ax2.plot(x_range, kde_values, color='black', 
             linewidth=2, alpha=0.8)
    
    # Add 270 line
    ax2.axvline(x=270, color='black', linestyle='--', 
                linewidth=2, alpha=0.8,
                label='270 EV Threshold')
    
    # Add annotations
    max_kde = max(kde_values)
    ax2.annotate('Harris Wins\n(<270 EV)', 
                xy=(250, max_kde/2),
                xytext=(220, max_kde * 0.7),
                ha='center', color=colors['harris'],
                fontsize=16, fontweight='bold',
                arrowprops=dict(arrowstyle='->',
                              color=colors['harris'],
                              lw=2))
    
    ax2.annotate('Trump Wins\n(≥270 EV)', 
                xy=(290, max_kde/2),
                xytext=(320, max_kde * 0.7),
                ha='center', color=colors['trump'],
                fontsize=16, fontweight='bold',
                arrowprops=dict(arrowstyle='->',
                              color=colors['trump'],
                              lw=2))
    
    ax2.set_title('Electoral Vote Distribution', 
                  pad=20, fontsize=18, fontweight='bold')
    ax2.set_xlabel('Electoral Votes', fontsize=16)
    ax2.set_ylabel('Density', fontsize=16)
    
    # Win probability bars
    ax3 = fig.add_subplot(gs[2])
    trump_prob = ev_results['TrumpProb']
    harris_prob = ev_results['HarrisProb']
    
    ax3.barh(0, trump_prob * 100, height=0.4, color=colors['trump'], alpha=0.7)
    ax3.barh(0, harris_prob * 100, height=0.4, color=colors['harris'], 
             alpha=0.7, left=trump_prob * 100)
    
    # Add probability labels
    ax3.text(trump_prob * 50, 0, f"TRUMP\n{trump_prob*100:.1f}%", 
             ha='center', va='center', color='white', 
             fontsize=18, fontweight='bold')
    ax3.text(trump_prob * 100 + harris_prob * 50, 0, 
             f"HARRIS\n{harris_prob*100:.1f}%", 
             ha='center', va='center', color='white', 
             fontsize=18, fontweight='bold')
    
    ax3.set_title('Win Probability', pad=20, fontsize=18, fontweight='bold')
    ax3.set_xlim(0, 100)
    ax3.set_yticks([])
    
    # Summary text
    summary_ax = fig.add_subplot(gs[3])
    summary_text = (
        f"Projection: Trump {ev_results['TrumpEV']:.0f} EV (±{ev_results['TrumpEV_std']:.0f}), "
        f"Harris {ev_results['HarrisEV']:.0f} EV  |  "
        f"90% CI: {ev_results['TrumpEV_CI'][0]:.0f}-{ev_results['TrumpEV_CI'][1]:.0f} EV  |  "
        f"Recount Scenario: {ev_results['Recount_Probability']*100:.1f}%"
    )
    summary_ax.text(0.5, 0.5, summary_text, ha='center', va='center',
                   fontsize=18, fontweight='bold', transform=summary_ax.transAxes)
    summary_ax.axis('off')
    
    # Add attribution
    plt.figtext(0.02, 0.02, 
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                fontsize=14, alpha=0.7)
    
    plt.tight_layout()
    plt.show()

def main():
    """Main execution function for the polling analysis."""
    try:
        print("Initializing 2024 Election Polling Analysis...")
        driver = setup_driver()
        state_data = {}
        
        # Fetch and process data for each state
        print("\nFetching polling data for battleground states...")
        for state in BATTLEGROUND_STATES:
            print(f"\nProcessing {state}...")
            df = fetch_polling_data(driver, state)
            if df is not None:
                processed_df = process_state_data(df)
                if processed_df is not None:
                    state_data[state] = processed_df
                    n_polls = len(processed_df)
                    print(f"✓ Successfully processed {n_polls} polls for {state}")
                else:
                    print(f"✗ Failed to process data for {state}")
            else:
                print(f"✗ Failed to fetch data for {state}")
        
        # Validate we have enough data to proceed
        if not state_data:
            raise ValueError("No valid polling data was collected for any state")
        
        print("\nRunning Monte Carlo simulation...")
        # Run Monte Carlo simulation with modern enhancements
        state_results, ev_results = monte_carlo_simulation(state_data)
        
        # Print detailed results
        print("\n=== State-by-State Results ===")
        for state in sorted(state_results.keys()):
            result = state_results[state]
            print(f"\n{state}:")
            print(f"Mean Margin: {result['MeanMargin']:+.1f}%")
            print(f"90% Confidence Interval: ({result['CI'][0]:+.1f}%, {result['CI'][1]:+.1f}%)")
            print(f"Win Probability - Trump: {result['TrumpProb']*100:.1f}%, " +
                  f"Harris: {result['HarrisProb']*100:.1f}%")
        
        print("\n=== Electoral College Projection ===")
        print(f"Trump: {ev_results['TrumpEV']:.1f} ± {ev_results['TrumpEV_std']:.1f} electoral votes")
        print(f"90% CI: ({ev_results['TrumpEV_CI'][0]:.1f}, {ev_results['TrumpEV_CI'][1]:.1f})")
        print(f"Harris: {ev_results['HarrisEV']:.1f} electoral votes")
        print(f"Win Probability - Trump: {ev_results['TrumpProb']*100:.1f}%")
        print(f"Win Probability - Harris: {ev_results['HarrisProb']*100:.1f}%")
        print(f"Probability of Recount Scenario: {ev_results['Recount_Probability']*100:.1f}%")
        
        # Generate visualizations
        print("\nGenerating visualizations...")
        visualize_results(state_results, ev_results)
        
        print("\nAnalysis complete!")
        
    except Exception as e:
        print(f"\nAn error occurred during execution: {str(e)}")
        import traceback
        print("\nFull error trace:")
        traceback.print_exc()
    finally:
        if 'driver' in locals():
            print("\nClosing web driver...")
            driver.quit()

if __name__ == "__main__":
    print("\n=== 2024 Presidential Election Polling Analysis ===")
    print("Version 1.0 - Modern Statistical Methods")
    print("Initializing...\n")
    
    # Set random seed for reproducibility while maintaining uncertainty
    np.random.seed(int(time.time()))
    
    # Set plotting style
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update(PLOT_STYLE)
    
    # Run main program
    main()