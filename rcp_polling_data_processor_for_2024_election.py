import os
import sys
import time
import re
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Data analysis imports
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import t  # Using t-distribution for fat tails
import itertools
import matplotlib.pyplot as plt

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
    'figure.figsize': (12, 15),  # Increased height
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

# Updated regional correlations for final week
REGIONAL_CORRELATIONS = {
    'Northeast': {'Northeast': 1.0, 'Midwest': 0.8, 'South': 0.7, 'Southwest': 0.6},
    'Midwest': {'Northeast': 0.8, 'Midwest': 1.0, 'South': 0.75, 'Southwest': 0.7},
    'South': {'Northeast': 0.7, 'Midwest': 0.75, 'South': 1.0, 'Southwest': 0.65},
    'Southwest': {'Northeast': 0.6, 'Midwest': 0.7, 'South': 0.65, 'Southwest': 1.0}
}

# Updated pollster ratings with more aggressive weighting
POLLSTER_RATINGS = {
    'A+': 1.0,
    'A': 0.85,
    'A-': 0.75,
    'B+': 0.65,
    'B': 0.55,
    'B-': 0.45,
    'C+': 0.35,
    'C': 0.25,
    'C-': 0.15,
    'D': 0.1
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
                
            table = tables[1]
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

def calculate_poll_weight(row: pd.Series) -> float:
    """Calculate comprehensive poll weight with improved methodology for final week."""
    try:
        # Sample size weighting using inverse square root relationship
        sample_str = str(row['SAMPLE']).replace('LV', '').replace('RV', '').strip()
        sample_size = float(sample_str) if sample_str.replace('.','').isdigit() else 0
        base_weight = 1 / np.sqrt(1/sample_size) if sample_size > 0 else 0
        
        # Calculate age and apply decay
        days_old = (datetime.now().date() - row['DATE'].date()).days
        
        # Double-check we're within MAX_POLL_AGE_DAYS (defensive programming)
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
    
    # Add uncertainty based on historical polling errors (reduced range)
    error_adjustment = np.tanh(historical_error / 10) * 0.04
    challenger_share = min(max(challenger_share - error_adjustment, 0.48), 0.56)
    
    # Calculate final allocation
    trump_allocation = undecided * challenger_share
    harris_allocation = undecided * (1 - challenger_share)
    
    return trump + trump_allocation, harris + harris_allocation

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
    
    # Apply both time-based filters:
    # 1. Nothing before Harris entered race
    # 2. Nothing older than MAX_POLL_AGE_DAYS
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
    
    # Allocate undecided voters
    df['TRUMP_ADJ'], df['HARRIS_ADJ'] = zip(*df.apply(
        lambda row: allocate_undecided_sophisticated(
            row['TRUMP (R)'], 
            row['HARRIS (D)'], 
            row['Undecided'],
            row['Historical_Error']
        ), 
        axis=1
    ))
    
    # Calculate weights
    df['Weight'] = df.apply(calculate_poll_weight, axis=1)
    
    # Remove invalid entries
    df = df.dropna(subset=['TRUMP_ADJ', 'HARRIS_ADJ', 'Weight', 'DATE'])
    df = df[df['Weight'] > 0]
    
    return df

def monte_carlo_simulation(
    state_data: Dict[str, pd.DataFrame], 
    n_sims: int = 250000
) -> Dict[str, Dict[str, float]]:
    """Run Monte Carlo simulation with improved methodology for realistic CIs."""
    results = {}
    
    # Create correlation matrix for states
    states = list(state_data.keys())
    n_states = len(states)
    correlation_matrix = np.zeros((n_states, n_states))
    
    for i, state1 in enumerate(states):
        for j, state2 in enumerate(states):
            region1 = BATTLEGROUND_STATES[state1]['region']
            region2 = BATTLEGROUND_STATES[state2]['region']
            correlation_matrix[i,j] = REGIONAL_CORRELATIONS[region1][region2]
    
    correlation_matrix += np.eye(n_states) * 1e-6
    
    # Generate correlated random effects using t-distribution with increased df
    df = 20  # Increased from 15 for even thinner tails
    random_effects = np.random.multivariate_normal(
        mean=np.zeros(n_states),
        cov=correlation_matrix,
        size=n_sims
    )
    random_effects = random_effects * np.sqrt((df-2)/df)
    
    # Add reduced systematic bias term
    systematic_bias = np.random.normal(0, 0.15, n_sims)  # Reduced from 0.2
    
    # Run simulations
    state_results = np.zeros((n_sims, n_states))
    
    for i, state in enumerate(states):
        df = state_data[state]
        if df is None or df.empty:
            continue
            
        # Calculate weighted mean and standard error
        weights = df['Weight']
        trump_mean = np.average(df['TRUMP_ADJ'], weights=weights)
        harris_mean = np.average(df['HARRIS_ADJ'], weights=weights)
        
        # Calculate effective sample size for uncertainty
        n_eff = sum(weights)**2 / sum(weights**2)
        
        # Calculate historical error with greater emphasis on recent elections
        hist_errors = BATTLEGROUND_STATES[state]['poll_errors']
        historical_error = (
            0.8 * hist_errors['2020'] +
            0.15 * hist_errors['2016'] +
            0.05 * hist_errors['2012']
        )
        
        # Combine multiple sources of uncertainty with reduced base error
        polling_error = 0.5  # Reduced from 0.6
        total_error = np.sqrt(
            polling_error**2 + 
            (historical_error * 0.3)**2 +  # Reduced from 0.4
            (100/n_eff)
        )
        
        # Generate simulated results
        state_results[:,i] = (
            trump_mean - harris_mean +
            random_effects[:,i] * total_error +
            systematic_bias * historical_error * 0.1  # Reduced from 0.15
        )
    
    # Calculate electoral votes and probabilities
    ev_results = calculate_electoral_votes(state_results, states)
    
    # Process state-level results
    for i, state in enumerate(states):
        mean_margin = np.mean(state_results[:,i])
        ci = np.percentile(state_results[:,i], [2.5, 97.5])
        trump_win_prob = np.mean(state_results[:,i] > 0)
        
        results[state] = {
            'MeanMargin': mean_margin,
            'CI': ci,
            'TrumpProb': trump_win_prob,
            'HarrisProb': 1 - trump_win_prob
        }
    
    return results, ev_results
    
def calculate_electoral_votes(
    state_results: np.ndarray, 
    states: List[str]
) -> Dict[str, float]:
    """Calculate electoral vote outcomes."""
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
    ev_ci = np.percentile(trump_ev, [2.5, 97.5])
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


def visualize_results(
    state_results: Dict[str, Dict[str, float]], 
    ev_results: Dict[str, float]
):
    """Create enhanced visualization of results using only matplotlib."""
    # Set style
    plt.style.use('fivethirtyeight')
    
    # Create figure with adjusted proportions
    fig = plt.figure(figsize=(15, 15))
    gs = plt.GridSpec(3, 1, height_ratios=[3, 1, 0.2], hspace=0.4)  # Increased hspace
    plt.subplots_adjust(left=0.25)
    
    # Top plot - State margins
    ax1 = fig.add_subplot(gs[0])
    
    # Sort states by absolute margin
    states = sorted(state_results.keys(), 
                   key=lambda x: abs(state_results[x]['MeanMargin']),
                   reverse=True)
    margins = [state_results[state]['MeanMargin'] for state in states]
    errors = [(state_results[state]['CI'][1] - state_results[state]['CI'][0])/2 
              for state in states]
    
    # Plot bars and error bars
    y_pos = np.arange(len(states))
    bars = ax1.barh(y_pos, margins, align='center', 
                    color=['red' if m > 0 else 'blue' for m in margins],
                    alpha=0.6, height=0.5)
    
    # Add CI error bars with caps
    error_bars = ax1.errorbar(margins, y_pos, xerr=errors, fmt='none',
                             color='darkgray', capsize=5, capthick=2.0,
                             elinewidth=2.0, label='95% Confidence Interval',
                             zorder=3)
    
    # Add CI numbers at ends of error bars
    for i, state in enumerate(states):
        result = state_results[state]
        # Add left CI number
        ax1.text(result['CI'][0] - 0.1, y_pos[i], 
                f"{result['CI'][0]:.1f}", 
                ha='right', va='center', 
                fontsize=11)
        # Add right CI number
        ax1.text(result['CI'][1] + 0.1, y_pos[i], 
                f"{result['CI'][1]:.1f}", 
                ha='left', va='center', 
                fontsize=11)
    
    # Create state labels with EVs and probabilities - standardized spacing
    labels = []
    for state in states:
        ev = BATTLEGROUND_STATES[state]['ev']
        label = f"{state:<15} ({ev} EV)"  # Fixed width for state names
        labels.append(label)
    
    # Add leading candidate probabilities on the left side - adjusted spacing
    for i, state in enumerate(states):
        prob = state_results[state]['TrumpProb']
        prob_pct = prob if prob >= 0.5 else (1-prob)
        candidate = "Trump" if prob >= 0.5 else "Harris"
        leading_text = f"{prob_pct*100:.0f}% {candidate}"
        ax1.text(-max(abs(min(margins)), abs(max(margins)))*1.1, 
                 y_pos[i] + 0.15,  # Slight upward adjustment
                 leading_text, 
                 ha='right', 
                 va='center',
                 fontsize=12)
    
    # Add margin values at end of bars
    for i, margin in enumerate(margins):
        ax1.text(margin + (0.1 if margin > 0 else -0.1),
                y_pos[i],
                f"{margin:+.1f}%",
                va='center',
                ha='left' if margin > 0 else 'right',
                fontsize=12)
    
    # Customize top plot
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels, fontsize=12)
    ax1.tick_params(axis='y', pad=20)
    ax1.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    ax1.set_title('2024 Battleground State Polling Margins, 95% C.I.', 
                  pad=20, fontsize=18)
    
    # Adjust xlabel position to prevent overlap
    ax1.set_xlabel('Margin (+ Trump Lead, - Harris Lead)', fontsize=14)
    ax1.xaxis.set_label_coords(0.5, -0.15)  # Moved down further
    
    # Set symmetric x-axis limits with space for error bars but not too wide
    max_abs_margin = max(abs(min(margins)), abs(max(margins)))
    max_abs_error = max(errors)
    ax1.set_xlim(-max_abs_margin*1.15 - max_abs_error, max_abs_margin*1.15 + max_abs_error)
    
    # Bottom plot - EV distribution
    ax2 = fig.add_subplot(gs[1])
    ev_dist = ev_results['EVDistribution']
    
    # Create histogram with KDE
    counts, bins, _ = ax2.hist(ev_dist, bins=50, density=True, alpha=0.3, 
                              color='purple', label='Electoral Vote Distribution')
    
    # Add KDE line
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(ev_dist)
    x_range = np.linspace(bins[0], bins[-1], 200)
    ax2.plot(x_range, kde(x_range), color='purple', alpha=0.8)
    
    # Add 270 line and annotation
    ax2.axvline(x=270, color='black', linestyle='--', alpha=0.7)
    ax2.text(270, ax2.get_ylim()[1], '270 EV\nNeeded to Win',
             ha='center', va='bottom', fontsize=14)
    
    # Customize bottom plot
    ax2.set_title('Electoral Vote Distribution', pad=40, fontsize=15)
    ax2.set_xlabel('Trump Electoral Votes', fontsize=14)
    ax2.set_ylabel('Probability Density', fontsize=14)
    ax2.tick_params(axis='both', labelsize=12)
    
    # Add summary text
    summary_ax = fig.add_subplot(gs[2])
    summary_text = (
        f"Projection: Trump {ev_results['TrumpEV']:.0f} EV (±{ev_results['TrumpEV_std']:.0f}), "
        f"Harris {ev_results['HarrisEV']:.0f} EV  |  "
        f"Win Probability: Trump {ev_results['TrumpProb']*100:.0f}%, "
        f"Harris {ev_results['HarrisProb']*100:.0f}%  |  "
        f"Recount Scenario: {ev_results['Recount_Probability']*100:.0f}%"
    )
    summary_ax.text(0.5, 0.5, summary_text, ha='center', va='center',
                   fontsize=14, transform=summary_ax.transAxes)
    summary_ax.axis('off')
    
    # Add timestamp and credit
    plt.figtext(0.02, 0.02, 
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} by Tatsu Ikeda",
                fontsize=11, alpha=0.7)
    plt.figtext(0.98, 0.02,
                "Based on RCP polling data with historical error adjustment",
                fontsize=11, alpha=0.7, ha='right')
    
    plt.tight_layout()
    # Adjust layout after tight_layout to maintain left margin
    plt.subplots_adjust(left=0.25)
    plt.show()

def main():
    try:
        driver = setup_driver()
        state_data = {}
        
        # Fetch and process data for each state
        for state in BATTLEGROUND_STATES:
            df = fetch_polling_data(driver, state)
            if df is not None:
                processed_df = process_state_data(df)
                if processed_df is not None:
                    state_data[state] = processed_df
                    print(f"Successfully processed data for {state}")
                else:
                    print(f"Failed to process data for {state}")
            else:
                print(f"Failed to fetch data for {state}")
        
        # Run Monte Carlo simulation
        state_results, ev_results = monte_carlo_simulation(state_data)
        
        # Print detailed results
        print("\nState-by-State Results:")
        for state in state_results:
            result = state_results[state]
            print(f"\n{state}:")
            print(f"Mean Margin: {result['MeanMargin']:.2f}%")
            print(f"95% CI: ({result['CI'][0]:.2f}%, {result['CI'][1]:.2f}%)")
            print(f"Win Probability - Trump: {result['TrumpProb']*100:.1f}%, " +
                  f"Harris: {result['HarrisProb']*100:.1f}%")
        
        print("\nElectoral College Projection:")
        print(f"Trump: {ev_results['TrumpEV']:.1f} ± {ev_results['TrumpEV_std']:.1f} electoral votes")
        print(f"95% CI: ({ev_results['TrumpEV_CI'][0]:.1f}, {ev_results['TrumpEV_CI'][1]:.1f})")
        print(f"Harris: {ev_results['HarrisEV']:.1f} electoral votes")
        print(f"Win Probability - Trump: {ev_results['TrumpProb']*100:.1f}%")
        print(f"Win Probability - Harris: {ev_results['HarrisProb']*100:.1f}%")
        print(f"Probability of Recount Scenario: {ev_results['Recount_Probability']*100:.1f}%")
        
        # Visualize results
        visualize_results(state_results, ev_results)
        
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if 'driver' in locals():
            driver.quit()

if __name__ == "__main__":
    main()