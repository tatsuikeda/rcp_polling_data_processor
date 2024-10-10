# Standard library imports
import os
import sys
import time
from datetime import datetime

# Third-party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import itertools

# Selenium imports
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

# Constants
BATTLEGROUND_STATES = {
    'Pennsylvania': 'pennsylvania',
    'North Carolina': 'north-carolina',
    'Georgia': 'georgia',
    'Wisconsin': 'wisconsin',
    'Michigan': 'michigan',
    'Arizona': 'arizona'
}
BASE_URL = 'https://www.realclearpolling.com/polls/president/general/2024/'
HARRIS_ENTRY_DATE = datetime(2024, 8, 1)

def setup_driver():
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
    
    chromedriver_path = "/usr/local/bin/chromedriver"
    
    try:
        service = Service(chromedriver_path)
        driver = webdriver.Chrome(service=service, options=chrome_options)
        print("Successfully set up ChromeDriver using new Selenium syntax.")
        return driver
    except Exception as e:
        print(f"Error setting up ChromeDriver with new syntax: {e}")
        try:
            driver = webdriver.Chrome(executable_path=chromedriver_path, options=chrome_options)
            print("Successfully set up ChromeDriver using old Selenium syntax.")
            return driver
        except Exception as e:
            print(f"Error setting up ChromeDriver with old syntax: {e}")
            print("Please ensure Chrome and ChromeDriver are installed and up to date.")
            print(f"Current ChromeDriver path: {chromedriver_path}")
            print("If you've moved it, please update the 'chromedriver_path' variable in the script.")
            raise

def fetch_polling_data(driver, state):
    url = f"{BASE_URL}{state}/trump-vs-harris"
    try:
        print(f"Fetching data from URL: {url}")
        driver.get(url)
        
        time.sleep(10)  # Wait for page to load
        
        # Find all tables on the page
        tables = driver.find_elements(By.TAG_NAME, 'table')
        
        if len(tables) < 2:
            print("Error: Less than two tables found on the page")
            return None
        
        # Select the second table
        table = tables[1]
        
        # Extract headers
        headers = [th.text.strip() for th in table.find_elements(By.TAG_NAME, 'th')]
        print(f"Headers found: {headers}")
        
        # Extract rows
        rows = []
        for tr in table.find_elements(By.TAG_NAME, 'tr')[1:]:  # Skip header row
            row = [td.text.strip() for td in tr.find_elements(By.TAG_NAME, 'td')]
            rows.append(row)
        
        print(f"Number of rows found: {len(rows)}")
        
        # Create DataFrame
        df = pd.DataFrame(rows, columns=headers)
        
        return df
    except Exception as e:
        print(f"Error fetching data for {state}: {e}")
        return None

def time_decay_weight(days, lambda_param=0.001):
    return 1 / (1 + lambda_param * days**2)

def sample_size_weight(sample_size, moe):
    try:
        sample_size = float(sample_size.replace('LV', '').replace('RV', '').strip())
        if pd.isna(moe) or moe == '—':
            moe = 1 / np.sqrt(sample_size)
        else:
            moe = float(moe)
        
        if sample_size <= 0 or moe <= 0:
            return 0
        
        return 1 / (moe**2 * sample_size)
    except (ValueError, AttributeError):
        return 0  # Return 0 weight for invalid inputs

def allocate_undecided(trump, harris, undecided, admin_approval=0.5):
    # Slightly favor the challenger (Trump)
    trump_share = 0.55 - 0.1 * admin_approval
    harris_share = 1 - trump_share
    return trump + undecided * trump_share, harris + undecided * harris_share

def estimate_moe(row, df):
    # If sample size is available, estimate MOE as 1 / sqrt(sample_size)
    if pd.notna(row['SAMPLE']) and row['SAMPLE'].replace('LV', '').replace('RV', '').strip().isdigit():
        sample_size = int(row['SAMPLE'].replace('LV', '').replace('RV', '').strip())
        return 1 / np.sqrt(sample_size)
    else:
        # If sample size is not available, use a default MOE (e.g., median of available MOEs)
        valid_moes = df['MOE'].dropna().astype(float)
        if not valid_moes.empty:
            return valid_moes.median()
        else:
            return 3.0  # Default MOE if no valid MOEs are available

def process_state_data(df):
    if df is None or df.empty:
        print("Input DataFrame is None or empty")
        return None

    print("Initial DataFrame shape:", df.shape)
    print("Initial DataFrame columns:", df.columns)
    print("Sample of initial data:")
    print(df.head())

    # Convert 'DATE' column to datetime
    def parse_date(date_str):
        if '-' in date_str:
            start, end = date_str.split('-')
            return pd.to_datetime(end.strip(), format='%m/%d')
        else:
            return pd.to_datetime(date_str, format='%m/%d')

    df['DATE'] = df['DATE'].apply(parse_date)
    
    # Add the current year to all dates
    current_year = datetime.now().year
    df['DATE'] = df['DATE'].apply(lambda x: x.replace(year=current_year))
    
    # Filter out future dates
    today = datetime.now().date()
    df = df[df['DATE'].dt.date <= today]
    
    # Sort the DataFrame by date in descending order
    df = df.sort_values('DATE', ascending=False)

    latest_date = df['DATE'].max()
    df['DaysSincePoll'] = (latest_date - df['DATE']).dt.days
    df['TimeWeight'] = df['DaysSincePoll'].apply(time_decay_weight)
    
    print("TimeWeight calculation complete")
    print("TimeWeight range:", df['TimeWeight'].min(), "-", df['TimeWeight'].max())
    
    df['SampleWeight'] = df.apply(lambda row: sample_size_weight(row['SAMPLE'], row['MOE']), axis=1)
    
    print("SampleWeight calculation complete")
    print("SampleWeight range:", df['SampleWeight'].min(), "-", df['SampleWeight'].max())
    
    df['TRUMP (R)'] = pd.to_numeric(df['TRUMP (R)'], errors='coerce')
    df['HARRIS (D)'] = pd.to_numeric(df['HARRIS (D)'], errors='coerce')
    
    print("Numeric conversion complete")
    print("TRUMP (R) range:", df['TRUMP (R)'].min(), "-", df['TRUMP (R)'].max())
    print("HARRIS (D) range:", df['HARRIS (D)'].min(), "-", df['HARRIS (D)'].max())
    
    # Convert 'MOE' to numeric, replacing non-numeric values with NaN
    df['MOE'] = pd.to_numeric(df['MOE'], errors='coerce')

    # Estimate MOE for rows with missing values
    df['MOE'] = df.apply(lambda row: estimate_moe(row, df) if pd.isna(row['MOE']) else row['MOE'], axis=1)
    
    # Allocate undecided voters
    df['Undecided'] = 100 - df['TRUMP (R)'] - df['HARRIS (D)']
    df['TRUMP_ADJ'], df['HARRIS_ADJ'] = zip(*df.apply(lambda row: allocate_undecided(row['TRUMP (R)'], 
                                                                                     row['HARRIS (D)'], 
                                                                                     row['Undecided']), axis=1))
    
    print("Undecided voter allocation complete")
    print("TRUMP_ADJ range:", df['TRUMP_ADJ'].min(), "-", df['TRUMP_ADJ'].max())
    print("HARRIS_ADJ range:", df['HARRIS_ADJ'].min(), "-", df['HARRIS_ADJ'].max())
    
    # Remove rows with NaN values
    initial_rows = len(df)
    df = df.dropna(subset=['TRUMP_ADJ', 'HARRIS_ADJ', 'TimeWeight', 'SampleWeight'])
    rows_after_nan_removal = len(df)
    print(f"Rows removed due to NaN values: {initial_rows - rows_after_nan_removal}")
    
    # Ensure weights are positive
    df['CombinedWeight'] = df['TimeWeight'] * df['SampleWeight']
    initial_rows = len(df)
    df = df[df['CombinedWeight'] > 0]
    rows_after_weight_filter = len(df)
    print(f"Rows removed due to non-positive weights: {initial_rows - rows_after_weight_filter}")
    
    if df.empty:
        print("Warning: All data filtered out due to invalid values.")
        return None
    
    print("Final DataFrame shape:", df.shape)
    print("Sample of final data:")
    print(df.head())
    
    return df

def aggregate_state_results(df):
    if df is None or df.empty:
        return None, None

    total_weight = df['CombinedWeight'].sum()
    if total_weight == 0:
        print("Warning: Total weight is zero.")
        return None, None

    trump_avg = (df['TRUMP_ADJ'] * df['CombinedWeight']).sum() / total_weight
    harris_avg = (df['HARRIS_ADJ'] * df['CombinedWeight']).sum() / total_weight
    return trump_avg, harris_avg

import numpy as np

def monte_carlo_simulation(df, n_simulations=10000, convergence_threshold=0.001):
    if df is None or df.empty:
        return None, (None, None)

    weights = df['CombinedWeight'].values
    trump_adj = df['TRUMP_ADJ'].values
    harris_adj = df['HARRIS_ADJ'].values
    moe = df['MOE'].values

    results = []
    running_mean = 0
    for i in range(1, n_simulations + 1):
        sample_indices = np.random.choice(len(df), size=len(df), p=weights/weights.sum())
        trump_sample = trump_adj[sample_indices] + np.random.normal(0, moe[sample_indices] / 2)
        harris_sample = harris_adj[sample_indices] + np.random.normal(0, moe[sample_indices] / 2)
        result = np.mean(trump_sample - harris_sample)
        results.append(result)
        
        new_mean = np.mean(results)
        if i > 1000 and abs(new_mean - running_mean) < convergence_threshold:
            print(f"Converged after {i} simulations")
            break
        running_mean = new_mean

    return np.mean(results), np.percentile(results, [2.5, 97.5])

def calculate_electoral_college(battleground_results):
   # Hard-coded EC votes based on 2024 allocation
    safe_ec = {
        'Trump': 215,
        'Harris': 227
    }
    
    battleground_ec = {
        'Arizona': 11, 'Georgia': 16, 'Michigan': 15, 
        'North Carolina': 16, 'Pennsylvania': 19, 'Wisconsin': 10
    }
    
    me_ne_ec = {
        'Maine': {'statewide': 2, 'ME-1': 1, 'ME-2': 1},
        'Nebraska': {'statewide': 2, 'NE-1': 1, 'NE-2': 1, 'NE-3': 1}
    }
    
    safe_ec['Harris'] += me_ne_ec['Maine']['statewide'] + me_ne_ec['Maine']['ME-1']
    safe_ec['Trump'] += me_ne_ec['Nebraska']['statewide'] + me_ne_ec['Nebraska']['NE-1'] + me_ne_ec['Nebraska']['NE-3']
    
    swing_districts = {'ME-2': 1, 'NE-2': 1}
    
    # Calculate most likely EC count and state probabilities
    most_likely_ec = safe_ec.copy()
    state_probabilities = {}
    for state, votes in battleground_ec.items():
        if state in battleground_results:
            mean_diff = battleground_results[state]['MeanDiff']
            ci = battleground_results[state]['CI']
            std_dev = (ci[1] - ci[0]) / (2 * 1.96)  # Assuming 95% CI
            z_score = mean_diff / std_dev
            trump_prob = 1 - stats.norm.cdf(z_score)
            state_probabilities[state] = trump_prob
            
            if mean_diff > 0:
                most_likely_ec['Trump'] += votes
            else:
                most_likely_ec['Harris'] += votes
        else:
            # If no polling data, use 2020 results
            if state in ['Arizona', 'Georgia', 'Michigan', 'Pennsylvania', 'Wisconsin']:
                most_likely_ec['Harris'] += votes
                state_probabilities[state] = 0.5  # Assume 50-50 if no data
            else:
                most_likely_ec['Trump'] += votes
                state_probabilities[state] = 0.5  # Assume 50-50 if no data
    
    # Allocate swing districts based on 2020 results
    most_likely_ec['Trump'] += swing_districts['ME-2']
    most_likely_ec['Harris'] += swing_districts['NE-2']
    
    # Calculate overall probability
    scenarios = [[0, 1]] * len(battleground_ec)
    total_probability = 0
    for scenario in itertools.product(*scenarios):
        scenario_ec = safe_ec.copy()
        scenario_prob = 1
        for i, (state, votes) in enumerate(battleground_ec.items()):
            if scenario[i] == 0:  # Trump wins
                scenario_ec['Trump'] += votes
                scenario_prob *= state_probabilities[state]
            else:  # Harris wins
                scenario_ec['Harris'] += votes
                scenario_prob *= (1 - state_probabilities[state])
        
        if scenario_ec['Trump'] > 269:
            total_probability += scenario_prob
    
    trump_probability = total_probability * 100
    harris_probability = (1 - total_probability) * 100
    
    return {
        'Trump': most_likely_ec['Trump'],
        'Harris': most_likely_ec['Harris'],
        'TrumpProbability': trump_probability,
        'HarrisProbability': harris_probability
    }

def main():
    try:
        driver = setup_driver()
    except Exception as e:
        print(f"Failed to set up WebDriver: {e}")
        return

    all_results = {}
    total_states = len(BATTLEGROUND_STATES)
    
    try:
        for index, (state, url_state) in enumerate(BATTLEGROUND_STATES.items(), 1):
            print(f"Processing {state}... ({index}/{total_states})")
            df = fetch_polling_data(driver, url_state)
            if df is not None:
                processed_df = process_state_data(df)
                if processed_df is not None:
                    trump_avg, harris_avg = aggregate_state_results(processed_df)
                    mean_diff, ci = monte_carlo_simulation(processed_df)
                    
                    all_results[state] = {
                        'TrumpAvg': trump_avg,
                        'HarrisAvg': harris_avg,
                        'MeanDiff': mean_diff,
                        'CI': ci
                    }
                    
                    print(f"{state} Results:")
                    print(f"Trump: {trump_avg:.2f}%, Harris: {harris_avg:.2f}%")
                    if mean_diff is not None and ci[0] is not None and ci[1] is not None:
                        print(f"Mean Difference: {mean_diff:.2f}% (95% CI: {ci[0]:.2f}% to {ci[1]:.2f}%)")
                    else:
                        print("Unable to calculate mean difference and confidence interval.")
                else:
                    print(f"No valid data for {state} after processing")
            else:
                print(f"No data fetched for {state}")
            print()
    except Exception as e:
        print(f"An error occurred during execution: {e}")
    finally:
        driver.quit()
    
    # Print summary of all results
    print("\nSummary of All Results:")
    for state, results in all_results.items():
        print(f"{state}:")
        if results['TrumpAvg'] is not None and results['HarrisAvg'] is not None:
            print(f"  Trump: {results['TrumpAvg']:.2f}%, Harris: {results['HarrisAvg']:.2f}%")
            if results['MeanDiff'] is not None and results['CI'][0] is not None and results['CI'][1] is not None:
                print(f"  Mean Difference: {results['MeanDiff']:.2f}% (95% CI: {results['CI'][0]:.2f}% to {results['CI'][1]:.2f}%)")
            else:
                print("  Unable to calculate mean difference and confidence interval.")
        else:
            print("  Unable to calculate averages.")
        print()
    
    # Calculate Electoral College results
    ec_results = calculate_electoral_college(all_results)
    print("\nElectoral College Projection:")
    print(f"Trump: {ec_results['Trump']} electoral votes")
    print(f"Harris: {ec_results['Harris']} electoral votes")
    print(f"Probability of Trump victory: {ec_results['TrumpProbability']:.2f}%")
    print(f"Probability of Harris victory: {ec_results['HarrisProbability']:.2f}%")

    # Verify total EC votes
    total_ec = ec_results['Trump'] + ec_results['Harris']
    if total_ec != 538:
        print(f"Warning: Total EC votes ({total_ec}) do not add up to 538.")
    else:
        print("Total EC votes correctly sum to 538.")

if __name__ == "__main__":
    main()