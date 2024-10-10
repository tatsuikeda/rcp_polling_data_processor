# RCP Polling Data Processor

This project is a Python script that processes polling data from RealClearPolitics for the 2024 US Presidential Election, focusing on battleground states. It aggregates poll results and calculates probabilities based on confidence intervals.

## Author

Tatsu Ikeda

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Requirements

- Python 3.11
- Chrome browser
- ChromeDriver

## Setup

1. Clone this repository:
   ```
   git clone https://github.com/your-username/rcp-polling-data-processor.git
   cd rcp-polling-data-processor
   ```

2. Create a virtual environment:
   ```
   python3.11 -m venv .venv
   ```

3. Activate the virtual environment:
   - On macOS and Linux:
     ```
     source .venv/bin/activate
     ```
   - On Windows:
     ```
     .venv\Scripts\activate
     ```

4. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

5. Install ChromeDriver:

   ### macOS
   
   Using Homebrew:
   ```
   brew install --cask chromedriver
   ```
   
   After installation, you may need to allow ChromeDriver in your security settings:
   1. Open "System Preferences" > "Security & Privacy" > "General"
   2. Click the lock to make changes
   3. Click "Allow Anyway" next to the ChromeDriver message

   ### Windows

   1. Download ChromeDriver from the [official site](https://sites.google.com/a/chromium.org/chromedriver/downloads)
   2. Choose the version that matches your Chrome browser version
   3. Extract the downloaded zip file
   4. Add the directory containing chromedriver.exe to your system PATH:
      - Right-click "This PC" and choose "Properties"
      - Click "Advanced system settings"
      - Click "Environment Variables"
      - Under "System variables", find and select "Path", then click "Edit"
      - Click "New" and add the directory path where you extracted chromedriver.exe
      - Click "OK" to save changes

   ### Linux

   1. Download ChromeDriver from the [official site](https://sites.google.com/a/chromium.org/chromedriver/downloads)
   2. Choose the version that matches your Chrome browser version
   3. Extract the downloaded file:
      ```
      unzip chromedriver_linux64.zip
      ```
   4. Move ChromeDriver to a directory in your PATH:
      ```
      sudo mv chromedriver /usr/local/bin/
      ```
   5. Make it executable:
      ```
      sudo chmod +x /usr/local/bin/chromedriver
      ```

## Usage

To run the script, ensure your virtual environment is activated, then execute:

```
python rcp_polling_data_processor_for_2024_election.py
```

The script will fetch polling data for battleground states, process it, and output a summary of polling results for each state.

## Mathematical and Statistical Methods

This script employs several mathematical and statistical techniques to process and analyze polling data:

### 1. Time Decay Weighting

We use a time decay function to give more weight to recent polls:

```python
def time_decay_weight(days, lambda_param=0.001):
    return 1 / (1 + lambda_param * days**2)
```

This quadratic decay function ensures that older polls have less influence on the final estimate. The `lambda_param` controls the rate of decay.

### 2. Sample Size Weighting

Polls with larger sample sizes are given more weight:

```python
def sample_size_weight(sample_size, moe):
    return 1 / (moe**2 * sample_size)
```

This formula is derived from the fact that the variance of a poll is proportional to 1 / (sample size).

### 3. Undecided Voter Allocation

Undecided voters are allocated using a simple model that slightly favors the challenger:

```python
def allocate_undecided(trump, harris, undecided, admin_approval=0.5):
    trump_share = 0.55 - 0.1 * admin_approval
    harris_share = 1 - trump_share
    return trump + undecided * trump_share, harris + undecided * harris_share
```

This model assumes that undecided voters are more likely to break for the challenger (Trump), but it's moderated by the administration's approval rating.

### 4. Weighted Averaging

The final estimate for each candidate is a weighted average of all polls:

```python
trump_avg = (df['TRUMP_ADJ'] * df['CombinedWeight']).sum() / total_weight
harris_avg = (df['HARRIS_ADJ'] * df['CombinedWeight']).sum() / total_weight
```

Where `CombinedWeight` is the product of time and sample size weights.

### 5. Confidence Interval-based Probability Calculation

We use the confidence intervals from aggregated polling data to calculate the probability of each candidate winning a state:

```python
mean_diff = battleground_results[state]['MeanDiff']
ci = battleground_results[state]['CI']
std_dev = (ci[1] - ci[0]) / (2 * 1.96)  # Assuming 95% CI
z_score = mean_diff / std_dev
trump_prob = 1 - stats.norm.cdf(z_score)
```

This method:
- Calculates the standard deviation based on the confidence interval
- Computes a z-score for the mean difference
- Uses the cumulative distribution function of the normal distribution to determine the probability

### 6. Overall Victory Probability Calculation

We calculate the overall probability of victory by considering all possible combinations of state outcomes:

```python
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
```

This approach:
- Generates all possible combinations of state outcomes
- Calculates the probability of each scenario
- Sums the probabilities of all scenarios where a candidate wins

### 7. Margin of Error Estimation

For polls missing MOE data, we estimate it based on the sample size:

```python
def estimate_moe(row, df):
    if pd.notna(row['SAMPLE']) and row['SAMPLE'].replace('LV', '').replace('RV', '').strip().isdigit():
        sample_size = int(row['SAMPLE'].replace('LV', '').replace('RV', '').strip())
        return 1 / np.sqrt(sample_size)
    else:
        valid_moes = df['MOE'].dropna().astype(float)
        return valid_moes.median() if not valid_moes.empty else 3.0
```

This uses the standard error formula for a proportion (1 / sqrt(n)) when sample size is available, or falls back to the median MOE or a default value.

### Limitations and Assumptions

- The confidence interval-based probability calculation assumes normally distributed polling errors.
- This model treats state outcomes as independent, which may not reflect real-world correlations between states.
- The method doesn't account for systematic polling errors or "unknown unknowns" that could affect all polls similarly.
- This approach is sensitive to the accuracy of the reported confidence intervals and may not capture all sources of uncertainty in polling.
- The time decay and sample size weighting methods are simplifications and may not capture all factors affecting poll reliability.
- The undecided voter allocation model is a simple heuristic and may not accurately reflect real voting behavior.

While this method provides a mathematically sound way to calculate probabilities based on polling data, it should be considered as one of many possible approaches to election forecasting. More sophisticated models might incorporate additional factors such as economic indicators, historical voting patterns, and demographic trends.

## Troubleshooting

If you encounter issues with ChromeDriver:

1. Ensure your Chrome browser is up to date
2. Verify that the ChromeDriver version matches your Chrome version
3. Check that ChromeDriver is correctly added to your system PATH
4. On macOS, you may need to run `xattr -d com.apple.quarantine /usr/local/bin/chromedriver` if you encounter security issues

## Note

This script uses Selenium WebDriver to scrape data from RealClearPolitics. Ensure you have the Chrome browser installed and the appropriate ChromeDriver for your Chrome version.

## Disclaimer

This tool is for educational and informational purposes only. Poll aggregation and electoral projections are complex topics, and this simple model should not be used for making predictions or decisions related to actual elections.