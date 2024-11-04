# RCP Polling Data Processor

This project is a Python script that processes polling data from RealClearPolitics for the 2024 US Presidential Election, focusing on battleground states. It aggregates poll results, calculates probabilities based on confidence intervals, and provides a visual representation of the polling analysis.

![Figure_1](https://github.com/user-attachments/assets/9398ee96-ce10-4937-a227-aeb500afbe4e)

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
   4. On macOS, you may need to run `xattr -d com.apple.quarantine /usr/local/bin/chromedriver` if you encounter security issues

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

The script will:
1. Fetch polling data for battleground states
2. Process and analyze the data
3. Output a summary of polling results for each state
4. Provide an Electoral College projection
5. Display a visualization of the battleground states polling analysis

## Mathematical and Statistical Methods

This script employs both traditional and modern statistical techniques to process and analyze polling data:

### 1. Traditional Poll Aggregation Methods

#### Pollster Quality Ratings
Pollsters are weighted based on their historical accuracy and methodology:

```python
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
```

#### Time Decay Weighting
Recent polls are given more weight using an exponential decay function:

```python
POLL_DECAY_RATE = 0.05  # Decay rate parameter
time_weight = 1 / (1 + POLL_DECAY_RATE * days_old)
```

#### Sample Size and Type Weighting
Polls are weighted based on sample size and likely voter (LV) vs. registered voter (RV) methodology:

```python
base_weight = 1 / np.sqrt(1/sample_size) if sample_size > 0 else 0
sample_type_weight = 1.5 if 'LV' in str(row['SAMPLE']) else 1.0
```

#### Historical Error Adjustment
Each state's historical polling errors are weighted to account for systematic bias:

```python
weighted_error = (
    0.7 * hist_errors['2020'] +
    0.2 * hist_errors['2016'] +
    0.1 * hist_errors['2012']
)
```

#### Regional Correlations
State outcomes are correlated based on geographic regions:

```python
REGIONAL_CORRELATIONS = {
    'Northeast': {'Northeast': 1.0, 'Midwest': 0.8, 'South': 0.7, 'Southwest': 0.6},
    'Midwest': {'Northeast': 0.8, 'Midwest': 1.0, 'South': 0.75, 'Southwest': 0.7},
    'South': {'Northeast': 0.7, 'Midwest': 0.75, 'South': 1.0, 'Southwest': 0.65},
    'Southwest': {'Northeast': 0.6, 'Midwest': 0.7, 'South': 0.65, 'Southwest': 1.0}
}
```

#### Undecided Voter Allocation
Sophisticated allocation of undecided voters considering:
- Challenger advantage (reduced in final weeks)
- Current polling strength
- Historical polling errors
- Diminishing returns on strength adjustments

```python
challenger_base = 0.52
strength_adjustment = np.tanh(relative_strength) * 0.08
error_adjustment = np.tanh(historical_error / 10) * 0.04
challenger_share = min(max(challenger_share - error_adjustment, 0.48), 0.56)
```

### 2. Modern Statistical Enhancements

#### Regularization Techniques
- LASSO (Least Absolute Shrinkage and Selection Operator)
- Ridge Regression
- Elastic Net
- Regularized correlation matrices for state relationships

```python
models = {
    'lasso': LassoCV(cv=5, random_state=42),
    'ridge': RidgeCV(cv=5),
    'elastic_net': ElasticNetCV(cv=5, random_state=42)
}
```

#### Machine Learning Ensemble
- Random Forest Regressor
- Gradient Boosting Regressor
- Weighted model averaging
- Cross-validation for model performance

```python
models.update({
    'rf': RandomForestRegressor(n_estimators=100, random_state=42),
    'gbm': GradientBoostingRegressor(n_estimators=100, random_state=42)
})
```

#### Hierarchical/Multilevel Modeling
- Region-level effects
- Shrinkage estimation
- Borrowing strength across similar states
- Prior strength adjustment

```python
shrinkage = weights[i] / (weights[i] + 100)  # 100 is prior strength
shrunk_estimate = (
    shrinkage * raw_estimate + 
    (1 - shrinkage) * (results.params[0] + region_effect)
)
```

#### Bootstrap Methods
- Resampling with replacement
- Robust confidence intervals
- Non-parametric uncertainty estimation

```python
def bootstrap_confidence_intervals(data: pd.DataFrame, n_bootstrap: int = 1000) -> dict:
    results = []
    for _ in range(n_bootstrap):
        bootstrap_sample = data.sample(n=len(data), replace=True)
        weighted_mean = np.average(
            bootstrap_sample['margin'], 
            weights=bootstrap_sample['Weight']
        )
        results.append(weighted_mean)
```

#### Missing Data Imputation
- Iterative imputation
- Multiple regression iterations
- Preservation of relationships between variables
- Bounded value constraints

```python
imputer = IterativeImputer(
    max_iter=10,
    random_state=42,
    min_value=0,
    max_value=100
)
```

### 3. Combined Monte Carlo Simulation

The simulation now integrates both traditional and modern methods:

#### Traditional Components
- Poll weights based on quality, recency, and sample size
- Historical error adjustments
- Regional correlations
- Undecided voter allocation

#### Modern Enhancements
- Ensemble model predictions
- Hierarchical state estimates
- Bootstrap confidence intervals
- Regularized correlation structure

#### Multiple Sources of Uncertainty
- Base polling error
- Historical error patterns
- Sampling uncertainty
- Model uncertainty
- Systematic bias

```python
state_results[:,i] = (
    mean_estimate + 
    random_effects[:,i] * uncertainty +
    systematic_bias * hist_error * 0.1
)
```

### 4. Final Estimation Formula

The complete weight calculation combines all factors:

```python
# Traditional poll weight
base_poll_weight = (
    base_weight * 
    time_weight * 
    sample_type_weight * 
    pollster_rating * 
    historical_error_adjustment
)

# Combined modern estimates
combined_estimates[state] = (
    0.4 * ensemble_predictions[state] +
    0.4 * hierarchical_estimates[state] +
    0.2 * bootstrap_results[state]['mean']
)
```

### 5. Recount Probability

Calculates the probability of extremely close electoral college outcomes:

```python
recount_zone = np.sum(np.abs(trump_ev - 269.5) <= 5) / n_sims
```

## Note on Statistical Methods

This implementation represents a hybrid approach combining traditional polling analysis with modern machine learning and statistical methods. The traditional methods provide a strong foundation in polling methodology, while the modern enhancements add robustness and sophisticated uncertainty quantification. This combination helps balance different sources of information and provides more reliable estimates of electoral outcomes.

## Visualization

The script now includes a comprehensive visualization of the battleground states polling analysis. This feature:

- Displays a horizontal bar chart showing the mean difference in polling between candidates for each battleground state
- Uses color coding (red for Trump lead, blue for Harris lead) to easily distinguish which candidate is leading in each state
- Includes error bars to represent the 90% confidence intervals of the polling data
- Orders states from largest lead to smallest, regardless of the leading candidate
- Shows electoral vote distribution with probability density
- Displays win probabilities and recount scenarios
- Provides clear regional and temporal correlations visualization

The visualization appears after the data processing and the final Electoral College projection.

## Note

This script uses Selenium WebDriver to scrape data from RealClearPolitics. Ensure you have the Chrome browser installed and the appropriate ChromeDriver for your Chrome version.

## Installing Homebrew

To install Homebrew on your Mac, follow these steps:

1. Open Terminal on your Mac.

2. Install Xcode Command Line Tools by running:
   ```
   xcode-select --install
   ```

3. Install Homebrew by pasting the following command in Terminal:
   ```
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

4. Follow the prompts in the Terminal to complete the installation.

5. After installation, add Homebrew to your PATH:
   - For Intel Macs:
     ```
     echo 'eval "$(/usr/local/bin/brew shellenv)"' >> ~/.zprofile
     ```
   - For Apple Silicon Macs:
     ```
     echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
     ```

6. Reload your shell configuration:
   ```
   source ~/.zprofile
   ```

7. Verify the installation by running:
   ```
   brew --version
   ```

You should now have Homebrew installed and ready to use on your Mac.

## Disclaimer

This tool is for educational and informational purposes only. Poll aggregation and electoral projections are complex topics, and this model should be used as one of many inputs for understanding election dynamics. The projections should not be used as the sole basis for decision-making related to actual elections.
