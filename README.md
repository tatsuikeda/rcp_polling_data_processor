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

This script employs sophisticated statistical techniques to process and analyze polling data:

### 1. Pollster Quality Ratings

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

### 2. Time Decay Weighting

Recent polls are given more weight using an exponential decay function:

```python
POLL_DECAY_RATE = 0.05  # Decay rate parameter
time_weight = 1 / (1 + POLL_DECAY_RATE * days_old)
```

### 3. Sample Size and Type Weighting

Polls are weighted based on sample size and likely voter (LV) vs. registered voter (RV) methodology:

```python
base_weight = 1 / np.sqrt(1/sample_size) if sample_size > 0 else 0
sample_type_weight = 1.5 if 'LV' in str(row['SAMPLE']) else 1.0
```

### 4. Historical Error Adjustment

Each state's historical polling errors are weighted to account for systematic bias:

```python
weighted_error = (
    0.7 * hist_errors['2020'] +
    0.2 * hist_errors['2016'] +
    0.1 * hist_errors['2012']
)
```

### 5. Regional Correlations

State outcomes are correlated based on geographic regions:

```python
REGIONAL_CORRELATIONS = {
    'Northeast': {'Northeast': 1.0, 'Midwest': 0.8, 'South': 0.7, 'Southwest': 0.6},
    'Midwest': {'Northeast': 0.8, 'Midwest': 1.0, 'South': 0.75, 'Southwest': 0.7},
    'South': {'Northeast': 0.7, 'Midwest': 0.75, 'South': 1.0, 'Southwest': 0.65},
    'Southwest': {'Northeast': 0.6, 'Midwest': 0.7, 'South': 0.65, 'Southwest': 1.0}
}
```

### 6. Undecided Voter Allocation

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

### 7. Monte Carlo Simulation

Comprehensive simulation incorporating:
- T-distribution with fat tails (df=20)
- Correlated state outcomes
- Systematic bias term
- Multiple sources of uncertainty:
  - Base polling error
  - Historical error
  - Sample size effects

### 8. Confidence Interval Calculation

90% confidence intervals are used for:
- State margins
- Electoral vote projections
- Win probabilities

This provides a balance between capturing uncertainty and maintaining practical interpretability.

### 9. Poll Aggregation Weight Formula

The final weight for each poll combines multiple factors:

```python
total_weight = (base_weight * 
               time_weight * 
               sample_type_weight * 
               pollster_rating * 
               historical_error_adjustment)
```

### 10. Recount Probability

Calculates the probability of extremely close electoral college outcomes:

```python
recount_zone = np.sum(np.abs(trump_ev - 269.5) <= 5) / n_sims
```

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
