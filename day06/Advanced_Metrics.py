
import numpy as np
import pandas as pd
import os
import requests
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split

# --- 1. 載入資料 ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(SCRIPT_DIR, 'Social_Network_Ads.csv')
DATA_URL = 'https://raw.githubusercontent.com/shivang98/Social-Network-ads-Boost/master/Social_Network_Ads.csv'

def load_or_download_data(local_path, url):
    if not os.path.exists(local_path):
        try:
            response = requests.get(url, verify=False)
            response.raise_for_status()
            with open(local_path, 'wb') as f:
                f.write(response.content)
        except Exception:
            return None
    return pd.read_csv(local_path)

df = load_or_download_data(DATA_FILE, DATA_URL)
if df is None:
    # Mock data if download fails
    df = pd.DataFrame({
        'Age': np.random.randint(18, 60, 400),
        'EstimatedSalary': np.random.randint(15000, 150000, 400),
        'Purchased': np.random.randint(0, 2, 400)
    })

X = df[['Age', 'EstimatedSalary']]
y = df['Purchased']

# --- 2. VIF 計算 (多重共線性) ---
print("--- VIF (Variance Inflation Factor) ---")
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
# VIF requires a constant term for correct calculation if not standardized, 
# but usually we check correlation. Here we just pass X directly.
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
print(vif_data)
print("-" * 30)

# --- 3. IV (Information Value) 計算 ---
def calculate_iv(df, feature, target, bins=5):
    # Binning continuous variables
    df['bin'] = pd.qcut(df[feature], q=bins, duplicates='drop')
    
    # Group by bin
    grouped = df.groupby('bin', observed=False)[target].agg(['count', 'sum'])
    grouped.columns = ['Total', 'Bad'] # Assuming target=1 is Bad (Purchased) or Good, doesn't matter for IV magnitude
    grouped['Good'] = grouped['Total'] - grouped['Bad']
    
    # Calculate distributions
    dist_bad = grouped['Bad'] / grouped['Bad'].sum()
    dist_good = grouped['Good'] / grouped['Good'].sum()
    
    # Calculate WoE and IV
    # Add small epsilon to avoid division by zero
    epsilon = 1e-10
    woe = np.log((dist_good + epsilon) / (dist_bad + epsilon))
    iv = (dist_good - dist_bad) * woe
    
    return iv.sum()

print("--- IV (Information Value) ---")
iv_age = calculate_iv(df.copy(), 'Age', 'Purchased')
iv_salary = calculate_iv(df.copy(), 'EstimatedSalary', 'Purchased')
print(f"Age IV: {iv_age:.4f}")
print(f"EstimatedSalary IV: {iv_salary:.4f}")
print("-" * 30)

# --- 4. PSI (Population Stability Index) 計算 ---
# 比較 Train set (Expected) 與 Test set (Actual) 的分佈
X_train, X_test, _, _ = train_test_split(X, y, test_size=0.25, random_state=0)

def calculate_psi(expected, actual, buckets=10):
    def scale_range (input, min, max):
        input += -(np.min(input))
        input /= np.max(input) / (max - min)
        input += min
        return input

    breakpoints = np.arange(0, buckets + 1) / (buckets) * 100
    breakpoints = np.percentile(expected, breakpoints)
    
    # Calculate counts in each bucket
    expected_percents = np.histogram(expected, breakpoints)[0] / len(expected)
    actual_percents = np.histogram(actual, breakpoints)[0] / len(actual)
    
    # Avoid zero division
    expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
    actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)
    
    psi_value = np.sum((expected_percents - actual_percents) * np.log(expected_percents / actual_percents))
    return psi_value

print("--- PSI (Population Stability Index) ---")
psi_age = calculate_psi(X_train['Age'], X_test['Age'])
psi_salary = calculate_psi(X_train['EstimatedSalary'], X_test['EstimatedSalary'])
print(f"Age PSI: {psi_age:.4f}")
print(f"EstimatedSalary PSI: {psi_salary:.4f}")
