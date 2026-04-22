# Importing required libraries 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import joblib # for saving the parameter of linear regression model
from statsmodels.tsa.stattools import adfuller, kpss

from fracdiff.sklearn import Fracdiff
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm.notebook import tqdm
from tqdm import tqdm 

import warnings
from statsmodels.tools.sm_exceptions import InterpolationWarning
warnings.filterwarnings("ignore", category=InterpolationWarning)
# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.use_deterministic_algorithms(True)

# Function to read the CSV file and preprocess the data
def reading_file(file_name):
    """
    Read a CSV with a 2-row header (e.g., ('Price','Ticker')) and a date index,
    keep only the first header level ('Price'), and covert all values to numeric.
    """
    data = pd.read_csv(
        file_name,
        header=[0, 1],        # 2 header rows: ('Price', 'Ticker')
        index_col=0,          # first column is the date index
        parse_dates=True,
    )
    # Keep only the first level of the columns ('Price')
    data.columns = data.columns.get_level_values(0)
    # Ensure all entries are numeric (coerce non-numeric values to NaN)
    data = data.apply(pd.to_numeric, errors="coerce")
    return data

# Function to compute stationarity test statistics
def stationarity_values(series):
    """
    Compute ADF and KPSS test statistics .
    The series is cleaned with dropna() and cast to float before testing.

    """
    # ADF regression model being estimated:
    # Δy_t = α + β * y_{t-1} + γ₁ * Δy_{t-1} + ε_t	( since regression='c' and maxlag=1, autolag is the lag added )
    series = series.dropna().astype(float)
    a_stat, a_p, *_ = adfuller(series, maxlag=1, regression='c', autolag=None)
    k_stat, k_p, *_ = kpss(series, regression='c', nlags=1)
    return a_stat, a_p, k_stat, k_p

# Function to choose the right order of (fractional) differentiation for stationarity
def sweep_fractional_orders(series, d_list):
    """
    For a given series compute fractional differences of ['Close'] for each d in d_list and compute ADF stats.

    """
    window_size = 10
    rows = []
    for d in d_list:
        frac = Fracdiff(d, window=window_size, window_policy="fixed") 	# the window size is now default=10
        X = series[['Close']]          
        X_fd = frac.fit_transform(X).ravel()
        X_fd[:(window_size - 1)] = np.nan
        s_fd = pd.Series(X_fd)        								
        a_stat, a_p, _ , _ = stationarity_values(s_fd)
        rows.append({
            "d": d, "series": s_fd,
            "adf_stat": a_stat, "adf_p": a_p,
        })
    return rows


# Function to add fractional and integer differencing column to the dataframe
def add_frac_diff_column(series, order_diff, cols, mode):
    """
    Add fractional differenced columns for each column in `cols` to `series`.
 
    """
    window_size = 10
    if mode == "fractional":
        fracdiff = Fracdiff(order_diff, window=window_size, window_policy="fixed") 		# the window size is now default=10
        for col in cols:											
            feature = series[[col]].astype(float).copy()			
            new_column = f"{col}_fd"
            series[new_column] = fracdiff.fit_transform(feature).ravel()
            series.iloc[:window_size-1, series.columns.get_loc(new_column)] = np.nan
    else:
        for col in cols:
            feature = series[col].copy()
            new_column = f"{col}_d"
            series[new_column] = feature.diff()
    return series


def plot_close_vs_frac(title, df, col_close="Close", col_frac="Close_fd"):
    """
    Plot the original, integer-differenced, and fractionally differenced series on the same figure using twin y-axes.
    Left axis → original series.
    Right axis → differenced series (integer + fractional).
    """
    fig, ax1 = plt.subplots(figsize=(12, 5))

    # Left axis: original prices 
    ax1.plot(df.index, df[col_close], lw=1.6, color="black", label="Original (Close)")
    ax1.set_ylabel("Original closing Price")
    ax1.set_xlabel("Date")

    # Right axis: integer and fractional differences 
    ax2 = ax1.twinx()
    ax2.plot(df.index, df[col_frac], lw=1.0, color="tab:orange", label="Frac diff (d*)")
    ax2.set_ylabel("Closing Price after differentiation")

    # Legend 
    lines = ax1.get_lines() + ax2.get_lines()
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left")

    ax1.set_title(title)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# Function to create the dataset
def prepare_data(series, cols):
    """
    Prepare (fractional- or integer-differenced) OHLC features and next-day target.
    - Drops rows with NaNs in features or target.
    - Splits into train/val/test by index (70/15/15) and normalizes features and target.
    """
    dataset = series[cols].copy()
    # As a target variable we are using the log-return of the next day
    # Meaning we are considering log(P_t+1) - log(P_t) = log(P_t+1 / P_t) = log( 1 + R_t ) circa R_t (using taylor expansion for small R_t)
    target = series['Close_d'].shift(-1)

    # mask valid rows
    mask = dataset.notna().all(axis=1) & target.notna()
    X = dataset.loc[mask].astype(float).values
    y = target.loc[mask].astype(float).values.reshape(-1, 1)
    
    # Split the data into training, validation, and test sets (70/15/15)
    train_size = int(len(X) * 0.70)
    val_size = int(len(X) * 0.15)

    train_data, val_data = X[:train_size], X[train_size:train_size + val_size]
    target_train, target_val = y[:train_size], y[train_size:train_size + val_size]
    test_data, target_test = X[train_size + val_size:], y[train_size + val_size:]

    # Normalize data (since the target are just the next-day prices, we scale them too)
    scaler_data = StandardScaler()
    train_data = scaler_data.fit_transform(train_data)
    val_data, test_data = scaler_data.transform(val_data), scaler_data.transform(test_data)
    
    return train_data, val_data, test_data, target_train, target_val, target_test


# Function for training and evaluating models : it returns test metrics
def train_evaluate(weight_filename, train_data, val_data, test_data, target_train, target_val, target_test, n_features):
    """
    Build, train, and evaluate a shallow MLP, using one hidden layer and ReLU activations.
    Uses validation split and early stopping.

    Returns :  [loss, rmse, mae].
    """
    # Define the MLP model
    nhidden = 32
    model = nn.Sequential(nn.Linear(n_features, nhidden),
                            nn.ReLU(),
                            nn.Linear(nhidden, 1))

    # Transforming data into tensor
    train_data = torch.tensor(train_data, dtype=torch.float32)
    val_data = torch.tensor(val_data, dtype=torch.float32)
    test_data = torch.tensor(test_data, dtype=torch.float32)
    target_train = torch.tensor(target_train, dtype=torch.float32).view(-1, 1)
    target_val = torch.tensor(target_val, dtype=torch.float32).view(-1, 1)
    target_test = torch.tensor(target_test, dtype=torch.float32).view(-1, 1)

    # Dataset e DataLoader
    train_dataset = TensorDataset(train_data, target_train)
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=False) # Shuffle False because of time series data

    # Defining loss, optimizer 
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Training loop
    n_epochs = 500
    train_curve, val_curve = [], []
    training_loop = tqdm(range(n_epochs)) 
    for epoch in training_loop:
        # Training mode
        model.train() 
        running = 0.0 
        for batch_data, batch_targets in train_loader:
            # Reset computation graph
            optimizer.zero_grad()
            # Forward pass
            outputs = model(batch_data)
            # Compute training loss on batch
            loss = criterion(outputs, batch_targets)
            # Compute gradient 
            loss.backward()
            # Gradient step
            optimizer.step()
            # Loss per batch 
            running += loss.item()
        # Loss per epoch
        train_epoch = running/len(train_loader)

        # Validation loss
        model.eval()
        with torch.no_grad():
            val_outputs = model(val_data)
            val_loss = criterion(val_outputs, target_val).item()

        # Store train and validation loss
        train_curve.append(train_epoch)
        val_curve.append(val_loss)

        # Print loss in the progression bar
        training_loop.set_postfix(train=train_epoch, val=val_loss, lr=optimizer.param_groups[0]['lr'])

    # Save the final model weights
    torch.save(model.state_dict(), weight_filename)
    # Plot losses
    plt.figure()
    plt.plot(train_curve, label='train')
    plt.plot(val_curve,   label='val')
    plt.yscale('log') 
    plt.legend(); plt.xlabel('epoch'); plt.ylabel('MSE')
    plt.show()			

    # Metrics evaluation 
    model.load_state_dict(torch.load(weight_filename)) 
    model.eval()
    with torch.no_grad():
        predictions = model(test_data).numpy() 
        test_loss = mean_squared_error(target_test.numpy(), predictions)
        test_rmse = np.sqrt(test_loss)
        test_mae = mean_absolute_error(target_test.numpy(), predictions)

    
    return [test_loss, test_rmse, test_mae]

# Function for training and evaluating models : it returns test metrics
def train_evaluate_sw(weight_filename, train_data, val_data, test_data, target_train, target_val, target_test, n_features):
    """
    Build, train, and evaluate a shallow MLP, using one hidden layer and ReLU activations.
    Uses validation split and early stopping.

    Returns :  [loss, rmse, mae].
    """
    # Define the MLP model
    nhidden = 32
    model = nn.Sequential(nn.Linear(n_features, nhidden),
                            nn.ReLU(),
                            nn.Linear(nhidden, 1))

    # Transforming data into tensor
    train_data = torch.tensor(train_data, dtype=torch.float32)
    val_data = torch.tensor(val_data, dtype=torch.float32)
    test_data = torch.tensor(test_data, dtype=torch.float32)
    target_train = torch.tensor(target_train, dtype=torch.float32).view(-1, 1)
    target_val = torch.tensor(target_val, dtype=torch.float32).view(-1, 1)
    target_test = torch.tensor(target_test, dtype=torch.float32).view(-1, 1)

    # Dataset e DataLoader
    train_dataset = TensorDataset(train_data, target_train)
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=False) # Shuffle False because of time series data

    # Defining loss, optimizer 
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)

    # Training loop
    n_epochs = 500
    train_curve, val_curve = [], []
    training_loop = tqdm(range(n_epochs)) 
    for epoch in training_loop:
        # Training mode
        model.train() 
        running = 0.0 
        for batch_data, batch_targets in train_loader:
            # Reset computation graph
            optimizer.zero_grad()
            # Forward pass
            outputs = model(batch_data)
            # Compute training loss on batch
            loss = criterion(outputs, batch_targets)
            # Compute gradient 
            loss.backward()
            # Gradient step
            optimizer.step()
            # Loss per batch 
            running += loss.item()
        # Loss per epoch
        train_epoch = running / len(train_loader)

        # Validation loss
        model.eval()
        with torch.no_grad():
            val_outputs = model(val_data)
            val_loss = criterion(val_outputs, target_val).item()

        # Store train and validation loss 
        train_curve.append(train_epoch)
        val_curve.append(val_loss)

        # Print loss in the progression bar
        training_loop.set_postfix(train=train_epoch, val=val_loss, lr=optimizer.param_groups[0]['lr'])

    # Save the final model weights
    torch.save(model.state_dict(), weight_filename)
    # Plot losses
    plt.figure()
    plt.plot(train_curve, label='train')
    plt.plot(val_curve,   label='val')
    plt.yscale('log')  
    plt.legend(); plt.xlabel('epoch'); plt.ylabel('MSE')
    plt.show()			

    # Metrics evaluation 
    model.load_state_dict(torch.load(weight_filename)) 
    model.eval()
    with torch.no_grad():
        predictions = model(test_data).numpy() 
        test_loss = mean_squared_error(target_test.numpy(), predictions)
        test_rmse = np.sqrt(test_loss)
        test_mae = mean_absolute_error(target_test.numpy(), predictions)

    
    return [test_loss, test_rmse, test_mae]

# This function prepares the dataset using a sliding window approach.
def prepare_data_sw(series, cols, window_size):
    """
    Prepare (fractional- or integer-differenced) OHLC features and next-day target using a sliding window approach:
    - The input dataset has dimensions (window_size x 4), where the window size determines the number of past time steps used.
    - Drops rows with NaNs in features or target.
    - Splits into train/val/test by index (70/15/15) and normalizes features and target.
    Note: The first `window_size` values cannot be used for training due to the sliding window mechanism, same for test set,
    reducing the usable dataset size by `window_size`.

    """
    dataset = series[cols].copy()
    # As a target variable we are using the log-return of the next day
    # Meaning we are considering log(P_t+1) - log(P_t) = log(P_t+1 / P_t) = log( 1 + R_t ) circa R_t (using taylor expansion for small R_t)
    target = series['Close_d'].shift(-1) 
    # mask valid rows
    mask = dataset.notna().all(axis=1) & target.notna()
    X = dataset.loc[mask].astype(float).values
    y = target.loc[mask].astype(float).values.reshape(-1, 1)
    
    # Create the new datasets using the sliding window approach: instead of having that n_samples x 4, we have (n_samples - window_size) x (window_size * 4)
    X_sw = np.array([ X[i:i + window_size].flatten()  for i in range(len(X) - window_size + 1) ])
    y_sw = y[window_size-1:]
    
    # Split the data into training, validation, and test sets (70/15/15)
    train_size = int(len(X_sw) * 0.70)
    val_size = int(len(X_sw) * 0.15)

    train_data, val_data = X_sw[:train_size], X_sw[train_size:train_size + val_size]
    target_train, target_val = y_sw[:train_size], y_sw[train_size:train_size + val_size]
    test_data, target_test = X_sw[train_size + val_size:], y_sw[train_size + val_size:]

    # Normalize data
    scaler_data = StandardScaler()
    train_data = scaler_data.fit_transform(train_data)
    val_data, test_data = scaler_data.transform(val_data), scaler_data.transform(test_data)

    
    return train_data, val_data, test_data, target_train, target_val, target_test

def lin_reg(weight_filename, train_set, test_set, target_train, target_test):
    model = LinearRegression()
    model.fit(train_set, target_train)
    # Save the linear regression model
    joblib.dump(model, weight_filename)
    # Make predictions
    pred = model.predict(test_set)
    # Evaluate the model
    rmse = np.sqrt(mean_squared_error(target_test, pred))
    mae = mean_absolute_error(target_test, pred)

    return [rmse, mae]
    
def compare_models(results, metric, mode):
    """
    Compare integer- vs fractional-differenced models across indices.
    DataFrame with per-index comparison and a dict with average improvements:
    - abs_improvement = integer - fractional  (positive => fractional better)
    - pct_improvement = 100 * (integer - fractional) / integer
    """
    if mode == 'loss':
        idx = {"loss":0, "rmse":1, "mae":2}[metric]
    else:
        idx = {"rmse":0, "mae":1}[metric]
    rows = []
    for name in ["sp500", "dax", "nikkei"]:
        base = results[f"{name}_d"][idx]     # integer-diff (baseline)
        cand = results[f"{name}_fd"][idx]    # fractional-diff (candidate)
        rows.append({
            "index": name.upper(),
            f"{metric}_integer": base,
            f"{metric}_fractional": cand,
            "abs_improvement": base - cand,                 # + => fractional better
            "pct_improvement": 100.0 * (base - cand) / base,
        })
    df = pd.DataFrame(rows)
    # Optional: add an average row for improvements
    avg = {
        "index": "AVERAGE",
        "abs_improvement": df["abs_improvement"].mean(),
        "pct_improvement": df["pct_improvement"].mean(),
    }
    return df, avg


def trading_signal_strategy(series, test_set, weight_file, mode, n_features, model):
    """
    Simulates a trading strategy based on model predictions.

    """
    # Ensure inputs are numpy arrays
    actual_price = series['Close'].iloc[-(len(test_set))-1:].values.astype(float)
    predictions = np.empty(len(test_set))

    if model == "neural_network":
        if mode == "sliding_window":
            nhidden = 32
            nn_model = nn.Sequential(nn.Linear(n_features, nhidden),
                        nn.ReLU(),
                        nn.Linear(nhidden, 1))
            nn_model.load_state_dict(torch.load(weight_file))
            nn_model.eval()
            with torch.no_grad():
                test_set = torch.tensor(test_set, dtype=torch.float32)
                predictions = nn_model(test_set).numpy()
        else:
            nhidden = 32
            nn_model = nn.Sequential(nn.Linear(n_features, nhidden),
                                    nn.ReLU(),
                                    nn.Linear(nhidden, 1))
            nn_model.load_state_dict(torch.load(weight_file))
            nn_model.eval()
            with torch.no_grad():
                test_set = torch.tensor(test_set, dtype=torch.float32)
                predictions = nn_model(test_set).numpy()
    else:  # linear regression
        lin_reg = LinearRegression() 
        lin_reg = joblib.load(weight_file)
        predictions = lin_reg.predict(test_set)

    # If instead of predicting the log-close, I predict the log-return
    # Initialize daily profit/loss (PnL)
    daily_pnl = []
    # If predictions[i] > 0 => P_t+1 > P_t  => long position
    # Simulate trading logic
    for i in range(len(test_set)):
        # Today's price and tomorrow's actual price
        today_price = actual_price[i]
        tomorrow_price = actual_price[i + 1]
        
        # Trading decision: long (1) or short (-1)
        position = 1 if predictions[i] > 0 else -1
        
        # Calculate profit/loss for the day
        pnl = position * (tomorrow_price - today_price)
        daily_pnl.append(pnl)
    
    # Calculate cumulative profit/loss
    cumulative_pnl = np.cumsum(daily_pnl)
    
    return cumulative_pnl 

def plot_results_trading_strategy(results, series_name, t_series):
    """
    Disegna in un unico grafico il cumulative return per ciascuna strategia in results.
    """
    # Colori per i modelli
    nn_color = "red"  # Neural networks
    lr_color = "blue"  # Linear regression
    labels_map = {
        "nn_fd_os": "NN • FD (One-step-ahead)",
        "nn_d_os": "NN • D (One-step-ahead)",
        "lr_fd_os": "LR • FD (One-step-ahead)",
        "lr_d_os": "LR • D (One-step-ahead)",
        "nn_fd_sw": "NN • FD (sliding window)",
        "nn_d_sw": "NN • D (sliding window)",
        "lr_fd_sw": "LR • FD (sliding window)",
        "lr_d_sw": "LR • D (sliding window)",
    }
    # Tipi di linea per i dataset
    line_styles = {
        "integer": "-",  # Linea continua
        "fractional": "--"  # Linea tratteggiata
    }
    plt.figure(figsize=(12, 6))
    for key, series in results.items():
        label = labels_map.get(key)
        # Determina il colore in base al modello
        if "nn" in key:  # Neural network strategies
            color = nn_color
        else: 
            color = lr_color        
        # Determina il tipo di linea in base al dataset
        if "_d_" in key:
            linestyle = line_styles["integer"]
        else:
            linestyle = line_styles["fractional"]
        plt.plot(series, label=label, color=color, linestyle=linestyle)
    # Passive investor	
    plt.plot(t_series['Close_d'].iloc[-len(results["nn_d_os"])-1:].to_numpy().cumsum(), label="Passive investor", color="black", linestyle="-", linewidth=2)
    plt.title(f"{series_name}Cumulative Returns from Trading Strategies")
    plt.xlabel("Date")
    plt.ylabel("Cumulative return")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()
