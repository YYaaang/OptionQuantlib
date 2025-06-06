#%%
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import time
import os
from scipy.stats import norm
import scipy.optimize as optimize
import QuantLib as ql
from src.QlCalendar import QlCalendar
from src.QlStocks import QlStocks

#%%
# Set random seed to ensure reproducibility
np.random.seed(42)
torch.manual_seed(42)
# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

#%%
n_paths = 2000
S0 = 100.0  # Initial stock price
V0 = 0.04  # Initial volatility
kappa = 2.0  # Mean reversion speed of volatility
theta = 0.04  # Long-term mean of volatility
sigma = 0.3  # Volatility of volatility
rho = -0.7  # Correlation between price and volatility
r = 0.05  # Risk-free rate
T = 1.0  # Option maturity (years)
K = 100.0  # Option strike price
trading_days = 252  # Trading days per year

#%%
# 初始化日历
start_date = ql.Date(1, 1, 2023)
ql_calendar = QlCalendar(init_date=start_date)
# 创建股票实例
ql_stocks = QlStocks(ql_calendar)
ql_stocks.heston([S0] * n_paths, v0=V0, kappa=kappa, theta=theta, rho=rho, sigma=sigma, dividend_rate=0.0)

#%%

# Deep Hedging model - using PyTorch's nn.Module
class DeepHedgingModel(nn.Module):
    def __init__(self, n_features, hidden_dim=64):
        super(DeepHedgingModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2)
        )

        # Use proper initialization methods
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.network(x)
#%%

# Train model with increased epochs (100 epochs)
def train_deep_hedging_model(model, params, n_paths, n_steps, epochs=100, batch_size=64, learning_rate=0.001):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # Updated scheduler for longer training - less aggressive reduction and longer patience
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=8, factor=0.7, verbose=True)

    # For tracking training progress
    loss_history = []
    val_loss_history = []  # Track validation loss
    best_loss = float('inf')
    best_model_state = None
    model.train()

    # Generate fixed validation set (10% of n_paths)
    val_size = int(0.1 * n_paths)
    S_val, V_val, integrated_var_val = simulate_heston(params, val_size, n_steps)
    option_payoff_val = put_option_payoff(S_val[:, -1], params.K)

    start_total_time = time.time()
    for epoch in range(epochs):
        start_time = time.time()

        # Generate new paths for training
        S, V, integrated_var = simulate_heston(params, n_paths, n_steps)

        # Calculate option payoffs
        option_payoff_np = put_option_payoff(S[:, -1], params.K)

        # Divide data into batches
        n_batches = n_paths // batch_size
        epoch_loss = 0

        for batch in range(n_batches):
            start_idx = batch * batch_size
            end_idx = (batch + 1) * batch_size

            # Initialize portfolio value and positions
            portfolio_values = torch.zeros(batch_size, device=device)
            stock_position = torch.zeros(batch_size, device=device)
            var_swap_position = torch.zeros(batch_size, device=device)
            total_cost = torch.tensor(0.0, device=device)

            optimizer.zero_grad()

            for t in range(n_steps):
                curr_t = t * params.T / n_steps
                remaining_T = params.T - curr_t

                # Current state (price, volatility, time, etc.)
                features = torch.cat([
                    torch.tensor(S[start_idx:end_idx, t], dtype=torch.float32).view(batch_size, 1),
                    torch.tensor(V[start_idx:end_idx, t], dtype=torch.float32).view(batch_size, 1),
                    torch.tensor(integrated_var[start_idx:end_idx, t], dtype=torch.float32).view(batch_size, 1),
                    torch.ones(batch_size, 1) * t / n_steps
                ], dim=1).to(device)

                # Predict new trade volumes
                trades = model(features)
                stock_trades = trades[:, 0]
                var_swap_trades = trades[:, 1]

                # Calculate trading costs
                step_cost = calculate_trading_cost(trades)
                total_cost += step_cost

                # Update positions
                stock_position += stock_trades
                var_swap_position += var_swap_trades

                # Calculate new variance swap values according to formula (31)
                var_swap_t = torch.tensor([
                    calculate_variance_swap(
                        integrated_var[start_idx + i, t],
                        V[start_idx + i, t],
                        curr_t,
                        params.T,
                        params
                    ) for i in range(batch_size)
                ], dtype=torch.float32, device=device)

                var_swap_t_plus_dt = torch.tensor([
                    calculate_variance_swap(
                        integrated_var[start_idx + i, t + 1],
                        V[start_idx + i, t + 1],
                        curr_t + params.T / n_steps,
                        params.T,
                        params
                    ) for i in range(batch_size)
                ], dtype=torch.float32, device=device)

                # Update portfolio value (including P&L for this time step)
                stock_pnl = stock_position * torch.tensor(S[start_idx:end_idx, t + 1] - S[start_idx:end_idx, t],
                                                          dtype=torch.float32, device=device)
                var_swap_pnl = var_swap_position * (var_swap_t_plus_dt - var_swap_t)

                portfolio_values += stock_pnl + var_swap_pnl - step_cost

            # Add option payoff, calculate final portfolio value
            option_payoff_batch = torch.tensor(option_payoff_np[start_idx:end_idx], dtype=torch.float32, device=device)
            final_portfolio = portfolio_values - option_payoff_batch

            # Calculate CVaR as loss function - now directly minimizing CVaR
            loss = compute_cvar(final_portfolio)  # We minimize CVaR directly (no negative sign)

            # Backpropagation and optimization
            loss.backward()
            # Gradient clipping to avoid explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / n_batches
        loss_history.append(avg_loss)

        # Validate on fixed validation set
        model.eval()
        with torch.no_grad():
            # Evaluate on validation set
            val_portfolio_values = torch.zeros(val_size, device=device)
            val_stock_position = torch.zeros(val_size, device=device)
            val_var_swap_position = torch.zeros(val_size, device=device)
            val_total_cost = torch.tensor(0.0, device=device)

            for t in range(n_steps):
                curr_t = t * params.T / n_steps
                remaining_T = params.T - curr_t

                val_features = torch.cat([
                    torch.tensor(S_val[:, t], dtype=torch.float32).view(val_size, 1),
                    torch.tensor(V_val[:, t], dtype=torch.float32).view(val_size, 1),
                    torch.tensor(integrated_var_val[:, t], dtype=torch.float32).view(val_size, 1),
                    torch.ones(val_size, 1) * t / n_steps
                ], dim=1).to(device)

                # Process in batches to avoid memory issues
                val_batch_size = 200
                all_trades = []

                for i in range(0, val_size, val_batch_size):
                    end_idx = min(i + val_batch_size, val_size)
                    batch_features = val_features[i:end_idx]
                    batch_trades = model(batch_features)
                    all_trades.append(batch_trades)

                # Combine results from all batches
                val_trades = torch.cat(all_trades, dim=0)
                val_stock_trades = val_trades[:, 0]
                val_var_swap_trades = val_trades[:, 1]

                # Calculate trading costs
                val_step_cost = calculate_trading_cost(val_trades)
                val_total_cost += val_step_cost

                # Update positions
                val_stock_position += val_stock_trades
                val_var_swap_position += val_var_swap_trades

                # Calculate new variance swap values
                val_var_swap_t = torch.tensor([
                    calculate_variance_swap(
                        integrated_var_val[i, t],
                        V_val[i, t],
                        curr_t,
                        params.T,
                        params
                    ) for i in range(val_size)
                ], dtype=torch.float32, device=device)

                val_var_swap_t_plus_dt = torch.tensor([
                    calculate_variance_swap(
                        integrated_var_val[i, t + 1],
                        V_val[i, t + 1],
                        curr_t + params.T / n_steps,
                        params.T,
                        params
                    ) for i in range(val_size)
                ], dtype=torch.float32, device=device)

                # Update portfolio value
                val_stock_pnl = val_stock_position * torch.tensor(S_val[:, t + 1] - S_val[:, t],
                                                                  dtype=torch.float32, device=device)
                val_var_swap_pnl = val_var_swap_position * (val_var_swap_t_plus_dt - val_var_swap_t)

                val_portfolio_values += val_stock_pnl + val_var_swap_pnl - val_step_cost

            # Final portfolio value (minus option payoff)
            val_option_payoff = torch.tensor(option_payoff_val, dtype=torch.float32, device=device)
            val_final_portfolio = val_portfolio_values - val_option_payoff

            # Calculate validation loss - directly compute CVaR
            val_loss = compute_cvar(val_final_portfolio)
            val_loss_history.append(val_loss.item())

        # Back to training mode
        model.train()

        # Save best model
        if val_loss.item() < best_loss:
            best_loss = val_loss.item()
            best_model_state = model.state_dict().copy()
            print(f"Epoch {epoch + 1}: New best model with validation loss: {best_loss:.4f}")

        # Use learning rate scheduler
        scheduler.step(val_loss.item())

        # Display training progress - more frequent updates for longer training
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Val Loss: {val_loss.item():.4f}, Time: {time.time() - start_time:.2f}s")

        # Every 25 epochs, show more detailed progress
        if (epoch + 1) % 25 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            elapsed_time = time.time() - start_total_time
            estimated_total = elapsed_time * (epochs / (epoch + 1))
            estimated_remaining = estimated_total - elapsed_time
            print(f"Progress: {epoch + 1}/{epochs} ({(epoch + 1) / epochs * 100:.1f}%)")
            print(f"Current learning rate: {current_lr:.6f}")
            print(f"Time elapsed: {elapsed_time / 60:.1f} minutes")
            print(f"Estimated remaining: {estimated_remaining / 60:.1f} minutes")

    # Load the best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model with validation loss: {best_loss:.4f}")

    # Total training time
    total_training_time = time.time() - start_total_time
    print(f"Total training time: {total_training_time / 60:.2f} minutes")

    return loss_history, val_loss_history


# Evaluate model
def evaluate_model(model, params, n_paths=1000, n_steps=30):
    model.eval()  # Set to evaluation mode
    print("\nStarting Deep Hedging Model evaluation...")

    # Generate new paths for evaluation
    print(f"Generating {n_paths} paths with {n_steps} steps for evaluation...")
    S, V, integrated_var = simulate_heston(params, n_paths, n_steps)

    # Calculate option payoffs
    option_payoff_np = put_option_payoff(S[:, -1], params.K)

    # Initialize
    portfolio_values = np.zeros(n_paths)
    stock_positions = np.zeros((n_paths, n_steps + 1))
    var_swap_positions = np.zeros((n_paths, n_steps + 1))
    trading_costs = np.zeros(n_paths)

    # Set up progress monitoring
    print(f"Running Deep Hedging Model on paths...")
    progress_interval = max(1, n_steps // 5)  # Update every 20% of time steps
    start_time = time.time()

    with torch.no_grad():  # Don't calculate gradients during evaluation
        for t in range(n_steps):
            if t % progress_interval == 0 or t == n_steps - 1:
                elapsed = time.time() - start_time
                progress = (t + 1) / n_steps * 100
                print(f"  Progress: {progress:.1f}% - Step {t + 1}/{n_steps}, Time: {elapsed:.1f}s")

            curr_t = t * params.T / n_steps
            remaining_T = params.T - curr_t

            # Feature vectors
            features = torch.cat([
                torch.tensor(S[:, t], dtype=torch.float32).view(n_paths, 1),
                torch.tensor(V[:, t], dtype=torch.float32).view(n_paths, 1),
                torch.tensor(integrated_var[:, t], dtype=torch.float32).view(n_paths, 1),
                torch.ones(n_paths, 1) * t / n_steps
            ], dim=1).to(device)

            # Process in batches to avoid memory issues
            batch_size = 200
            all_trades = []

            for i in range(0, n_paths, batch_size):
                end_idx = min(i + batch_size, n_paths)
                batch_features = features[i:end_idx]
                batch_trades = model(batch_features)
                all_trades.append(batch_trades)

            # Combine results from all batches
            trades = torch.cat(all_trades, dim=0).cpu().numpy()
            stock_trades = trades[:, 0]
            var_swap_trades = trades[:, 1]

            # Calculate trading costs
            cost = 0.001 * np.sum(np.abs(trades), axis=1)
            trading_costs += cost

            # Update positions
            stock_positions[:, t + 1] = stock_positions[:, t] + stock_trades
            var_swap_positions[:, t + 1] = var_swap_positions[:, t] + var_swap_trades

            # Calculate variance swap values
            var_swap_t = np.array([
                calculate_variance_swap(
                    integrated_var[i, t],
                    V[i, t],
                    curr_t,
                    params.T,
                    params
                ) for i in range(n_paths)
            ])

            var_swap_t_plus_dt = np.array([
                calculate_variance_swap(
                    integrated_var[i, t + 1],
                    V[i, t + 1],
                    curr_t + params.T / n_steps,
                    params.T,
                    params
                ) for i in range(n_paths)
            ])

            # Update portfolio value
            stock_pnl = stock_positions[:, t + 1] * (S[:, t + 1] - S[:, t])
            var_swap_pnl = var_swap_positions[:, t + 1] * (var_swap_t_plus_dt - var_swap_t)

            portfolio_values += stock_pnl + var_swap_pnl - cost

    # Final portfolio value (minus option payoff)
    final_portfolio = portfolio_values - option_payoff_np

    # Calculate statistics
    mean_pnl = np.mean(final_portfolio)
    std_pnl = np.std(final_portfolio)

    # Calculate VaR and CVaR - correctly as positive values for losses
    alpha = 0.05
    sorted_pnl = np.sort(final_portfolio)
    var_index = int(np.ceil(alpha * n_paths)) - 1
    var_index = max(0, var_index)
    var = max(0, -sorted_pnl[var_index])
    cvar = max(0, -np.mean(sorted_pnl[:var_index + 1]))

    total_time = time.time() - start_time
    print(f"Deep Hedging evaluation completed in {total_time:.2f} seconds")
    print(f"Deep Hedging Evaluation Results:")
    print(f"Mean P&L: {mean_pnl:.4f}")
    print(f"P&L Standard Deviation: {std_pnl:.4f}")
    print(f"VaR(95%): {var:.4f}")
    print(f"CVaR(95%): {cvar:.4f}")
    print(f"Average Trading Cost: {np.mean(trading_costs):.4f}")

    return final_portfolio, stock_positions, var_swap_positions
#%%

#%%

#%%

#%%

