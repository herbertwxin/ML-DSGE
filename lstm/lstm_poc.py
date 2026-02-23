import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


# ==========================================
# 1. The Data Generating Process (DGP)
# ==========================================
def generate_structural_break_data(T=10000, break_points=None):
    if break_points is None:
        break_points = range(500, 9500, 500)

    y = np.zeros(T, dtype=np.float32)
    y[0] = 0.0
    # mu is the intercept of the process
    mu = np.zeros(T, dtype=np.float32)
    mu[0] = 1.0
    phi = 0.9  # Fixed stable AR(1) coefficient

    np.random.seed(55)
    epsilon = 0.5 * np.random.randn(T).astype(np.float32)

    for t in range(1, T):
        mu[t] = mu[t - 1]

        if t in break_points:
            # Shift the mean significantly at break points
            mu[t] = np.random.uniform(-10.0, 10.0)

        # Without time trend
        #y[t] = mu[t] + phi * y[t - 1] + epsilon[t]
        # With time trend
        y[t] = mu[t] + phi * y[t - 1] + 0.01 * t + epsilon[t]
    return y, list(break_points)


# ==========================================
# 2. The LSTM Agent
# ==========================================
class LSTMAgent(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=12, output_dim=1):
        super(LSTMAgent, self).__init__()
        self.hidden_dim = hidden_dim
        # PyTorch LSTM: (input_size, hidden_size, num_layers)
        # batch_first=True -> (batch, seq, feature)
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x shape: (batch, seq, feature)
        lstm_out, _ = self.lstm(x)
        # We only care about the last step's output for prediction
        last_step = lstm_out[:, -1, :]
        out = self.fc(last_step)
        return out


# ==========================================
# 3. Training & Simulation
# ==========================================
def run_experiment():
    # Hyperparameters
    T = 10000
    lookback = 5
    hidden_dim = 12
    learning_rate = 0.02

    # Check for Apple Silicon (MPS)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU (MPS) acceleration.")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU.")
    else:
        device = torch.device("cpu")
        print("Using CPU.")

    torch.manual_seed(55)

    data, bps = generate_structural_break_data(T)

    # Normalize
    scale_factor = 100.0
    data_scaled = data / scale_factor

    model = LSTMAgent(input_dim=1, hidden_dim=hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # --- Data Preparation for Phase 1 ---
    # Create sliding windows for the first 5000 points
    train_len = 5000
    X_train = []
    y_train = []

    for i in range(lookback, train_len):
        # Window: [i-lookback ... i-1] -> Target: i
        window = data_scaled[i - lookback : i]
        target = data_scaled[i]
        X_train.append(window.reshape(lookback, 1))
        y_train.append(target)

    X_train = torch.tensor(np.array(X_train), dtype=torch.float32).to(device)
    y_train = (
        torch.tensor(np.array(y_train), dtype=torch.float32).view(-1, 1).to(device)
    )

    # Batching using DataLoader
    batch_size = 64
    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    # --- Phase 1: Offline Training ---
    print(f"Phase 1: Offline Training on first {train_len} points...")
    epochs = 20
    model.train()

    for epoch in tqdm(range(epochs), desc="Training Epochs"):
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

    # --- Phase 2: Online Simulation ---
    print(f"\nPhase 2: Online Simulation (t > {train_len})...")

    forecasts_scaled = np.zeros(T)
    errors = np.zeros(T)

    # For plotting logic later, fill initial part with actuals or 0
    forecasts_scaled[:train_len] = data_scaled[:train_len]

    # OPTIMIZATION FOR M3 ULTRA / GPU:
    # Move the ENTIRE dataset to the GPU once.
    # This prevents moving small 5-item chunks from CPU->GPU 5000 times.
    all_data_gpu = torch.tensor(data_scaled, dtype=torch.float32).to(device).view(-1, 1)

    # Switch to single-instance mode but keep training enabled (model.train())
    # because we want to update weights.

    for t in tqdm(range(train_len, T - 1), desc="Online Sim"):
        # Prepare input: [t-lookback+1 ... t]
        # Current time is t. We want to predict t+1.

        start_idx = t - lookback + 1
        end_idx = t + 1

        # SLICE DIRECTLY ON GPU (No CPU Transfer)
        # shape needs to be (batch=1, seq=lookback, feature=1)
        input_tensor = all_data_gpu[start_idx:end_idx].view(1, lookback, 1)

        # 1. Forecast
        optimizer.zero_grad()
        pred = model(input_tensor)

        # We only move the SCALAR result back to CPU for storage
        pred_item = pred.item()
        forecasts_scaled[t + 1] = pred_item

        # 2. Realization
        # true_val is also on GPU
        true_tensor = all_data_gpu[t + 1].view(1, 1)

        # 3. Error Tracking (Original Scale)
        true_val_item = true_tensor.item()
        errors[t + 1] = (
            (pred_item * scale_factor) - (true_val_item * scale_factor)
        ) ** 2

        # 4. Online Update
        # Calculate loss on this new observation
        loss = criterion(pred, true_tensor)
        loss.backward()
        optimizer.step()

    # --- Visualization ---
    # Save simple plot
    forecasts = forecasts_scaled * scale_factor

    print(f"Data range: {data.min():.3f} to {data.max():.3f}")
    print(f"Forecast range (test phase): {forecasts[train_len:].min():.3f} to {forecasts[train_len:].max():.3f}")
    print(f"Mean Squared Error (test phase): {errors[train_len+1:].mean():.6f}")

    # Specifically target the break points the user is interested in plus some others
    test_bps = [bp for bp in bps if bp > train_len and bp < T - 150]
    # Filter or select specific ones: e.g., 5500, 7000, 8000, 8500
    display_bps = [5500, 7000, 8000, 8500]
    num_zoom_plots = len(display_bps)
    
    plt.figure(figsize=(15, 18))
    
    # Plot several zoom-ins around breaks
    for i, bp in enumerate(display_bps):
        plt.subplot(num_zoom_plots + 1, 1, i + 1)
        window = range(bp - 100, bp + 151)
        plt.plot(window, data[window], label="True Process", color="black", alpha=0.7)
        plt.plot(window, forecasts[window], label="LSTM Forecast", color="red", linestyle="--")
        plt.axvline(x=bp, color="blue", linestyle=":", label="Break")
        plt.title(f"Adaptation at Break t={bp}")
        plt.grid(True, alpha=0.3)
        if i == 0:
            plt.legend()

    # Plot 2: Error
    plt.subplot(num_zoom_plots + 1, 1, num_zoom_plots + 1)
    test_range = range(train_len + 1, T)
    plt.plot(test_range, errors[train_len + 1:], label="Sq Error", color="red", alpha=0.6)

    for bp in test_bps:
        plt.axvline(x=bp, color="blue", alpha=0.2, linestyle="--")

    plt.title("Squared Error (Test Phase)")
    plt.tight_layout()
    plt.savefig("lstm_results.png")
    print("Figure saved as lstm_results.png")


if __name__ == "__main__":
    run_experiment()
