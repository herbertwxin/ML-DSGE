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
    mu = np.zeros(T, dtype=np.float32)
    mu[0] = 1.0
    phi = 0.9

    np.random.seed(55)
    epsilon = 0.5 * np.random.randn(T).astype(np.float32)

    for t in range(1, T):
        mu[t] = mu[t - 1]
        if t in break_points:
            mu[t] = np.random.uniform(-10.0, 10.0)
        y[t] = mu[t] + phi * y[t - 1] + 0.01 * t + epsilon[t]
    return y, list(break_points)


# ==========================================
# 2. Stateful LSTM Agent (The Model)
# ==========================================
class StatefulLSTMAgent(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=12, output_dim=1):
        super(StatefulLSTMAgent, self).__init__()
        self.hidden_dim = hidden_dim
        # batch_first=True means we use (Batch, Sequence, Features)
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
        # 'hidden_state' will store the (h, c) tensors between calls.
        # This is what makes it "Stateful".
        self.hidden_state = None

    def reset_state(self):
        """Manually clear the 'memory' of the LSTM."""
        self.hidden_state = None

    def forward(self, x):
        # x shape: (Batch Size, Sequence Length, Features)
        
        # CRITICAL FOR STATEFUL: 
        # When training on long sequences, we 'detach' the hidden state from the 
        # previous calculation. This stops PyTorch from trying to calculate 
        # gradients all the way back to the beginning of time (t=0).
        if self.hidden_state is not None:
            h_0, c_0 = self.hidden_state
            self.hidden_state = (h_0.detach(), c_0.detach())
        
        # We pass the input AND the previous hidden_state into the LSTM
        lstm_out, self.hidden_state = self.lstm(x, self.hidden_state)
        
        # Pass the result through the final linear layer
        out = self.fc(lstm_out)
        return out


# ==========================================
# 3. Training & Simulation
# ==========================================
def run_experiment():
    T = 10000
    hidden_dim = 12
    learning_rate = 0.01 # Slightly lower for stateful online learning stability
    train_len = 5000

    if torch.backends.mps.is_available():
        # device = torch.device("mps") # Skipping MPS for stateful LSTM due to known issues with hidden states
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    torch.manual_seed(55)
    data, bps = generate_structural_break_data(T)

    # Normalize
    scale_factor = 100.0
    data_scaled = data / scale_factor

    model = StatefulLSTMAgent(input_dim=1, hidden_dim=hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # --- Phase 1: Offline Training (Sequential) ---
    # In stateful training, we don't shuffle! Order matters because the model
    # remembers what happened in the previous batch.
    print(f"Phase 1: Offline Training on first {train_len} points...")
    
    # We split the 5000 points into chunks of 50. 
    # The LSTM carries its memory from chunk 1 to chunk 2.
    chunk_size = 50 
    epochs = 40 

    model.train()
    for epoch in tqdm(range(epochs), desc="Training Epochs"):
        model.reset_state() # Start each epoch with a clean slate
        
        for i in range(0, train_len - 1, chunk_size):
            optimizer.zero_grad()
            
            # Sequence: Today's values
            # Target: Tomorrow's values (offset by 1)
            end_idx = min(i + chunk_size, train_len - 1)
            
            seq_x = torch.tensor(data_scaled[i:end_idx], dtype=torch.float32).view(1, -1, 1).to(device)
            seq_y = torch.tensor(data_scaled[i+1:end_idx+1], dtype=torch.float32).view(1, -1, 1).to(device)
            
            output = model(seq_x)
            loss = criterion(output, seq_y)
            loss.backward()
            optimizer.step()
            
    # --- Phase 2: Online Simulation (Stateful) ---
    # This simulates "Living through time" where we only see one new point at a time.
    print(f"\nPhase 2: Online Simulation (t > {train_len})...")
    
    forecasts_scaled = np.zeros(T)
    errors = np.zeros(T)
    forecasts_scaled[:train_len] = data_scaled[:train_len]

    # Warm up: Give the model all the training data at once so its 
    # hidden state matches the end of the training period.
    model.eval() 
    model.reset_state()
    with torch.no_grad():
        warmup_x = torch.tensor(data_scaled[:train_len], dtype=torch.float32).view(1, -1, 1).to(device)
        _ = model(warmup_x)

    # Online loop: Process one step at a time
    model.train() 
    all_data_gpu = torch.tensor(data_scaled, dtype=torch.float32).to(device).view(-1, 1)

    for t in tqdm(range(train_len, T - 1), desc="Online Sim"):
        # 1. FORECAST: Predict t+1 using the single value at time t.
        # The model uses its internal 'hidden_state' to remember t-1, t-2, etc.
        optimizer.zero_grad()
        
        current_val = all_data_gpu[t].view(1, 1, 1) # Just one point!
        
        pred = model(current_val) 
        
        pred_item = pred.item()
        forecasts_scaled[t + 1] = pred_item

        # 2. REALIZATION: See the actual value
        true_tensor = all_data_gpu[t + 1].view(1, 1, 1)
        true_val_item = true_tensor.item()

        # 3. TRACK ERROR
        errors[t + 1] = ((pred_item * scale_factor) - (true_val_item * scale_factor)) ** 2

        # 4. ONLINE UPDATE: Learn from the mistake immediately
        loss = criterion(pred, true_tensor)
        loss.backward()
        optimizer.step()

    # --- Visualization ---
    forecasts = forecasts_scaled * scale_factor
    print(f"Mean Squared Error (test phase): {errors[train_len+1:].mean():.6f}")

    plt.figure(figsize=(15, 18))
    display_bps = [5500, 7000, 8000, 8500]
    num_zoom_plots = len(display_bps)
    
    for i, bp in enumerate(display_bps):
        plt.subplot(num_zoom_plots + 1, 1, i + 1)
        window = range(bp - 100, bp + 151)
        plt.plot(window, data[window], label="True Process", color="black", alpha=0.7)
        plt.plot(window, forecasts[window], label="Stateful LSTM Forecast", color="green", linestyle="--")
        plt.axvline(x=bp, color="blue", linestyle=":", label="Break")
        plt.title(f"Adaptation at Break t={bp} (Stateful)")
        plt.grid(True, alpha=0.3)
        if i == 0: plt.legend()

    plt.subplot(num_zoom_plots + 1, 1, num_zoom_plots + 1)
    test_range = range(train_len + 1, T)
    plt.plot(test_range, errors[train_len + 1:], label="Sq Error", color="red", alpha=0.6)
    plt.title("Squared Error (Test Phase)")
    plt.tight_layout()
    plt.savefig("lstm_stateful_results.png")
    print("Figure saved as lstm_stateful_results.png")

if __name__ == "__main__":
    run_experiment()
