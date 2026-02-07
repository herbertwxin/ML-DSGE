# ==========================================
# LSTM POOR-MAN'S PROOF OF CONCEPT (ANNOTATED)
# This version is designed for learning Python & PyTorch
# ==========================================

# 1. Imports: Bringing in external libraries
# 'import' is like 'using' in Julia.
# 'as' creates an alias (a shortcut) for easier typing.
import matplotlib.pyplot as plt  # For plotting graphs
import numpy as np               # For numerical arrays (like Julia's Arrays)
import torch                     # The main PyTorch library (Neural Networks)
import torch.nn as nn            # Neural Network layers (Linear, LSTM, etc.)
import torch.optim as optim     # Optimization algorithms (Adam, SGD, etc.)
from tqdm import tqdm            # A library for showing progress bars in loops


# ==========================================
# 1. The Data Generating Process (DGP)
# ==========================================
# 'def' defines a function. Notice the colon ':' at the end.
# Arguments can have default values (T=10000).
def generate_structural_break_data(T=10000, break_points=None):
    # 'if ... is None' is a common Python check for optional arguments.
    if break_points is None:
        # range(start, stop, step) generates numbers.
        # This creates breaks every 500 steps.
        break_points = range(500, 9500, 500)

    # np.zeros(T) creates an array of zeros of length T.
    # dtype=np.float32 specifies 32-bit floats (standard for Deep Learning).
    y = np.zeros(T, dtype=np.float32)
    y[0] = 0.0  # Python indexing starts at 0! (Julia starts at 1)
    
    # mu will store the 'mean' or 'intercept' of our process at each point in time
    mu = np.zeros(T, dtype=np.float32)
    mu[0] = 1.0
    
    phi = 0.9  # This is our AR(1) coefficient (how much today depends on yesterday)

    # Set a seed so the 'random' numbers are the same every time we run it.
    np.random.seed(55)
    # Generate random 'noise' (epsilon) from a standard normal distribution.
    epsilon = 0.5 * np.random.randn(T).astype(np.float32)

    # The Loop: 'for t in range(1, T)' goes from 1 to T-1.
    # Note the indentation: everything inside the loop is pushed to the right.
    for t in range(1, T):
        # By default, the mean stays the same as the previous step
        mu[t] = mu[t - 1]

        # Check if current time 't' is one of our 'break points'
        if t in break_points:
            # Significant shift: pick a new random mean between -10 and 10
            mu[t] = np.random.uniform(-10.0, 10.0)

        # The core equation: 
        # y_t = Intercept + (phi * y_{t-1}) + TimeTrend + RandomNoise
        # 0.01 * t makes the data drift upwards over time.
        y[t] = mu[t] + phi * y[t - 1] + 0.01 * t + epsilon[t]
    
    # Return the generated data and the list of breaks.
    return y, list(break_points)


# ==========================================
# 2. The LSTM Agent (The Model)
# ==========================================
# 'class' defines a blueprint for an object.
# 'LSTMAgent(nn.Module)' means our class inherits from PyTorch's base model.
class LSTMAgent(nn.Module):
    # __init__ is the 'Constructor'. It runs once when you create the model.
    # 'self' refers to the specific instance of the model being created.
    def __init__(self, input_dim=1, hidden_dim=12, output_dim=1):
        # 'super' initializes the parent PyTorch class.
        super(LSTMAgent, self).__init__()
        
        # Store the hidden dimension size (the 'memory' size)
        self.hidden_dim = hidden_dim
        
        # Define the LSTM layer:
        # input_dim: how many features per time step (here, 1: just the value of y)
        # hidden_dim: internal complexity of the LSTM
        # batch_first=True: tells PyTorch our data is organized as (Batch, Time, Features)
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        
        # Define the Linear (Dense) layer:
        # It takes the hidden output of the LSTM and turns it into a single prediction.
        self.fc = nn.Linear(hidden_dim, output_dim)

    # 'forward' defines how data flows through the network.
    # This is called automatically when you do 'model(x)'.
    def forward(self, x):
        # x shape: (Batch Size, Sequence Length, Features)
        
        # Pass input through LSTM. 
        # lstm_out contains the 'hidden states' for EVERY time step in the sequence.
        # the '_' represents internal cell states we don't need here.
        lstm_out, _ = self.lstm(x)
        
        # We only want to predict the NEXT value based on the WHOLE sequence.
        # So we take the output of the VERY LAST step in the sequence: [:, -1, :]
        # -1 in Python indexing means 'the last item'.
        last_step_output = lstm_out[:, -1, :]
        
        # Pass that last hidden state through the linear layer to get the final number.
        prediction = self.fc(last_step_output)
        
        return prediction


# ==========================================
# 3. Training & Simulation (The Experiment)
# ==========================================
def run_experiment():
    # --- Configuration ---
    T = 10000          # Total time steps
    lookback = 5       # Look at 5 previous steps to predict the next
    hidden_dim = 12    # Number of 'neurons' in the LSTM
    learning_rate = 0.02

    # --- Device Selection (Hardware) ---
    # This checks if you have a GPU (Apple Silicon or Nvidia) to speed things up.
    if torch.backends.mps.is_available():
        device = torch.device("mps")  # Mac GPU
        print("Using Apple Silicon GPU (MPS) acceleration.")
    elif torch.cuda.is_available():
        device = torch.device("cuda") # Nvidia GPU
        print("Using CUDA GPU.")
    else:
        device = torch.device("cpu")  # Just use CPU
        print("Using CPU.")

    # Fix the PyTorch random seed for reproducibility.
    torch.manual_seed(55)

    # Generate our synthetic data using the function we wrote above.
    data, bps = generate_structural_break_data(T)

    # --- Normalization ---
    # Neural Networks HATE large numbers (like 100).
    # We divide by 100 to keep the data roughly between -1 and 1.
    scale_factor = 100.0
    data_scaled = data / scale_factor

    # Create the model and move it to the device (CPU or GPU).
    model = LSTMAgent(input_dim=1, hidden_dim=hidden_dim).to(device)
    
    # The Optimizer: Adam is a smart version of Gradient Descent.
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # The Criterion: How we measure error. MSE = Mean Squared Error.
    criterion = nn.MSELoss()

    # --- Data Preparation (Phase 1: Offline) ---
    # We will use the first 5000 points to teach the model the basic trend.
    train_len = 5000
    X_train = [] # Features (sequences of 5)
    y_train = [] # Targets (the 6th value)

    # range(lookback, train_len) -> starts at index 5, ends at 4999.
    for i in range(lookback, train_len):
        # Slicing: data[0:5] gives indices 0, 1, 2, 3, 4. (Excludes 5!)
        window = data_scaled[i - lookback : i]
        target = data_scaled[i]
        
        # Reshape to (5, 1) because PyTorch expects (Time, Features).
        X_train.append(window.reshape(lookback, 1))
        y_train.append(target)

    # Convert the lists of NumPy arrays into PyTorch Tensors.
    # Tensors are just 'Smart Arrays' that can live on the GPU.
    X_train = torch.tensor(np.array(X_train), dtype=torch.float32).to(device)
    y_train = torch.tensor(np.array(y_train), dtype=torch.float32).view(-1, 1).to(device)

    # DataLoader: A helper that splits our data into small 'batches' (64 at a time).
    # This is more efficient than looking at 1 point or all 5000 points at once.
    batch_size = 64
    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # --- Phase 1: Offline Training ---
    print(f"Phase 1: Offline Training on first {train_len} points...")
    epochs = 20 # How many times the model sees the entire 5000-point dataset
    model.train() # Put model in 'training mode' (enables weight updates)

    for epoch in tqdm(range(epochs), desc="Training Epochs"):
        # Loop through each batch of 64 examples
        for batch_X, batch_y in dataloader:
            # 1. Reset gradients (forget the errors from the previous batch)
            optimizer.zero_grad()
            
            # 2. Forward Pass: Get the model's current guess
            output = model(batch_X)
            
            # 3. Compute Loss: How far off was the guess?
            loss = criterion(output, batch_y)
            
            # 4. Backward Pass: Calculate how to change weights to fix the error
            loss.backward()
            
            # 5. Optimizer Step: Actually update the weights
            optimizer.step()

    # --- Phase 2: Online Simulation (Real-time Learning) ---
    print(f"\nPhase 2: Online Simulation (t > {train_len})...")
    # This phase simulates a robot living through time, seeing one new point at a time.

    # Arrays to store our results for plotting
    forecasts_scaled = np.zeros(T)
    errors = np.zeros(T)
    # Fill the first part with 'known' data (since we aren't predicting it here)
    forecasts_scaled[:train_len] = data_scaled[:train_len]

    # Optimization: Move all data to GPU once to avoid slow transfers inside the loop.
    all_data_gpu = torch.tensor(data_scaled, dtype=torch.float32).to(device).view(-1, 1)

    # Loop from the end of training to the end of the dataset.
    for t in tqdm(range(train_len, T - 1), desc="Online Sim"):
        # We are at time 't'. We want to predict 't+1'.
        # We look at the sequence: [t-4, t-3, t-2, t-1, t] (length of 5)
        start_idx = t - lookback + 1
        end_idx = t + 1 # Slice is exclusive, so this goes up to index 't'
        
        # Prepare the input sequence directly on the GPU.
        # .view(1, lookback, 1) adds the 'Batch' dimension (Batch=1).
        input_tensor = all_data_gpu[start_idx:end_idx].view(1, lookback, 1)

        # 1. FORECAST: Ask the model for its prediction
        optimizer.zero_grad()
        pred = model(input_tensor)
        
        # .item() converts a 1-element Tensor back into a standard Python float.
        pred_item = pred.item()
        forecasts_scaled[t + 1] = pred_item

        # 2. REALIZATION: Now 'time moves forward' and we see the ACTUAL value.
        true_tensor = all_data_gpu[t + 1].view(1, 1)
        true_val_item = true_tensor.item()

        # 3. TRACK ERROR: Calculate squared error in the ORIGINAL scale (multiply by 100)
        errors[t + 1] = ((pred_item * scale_factor) - (true_val_item * scale_factor)) ** 2

        # 4. ONLINE UPDATE: This is the 'learning' part.
        # Even though we are 'testing', we update the model weights using this new point.
        # This allows the model to adapt to structural breaks as they happen!
        loss = criterion(pred, true_tensor)
        loss.backward()
        optimizer.step()

    # --- Visualization (Generating Graphs) ---
    # Un-scale the forecasts to get back to original values
    forecasts = forecasts_scaled * scale_factor

    print(f"Mean Squared Error (test phase): {errors[train_len+1:].mean():.6f}")

    # Set up the plot (size: 15 inches wide, 18 inches tall)
    plt.figure(figsize=(15, 18))
    
    # We will create 4 'Zoom' plots at specific break points and 1 'Error' plot.
    display_bps = [5500, 7000, 8000, 8500]
    num_zoom_plots = len(display_bps)
    
    for i, bp in enumerate(display_bps):
        # subplot(rows, cols, index)
        plt.subplot(num_zoom_plots + 1, 1, i + 1)
        
        # Focus on a window of 250 points around the break
        window = range(bp - 100, bp + 151)
        
        # Plot the True data vs the LSTM prediction
        plt.plot(window, data[window], label="True Process", color="black", alpha=0.7)
        plt.plot(window, forecasts[window], label="LSTM Forecast", color="red", linestyle="--")
        
        # Draw a dotted blue line at the exact moment of the break
        plt.axvline(x=bp, color="blue", linestyle=":", label="Break")
        
        plt.title(f"Adaptation at Break t={bp}")
        plt.grid(True, alpha=0.3)
        if i == 0: plt.legend()

    # Final plot: The Squared Error over time
    plt.subplot(num_zoom_plots + 1, 1, num_zoom_plots + 1)
    test_range = range(train_len + 1, T)
    plt.plot(test_range, errors[train_len + 1:], label="Sq Error", color="red", alpha=0.6)
    plt.title("Squared Error (Test Phase)")
    
    plt.tight_layout() # Fixes spacing between plots
    plt.savefig("lstm_results_annotated.png") # Save the image
    print("Figure saved as lstm_results_annotated.png")


# This is the entry point of the script. 
# It says: 'If this file is run directly (not imported), then execute run_experiment()'
if __name__ == "__main__":
    run_experiment()
