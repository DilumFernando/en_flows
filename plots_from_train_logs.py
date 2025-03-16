import re
import matplotlib.pyplot as plt

# Paths to the training log files
n_data = 100
data = 'lj13'
log_file_1 = f"{data}_logs/training_log_n_data_{n_data}.txt"  
log_file_2 = f"{data}_logs/training_log_inner_n_data_{n_data}.txt"  
plot_file = f"{data}_images/training_losses_plot_n_data_{n_data}.png"  

# Initialize lists to store extracted data for both log files
epochs_1 = []
mean_train_losses_1 = []

epochs_2 = []
mean_train_losses_2 = []

# Regular expression pattern to match the epoch and mean training NLL
pattern = re.compile(r"Epoch (\d+) - Mean Train NLL: (-?[\d.]+)")

# Read the first log file and extract relevant information
with open(log_file_1, "r") as file:
    for line in file:
        match = pattern.search(line)
        if match:
            epoch = int(match.group(1))
            loss = float(match.group(2))
            epochs_1.append(epoch)
            mean_train_losses_1.append(loss)

# Read the second log file and extract relevant information
with open(log_file_2, "r") as file:
    for line in file:
        match = pattern.search(line)
        if match:
            epoch = int(match.group(1))
            loss = float(match.group(2))
            epochs_2.append(epoch)
            mean_train_losses_2.append(loss)

# Plot the losses from both log files on the same plot
plt.figure(figsize=(8, 5))
plt.plot(epochs_1, mean_train_losses_1, marker='o', linestyle='-', color='b', label='distance based - Mean Train NLL')
plt.plot(epochs_2, mean_train_losses_2, marker='x', linestyle='--', color='r', label='inner product based - Mean Train NLL')
plt.xlabel('Epochs')
plt.ylabel('Negative Log Likelihood (NLL)')
plt.title('Training Loss Comparison over Epochs')
plt.legend()
plt.grid(True)
plt.ylim(bottom=-40)
plt.savefig(plot_file)  # Save plot as an image
plt.close()  # Close the figure to free memory

print(f"Training loss comparison plot saved as {plot_file}")
