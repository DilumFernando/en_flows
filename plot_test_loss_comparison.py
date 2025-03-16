# import re
# import matplotlib.pyplot as plt

# # Paths to log files
# log_file_1 = "lj_13_logs/training_log_100_epochs.txt"  
# log_file_2 = "lj_13_logs/training_log_inner_100_epochs.txt"  
# plot_file = "lj_13_best_test_loss_plot_100_epochs.png"  

# # Initialize lists
# epochs = []
# best_test_losses = []

# # Regex patterns
# epoch_pattern = re.compile(r"Epoch (\d+) - Mean Train NLL")  # To capture the epoch number
# best_test_loss_pattern = re.compile(r"Best val loss: [-\d.]+\s+Best test loss: ([-\d.]+)")  # Extract best test loss

# # Function to extract best test losses
# def extract_best_test_losses(log_file):
#     current_epoch = None  # Store the last seen epoch number

#     with open(log_file, "r") as file:
#         for line in file:
#             epoch_match = epoch_pattern.search(line)
#             if epoch_match:
#                 current_epoch = int(epoch_match.group(1))  # Store epoch from last match

#             test_loss_match = best_test_loss_pattern.search(line)
#             if test_loss_match and current_epoch is not None:
#                 test_loss = float(test_loss_match.group(1))
#                 epochs.append(current_epoch)
#                 best_test_losses.append(test_loss)
#                 # print(f"Extracted -> Epoch: {current_epoch}, Best Test Loss: {test_loss}")  # Debug print

#     return epochs, best_test_losses
# # Extract from both logs
# epochs_1, best_test_losses_1 = extract_best_test_losses(log_file_1)
# epochs_2, best_test_losses_2 = extract_best_test_losses(log_file_2)

# print(epochs_1)
# # Check if we got any data
# if not epochs:
#     print("No matches found. Check log format and regex.")
# else:
#     # Plot the best test losses over epochs
#     plt.figure(figsize=(8, 5))
#     plt.plot(epochs_1, best_test_losses_1, marker='o', linestyle='-', color='g', label='distance based Best Test Loss')
#     # plt.plot(epochs_2, best_test_losses_2, marker='x', linestyle='--', color='r', label='inner product based Best Test Loss')
#     plt.xlabel('Epochs')
#     plt.ylabel('Best Test Loss')
#     plt.title('Best Test Loss Over Epochs')
#     plt.legend()
#     plt.grid(True)
#     plt.savefig(plot_file)  
#     plt.close()  
#     print(f"Best test loss plot saved as {plot_file}")

import re
import matplotlib.pyplot as plt

# Paths to log files
n_data = 100
log_file_1 = f"training_log_n_data_{n_data}.txt"  
log_file_2 = f"training_log_inner_n_data_{n_data}.txt"  
plot_file = f"dw4_images/training_losses_plot_n_data_{n_data}.png"  

# Dictionary to store unique epochs and their best test losses

# Regex patterns
epoch_pattern = re.compile(r"Epoch (\d+) - Mean Train NLL")  # Match epoch numbers
best_test_loss_pattern = re.compile(r"Best val loss: [-\d.]+\s+Best test loss: ([-\d.]+)")  # Extract best test loss

# Function to extract best test losses
def extract_best_test_losses(log_file):
    epoch_to_test_loss = {}
    current_epoch = None  # Store last seen epoch number

    with open(log_file, "r") as file:
        for line in file:
            epoch_match = epoch_pattern.search(line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))  # Store current epoch

            test_loss_match = best_test_loss_pattern.search(line)
            if test_loss_match and current_epoch is not None:
                test_loss = float(test_loss_match.group(1))
                
                # Only record if epoch isn't already in dictionary
                if current_epoch not in epoch_to_test_loss:
                    epoch_to_test_loss[current_epoch] = test_loss
                    print(f"Extracted -> Epoch: {current_epoch}, Best Test Loss: {test_loss}")  # Debug print
    return epoch_to_test_loss
# # Extract from both logs
epoch_to_test_loss_1 = extract_best_test_losses(log_file_1)
epoch_to_test_loss_2 = extract_best_test_losses(log_file_2)

# Sort epochs before plotting
sorted_epochs_1 = sorted(epoch_to_test_loss_1.keys())
sorted_test_losses_1 = [epoch_to_test_loss_1[epoch] for epoch in sorted_epochs_1]

sorted_epochs_2 = sorted(epoch_to_test_loss_2.keys())
sorted_test_losses_2 = [epoch_to_test_loss_2[epoch] for epoch in sorted_epochs_2]


# Plot the best test losses over epochs
plt.figure(figsize=(8, 5))
plt.plot(sorted_epochs_1, sorted_test_losses_1, marker='o', linestyle='-', color='g', label='distance based Best Test Loss')
plt.plot(sorted_epochs_2, sorted_test_losses_2, marker='x', linestyle='--', color='r', label='inner product based Best Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Best Test Loss')
plt.title('Best Test Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.savefig(plot_file)  
plt.close()  
print(f"Best test loss plot saved as {plot_file}")
