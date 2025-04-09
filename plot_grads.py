import re
import matplotlib.pyplot as plt

# Read the log file
log_file = "dw4_logs/training_log_normalized_inner_n_data_100_0.txt"  # Replace with the actual path

# Initialize a list to store the gradient norms
gradient_norms = []

# Define a regular expression pattern to capture the gradient norms
pattern = r"Total Grad Norm: (\d+\.\d+)"

# Open and read the log file
with open(log_file, "r") as f:
    for line in f:
        # Search for lines containing the gradient norm
        match = re.search(pattern, line)
        if match:
            # Append the gradient norm to the list
            gradient_norms.append(float(match.group(1)))

# Plot the gradient norms
plt.plot(gradient_norms)
plt.xlabel('Iteration')
plt.ylabel('Gradient Norm')
plt.title('Gradient Norms During Training')
plt.savefig('grad_norms_normalized_inner_100_1')
plt.close()