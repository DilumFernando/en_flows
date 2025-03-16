import torch
import numpy as np

def double_well_potential(x, a=1.0, b=0.0, c=0.0):
    """
    Compute the double well potential energy for a single particle.
    
    Parameters:
    - x: The position of the particle.
    - a: The parameter controlling the steepness of the potential.
    - b: The parameter controlling the location of the wells.
    - c: The offset parameter of the potential.
    
    Returns:
    - energy: The potential energy for the particle at position x.
    """
    return a * (x**2 - b)**2 + c


def compute_normalized_nll(samples, a=1.0, b=0.0, c=0.0, k_B_T=1.0):
    """
    Compute the Normalized Negative Log Likelihood for the generated samples based on the 
    double well potential energy function using the Boltzmann distribution.
    
    Parameters:
    - samples: A tensor of shape (n_samples, num_particles, dimensions) containing the generated samples.
    - a, b, c: Parameters for the double well potential energy function.
    - k_B_T: Temperature scaled by the Boltzmann constant (default is 1.0).
    
    Returns:
    - nll: The normalized negative log likelihood of the generated samples.
    """
    energies = []

    # Loop over all generated samples and compute energies for each particle
    for sample in samples:
        energies.append(double_well_potential(sample, a, b, c))

    # Stack all energies into a single tensor and flatten it
    energies = torch.cat(energies, dim=0)
    
    n_samples = len(energies)

    # Move the tensor to CPU before performing any numpy operation
    energies_cpu = energies.cpu()
    
    # Calculate NLL using the formula
    avg_energy = torch.mean(energies_cpu)  # Average energy
    partition_function = torch.sum(torch.exp(-energies_cpu / k_B_T)).cpu().numpy()  # Partition function approximation
    
    # Use numpy for partition function calculation
    nll = (1 / n_samples) * ((1 / k_B_T) * avg_energy + n_samples * np.log(partition_function))

    return nll


# Example usage
if __name__ == "__main__":
    # Generate some sample data (replace this with your flow model-generated samples)
    n_samples = 100000
    dim = 2  # Number of dimensions
    num_particles = 4  # Number of particles
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    
    # Random samples as an example
    samples = torch.randn(n_samples, num_particles, dim).to(device)
    a = 0.9
    b = -4
    c = 0
    
    # Compute the Negative Log Likelihood for the generated samples
    nll_value = compute_normalized_nll(samples, a, b, c, k_B_T=1.0)
    
    print(f"Negative Log Likelihood (NLL): {nll_value.item()}")
