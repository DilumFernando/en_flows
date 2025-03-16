from deprecated.eqnode.distances import distance_vectors, distances_from_vectors
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from dw4_experiment.models import get_model
from flows.distributions import PositionPrior
from dw4_experiment.dataset import get_data
from main_dw4_lj13 import test


def load_and_test_model(args, model_path, n_particles, n_dims):
    """ Loads the model from the given path and runs the test function. """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dim = n_particles * n_dims 

    # Load the model
    flow = get_model(args, dim, n_particles).to(device)
    flow.load_state_dict(torch.load(model_path, map_location=device))
    flow.eval() 

    # Load the test data
    data_test, batch_iter_test = get_data(args, 'test', batch_size=1000)

    # Initialize the prior distribution
    prior = PositionPrior()

    # Calculate test loss
    test_loss = test(args, data_test, batch_iter_test, flow, prior, epoch=0, partition='test')
    print(f"Test Loss for {model_path}: {test_loss:.4f}")

    return test_loss

def plot_distance_hist(x, n_particles, n_dimensions, bins=100, xs=None, ys=None):
    if not isinstance(x, torch.Tensor):
        x = torch.Tensor(x)
    x = x.contiguous()
    plt.figure(figsize=(10, 10))
    dists = distances_from_vectors(
        distance_vectors(x.view(-1, n_particles, n_dimensions))
    ).cpu().detach().numpy()
    plot_idx = 1
    for i in range(n_particles):
        for j in range(n_particles - 1):
            plt.subplot(n_particles, n_particles - 1, plot_idx)
            plt.hist(dists[:, i, j], bins=bins, density=True)
            if xs is not None and ys     is not None:
                plt.plot(xs, np.exp(-ys) / xs**(n_dimensions - 1) / 32.6)
            plt.yticks([], [])
            plot_idx += 1


def plot_data(data, n_samples):
    """ Plot the first few samples of data """
    data = data.view(n_samples, 4, 2)  # Reshape if necessary to 4 particles and 2 dimensions
    data = data[:10, :, :]
    for i in range(len(data)):
        plt.scatter(data[i][:, 0].cpu().numpy(), data[i][:, 1].cpu().numpy())
        plt.title(f"Sample {i+1}")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.savefig("samples")  # Save plot as an image
        plt.close() 


def main():
    """ Iterates over different values of n_data, loading and testing models. """
    parser = argparse.ArgumentParser(description='SE3 Test')
    parser.add_argument('--model', type=str, default='simple_dynamics',
                        help='our_dynamics | schnet | simple_dynamics | kernel_dynamics | kernel_dynamics_inner | egnn_dynamics | gnn_dynamics')
    parser.add_argument('--n_data_list', type=int, nargs='+', required=True, help='List of n_data values to test')
    parser.add_argument('--trace', type=str, default='hutch', help='hutch | exact')
    parser.add_argument('--data', type=str, default='lj13', help='dw4 | lj13')

    args = parser.parse_args()

    # Set dataset and model directory
    if args.data == 'dw4':
        n_particles, n_dims = 4, 2
        model_dir = 'saved_models_dw4'
    elif args.data == 'lj13':
        n_particles, n_dims = 13, 3
        model_dir = 'saved_models_lj13'
    else:
        raise ValueError("Invalid dataset choice.")

    test_losses = {}
    results_file = f"test_results_{args.model}_n_data_{args.data}.txt"

    with open(results_file, "a") as f:
        # f.write(f"\n### Testing Results for Model: {args.model}, Data: {args.data} ###\n")

        for n_data in args.n_data_list:
            if 'inner' in args.model:
                model_path = f"{model_dir}/best_model_inner_n_data_{n_data}.pth"
            else:
                model_path = f"{model_dir}/best_model_n_data_{n_data}.pth"

            args.n_data = n_data
            print(f"\n### Testing model trained with n_data = {n_data} ###")
            test_loss = load_and_test_model(args, model_path, n_particles, n_dims)
            test_losses[n_data] = test_loss

            # f.write(f"n_data = {n_data}, Test Loss = {test_loss:.4f}\n")

            # Load test data and plot it
            data_test, _ = get_data(args, 'test', batch_size=1000)
            plot_data(data_test, n_samples=1000)  # Plot first 5 samples

    print(f"\nTest results saved to {results_file}")


if __name__ == "__main__":
    main()
