import torch
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from main_dw4_lj13 import test

from deprecated.eqnode.distances import distance_vectors, distances_from_vectors
from deprecated.eqnode.train_utils import IndexBatchIterator, LossReporter, BatchIterator
from deprecated.eqnode.densenet import DenseNet
from deprecated.eqnode.dynamics import KernelDynamics
#from se3_dynamics.dynamics import OurDynamics

from deprecated.eqnode.flows2 import HutchinsonEstimator
from deprecated.eqnode.kernels import RbfEncoder

from deprecated.eqnode.test_systems import MultiDoubleWellPotential
from deprecated.eqnode.bg import DiffEqFlow
from deprecated.eqnode.particle_utils import remove_mean
from deprecated.eqnode.priors import MeanFreeNormalPrior
from dw4_experiment import losses
from dw4_experiment.dataset import get_data
from dw4_experiment.models import get_model
from flows.distributions import PositionPrior



def load_and_test_model(args, model_path, n_particles, n_dims):
    """ Loads the model from the given path and runs the test function. """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dim = n_particles * n_dims 

    # Load the model
    flow = get_model(args, dim, n_particles).to(device)
    flow.load_state_dict(torch.load(model_path, map_location=device))

    # Load the test data
    data_test, batch_iter_test = get_data(args, 'test', batch_size=1000)

    # Initialize the prior distribution
    # prior = MeanFreeNormalPrior(dim, n_particles)
    # prior = PositionPrior()

    # Calculate test loss
    # test_loss = test(args, data_test, batch_iter_test, flow, prior, epoch=0, partition='test')
    # print(f"Test Loss for {model_path}: {test_loss:.4f}")

    return flow

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


def plot_data(data, energies, save_path):
    """ Plot the first few samples of data """
    data = data.view(-1, 4, 2).cpu().detach().numpy()  # Reshape if necessary to 4 particles and 2 dimensions
    for i in range(len(data)):
        plt.scatter(data[i][:, 0], data[i][:, 1])
        plt.title(f"U(x) : {energies[i]:.2f}")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        sample_path = save_path + f"/Sample_U_{energies[i]:.0f}"
        os.makedirs(os.path.dirname(sample_path), exist_ok=True)
        plt.savefig(sample_path)  # Save plot as an image
        plt.close() 

def dw4_data_and_target():
    # first define system dimensionality and a target energy/distribution
    dim = 8
    n_particles = 4

    # DW parameters
    a = 0.9
    b = -4
    c = 0
    offset = 4

    data, idx = np.load("dw4_experiment/data/dw4-dataidx.npy", allow_pickle=True)
    idx = np.random.choice(len(data), len(data), replace=False)
    data = data.reshape(-1, dim)
    data  = remove_mean(data, n_particles, dim // n_particles)

    # Target energy function
    target = MultiDoubleWellPotential(dim, n_particles, a, b, c, offset)
    return data, target 

def calculate_nll(energy_histogram, bin_edges):
    """
    Calculate the Negative Log-Likelihood (NLL) for a histogram without assuming any specific model.
    
    Parameters:
        hist_data (array): The histogram frequencies (counts per bin).
        bin_edges (array): The edges of the histogram bins.
    
    Returns:
        float: The Negative Log-Likelihood value.
    """
    
    # Total number of observations (sum of all frequencies)
    total_counts = np.sum(energy_histogram)
    
    # Calculate bin widths
    bin_widths = np.diff(bin_edges)
    
    # Normalize histogram to get the probability density
    prob_density = energy_histogram / (total_counts * bin_widths)
    
    # Avoid log(0) by ensuring no bin has zero probability
    prob_density[prob_density == 0] = 1e-16  # small value to avoid log(0)
    
    # Calculate the Negative Log-Likelihood
    nll = -np.sum(energy_histogram * np.log(prob_density)) / total_counts
    
    return nll

def plot_hist(energies_bg, energies_data, min_energy, ax=None, label=None):
    if ax is None:
        fig, ax = plt.subplots()
    # efac = 1 
        ax.hist(energies_bg, bins=100, density=True, range=(min_energy, 50), alpha=0.4, histtype='step', linewidth=1,
                color="r", label="samples");

    ax.hist(energies_data, bins=100, density=True, range=(min_energy, 50),  alpha=0.4, color="g", histtype='step',
            linewidth=4,
            label=label);
    
    # plt.hist(energies_bg, bins=100, density=True, range=(min_energy, 0), alpha=0.4, histtype='step', linewidth=4,
    #          color="b", label="weighted samples", weights=np.exp(-log_w));

    ax.set_xlabel("u(x)", fontsize=10)
    # ax.set_xticks([...])
    # ax.set_yticks([...])
    ax.set_title('Energy Histogram')
    ax.legend(fontsize=10)

def plot_generating_flow(args, data, flow, prior, target, latent, samples, ax=None, label=None, epoch=None):
    # use OTD in the evaluation process
    
    flow._use_checkpoints = False # Testing mode
    flow.eval()
    
    if "kernel" or "new" in args.model:
        x, dlogp = flow(latent, inverse=True)
        x = x.view(samples, -1)
    else:
        sampling = True
        x = flow.reverse(latent)
        x = x.view(samples, -1)
         
    latent = latent.view(samples, -1)
    
    energies_data = target.energy(data[:samples]).numpy()
    energies_bg = target.energy(x).cpu().detach().view(-1).numpy()
    # energies_prior = target.energy(latent).cpu().detach().numpy()
    min_energy = min(energies_data.min(), energies_bg.min())
    # max_energy = max(energies_data.max(), energies_bg.max())

    # log_w = target.energy(x).view(-1) - prior(latent.view(samples, 4, 2)).view(-1) + dlogp.view(-1)
    # log_w = log_w.view(-1).cpu().detach()

    if epoch == None:
        ax.hist(energies_bg, bins=100, density=True, range=(min_energy, 50), alpha=0.4, histtype='step', linewidth=1,
                color="r", label="True samples");
        plot_hist(energies_bg, energies_data, min_energy, ax=ax, label=label)
    else:
        plot_hist(energies_bg, energies_data, min_energy, ax=ax, label=label)

        save_inner_epoch_path = f"generated_flows/{args.model}/n_data_{args.n_data}_{args.model_num}/epoch_{epoch}"
        os.makedirs(os.path.dirname(save_inner_epoch_path), exist_ok=True)
        plt.savefig(save_inner_epoch_path)
        
        plt.close()

def main():
    """ Iterates over different values of n_data, loading and testing models. """
    parser = argparse.ArgumentParser(description='SE3 Test')
    parser.add_argument('--model', type=str, default='simple_dynamics',
                        help='our_dynamics | schnet | simple_dynamics | kernel_dynamics | kernel_dynamics_inner | egnn_dynamics | gnn_dynamics')
    parser.add_argument('--n_data', type=int, default=100, help=100)
    parser.add_argument('--trace', type=str, default='hutch', help='hutch | exact')
    parser.add_argument('--data', type=str, default='lj13', help='dw4 | lj13')
    parser.add_argument('--plot', type=bool, default=False)
    parser.add_argument('--model_num_list', nargs='+', default='0')  

    args = parser.parse_args()

    # Set dataset and model directory
    if args.data == 'dw4':
        n_particles, n_dims = 4, 2
        model_dir = 'saved_models_dw4'
    elif args.data == 'lj13':
        n_particles, n_dims = 13, 3
        model_dir = 'incorrect_models_lj13'
    else:
        raise ValueError("Invalid dataset choice.")

    if not args.plot:
        test_losses = {}
        results_file = f"test_results_{args.model}_n_data_{args.n_data}.txt"

        with open(results_file, "a") as f:
            # f.write(f"\n### Testing Results for Model: {args.model}, Data: {args.data} ###\n")

            for n_data in args.n_data_list:
                if 'inner' in args.model:
                    model_path = f"{model_dir}/best_model_corrected_inner_n_data_{n_data}.pth"
                else:
                    model_path = f"{model_dir}/best_model_n_data_{n_data}.pth"

                args.n_data = n_data
                print(f"\n### Testing model trained with n_data = {n_data} ###")
                test_loss = load_and_test_model(args, model_path, n_particles, n_dims)
                test_losses[n_data] = test_loss

                # f.write(f"n_data = {n_data}, Test Loss = {test_loss:.4f}\n")

                # Load test data and plot it
                data_test, _ = get_data(args, 'test', batch_size=1000)

        print(f"\nTest results saved to {results_file}")
    else:
        print("plotting...")
        fig, ax = plt.subplots()
        prior = PositionPrior()
        samples = 10000
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        latent = prior.sample(size=[samples, 4, 2], device=device)
        model_path = f"{model_dir}/best_model_{args.model}/n_data_{args.n_data}/{args.model_num_list[0]}.pth"
        # saved_models_dir = f"{model_dir}/best_model_{args.model}/n_data_{args.n_data}"
        # saved_models_dir = os.path.join(os.getcwd(), models_path)

        # model_paths = sorted(
            # [
                # os.path.join(saved_models_dir, fname)
                # for fname in os.listdir(saved_models_dir)
                # if os.path.isfile(os.path.join(saved_models_dir, fname))
            # ],
            # key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

        # model_paths = [
            # os.path.join(saved_models_dir, fname)
            # for fname in os.listdir(saved_models_dir)
            # if os.path.isfile(os.path.join(saved_models_dir, fname))]

        # print(model_paths)  

        # for model_path in model_paths:
            # flow = load_and_test_model(args, model_path, n_particles, n_dims)
            # data, target = dw4_data_and_target()
            # label = os.path.basename(model_path)[0]
            # plot_generating_flow(args, data, flow, prior, target, latent, samples, ax=ax, label=label)
        
        flow = load_and_test_model(args, model_path, n_particles, n_dims)
        data, target = dw4_data_and_target()
        label = 'Generated samples - model_{args.model_num_list[0]}'
        plot_generating_flow(args, data, flow, prior, target, latent, samples, ax=ax, label=label)

        save_best_path = f"generated_hists/model_{args.model_num_list[0]}.png"
        os.makedirs(os.path.dirname(save_best_path), exist_ok=True)
        plt.savefig(save_best_path)
        print('done')


if __name__ == "__main__":
    main()
