import torch
import argparse
from egnn.models import EGNN_dynamics
from deprecated.eqnode.kernels import compute_gammas
from deprecated.eqnode.flows2 import HutchinsonEstimator, RegularizedHutchinsonEstimator
from deprecated.eqnode.bg import DiffEqFlow, RegularizedDiffEqFlow
from deprecated.eqnode.flows import ContinuousNormalizingFlow
from deprecated.eqnode.densenet import DenseNet
from deprecated.eqnode.dynamics import SchNet, SimpleEqDynamics
from flows.ffjord import FFJORD
from deprecated.eqnode.kernels import RbfEncoder
from deprecated.eqnode.dynamics import KernelDynamics, KernelDynamics_inner, KernelDynamics_inner_old
from deprecated.eqnode.distances import distance_vectors, distances_from_vectors, distance_vectors_v2, diagonal_filter, inner_prods

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Ensures deterministic behavior

def main():

    n_rbfs = 50
    d_max = 16
    n_features = 16
    n_rbfs = 50
    dim = 8
    n_particles = 4

    kernel_mus = torch.linspace(0., 8., n_rbfs)
    kernel_gammas = torch.ones(n_rbfs) * 0.5

    if torch.cuda.is_available():
        kernel_mus = kernel_mus.cuda()
        kernel_gammas = kernel_gammas.cuda()

    rbf_encoder = RbfEncoder(kernel_mus, kernel_gammas.log(), trainable=False)


    simple_dynamics = SimpleEqDynamics(
            transformation=DenseNet([n_rbfs, 64, 32, 1],
                                    activation=torch.nn.Tanh()),
            rbf_encoder=rbf_encoder,
            n_particles=n_particles,
            n_dimension=dim // n_particles,
            n_rbfs=n_rbfs,
        )
    n_dimension = dim // n_particles
    d_max = 8
    n_rbfs = 50
    mus = torch.linspace(0, d_max, n_rbfs)
    mus.sort()
    gammas = 0.5 * torch.ones(n_rbfs)
    mus_time = torch.linspace(0, 1, 10)
    gammas_time = 0.3 * torch.ones(10)

    set_seed(42)
    kernel_dynamics = KernelDynamics(n_particles, n_dimension, mus, gammas,
                                optimize_d_gammas=True,
                                optimize_t_gammas=True,
                                mus_time=mus_time,
                                gammas_time=gammas_time)
    set_seed(42)
    kernel_dynamics_inner = KernelDynamics_inner_old(n_particles, n_dimension, mus, gammas,
                                optimize_d_gammas=True,
                                optimize_t_gammas=True,
                                mus_time=mus_time,
                                gammas_time=gammas_time)
    
    batch_size = 1
    input = torch.randn(batch_size, dim)

    t = 0.5
    output_1 = kernel_dynamics(t, input)
    output_2 = kernel_dynamics_inner(t, input)

    print(output_1, output_2)



if __name__ == '__main__':
    main()