import torch
import argparse
from dw4_experiment.models import get_model
from flows.distributions import PositionPrior
from dw4_experiment.dataset import get_data
from main_dw4_lj13 import test

def load_and_test_model(args, model_path, n_particles, n_dims):
    """ Loads the model from the given path and runs the test function. """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dim = n_particles * n_dims 


    flow = get_model(args, dim, n_particles).to(device)
    flow.load_state_dict(torch.load(model_path, map_location=device))
    flow.eval() 


    data_test, batch_iter_test = get_data(args, 'test', batch_size=1000)


    prior = PositionPrior()


    test_loss = test(args, data_test, batch_iter_test, flow, prior, epoch=0, partition='test')
    print(f"Test Loss for {model_path}: {test_loss:.4f}")
    
    return test_loss

def main():
    """ Iterates over different values of n_data, loading and testing models. """
    parser = argparse.ArgumentParser(description='SE3 Test')
    parser.add_argument('--model', type=str, default='simple_dynamics',
                        help='our_dynamics | schnet | simple_dynamics | kernel_dynamics | kernel_dynamics_inner | egnn_dynamics | gnn_dynamics')
    parser.add_argument('--n_data_list', type=int, nargs='+', required=True, help='List of n_data values to test')
    parser.add_argument('--trace', type=str, default='hutch', help='hutch | exact')
    parser.add_argument('--data', type=str, default='lj13', help='dw4 | lj13')

    args = parser.parse_args()


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
        f.write(f"\n### Testing Results for Model: {args.model}, Data: {args.data} ###\n")

        for n_data in args.n_data_list:
            if 'inner' in args.model:
                model_path = f"{model_dir}/best_model_inner_n_data_{n_data}.pth"
            else:
                model_path = f"{model_dir}/best_model_n_data_{n_data}.pth"

            args.n_data = n_data
            print(f"\n### Testing model trained with n_data = {n_data} ###")
            test_loss = load_and_test_model(args, model_path, n_particles, n_dims)
            test_losses[n_data] = test_loss

            f.write(f"n_data = {n_data}, Test Loss = {test_loss:.4f}\n")

        # f.write("\n### Summary of Test Losses ###\n")
        # for n_data, loss in test_losses.items():
        #     f.write(f"n_data = {n_data}: Test Loss = {loss:.4f}\n")

    print(f"\nTest results saved to {results_file}")

if __name__ == "__main__":
    main()