import argparse
import torch
import utils
import wandb
import time
import logging
import numpy as np
from dw4_experiment import losses
from dw4_experiment.dataset import get_data
from dw4_experiment.models import get_model
from flows.distributions import PositionPrior
from flows.utils import remove_mean


def main():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"using devide: {device}")
    prior = PositionPrior()  # set up prior

    flow = get_model(args, dim, n_particles)
    flow = flow.to(device)

    # Log all args to wandb
    wandb.init(entity=args.wandb_usr, project='se3flows', name=args.name, config=args)
    wandb.save('*.txt')
    # logging.write_info_file(model=dynamics, FLAGS=args,
    #                         UNPARSED_ARGV=unparsed_args,
    #                         wandb_log_dir=wandb.run.dir)

    data_train, batch_iter_train = get_data(args, 'train', args.batch_size)
    data_val, batch_iter_val = get_data(args, 'val', batch_size=100)
    data_test, batch_iter_test = get_data(args, 'test', batch_size=100)

    print("Max")
    print(torch.max(data_train))

    # initial training with likelihood maximization on data set
    optim = torch.optim.AdamW(flow.parameters(), lr=args.lr, amsgrad=True,
                              weight_decay=args.weight_decay)
    print(flow)

    best_val_loss = 1e8
    best_test_loss = 1e8

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,  # Set log level to INFO to capture detailed information
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{args.data}_logs/training_log_n_data_{args.n_data}.txt"),  # Log to a file named 'training_log.txt'
            logging.StreamHandler()  # Log to the console
        ]
    )

    for epoch in range(args.n_epochs):
        start_epoch_time = time.time()  # Start time for this epoch
        nll_epoch = []
        flow.set_trace(args.trace)
        
        logging.info(f"Starting Epoch {epoch}/{args.n_epochs}...")
        
        for it, idxs in enumerate(batch_iter_train):
            batch = data_train[idxs]
            assert batch.size(0) == args.batch_size

            batch = batch.view(batch.size(0), n_particles, n_dims)
            batch = remove_mean(batch)

            if args.data_augmentation:
                batch = utils.random_rotation(batch).detach()

            batch = batch.to(device)

            optim.zero_grad()

            # transform batch through flow
            if 'kernel_dynamics' in args.model:
                loss, nll, reg_term, mean_abs_z = losses.compute_loss_and_nll_kerneldynamics(args, flow, prior, batch, n_particles, n_dims)
            else:
                loss, nll, reg_term, mean_abs_z = losses.compute_loss_and_nll(args, flow, prior, batch)
            # standard nll from forward KL

            loss.backward()

            optim.step()

            if it % args.n_report_steps == 0:
                logging.info(f"Epoch: {epoch}, Iter: {it}/{len(batch_iter_train)}, NLL: {nll.item():.4f}, Reg term: {reg_term.item():.3f}")

            nll_epoch.append(nll.item())

        # Log Epoch NLL
        logging.info(f"Epoch {epoch} - Mean Train NLL: {np.mean(nll_epoch):.4f}")

        if epoch % args.test_epochs == 0:
            val_loss = test(args, data_val, batch_iter_val, flow, prior, epoch, partition='val')
            test_loss = test(args, data_test, batch_iter_test, flow, prior, epoch, partition='test')
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_test_loss = test_loss
                save_dir_best = f'saved_models_{args.data}/best_model_n_data_{args.n_data}.pth'
                torch.save(flow.state_dict(), save_dir_best)  
                logging.info(f"Model saved at epoch {epoch} with best validation loss.")
            
            logging.info(f"Best val loss: {best_val_loss:.4f} \t Best test loss: {best_test_loss:.4f}")

        # End time for this epoch
        end_epoch_time = time.time()
        epoch_time = end_epoch_time - start_epoch_time  # Calculate the epoch time
        logging.info(f"Epoch {epoch} completed in {epoch_time:.2f} seconds.")

        logging.info("-" * 50)  # Separator for each epoch

    return best_test_loss


def test(args, data_test, batch_iter_test, flow, prior, epoch, partition='test'):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # use OTD in the evaluation process
    flow._use_checkpoints = False
    if args.data == 'dw4':
        flow.set_trace('exact')

    print('Testing %s partition ...' % partition)
    data_nll = 0.
    # batch_iter = BatchIterator(len(data_smaller), n_batch)
    with torch.no_grad():
        for it, batch_idxs in enumerate(batch_iter_test):
            batch = torch.Tensor(data_test[batch_idxs])
            batch = batch.to(device)
            batch = batch.view(batch.size(0), n_particles, n_dims)
            if 'kernel_dynamics' in args.model:
                loss, nll, reg_term, mean_abs_z = losses.compute_loss_and_nll_kerneldynamics(args, flow, prior, batch,
                                                                                             n_particles, n_dims)
            else:
                loss, nll, reg_term, mean_abs_z = losses.compute_loss_and_nll(args, flow, prior, batch)
            print("\r{}".format(it), nll, end="")
            data_nll += nll.item()
        data_nll = data_nll / (it + 1)

        print()
        print(f'%s nll {data_nll}' % partition)
        # wandb.log({"Test NLL": data_nll}, commit=False)

        # TODO: no evaluation on hold out data yet
    flow.set_trace(args.trace)

    return data_nll

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SE3')
    parser.add_argument('--model', type=str, default='simple_dynamics',
                        help='our_dynamics | schnet | simple_dynamics | kernel_dynamics | kernel_dynamics_inner | egnn_dynamics | gnn_dynamics')
    parser.add_argument('--data', type=str, default='lj13',
                        help='dw4 | lj13')
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--n_data', type=int, default=100,
                        help="Number of training samples")
    parser.add_argument('--sweep_n_data', type=eval, default=False,
                        help="sweep n_data instead of using the provided parameter")
    parser.add_argument('--condition_time', type=eval, default=True,
                        help='True | False')
    parser.add_argument('--trace', type=str, default='hutch',
                        help='hutch | exact')
    parser.add_argument('--tanh', type=eval, default=True,
                        help='use tanh in the coord_mlp')
    parser.add_argument('--hutch_noise', type=str, default='bernoulli',
                        help='gaussian | bernoulli')
    parser.add_argument('--nf', type=int, default=32,
                        help='number of layers')
    parser.add_argument('--n_layers', type=int, default=3,
                        help='number of layers')
    parser.add_argument('--name', type=str, default='debug')
    parser.add_argument('--wandb_usr', type=str, default='')
    parser.add_argument('--n_report_steps', type=int, default=1)
    parser.add_argument('--test_epochs', type=int, default=10)
    parser.add_argument('--attention', type=eval, default=True,
                        help='use attention in the EGNN')
    parser.add_argument('--data_augmentation', type=eval, default=False,
                        help='use data augmentation')
    parser.add_argument('--weight_decay', type=float, default=1e-12,
                        help='use data augmentation')
    parser.add_argument('--ode_regularization', type=float, default=0)
    parser.add_argument('--x_aggregation', type=str, default='sum',
                        help='sum | mean')

    args, unparsed_args = parser.parse_known_args()
    if args.model == 'kernel_dynamics' and args.data == 'lj13':
        args.model = 'kernel_dynamics_lj13'
    print(args)

    if args.data == 'dw4':
        print(args.n_data)
        n_particles = 4
        n_dims = 2
        dim = n_particles * n_dims  # system dimensionality
    elif args.data == 'lj13':
        n_particles = 13
        n_dims = 3
        dim = n_particles * n_dims
    else:
        raise Exception('wrong data partition: %s' % args.data)
    
    if args.sweep_n_data:
        optimal_losses = []

        if args.data == "dw4":
            sweep_n_data = [100, 1000, 10000, 100000]
            epochs = [1000, 300, 50, 6]
            test_epochs = [20, 5, 1, 1]
            batch_sizes = [100, 100, 100, 100]
        elif args.data == "lj13":
            sweep_n_data = [10, 100, 1000, 10000]
            epochs = [500, 1000, 300, 50]
            test_epochs = [10, 25, 5, 1]
            batch_sizes = [10, 100, 100, 100]
        else:
            raise Exception("Not working")
        for n_data, n_epochs, test_epoch, batch_size in zip(sweep_n_data, epochs, test_epochs, batch_sizes):
            args.n_data = n_data
            args.n_epochs = n_epochs
            args.test_epochs = test_epoch
            args.batch_size = batch_size
            print("\n###########################" +
                  ("\nSweeping experiment, number of training samples --> %d \n" % n_data) +
                  "###########################\n")
            nll_loss = main()
            optimal_losses.append(nll_loss)
            print("Optimal losses")
            print(optimal_losses)
            print("\n###########################" +
                  ("\nOptimal test loss for %d training samples --> %.4f \n" % (n_data, nll_loss)) +
                  "###########################\n")
    else:
        main()